import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import transform_for_maf, get_weight_path
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
import torchvision.models as models
import torch.nn.utils.spectral_norm as spectral_norm

# --- DAEDL density estimation utilities ---
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))
from density_estimation import fit_gda, gmm_forward


class WideBasicBlock(nn.Module):
    """Pre-activation residual block for WideResNet."""
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = spectral_norm(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False))
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    WideResNet-28-10 backbone for CIFAR-10.
    Spectral norm on all conv/linear layers; .feature populated on every forward pass.
    """
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6
        nch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = spectral_norm(nn.Conv2d(3, nch[0], kernel_size=3, stride=1, padding=1, bias=False))
        self.layer1 = self._make_layer(n, nch[0], nch[1], stride=1, dr=dropout_rate)
        self.layer2 = self._make_layer(n, nch[1], nch[2], stride=2, dr=dropout_rate)
        self.layer3 = self._make_layer(n, nch[2], nch[3], stride=2, dr=dropout_rate)
        self.bn = nn.BatchNorm2d(nch[3])
        self.fc = spectral_norm(nn.Linear(nch[3], num_classes))
        self.feature_dim = nch[3]  # 640 for widen_factor=10
        self.feature = None

    def _make_layer(self, n, in_planes, out_planes, stride, dr):
        layers = [WideBasicBlock(in_planes, out_planes, stride, dr)]
        for _ in range(n - 1):
            layers.append(WideBasicBlock(out_planes, out_planes, 1, dr))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        self.feature = out  # (B, 640) — required by fit_gda
        return self.fc(out)

    def get_features(self, x):
        self.forward(x)
        return self.feature


class StandardCNN(nn.Module):
    """
    ResNet-18 backbone adapted for small images (MNIST: 28×28, single channel).
    Spectral norm on all conv/linear layers; .feature populated on every forward pass.
    """
    def __init__(self, num_classes=10, input_dims=(1, 28, 28)):
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        channels = self.input_dims[0]

        self.backbone = models.resnet18(weights=None)

        # Replace Conv7×7 (stride 2) + MaxPool with Conv3×3 (stride 1) to keep
        # spatial resolution on small 28×28 inputs.
        self.backbone.conv1 = spectral_norm(nn.Conv2d(
            channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        ))
        self.backbone.maxpool = nn.Identity()

        def apply_sn(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                return spectral_norm(m)
            return m

        self.backbone.layer1.apply(apply_sn)
        self.backbone.layer2.apply(apply_sn)
        self.backbone.layer3.apply(apply_sn)
        self.backbone.layer4.apply(apply_sn)

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = spectral_norm(nn.Linear(num_ftrs, self.num_classes))

        self.feature = None  # populated on forward — required by fit_gda

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        self.feature = x
        return self.backbone.fc(x)

    def get_features(self, x):
        self.forward(x)
        return self.feature


class StandardCNN1d(nn.Module):
    """
    1-D backbone for spectral data (e.g. LAMOST).
    .feature is populated on every forward pass for GDA fitting.
    """
    def __init__(self, num_classes: int = 2, input_dims: tuple = (1, 556)):
        super().__init__()
        self.num_classes = num_classes
        from .spectral_backbone import Conv1dMultiBranchNet
        self._net = Conv1dMultiBranchNet(
            input_shape=input_dims[1],
            num_classes=num_classes,
        )
        self.feature = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._net(x)
        self.feature = self._net.penultimate_feature  # 32-dim penultimate features
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        self.forward(x)
        return self.feature


class dip_EDL(nn.Module):
    """
    DIP-EDL: Density-Informed Prior for Evidential Deep Learning.

    alpha = 1 + N * gamma * p(x) * P(y|x)

    Backbone:
      - MNIST:    ResNet-18 (StandardCNN) + MAF density on pixel space
      - CIFAR-10: WideResNet-28-10 + GDA density on backbone features
      - LAMOST:   Conv1d (StandardCNN1d) + GDA density on penultimate features
    """
    def __init__(self,
                 num_classes=10,
                 input_dims=(1, 28, 28),
                 maf_hidden_features=1024,
                 maf_num_layers=10,
                 maf_blocks=20,
                 n_train_samples=50000):

        super().__init__()
        self.n = float(n_train_samples)
        self.input_dims = tuple(input_dims)
        self.num_classes = num_classes
        self.min_alpha = 1.0

        if input_dims == (1, 28, 28):
            self.dataset_mode = 'mnist'
            self.cnn = StandardCNN(num_classes, input_dims)
            self.flat_dim = int(np.prod(input_dims))
            self.maf = MaskedAutoregressiveFlow(
                features=self.flat_dim,
                hidden_features=maf_hidden_features,
                num_layers=maf_num_layers,
                num_blocks_per_layer=maf_blocks,
                batch_norm_within_layers=True,
                batch_norm_between_layers=True
            )
        elif input_dims == (3, 32, 32):
            self.dataset_mode = 'cifar10'
            self.cnn = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)
            self.embedding_dim = self.cnn.feature_dim  # 640
            self.gda = None
        elif len(input_dims) == 2:
            # 1-D spectral data (LAMOST): GDA on 32-dim penultimate features.
            self.dataset_mode = 'lamost'
            self.cnn = StandardCNN1d(num_classes, input_dims)
            self.embedding_dim = self.cnn._net.dense[-1].in_features  # 32-dim
            self.gda = None
        else:
            raise ValueError(f"Unsupported input_dims: {input_dims}")

        self.register_buffer('log_prob_mean', torch.tensor(0.0))
        self.register_buffer('log_prob_std', torch.tensor(1.0))

        # Scaling factor for evidence: N * gamma * p(x) * P(y|x).
        # Default 1.0. Set externally for ablation experiments.
        self.gamma = 1.0

        # Noise std for density corruption ablation.
        # 0.0 = clean density (default). float('inf') = pure random density.
        self.density_noise_std = 0.0

    def fit_density(self, train_loader, device):
        """Fits GDA on backbone features (CIFAR-10 and LAMOST)."""
        if self.dataset_mode == 'mnist':
            print("fit_density skipped (MNIST uses MAF, not GDA).")
            return

        print("Fitting GDA density estimator...")
        self.cnn.eval()
        self.cnn.to(device)

        self.gda, p_z_train = fit_gda(
            self.cnn, train_loader, self.num_classes, self.embedding_dim, device
        )

        marginal_lp = p_z_train
        mean_lp = marginal_lp.mean().item()
        std_lp = marginal_lp.std().item()
        self.log_prob_mean.fill_(mean_lp)
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)

        print(f"GDA fitted. Stats: mean={mean_lp:.4f}, std={std_lp:.4f}")

    def calibrate_density(self, loader, device):
        """Z-score calibration for the MAF density estimator (MNIST only)."""
        if self.dataset_mode != 'mnist':
            return

        print("Calibrating MAF density for DIP-EDL (MNIST)...")
        self.maf.eval()
        self.cnn.eval()

        log_probs = []
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                x = x.to(device)
                maf_input = transform_for_maf(x)
                lp = self.maf.log_prob(maf_input)
                log_probs.append(lp.cpu().numpy())
                if i > 200:
                    break

        all_lp = np.concatenate(log_probs)
        mean_lp = float(np.mean(all_lp))
        std_lp = float(np.std(all_lp))
        self.log_prob_mean.fill_(mean_lp)
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)
        print(f"Calibration done. mean={mean_lp:.4f}, std={std_lp:.4f}")

    def forward(self, x):
        logits = self.cnn(x)
        probs = F.softmax(logits, dim=1)

        if self.dataset_mode == 'cifar10':
            if self.gda is None:
                return torch.exp(logits) + 1e-6
            with torch.no_grad():
                log_probs_class = self.gda.log_prob(self.cnn.feature[:, None, :])
                log_prob = torch.logsumexp(log_probs_class, dim=-1)
        elif self.dataset_mode == 'lamost':
            if self.gda is None:
                return torch.exp(logits) + 1e-6
            with torch.no_grad():
                log_probs_class = self.gda.log_prob(self.cnn.feature[:, None, :])
                log_prob = torch.logsumexp(log_probs_class, dim=-1)
        else:
            # MNIST: MAF on pixel space
            x_maf = transform_for_maf(x, self.dataset_mode)
            log_prob = self.maf.log_prob(x_maf)

        # Z-score normalization
        log_prob_scaled = (log_prob - self.log_prob_mean) / self.log_prob_std

        # Density corruption (ablation only)
        if self.density_noise_std == float('inf'):
            log_prob_scaled = torch.randn_like(log_prob_scaled)
        elif self.density_noise_std > 0.0:
            log_prob_scaled = log_prob_scaled + torch.randn_like(log_prob_scaled) * self.density_noise_std

        likelihood = torch.exp(log_prob_scaled).unsqueeze(1)

        evidence = self.n * self.gamma * likelihood * probs
        alpha = evidence + 1
        return alpha

    def calculate_uncertainties(self, x):
        with torch.no_grad():
            alpha = self.forward(x)
            S = torch.sum(alpha, dim=1, keepdim=True)
            prediction = alpha / S
            variance = alpha * (S - alpha) / (S * S * (S + 1))
            epistemic = torch.sum(variance, dim=1)
            aleatoric = -torch.sum(prediction * torch.log(prediction + 1e-9), dim=1)
            K = alpha.shape[1]
            total_u = K / S.squeeze()
            return prediction, epistemic, aleatoric, total_u

    def load_cnn_weights(self, cnn_filename, device='cpu'):
        cnn_path = get_weight_path(cnn_filename)
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        self.cnn.to(device)
        self.cnn.eval()
        print(f"[Weights] CNN loaded from {cnn_filename}")

    def load_maf_weights(self, maf_filename, device='cpu'):
        """Loads MAF weights (MNIST only)."""
        if self.dataset_mode != 'mnist':
            print("Note: GDA mode — no MAF weights to load.")
            return
        maf_path = get_weight_path(maf_filename)
        self.maf.load_state_dict(torch.load(maf_path, map_location=device))
        self.maf.to(device)
        self.maf.eval()
        print(f"[Weights] MAF loaded from {maf_filename}")

    def load_pretrained_weights(self, cnn_filename, maf_filename, device='cpu'):
        self.load_cnn_weights(cnn_filename, device)
        self.load_maf_weights(maf_filename, device)
