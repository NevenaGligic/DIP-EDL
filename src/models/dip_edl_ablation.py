import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import transform_for_maf, get_weight_path
from nflows.flows.autoregressive import MaskedAutoregressiveFlow

# --- DAEDL density estimation utilities ---
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))
from density_estimation import fit_gda, gmm_forward

from src.models.dip_edl import WideResNet, StandardCNN


class dip_EDL(nn.Module):
    """
    DIP-EDL ablation variant for decomposing model components.

    self.task selects which evidence formula is used in forward():
      '1a' : N only              (prior count)
      '1b' : density only        (p(x))
      '1c' : classifier only     (P(y|x))
      '2a' : N × density
      '2b' : N × classifier
      '2c' : density × classifier
      '3'  : full model (default)
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
        self.task = '3'  # set externally before evaluation
        self.min_alpha = 1.0

        if input_dims == (1, 28, 28):
            self.flat_dim = int(np.prod(input_dims))
            self.dataset_mode = 'mnist'
            self.cnn = StandardCNN(num_classes, input_dims)
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
        else:
            raise ValueError(f"Unsupported input_dims: {input_dims}")

        self.register_buffer('log_prob_mean', torch.tensor(0.0))
        self.register_buffer('log_prob_std', torch.tensor(1.0))

    def fit_density(self, train_loader, device):
        """Fits GDA density estimator (CIFAR-10 only)."""
        if self.dataset_mode != 'cifar10':
            print("fit_density called but model is not in GDA mode (MNIST?). Skipping.")
            return

        print("Fitting GDA density estimator for ablation DIP-EDL...")
        self.cnn.eval()
        self.cnn.to(device)

        self.gda, p_z_train = fit_gda(
            self.cnn,
            train_loader,
            self.num_classes,
            self.embedding_dim,
            device
        )

        mean_lp = p_z_train.mean().item()
        std_lp = p_z_train.std().item()
        self.log_prob_mean.fill_(mean_lp)
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)

        print(f"GDA fitted. Stats: mean={mean_lp:.4f}, std={std_lp:.4f}")

    def calibrate_density(self, loader, device):
        """Z-score calibration for MAF density estimator (MNIST only)."""
        if self.dataset_mode == 'cifar10':
            print("--- Calibration skipped for GDA (handled in fit_density) ---")
            return

        print("--- Calibrating MAF density for ablation DIP-EDL ---")
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
        mean_lp = np.mean(all_lp)
        std_lp = np.std(all_lp)
        self.log_prob_mean.fill_(mean_lp)
        self.log_prob_std.fill_(std_lp if std_lp > 1e-6 else 1.0)

        print(f"   Train Stats: Mean {mean_lp:.4f} | Std {std_lp:.4f}")

    def forward(self, x):
        task = self.task

        logits = self.cnn(x)
        probs = F.softmax(logits, dim=1)

        if self.dataset_mode == 'cifar10':
            if self.gda is None:
                likelihood = torch.ones(x.size(0), 1, device=x.device)
            else:
                with torch.no_grad():
                    log_probs_class = self.gda.log_prob(self.cnn.feature[:, None, :])
                    log_prob = torch.logsumexp(log_probs_class, dim=-1)
                log_prob_scaled = (log_prob - self.log_prob_mean) / self.log_prob_std
                likelihood = torch.exp(log_prob_scaled).unsqueeze(1)
        else:
            x_maf = transform_for_maf(x)
            log_prob = self.maf.log_prob(x_maf)
            log_prob_scaled = (log_prob - self.log_prob_mean) / self.log_prob_std
            likelihood = torch.exp(log_prob_scaled).unsqueeze(1)

        if task == '1a':
            evidence = torch.full_like(probs, self.n)
        elif task == '1b':
            evidence = likelihood.expand_as(probs)
        elif task == '1c':
            evidence = probs
        elif task == '2a':
            evidence = self.n * likelihood.expand_as(probs)
        elif task == '2b':
            evidence = self.n * probs
        elif task == '2c':
            evidence = likelihood * probs
        else:  # '3' — full model
            evidence = self.n * likelihood * probs

        return evidence + 1

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
        if self.dataset_mode == 'mnist':
            maf_path = get_weight_path(maf_filename)
            self.maf.load_state_dict(torch.load(maf_path, map_location=device))
            self.maf.to(device)
            self.maf.eval()
            print(f"[Weights] MAF loaded from {maf_filename}")
        else:
            print("Note: GDA mode — no MAF weights to load.")

    def load_pretrained_weights(self, cnn_filename, maf_filename, device='cpu'):
        self.load_cnn_weights(cnn_filename, device)
        self.load_maf_weights(maf_filename, device)
