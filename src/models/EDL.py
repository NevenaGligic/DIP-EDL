import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import get_weight_path


class EDL(nn.Module):
    """
    EDL backbone supporting MNIST, CIFAR-10, and LAMOST.

    - MNIST:    LeNet-style architecture (2 conv layers + FC).
    - CIFAR-10: WideResNet-28-10 with spectral normalization.
    - LAMOST:   1-D Conv multi-branch network.
    """
    def __init__(self, num_classes=10, input_dims=(1, 28, 28), cifar_backbone='wrn28_10'):
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = tuple(input_dims)
        self.min_alpha = 1.0
        self.feature = None

        if self.input_dims == (1, 28, 28):
            self.mode = 'lenet'
            filters = [20, 50]
            hidden_units = 500
            flatten_dim = 50 * 5 * 5

            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, filters[0], kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(filters[0], filters[1], kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(flatten_dim, hidden_units),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_units, self.num_classes)
            )

        elif self.input_dims == (3, 32, 32):
            self.mode = 'wrn'
            from .dip_edl import WideResNet
            self.backbone = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)

        elif len(self.input_dims) == 2:
            self.mode = '1dcnn'
            from .spectral_backbone import Conv1dMultiBranchNet
            self.backbone_1d = Conv1dMultiBranchNet(
                input_shape=self.input_dims[1],
                num_classes=self.num_classes,
            )

        else:
            raise ValueError(f"EDL: Unsupported input_dims {input_dims}")

    def forward(self, x):
        if self.mode == 'lenet':
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            logits = self.fc_layers(x)
        elif self.mode == '1dcnn':
            logits = self.backbone_1d(x)
            self.feature = self.backbone_1d.feature
        else:  # wrn
            logits = self.backbone(x)
            self.feature = self.backbone.feature

        evidence = F.relu(logits)
        alpha = evidence + 1
        return alpha

    def load_edl_weights(self, edl_filename, device='cpu'):
        edl_path = get_weight_path(edl_filename)
        self.load_state_dict(torch.load(edl_path, map_location=device))
        self.to(device)
        self.eval()
        print(f"[Weights] EDL loaded from {edl_filename}")

    def calculate_uncertainties(self, x):
        with torch.no_grad():
            alpha = self.forward(x)
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
            uncertainty = self.num_classes / S.squeeze()
            return probs, uncertainty


# ----------------------------------------
# EDL Loss Function
# ----------------------------------------
def KL_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_beta_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_beta_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg_S_alpha = torch.digamma(S_alpha)
    dg_alpha = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg_alpha - dg_S_alpha), dim=1, keepdim=True) + ln_beta_alpha + ln_beta_beta
    return kl


def EDL_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """MSE + variance + KL annealing loss from Sensoy et al. 2018."""
    alpha = output
    evidence = alpha - 1
    y = F.one_hot(target, num_classes=num_classes)
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S

    mse_loss = torch.sum((y - p)**2, dim=1)
    variance_loss = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device)
    )
    alpha_tilde = (evidence * (1 - y)) + 1
    kl_reg = annealing_coef * KL_divergence(alpha_tilde, num_classes, device=device)

    total_loss = mse_loss + variance_loss + kl_reg.squeeze()
    return torch.mean(total_loss)
