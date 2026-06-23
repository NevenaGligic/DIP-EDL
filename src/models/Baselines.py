"""
Deep Ensembles and MC Dropout baselines using torch-uncertainty.

Both models output EDL-compatible pseudo-alpha values so they plug directly
into run_benchmark without any changes to the evaluation pipeline.

Uncertainty mapping:
    mean_probs = mean softmax over estimators          shape [batch, K]
    H          = predictive entropy                    shape [batch]
    norm_H     = H / log(K)  ∈ [0, 1]                 (0=certain, 1=maximally uncertain)
    S          = K / (norm_H + eps)
    alpha      = mean_probs * S[:,None]

This satisfies:
    alpha / sum(alpha) = mean_probs          (correct probabilities)
    K / sum(alpha)     = norm_H              (uncertainty compatible with K/S metric)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from tqdm import tqdm
from torch_uncertainty.models import deep_ensembles, mc_dropout


# ---------------------------------------------------------------------------
# Backbone architectures
# ---------------------------------------------------------------------------

class _LeNetBackbone(nn.Module):
    """LeNet-style backbone for MNIST (1x28x28), matching EDL's architecture."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(50 * 5 * 5, 500),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(500, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class _ResNetBackbone(nn.Module):
    """ResNet-18 backbone for CIFAR-10 (3x32x32) — kept for reference/fallback."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class _WRNBackbone(nn.Module):
    """WideResNet-28-10 backbone for CIFAR-10 (3x32x32), with spectral norm."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        from .dip_edl import WideResNet
        self._wrn = WideResNet(depth=28, widen_factor=10,
                               dropout_rate=dropout_rate, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._wrn(x)


class _Conv1dBackbone(nn.Module):
    """1-D CNN backbone for spectral data, e.g. LAMOST.

    Wraps Conv1dMultiBranchNet (PyTorch port of CNN_Model_1D) so that
    ensemble / dropout models share the same architecture as EDL and nu_EDL.
    dropout_rate > 0 adds Dropout in the dense head for MC Dropout.
    """

    def __init__(self, num_classes: int = 2, input_dims: tuple = (1, 556),
                 dropout_rate: float = 0.5):
        super().__init__()
        from .spectral_backbone import Conv1dMultiBranchNet
        self._net = Conv1dMultiBranchNet(
            input_shape=input_dims[1],
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ---------------------------------------------------------------------------
# Utility: convert mean softmax probs to EDL-compatible pseudo-alpha
# ---------------------------------------------------------------------------

def _probs_to_pseudo_alpha(mean_probs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Map mean softmax probabilities to pseudo-alpha via predictive entropy.

    alpha / S  == mean_probs      (correct normalised probabilities)
    K / S      == normalised H    (uncertainty ∈ [0, 1], 0=certain, 1=uniform)
    """
    eps = 1e-8
    H = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)      # [B]
    H_max = torch.log(torch.tensor(float(num_classes), device=mean_probs.device))
    norm_H = (H / (H_max + eps)).clamp(min=eps, max=1.0)                 # [B]

    S = num_classes / norm_H                                              # [B]
    alpha = mean_probs * S.unsqueeze(1)                                   # [B, K]
    return alpha


# ---------------------------------------------------------------------------
# Deep Ensembles
# ---------------------------------------------------------------------------

class DeepEnsembleModel(nn.Module):
    """Deep Ensemble using torch-uncertainty.

    Trains ``num_estimators`` independent models; at inference averages their
    softmax predictions and converts the result to pseudo-alpha.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_dims: tuple = (1, 28, 28),
        num_estimators: int = 5,
        cifar_backbone: str = 'wrn28_10',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.num_estimators = num_estimators
        self.min_alpha = 1e-6

        if len(input_dims) == 2:                  # 1-D spectral (e.g. LAMOST)
            backbone = _Conv1dBackbone(num_classes, input_dims, dropout_rate=0.0)
        elif input_dims[0] == 1:                  # MNIST
            backbone = _LeNetBackbone(num_classes, dropout_rate=0.0)
        elif cifar_backbone == 'wrn28_10':        # CIFAR-10 — WideResNet-28-10
            backbone = _WRNBackbone(num_classes, dropout_rate=0.0)
        else:                                     # CIFAR-10 — original ResNet-18
            backbone = _ResNetBackbone(num_classes, dropout_rate=0.0)

        # deep_ensembles randomly reinitialises copies when given a single model
        self.ensemble = deep_ensembles(
            backbone,
            num_estimators=num_estimators,
            reset_model_parameters=True,
        )

    def _ensemble_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return [M, B, K] logits with unambiguous layout (no reshape assumptions)."""
        return torch.stack([m(x) for m in self.ensemble.core_models], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._ensemble_logits(x)        # [M, B, K]
        probs = F.softmax(logits, dim=-1)        # [M, B, K]
        mean_probs = probs.mean(dim=0)           # [B, K]
        return _probs_to_pseudo_alpha(mean_probs, self.num_classes)

    # ------------------------------------------------------------------
    # Weight I/O helpers (one file per ensemble member)
    # ------------------------------------------------------------------

    def save_member_weights(self, path_prefix: str, save_fn=None):
        """Save each member to ``{path_prefix}_{i}.pth``."""
        for i, member in enumerate(self.ensemble.core_models):
            path = f"{path_prefix}_{i}.pth"
            if save_fn:
                save_fn(member, path)
            else:
                torch.save(member.state_dict(), path)

    def predict_probs(self, x: torch.Tensor):
        """Return mean softmax probs and epistemic uncertainty (mutual information).

        Returns:
            mean_probs: np.ndarray [batch, K]
            mi:         np.ndarray [batch]  — mutual information (epistemic uncertainty)
        """
        eps = 1e-8
        logits = self._ensemble_logits(x)          # [M, B, K]
        probs = F.softmax(logits, dim=-1)          # [M, B, K]
        mean_probs = probs.mean(dim=0)             # [B, K]

        H_mean = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)   # [B]
        H_members = -torch.sum(probs * torch.log(probs + eps), dim=-1)          # [M, B]
        mi = (H_mean - H_members.mean(dim=0)).clamp(min=0.0)                    # [B]

        return mean_probs.cpu().numpy(), mi.cpu().numpy()

    def load_weights(self, path_prefix: str, device):
        """Load each member from ``{path_prefix}_{i}.pth``."""
        from ..utils import get_weight_path
        for i, member in enumerate(self.ensemble.core_models):
            path = f"{path_prefix}_{i}.pth"
            full_path = get_weight_path(path)
            member.load_state_dict(torch.load(full_path, map_location=device))
            member.eval()
            print(f"Loaded ensemble member {i} from {full_path}")


# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------

class MCDropoutModel(nn.Module):
    """MC Dropout using torch-uncertainty.

    Keeps dropout active at inference time and averages ``num_estimators``
    stochastic forward passes, then converts to pseudo-alpha.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_dims: tuple = (1, 28, 28),
        num_estimators: int = 50,
        dropout_rate: float = 0.5,
        cifar_backbone: str = 'wrn28_10',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.num_estimators = num_estimators
        self.min_alpha = 1e-6

        if len(input_dims) == 2:                  # 1-D spectral (e.g. LAMOST)
            backbone = _Conv1dBackbone(num_classes, input_dims, dropout_rate=dropout_rate)
        elif input_dims[0] == 1:                  # MNIST
            backbone = _LeNetBackbone(num_classes, dropout_rate=dropout_rate)
        elif cifar_backbone == 'wrn28_10':        # CIFAR-10 — WideResNet-28-10
            backbone = _WRNBackbone(num_classes, dropout_rate=dropout_rate)
        else:                                     # CIFAR-10 — original ResNet-18
            backbone = _ResNetBackbone(num_classes, dropout_rate=dropout_rate)

        # on_batch=True tiles the batch → single GPU kernel, faster
        self.mc_model = mc_dropout(
            backbone,
            num_estimators=num_estimators,
            on_batch=True,
        )

    def _mc_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return [M, B, K] logits with dropout guaranteed active.

        Sets the entire core model to eval (so BatchNorm uses running stats), then
        selectively flips only Dropout layers back to train mode so they fire on
        every pass.  Original per-module training state is restored afterwards.
        """
        core = self.mc_model.core_model
        prev_states = {m: m.training for m in core.modules()}
        core.eval()
        for m in core.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        try:
            return torch.stack([core(x) for _ in range(self.num_estimators)], dim=0)
        finally:
            for m, state in prev_states.items():
                m.train(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._mc_logits(x)              # [M, B, K]
        probs = F.softmax(logits, dim=-1)        # [M, B, K]
        mean_probs = probs.mean(dim=0)           # [B, K]
        return _probs_to_pseudo_alpha(mean_probs, self.num_classes)

    # ------------------------------------------------------------------
    # Weight I/O helpers (single core model file)
    # ------------------------------------------------------------------

    def predict_probs(self, x: torch.Tensor):
        """Return mean softmax probs and epistemic uncertainty (mutual information).

        Dropout is guaranteed active via _mc_logits (core model forced to train
        mode), so each call produces M independent stochastic forward passes.

        Returns:
            mean_probs: np.ndarray [batch, K]
            mi:         np.ndarray [batch]  — mutual information (epistemic uncertainty)
        """
        eps = 1e-8
        logits = self._mc_logits(x)                # [M, B, K]
        probs = F.softmax(logits, dim=-1)          # [M, B, K]
        mean_probs = probs.mean(dim=0)             # [B, K]

        H_mean = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)   # [B]
        H_members = -torch.sum(probs * torch.log(probs + eps), dim=-1)          # [M, B]
        mi = (H_mean - H_members.mean(dim=0)).clamp(min=0.0)                    # [B]

        return mean_probs.cpu().numpy(), mi.cpu().numpy()

    def save_weights(self, path: str, save_fn=None):
        if save_fn:
            save_fn(self.mc_model.core_model, path)
        else:
            torch.save(self.mc_model.core_model.state_dict(), path)

    def load_weights(self, path: str, device):
        from ..utils import get_weight_path
        full_path = get_weight_path(path)
        state = torch.load(full_path, map_location=device)
        self.mc_model.core_model.load_state_dict(state)
        self.mc_model.core_model.to(device)
        print(f"Loaded MC Dropout weights from {full_path}")


# ---------------------------------------------------------------------------
# Baseline benchmark (Deep Ensembles / MC Dropout)
# ---------------------------------------------------------------------------

def run_benchmark_baseline(
    model: nn.Module,
    test_loader_id,
    ood_loaders: dict,
    device: torch.device,
    num_classes: int,
):
    """Evaluate DE or MC Dropout without any EDL/Dirichlet assumptions.

    Uncertainty is measured via mutual information (epistemic uncertainty),
    which is the natural analogue of EDL's K/S vacuity:

        MI = H[mean_probs] - (1/M) Σ_m H[probs_m]

    MI is high when ensemble members disagree (OOD) and low when they agree
    (confident ID prediction), matching the role of K/S in the EDL pipeline.

    Uses the same metric functions as the EDL benchmark:
        - calculate_classification_accuracy
        - calculate_brier_score  (ID and OOD)
        - calculate_ood_auroc    (with MI as the uncertainty score)

    Returns the same tuple as experiments.run_benchmark for drop-in compatibility:
        id_accuracy, id_brier, ood_results, all_id_probs, all_id_targets
    """
    from src.metrics import (
        calculate_classification_accuracy,
        calculate_brier_score,
        calculate_ood_auroc,
    )

    model.eval()

    # --- 1. ID inference ---
    all_id_probs, all_id_mi, all_id_targets = [], [], []

    print("Collecting ID Data (probs + MI)...")
    with torch.no_grad():
        for data, target in tqdm(test_loader_id, desc="ID Inference"):
            probs, mi = model.predict_probs(data.to(device))
            all_id_probs.append(probs)
            all_id_mi.append(mi)
            all_id_targets.append(target.numpy())

    all_id_probs   = np.concatenate(all_id_probs,   axis=0)
    all_id_mi      = np.concatenate(all_id_mi,      axis=0)
    all_id_targets = np.concatenate(all_id_targets, axis=0)

    # --- 2. ID metrics ---
    id_accuracy = calculate_classification_accuracy(all_id_probs, all_id_targets)
    id_brier    = calculate_brier_score(all_id_probs, all_id_targets, num_classes, is_ood=False)
    print(f"Accuracy ID: {id_accuracy}")
    print(f"Brier ID:    {id_brier}")

    # --- 3. OOD evaluation ---
    ood_results = {}

    for ood_name, ood_loader in ood_loaders.items():
        all_ood_probs, all_ood_mi = [], []

        print(f"Collecting OOD Data for {ood_name}...")
        with torch.no_grad():
            for data, _ in tqdm(ood_loader, desc=f"OOD Inference ({ood_name})"):
                probs, mi = model.predict_probs(data.to(device))
                all_ood_probs.append(probs)
                all_ood_mi.append(mi)

        all_ood_probs = np.concatenate(all_ood_probs, axis=0)
        all_ood_mi    = np.concatenate(all_ood_mi,    axis=0)

        # AUROC/AUPR: MI as epistemic uncertainty score (higher = more OOD)
        auroc, aupr = calculate_ood_auroc(all_id_mi, all_ood_mi)

        # OOD Brier: how far the model's probs are from the uniform distribution
        dummy_targets = np.zeros(all_ood_probs.shape[0], dtype=int)
        ood_brier = calculate_brier_score(all_ood_probs, dummy_targets, num_classes, is_ood=True)

        ood_results[ood_name] = {'AUROC': auroc, 'AUPR': aupr, 'OOD_Brier': ood_brier}

    return id_accuracy, id_brier, ood_results, all_id_probs, all_id_targets
