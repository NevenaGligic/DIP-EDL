import torch
import torch.nn as nn
import os
import importlib.util

# Load ModifiedEvidentialNet directly from Re-EDL repo by file path to avoid
# module cache collisions with the ICLR2024-REDL version (same module name,
# different API — Re-EDL adds configurable kl_c and multiple loss types).
_REEDL_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "Re-EDL",
    "code_classical", "models", "ModifiedEvidentialN.py"
))

# We also need the architecture modules that ModifiedEvidentialN imports.
# Temporarily insert Re-EDL's code_classical onto sys.path for that import,
# then remove it so it doesn't affect other imports.
import sys as _sys
_REEDL_ROOT = os.path.abspath(os.path.join(os.path.dirname(_REEDL_MODEL_PATH), ".."))
_sys.path.insert(0, _REEDL_ROOT)
_spec = importlib.util.spec_from_file_location("reedl_model", _REEDL_MODEL_PATH)
_reedl_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reedl_module)
_sys.path.pop(0)

ModifiedEvidentialNet = _reedl_module.ModifiedEvidentialNet


class ReEDL(nn.Module):
    """
    Wraps ModifiedEvidentialNet (Re-EDL, TPAMI 2025) for use as a baseline.

    Re-EDL differs from R-EDL in two key hyperparameter choices identified by
    Chen et al. (2025) as essential:
      - lamb2: raised to 0.8 for CIFAR-10 (0.1 for MNIST) — larger prior
        concentration improves calibration.
      - kl_c=0.0: KL regularization is removed (found to be nonessential and
        sometimes harmful).

    The forward pass returns alpha = evidence + lamb2, compatible with the
    standard EDL evaluation pipeline (run_benchmark).
    """

    def __init__(self,
                 num_classes=10,
                 input_dims=(1, 28, 28),
                 architecture='conv',
                 batch_size=128,
                 lr=1e-3,
                 lamb1=1.0,
                 lamb2=0.8,
                 kl_c=0.0,
                 fisher_c=0.0,
                 kernel_dim=5,
                 hidden_dims=[64, 64, 64],
                 clf_type='softplus',
                 seed=123):
        super().__init__()

        self.model = ModifiedEvidentialNet(
            input_dims=input_dims,
            output_dim=num_classes,
            architecture=architecture,
            hidden_dims=hidden_dims,
            kernel_dim=kernel_dim,
            batch_size=batch_size,
            lr=lr,
            loss='MEDL',
            clf_type=clf_type,
            fisher_c=fisher_c,
            kl_c=kl_c,
            lamb1=lamb1,
            lamb2=lamb2,
            seed=seed,
        )

        self.num_classes = num_classes
        self.input_dims = input_dims
        self.min_alpha = lamb2  # alpha = evidence + lamb2, so lamb2 is the minimum

        # ModifiedEvidentialNet only creates self.scheduler for architecture='conv'.
        # Their train.py calls model.scheduler.step() unconditionally, so add a
        # no-op scheduler for architectures that skip it (e.g. 'linear', 'vgg').
        if not hasattr(self.model, 'scheduler'):
            class _NoOpScheduler:
                def step(self): pass
            self.model.scheduler = _NoOpScheduler()

    def forward(self, x):
        """Returns alpha (Dirichlet parameters) compatible with run_benchmark."""
        return self.model(x, return_output='alpha', compute_loss=False)

    def load_weights(self, weight_path, device='cpu'):
        """
        Load weights from a checkpoint file (local or Hugging Face).
        Supports both {'model_state_dict': ...} format and plain state_dict.
        """
        from ..utils import get_weight_path
        full_path = get_weight_path(weight_path)
        checkpoint = torch.load(full_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        print(f"Loaded Re-EDL weights from {full_path}")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
