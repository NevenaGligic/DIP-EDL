import torch
import torch.nn as nn
import os
import sys

# --- Import DAEDL Modules ---
# Point to the root of the DAEDL repo
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))

# Import their official functions
from utility import load_model  # <--- Using their loader
from density_estimation import fit_gda, gmm_forward

class DAEDL(nn.Module):
    """
    Wrapper for Density Aware Evidential Deep Learning.
    
    Combines the backbone (loaded via DAEDL's utility.load_model)
    with the GDA density estimator.
    """
    
    def __init__(self, 
                 num_classes=10, 
                 dataset='MNIST',
                 input_dims=(1, 28, 28),
                 dropout_rate=0.5,
                 device='cuda'):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device

        print(f"DAEDL Wrapper: Loading backbone for {dataset}...")
        # For LAMOST the feature length is determined by the PCA + feature
        # extraction pipeline, so we read it from input_dims at init time.
        input_len = input_dims[1] if (len(input_dims) == 2) else None

        self.model = load_model(
            ID_dataset=dataset,
            pretrained=False,
            index=0,
            dropout_rate=dropout_rate,
            device=device,
            num_classes=num_classes,
            input_len=input_len,
        )

        # Infer embedding dimension via a dummy forward pass.
        # All DAEDL backbones (ConvLinSeq, VGG, ConvLinSeq1d) set self.feature
        # as a side-effect of forward(), so this works universally.
        with torch.no_grad():
            self.model.eval()
            dummy_shape = (1, *input_dims)
            self.model(torch.randn(dummy_shape).to(device))
            self.embedding_dim = self.model.feature.shape[1]
            print(f"DAEDL Wrapper: Inferred embedding dim: {self.embedding_dim}")

        # Storage for the GDA (Gaussian Discriminant Analysis) estimator
        self.gda = None
        self.p_z_min = None
        self.p_z_max = None
        self.min_alpha = 1e-6  # alpha = exp(z * density), no hard lower bound

    def fit_density(self, train_loader):
        """
        Fits the GDA on training data.
        Equivalent to 'fit_gda' call in DAEDL's main.py.
        """
        print("Fitting DAEDL Density Estimator (GDA)...")
        self.model.eval()
        
        # Their fit_gda function handles embedding extraction and GMM fitting
        # Returns: gda (model), p_z_train (log-likelihoods of training data)
        self.gda, p_z_train = fit_gda(
            self.model, 
            train_loader, 
            self.num_classes, 
            self.embedding_dim, 
            self.device
        )
        
        # Store min/max for normalization (crucial for inference)
        # See logic in conf_calibration.py
        self.p_z_min = p_z_train.min().item()
        self.p_z_max = p_z_train.max().item()
        print(f"GDA Fitted. Density Range: [{self.p_z_min:.4f}, {self.p_z_max:.4f}]")

    def forward(self, x):
        """
        Forward pass implementing the Density-Aware logic.
        """
        # 1. Get Backbone Logits
        z = self.model(x)
        
        # 2. If training or GDA not ready, return standard exponential evidence
        # This matches 'eval_daedl' in their train.py
        if self.training or self.gda is None:
            return torch.exp(z) + 1e-6
        
        # 3. Density-Aware Inference
        # Replicates logic from 'conf_calibration.py' lines 28-36
        with torch.no_grad():
            log_probs = gmm_forward(self.model, self.gda, x)
            p_z = torch.logsumexp(log_probs, dim=-1)
            
            # Normalize density
            p_z = torch.clamp(p_z, min=self.p_z_min)
            p_z_norm = (p_z - self.p_z_min) / (self.p_z_max - self.p_z_min + 1e-12)
            
        # alpha = exp(logits * density)
        alpha = torch.exp(z * p_z_norm.view(-1, 1))
        return alpha

    def load_weights(self, path):
        from ..utils import get_weight_path
        full_path = get_weight_path(path)
        state = torch.load(full_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)
        print(f"Loaded DAEDL weights from {full_path}")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)