"""
Vacuity under perfect interpolation for standard EDL.

Trains an EDL model on a 2-class Gaussian mixture and measures in-sample
vacuity (K/S) for varying regularisation strengths nu. Demonstrates that
vacuity concentrates below the theoretical upper bound K / (alpha_0 + nu)
when the model perfectly fits the training data.

Outputs (saved to ../figures/ and ../tables/):
  - Per-nu vacuity histogram and feature-space scatter plot (PDF)
  - Summary statistics table (CSV)

Usage:
    python vacuity_perfect_interpolation.py
"""

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "../figures")
TABLES_DIR  = os.path.join(SCRIPT_DIR, "../tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)


# ==========================================
# 1. SEED / REPRODUCIBILITY
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ==========================================
# 2. MODEL AND LOSS
# ==========================================

class EDLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1    = nn.Linear(input_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim)
        self.fc3    = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.relu(self.output(x))  # evidence must be non-negative


def edl_loss(evidence, target, alpha_prior, lam):
    alpha_post = evidence + alpha_prior
    S = torch.sum(alpha_post, dim=1, keepdim=True)

    target_one_hot = F.one_hot(target, num_classes=alpha_prior.shape[0]).float()
    data_loss = torch.sum(
        target_one_hot * (torch.digamma(S) - torch.digamma(alpha_post)),
        dim=1
    )

    # KL(Dir(alpha_post) || Dir(alpha_prior))
    reg_loss = kl_divergence(Dirichlet(alpha_post), Dirichlet(alpha_prior))

    return torch.mean(data_loss + lam * reg_loss)


# ==========================================
# 3. EXPERIMENT
# ==========================================

def run_experiment(nu, X_tensor, y_tensor, alpha_prior, K, n_epochs, lr, frequency, seed):
    set_seed(seed)

    alpha_0 = torch.sum(alpha_prior)
    model   = EDLModel(input_dim=2, hidden_dim=64, num_classes=K)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lam = 1 / nu

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = edl_loss(model(X_tensor), y_tensor, alpha_prior, lam)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (n_epochs // frequency) == 0:
            print(f"nu={nu}, Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        alpha_post = model(X_tensor) + alpha_prior + 1e-8
        S          = torch.sum(alpha_post, dim=1, keepdim=True)
        uncertainty = K / S

    # Vacuity histogram
    threshold = (K / (alpha_0 + nu)).item()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(uncertainty.numpy(), bins=100, alpha=0.7, color='blue')
    ax.axvline(x=threshold, color='red', linestyle='--',
               label=r'$\frac{K}{\alpha_0 + \nu}$' + f' = {threshold:.4f}')
    ax.set_xlabel('Vacuity')
    ax.set_ylabel('Frequency')
    ax.set_title(r'Vacuity Distribution $\nu = $' + str(nu))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'vacuity_perfect_interpolation_nu_{nu}_histogram.pdf'))
    plt.close(fig)

    # Vacuity feature-space map
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_tensor[:, 0], X_tensor[:, 1],
                    c=uncertainty.numpy(), cmap='viridis', alpha=0.6,
                    vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(0, 1.1, 0.1))
    cbar.set_label('Vacuity')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(r'Vacuity Map $\nu = $' + str(nu))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'vacuity_perfect_interpolation_nu_{nu}_map.pdf'))
    plt.close(fig)

    uncertainty_np = uncertainty.numpy()
    return {
        'nu':                 nu,
        'theoretical_anchor': threshold,
        'mean':               float(np.mean(uncertainty_np)),
        'median':             float(np.median(uncertainty_np)),
        'std':                float(np.std(uncertainty_np)),
        'min':                float(np.min(uncertainty_np)),
        'max':                float(np.max(uncertainty_np)),
    }


# ==========================================
# 4. MAIN
# ==========================================

if __name__ == '__main__':
    seed = 12
    set_seed(seed)

    K          = 2
    NU_VALUES  = [1, 5, 10, 50, 100, 500, 1000]
    N_EPOCHS   = 3000
    LR         = 0.005
    N_SAMPLES  = 100000
    alpha_prior = torch.ones(K)

    # Two well-separated Gaussians (perfect interpolation regime)
    g1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_SAMPLES // 2)
    g2 = np.random.multivariate_normal([7, 7], [[1, 0], [0, 1]], N_SAMPLES // 2)
    X  = np.vstack([g1, g2])
    y  = np.hstack([np.zeros(N_SAMPLES // 2, dtype=int),
                    np.ones(N_SAMPLES  // 2, dtype=int)])
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Classification Data: Mixture of Two Gaussians')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'vacuity_perfect_interpolation_Gaussian_mixture.pdf'))
    plt.close()

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    mp.set_start_method('spawn', force=True)
    worker_fn = partial(
        run_experiment,
        X_tensor=X_tensor,
        y_tensor=y_tensor,
        alpha_prior=alpha_prior,
        K=K,
        n_epochs=N_EPOCHS,
        lr=LR,
        frequency=10,
        seed=seed,
    )

    with mp.Pool(processes=len(NU_VALUES)) as pool:
        results = pool.map(worker_fn, NU_VALUES)

    print("All experiments complete.")

    df = pd.DataFrame(results).sort_values('nu')
    print("\nSummary Statistics of In-Sample Vacuity:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    df.to_csv(os.path.join(TABLES_DIR, 'vacuity_perfect_interpolation_results.csv'), index=False)
