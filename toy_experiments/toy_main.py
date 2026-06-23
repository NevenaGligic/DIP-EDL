import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import os
import sys
import argparse

# Allow imports from the parent Shrinkage_Gated_Architecture directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Directory where this script lives — plots saved here
TOY_DIR = os.path.dirname(os.path.abspath(__file__))

from src.train import train_EDL
from src.models.EDL import EDL_mse_loss

# ==========================================
# 1. HELPER FUNCTIONS (KL Divergence)
# ==========================================
def kl_divergence(alpha, num_classes, device=None):
    if device is None: device = alpha.device
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

# ==========================================
# 2. ADAPTED MODEL (Wrapper Logic)
# ==========================================

class TinyREDL(nn.Module):
    """
    Adapts the logic of the original R-EDL snippet.
    - Encapsulates optimizer.
    - Calculates loss internally during forward pass if compute_loss=True.
    """
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. The Neural Network Backbone
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # 2. Internal Optimizer (As implied by 'model.step()')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Internal state for loss tracking
        self.grad_loss = 0.0

    def forward(self, x, y=None, compute_loss=False, epoch=0, return_output='soft'):
        # A. Inference
        logits = self.net(x)
        evidence = F.relu(logits)
        alpha = evidence + 1
        
        # B. Loss Calculation (The "Robust" Digamma Loss)
        if compute_loss and y is not None:
            # 1. Digamma Loss (Maximizing Log-Likelihood)
            # L = Σ y * (digamma(S) - digamma(alpha))
            S = torch.sum(alpha, dim=1, keepdim=True)
            loss_nll = torch.sum(F.one_hot(y, self.num_classes) * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
            loss_nll = torch.mean(loss_nll)

            # 2. KL Regularization
            annealing_coef = min(1, epoch / 10) # 10 epoch annealing
            kl_alpha = (alpha - 1) * (1 - F.one_hot(y, self.num_classes)) + 1
            loss_kl = annealing_coef * torch.mean(kl_divergence(kl_alpha, self.num_classes))
            
            # 3. Store total loss for the step function
            self.grad_loss = loss_nll + loss_kl
        
        if return_output == 'hard':
            return torch.argmax(alpha, dim=1)
        
        return alpha

    def step(self):
        """Performs the optimization step."""
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

# ==========================================
# 3. ADAPTED TRAINING LOOP (From your snippet)
# ==========================================

def compute_loss_accuracy_adapted(model, loader, epoch, device):
    model.eval()
    total_loss_ = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            
            # Call model to compute loss internally
            # Note: We don't use the return value for prediction here to match your snippet logic strictly,
            # but we need predictions for accuracy.
            probs = model(X, Y, compute_loss=True, epoch=epoch, return_output='soft')
            
            # Accumulate the loss stored in the model
            total_loss_ += model.grad_loss.item()
            
            # Calculate Accuracy
            _, predicted = torch.max(probs, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

    avg_loss = total_loss_ / len(loader)
    accuracy = correct / total
    return accuracy, avg_loss

def train_redl_adapted(model, train_loader, val_loader, max_epochs=50, device=torch.device("cpu")):
    print("Starting Adapted R-EDL Training...")
    model.to(device)
    model.train()
    
    val_losses = []
    
    for epoch in range(max_epochs):
        # 1. Training Loop
        for X_train, Y_train in train_loader:
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            
            model.train()
            # Forward + Compute Loss
            model(X_train, Y_train, compute_loss=True, epoch=epoch)
            # Optimize
            model.step()
        
        # 2. Scheduler Step
        model.scheduler.step()
        
        # 3. Validation / Logging (Every 10 epochs)
        if (epoch + 1) % 10 == 0:
            val_acc, val_loss = compute_loss_accuracy_adapted(model, val_loader, epoch, device)
            val_losses.append(val_loss)
            
            print(f"\033[34m Epoch {epoch+1} -> Val loss {val_loss:.4f} | Val Acc. {val_acc*100:.2f}%\033[0m")
            
            if np.isnan(val_loss):
                print("Detected NaN Loss - Stopping")
                break

    print("Training Finished.")

# ==========================================
# 1. Tiny Models for 2D Data
# ==========================================

class TinyEDL(nn.Module):
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        logits = self.net(x)
        evidence = torch.relu(logits)
        alpha = evidence + 1
        return alpha

class TinyMAF(nn.Module):
    def __init__(self, input_dim=2, hidden_features=64, num_layers=5):
        super().__init__()
        base_dist = StandardNormal(shape=[input_dim])
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=input_dim, 
                hidden_features=hidden_features
            ))
        self.transform = CompositeTransform(transforms)
        self.flow = Flow(self.transform, base_dist)

    def log_prob(self, x):
        return self.flow.log_prob(x)


class TinyClassifier(nn.Module):
    """
    A standard MLP classifier.
    Outputs: Raw logits (CrossEntropyLoss applies softmax internally).
    """
    def __init__(self, input_dim=2, num_classes=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class dip_EDL(nn.Module):
    """
    Implements the formula: Alpha = 1 + n * p(x) * P(y|x)
    """
    def __init__(self, classifier, maf, n_samples):
        super().__init__()
        self.classifier = classifier
        self.maf = maf
        self.n = float(n_samples)
    
    def forward(self, x):
        # 1. Get P(y|x) from standard classifier
        probs = F.softmax(self.classifier(x), dim=1)
        
        # 2. Get p(x) from MAF
        # MAF gives log_prob, so we exp() it to get likelihood
        # Detach to ensure we don't backprop into MAF/Classifier during inference visualization
        log_prob = self.maf.log_prob(x).detach()
        likelihood = torch.exp(log_prob).unsqueeze(1) # Shape [Batch, 1]
        
        # 3. Apply Formula: alpha = 1 + n * p(x) * P(y|x)
        evidence = self.n * likelihood * probs
        alpha = evidence + 1
        
        return alpha

# ==========================================
# 2. Visualization
# ==========================================
def _get_uncertainty_grid(model, X, device):
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    model.eval()
    with torch.no_grad():
        alphas = model(grid_tensor)
        S = torch.sum(alphas, dim=1)
        uncertainty = (2.0 / S).cpu().reshape(xx.shape).numpy()
    return xx, yy, uncertainty


def _get_all_uncertainty_grids(model, X, device):
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    model.eval()
    with torch.no_grad():
        alphas = model(grid_tensor)
        K = alphas.shape[1]
        S = alphas.sum(dim=1, keepdim=True)
        probs = alphas / S

        # Vacuity: K/S (Subjective Logic total uncertainty)
        vacuity = (K / S.squeeze()).cpu().reshape(xx.shape).numpy()

        # Total uncertainty: entropy of the mean prediction
        total = (-torch.sum(probs * torch.log(probs.clamp(min=1e-8)), dim=1)
                 ).cpu().reshape(xx.shape).numpy()

        # Aleatoric: expected entropy of Dirichlet sample (closed form via digamma)
        # E[H[Cat(p)]] = sum_k (alpha_k/S) * (psi(S+1) - psi(alpha_k+1))
        aleatoric = (torch.sum(
            probs * (torch.digamma(S + 1) - torch.digamma(alphas + 1)), dim=1
        )).cpu().reshape(xx.shape).numpy()

        # Epistemic: mutual information = total - aleatoric
        epistemic = total - aleatoric

    return xx, yy, vacuity, aleatoric, epistemic


def plot_uncertainty_surface(model, X, y, title, filename):
    device = next(model.parameters()).device
    xx, yy, uncertainty = _get_uncertainty_grid(model, X, device)

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(xx, yy, uncertainty, levels=np.linspace(0, 1.1, 20), cmap='magma')
    cbar = plt.colorbar(contour)
    cbar.set_label('Vacuity (K/S)', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='#56B4E9', edgecolors='#1a4a6b', linewidths=0.5, s=20, alpha=0.9, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='white',   edgecolors='#222222',  linewidths=0.8, s=20, alpha=0.9, label='Class 1')
    plt.title(title, fontsize=22)
    plt.legend(loc='upper right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_comparison(edl_model, dip_edl_model, X, y, filename):
    device = next(edl_model.parameters()).device

    xx, yy, vac_edl, alea_edl, epi_edl = _get_all_uncertainty_grids(edl_model,    X, device)
    _,  _,  vac_nu,  alea_nu,  epi_nu  = _get_all_uncertainty_grids(dip_edl_model, X, device)

    col_titles = ['Vacuity (K/S)', 'Aleatoric', 'Epistemic (MI)']
    model_rows = [
        ('EDL',     [vac_edl, alea_edl, epi_edl]),
        ('DIP-EDL', [vac_nu,  alea_nu,  epi_nu]),
    ]
    # Single shared scale across all panels so magnitudes are directly comparable
    global_vmax = max(g.max() for grids in [model_rows[0][1], model_rows[1][1]] for g in grids)
    # Explicit levels so every contourf uses identical boundaries (not per-panel data range)
    shared_levels = np.linspace(0, global_vmax, 21)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle('Uncertainty decomposition: EDL vs DIP-EDL',
                 fontsize=22, fontweight='bold', y=0.97)

    last_cf = None
    for row_idx, (model_name, grids) in enumerate(model_rows):
        for col_idx, (unc_name, grid) in enumerate(zip(col_titles, grids)):
            ax = axes[row_idx, col_idx]
            cf = ax.contourf(xx, yy, grid, levels=shared_levels, cmap='magma')
            last_cf = cf

            ax.scatter(X[y==0, 0], X[y==0, 1], c='#56B4E9', edgecolors='#1a4a6b',
                       linewidths=0.5, s=15, alpha=0.9, label='Class 0')
            ax.scatter(X[y==1, 0], X[y==1, 1], c='white',   edgecolors='#222222',
                       linewidths=0.8, s=15, alpha=0.9, label='Class 1')

            if row_idx == 0:
                ax.set_title(unc_name, fontsize=16, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel('$x_2$', fontsize=14)
                ax.text(-0.18, 0.5, model_name, transform=ax.transAxes,
                        fontsize=22, fontweight='bold', va='center', ha='center', rotation=90)
            if row_idx == 1:
                ax.set_xlabel('$x_1$', fontsize=14)
            # Legend on top-right panel only
            if row_idx == 0 and col_idx == 2:
                leg = ax.legend(loc='upper right', fontsize=15)
                for handle in leg.legend_handles:
                    handle.set_sizes([80])
            ax.tick_params(axis='both', labelsize=11)

    # Single colorbar in its own axes — doesn't disturb the subplot grid
    fig.subplots_adjust(right=0.88, top=0.90, left=0.1, wspace=0.08, hspace=0.06)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(last_cf, cax=cbar_ax)
    cbar.set_label('Uncertainty', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved comparison plot: {filename}")
    plt.close()

def plot_vacuity_comparison(edl_model, dip_edl_model, X, y, filename):
    device = next(edl_model.parameters()).device

    xx, yy, vac_edl, _, _ = _get_all_uncertainty_grids(edl_model,    X, device)
    _,  _,  vac_nu,  _, _ = _get_all_uncertainty_grids(dip_edl_model, X, device)

    shared_levels = np.linspace(0, 1.0, 21)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle('Vacuity comparison (toy example)',
                 fontsize=22, fontweight='bold', y=1.02)

    for ax, grid, model_name in zip(axes, [vac_edl, vac_nu], ['EDL', 'DIP-EDL']):
        cf = ax.contourf(xx, yy, grid, levels=shared_levels, cmap='magma')
        ax.scatter(X[y==0, 0], X[y==0, 1], c='#56B4E9', edgecolors='#1a4a6b',
                   linewidths=0.5, s=18, alpha=0.9, label='Class 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='white',   edgecolors='#222222',
                   linewidths=0.8, s=18, alpha=0.9, label='Class 1')
        ax.set_title(model_name, fontsize=20, fontweight='bold')
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.tick_params(axis='both', labelsize=13)

    axes[0].set_ylabel('$x_2$', fontsize=16)
    leg = axes[1].legend(loc='upper right', fontsize=14)
    for handle in leg.legend_handles:
        handle.set_sizes([80])

    fig.subplots_adjust(right=0.88, wspace=0.06)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Vacuity (K/S)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved vacuity comparison plot: {filename}")
    plt.close()


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help="Random seed") # or 73
    parser.add_argument('--model_type', type=str, default='edl', choices=['edl', 'redl', 'dip_edl', 'both'], help="Model to train. 'both' trains EDL and DIP-EDL and produces a side-by-side comparison plot.")
    args = parser.parse_args()

    # 1. Set Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Toy Experiment on {device} (Seed={args.seed})")

    # 2. Data
    n_samples = 1000
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=args.seed)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    dataset = TensorDataset(X, y)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 3. Initialize Models
    if args.model_type == 'redl':
        print("Using R-EDL backbone")
        edl_model = TinyREDL().to(device)
    else:
        print("Using Standard EDL backbone")
        edl_model = TinyEDL().to(device)

    if args.model_type in ('dip_edl', 'both'):
        maf_model = TinyMAF().to(device)
        classifier = TinyClassifier().to(device)

    # 4. Train
    print(f"\n--- Training Tiny {args.model_type.upper()} ---")
    if args.model_type == 'redl':
        train_redl_adapted(
            model=edl_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=100,
            device=device
        )

    if args.model_type in ('edl', 'both'):
        train_EDL(
            model_EDL=edl_model,
            train_loader=train_loader,
            device=device,
            EDL_mse_loss=EDL_mse_loss,
            num_classes=2,
            file_path=None,
            epochs=100,
            annealing_step=20,
            lr=1e-3,
            save_and_upload_model_fn=lambda m, p: None,
        )

    if args.model_type in ('dip_edl', 'both'):
        print("\n--- Training Standard Classifier (Cross Entropy) ---")
        optimizer_cls = optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(100):
            total_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                logits = classifier(bx)
                loss = criterion(logits, by)

                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"  Classifier Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

        print("\n--- Training Tiny MAF ---")
        optimizer_maf = optim.Adam(maf_model.parameters(), lr=1e-3)
        maf_model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)
                loss = -maf_model.log_prob(batch_x).mean()

                optimizer_maf.zero_grad()
                loss.backward()
                optimizer_maf.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"  MAF Epoch {epoch+1}: Loss {total_loss / len(train_loader):.4f}")

    # 5. Assemble final model(s) and visualize
    X_np, y_np = X.cpu().numpy(), y.cpu().numpy()
    print("\n--- Generating Plots ---")

    if args.model_type == 'both':
        dip_edl_model = dip_EDL(classifier, maf_model, n_samples=train_size).to(device)
        plot_comparison(
            edl_model, dip_edl_model, X_np, y_np,
            filename=os.path.join(TOY_DIR, "toy_comparison.pdf")
        )
        plot_vacuity_comparison(
            edl_model, dip_edl_model, X_np, y_np,
            filename=os.path.join(TOY_DIR, "toy_vacuity_comparison.pdf")
        )
    elif args.model_type == 'dip_edl':
        final_model = dip_EDL(classifier, maf_model, n_samples=train_size).to(device)
        plot_uncertainty_surface(final_model, X_np, y_np, title='DIP-EDL',
                                 filename=os.path.join(TOY_DIR, "toy_dip_edl.pdf"))
    else:
        plot_uncertainty_surface(edl_model, X_np, y_np, title=args.model_type.upper(),
                                 filename=os.path.join(TOY_DIR, f"toy_{args.model_type}.pdf"))

    print("\nExperiment Complete. Check the .pdf files.")

if __name__ == "__main__":
    main()