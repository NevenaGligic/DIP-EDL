import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import torch.optim as optim
from torch.amp import GradScaler, autocast
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nflows.flows.autoregressive import MaskedAutoregressiveFlow # For type hinting
from src.utils import transform_for_maf

# -----------------------------------------------------------
# 1. EDL Training Function (Prior Network Head)
# -----------------------------------------------------------
def train_EDL(
    model_EDL: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    EDL_mse_loss: callable,
    num_classes: int,
    file_path: str,
    epochs: int = 50,
    annealing_step: float = 10.0,
    lr: float = 1e-3,
    weight_decay: float = 0.005,
    save_and_upload_model_fn=None,
    dataset_name: str = "mnist",
    optimizer_type: str = 'adam',   # 'adam' or 'sgd'
    scheduler_type: str = 'step',   # 'step' or 'cosine'
):
    """Trains the Evidential LeNet model (prior network head)."""
    print(f"Starting EDL Training on {device} for {epochs} epochs "
          f"(opt={optimizer_type}, sched={scheduler_type})...")

    trainable_params = [p for p in model_EDL.parameters() if p.requires_grad]
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    else:
        scheduler = None  # original: no scheduler in train_EDL

    model_EDL.to(device)

    # Initialize Automatic Mixed Precision (AMP) GradScaler
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(device.type, enabled=use_amp)

    model_EDL.train()
    
    # Store losses for potential plotting
    loss_history = []
    
    for epoch in tqdm(range(epochs), desc="EDL Training"):
        total_loss = 0
        
        # Use explicit iterator with length for tqdm
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass and Loss calculation using AMP
            with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = model_EDL(data)
                # Pass necessary arguments to the external loss function
                loss = EDL_mse_loss(output, target, epoch, num_classes, annealing_step, device=device)
            
            # Backward pass and Optimization using GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0] if scheduler is not None else lr
            tqdm.write(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f} | LR {lr_now:.2e}")

    # Final save and upload
    if save_and_upload_model_fn:
        save_and_upload_model_fn(model_EDL, file_path)
    else:
        # Fallback to local save if no upload function is provided
        torch.save(model_EDL.state_dict(), file_path)
        print(f"Model trained, saved locally to {file_path}")

    return model_EDL

# -----------------------------------------------------------
# 2. Density Estimation Training Function (Log-density Head)
# -----------------------------------------------------------
def train_density_estimator(
    model_wrapper,  # Can be MaskedAutoregressiveFlow or GMMDensity
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
    learning_rate: float = 1e-3, 
    scheduler_step_size: int = 5, 
    scheduler_gamma: float = 0.5, 
    num_epochs: int = 50, 
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 3, 
    save_and_upload_model_fn = None,
    dataset_name: str = "mnist"
):
    """
    Unified trainer for the Density Estimator p(x).
    - CIFAR-10: Fits GDA (Gaussian Discriminant Analysis).
    - MNIST: Trains MAF (Masked Autoregressive Flow).
    """
    is_feature_space = (dataset_name == 'cifar10')
    
    # --- BRANCH 1: GDA Fitting (CIFAR-10) ---
    if dataset_name == 'cifar10':
        print(f">> MODE: CIFAR-10 Detected. Fitting GDA Density Estimator...")
        
        if not hasattr(model_wrapper, 'fit_density'):
            raise ValueError("Model does not have 'fit_density'. Ensure you are passing a dip_EDL model wrapper.")
            
        # Run the fitting (One-shot)
        model_wrapper.fit_density(train_loader, device)
        
        # Save the GDA stats (min/max/params) inside the main model state
        if save_and_upload_model_fn:
            save_and_upload_model_fn(model_wrapper, file_path)
        else:
            torch.save(model_wrapper.state_dict(), file_path)
            
        print(f"✅ GDA Fitted and stats saved to {file_path}")
        return model_wrapper
    
    # --- BRANCH 2: MAF Training (MNIST / LAMOST MAF) ---
    is_lamost_maf = (dataset_name == 'lamost')
    if is_lamost_maf:
        print(">> MODE: LAMOST MAF — training on 32-dim penultimate backbone features...")
        model_wrapper.cnn.eval()
        model_wrapper.cnn.to(device)
    else:
        print(">> MODE: MNIST Detected. Training MAF Flow...")

    def _get_maf_input(raw):
        """Returns MAF input: penultimate backbone features (LAMOST) or transformed pixels (MNIST)."""
        if is_lamost_maf:
            with torch.no_grad():
                model_wrapper.cnn(raw)
            return model_wrapper.cnn._net.penultimate_feature.detach()
        return transform_for_maf(raw, dataset_name)

    # For MNIST/LAMOST, we work with the sub-module 'maf'
    model_MAF = model_wrapper.maf
    model_MAF.to(device)  # ensure MAF is on the right device before training

    # Training parameters with L2 regularization
    optimizer = optim.Adam(model_MAF.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Initialize Automatic Mixed Precision (AMP) GradScaler
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(device.type, enabled=use_amp)
    
    # Training loop state
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in tqdm(range(num_epochs), desc="MAF Training"):
        # Training phase
        model_MAF.train()
        epoch_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            data = data.to(device)
            # data = transform_for_maf(data)
            
            data = _get_maf_input(data)

            optimizer.zero_grad()
            
            # Forward pass and Loss calculation using AMP
            with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = -model_MAF.log_prob(data).mean()
            
            # Backward pass, unscale, clip, and step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Must unscale before clipping
            torch.nn.utils.clip_grad_norm_(model_MAF.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model_MAF.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in tqdm(val_loader, desc="Validation"):
                data = data.to(device)
                # data = transform_for_maf(data)
                data = _get_maf_input(data)

                with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    loss_val = -model_MAF.log_prob(data).mean()
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
                   f'Training Loss: {avg_train_loss:.4f}, '
                   f'Validation Loss: {avg_val_loss:.4f}, '
                   f'LR: {scheduler.get_last_lr()[0]:.2e}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Store copy of the best state dict
            best_model_state = {k: v.cpu().clone() for k, v in model_MAF.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            tqdm.write(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Restore best model
    if best_model_state is not None:
        model_MAF.load_state_dict(best_model_state)
    
    # Final save and upload (plotting logic can go in a separate 'report' script)
    if save_and_upload_model_fn:
        save_and_upload_model_fn(model_MAF, file_path)
    else:
        torch.save(model_MAF.state_dict(), file_path)
        print(f"Model trained, saved locally to {file_path}")

    return model_MAF


# -----------------------------------------------------------
# 3. StandardCNN Training Function (nu-EDL)
# -----------------------------------------------------------

def train_CNN(
    model_CNN,
    train_loader,
    device,
    file_path,
    epochs=50,
    lr=1e-3,
    save_and_upload_model_fn=None,
    class_weights=None,
    optimizer_type='adam',   # 'adam' (default) or 'sgd'
    scheduler_type='step',   # 'step' (default) or 'cosine'
):
    """
    Trains a StandardCNN using Cross Entropy Loss.
    optimizer_type='sgd'  + scheduler_type='cosine' : WideResNet recipe
    optimizer_type='adam' + scheduler_type='step'   : original recipe (all other datasets)
    """
    print(f"--- Starting Standard Classifier Training ({epochs} epochs, "
          f"opt={optimizer_type}, sched={scheduler_type}) ---")
    model_CNN.to(device)
    model_CNN.train()

    trainable_params = [p for p in model_CNN.parameters() if p.requires_grad]
    w = class_weights.to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=lr,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    use_amp = (device.type == 'cuda')
    scaler = GradScaler(device.type, enabled=use_amp)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model_CNN(data)
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            acc = 100 * correct / total
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f} | Acc {acc:.2f}% | LR {current_lr:.2e}")

    if save_and_upload_model_fn:
        save_and_upload_model_fn(model_CNN, file_path)
    else:
        torch.save(model_CNN.state_dict(), file_path)
        print(f"Classifier trained, saved locally to {file_path}")

    return model_CNN


# -----------------------------------------------------------
# 4. Deep Ensemble Training (trains each member independently)
# -----------------------------------------------------------

def train_deep_ensemble(
    ensemble_model,
    train_loader,
    device,
    file_path_prefix: str,
    epochs: int = 50,
    lr: float = 1e-3,
    save_and_upload_model_fn=None,
    optimizer_type: str = 'adam',
    scheduler_type: str = 'step',
):
    """Train each member of a DeepEnsembleModel with cross-entropy loss.

    Weights for member i are saved to ``{file_path_prefix}_{i}.pth``.
    """
    criterion = nn.CrossEntropyLoss()
    use_amp = (device.type == 'cuda')

    for i, member in enumerate(ensemble_model.ensemble.core_models):
        print(f"\n--- Training Ensemble Member {i + 1}/{ensemble_model.num_estimators} ---")
        member.to(device)
        member.train()

        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(member.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            optimizer = torch.optim.Adam(member.parameters(), lr=lr)

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        scaler = GradScaler(device.type, enabled=use_amp)

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    logits = member(data)
                    loss = criterion(logits, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                acc = 100 * correct / total
                print(f"  Member {i}, Epoch {epoch + 1}: Loss {avg_loss:.4f} | Acc {acc:.2f}%")

        member_path = f"{file_path_prefix}_{i}.pth"
        if save_and_upload_model_fn:
            save_and_upload_model_fn(member, member_path)
        else:
            torch.save(member.state_dict(), member_path)
            print(f"Ensemble member {i} saved to {member_path}")

    return ensemble_model