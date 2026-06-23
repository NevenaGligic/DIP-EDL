import argparse
import os
import sys
import torch
import numpy as np
import json

# --- Project Modules ---
from src.dataloaders import get_experiment_loaders
from src.train import train_EDL, train_density_estimator, train_CNN, train_deep_ensemble
from src.models.dip_edl import dip_EDL
from src.models.EDL import EDL, EDL_mse_loss
from src.models.R_EDL import REDL
from src.models.Re_EDL import ReEDL
from src.models.DAEDL import DAEDL
from src.models.PostNet import PostNet, PostNetLoaderWrapper
from src.models.Baselines import DeepEnsembleModel, MCDropoutModel, run_benchmark_baseline
from src.utils import get_best_device, save_model
from src.experiments import run_benchmark

# --- External Training Loops ---
# R-EDL / Re-EDL
REDL_ROOT = os.path.join(os.path.dirname(__file__), "ICLR2024-REDL", "code_classical")
sys.path.append(os.path.abspath(REDL_ROOT))
from train import train as redl_train

# DAEDL
DAEDL_ROOT = os.path.join(os.path.dirname(__file__), "DAEDL")
sys.path.append(os.path.abspath(DAEDL_ROOT))
from train_DAEDL import train_daedl

# PostNet
POSTNET_ROOT = os.path.join(os.path.dirname(__file__), "Posterior-Network")
sys.path.append(os.path.abspath(POSTNET_ROOT))
from src_postnet.posterior_networks.train_postnet import train_postnet


# --- Centralized Configuration ---
CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_dims': (1, 28, 28),
        'maf': {
            'batch_size': 128,
            'lr': 1e-4,
            'epochs': 50,
            'maf_hidden_features': 1024,
            'maf_num_layers': 10,
            'maf_blocks': 20,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'scheduler_step_size': 5,
        },
        'edl': {
            'batch_size': 128,
            'lr': 1e-3,
            'weight_decay': 0.005,
            'epochs': 50,
            'annealing_step': 10,
            'filters': [20, 50],
            'hidden_units': 500
        },
        'cnn': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
        },
        'redl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 60,
            'lamb1': 1.0,
            'lamb2': 0.1,
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64],
            'kernel_dim': 5,
            'architecture': 'conv'
        },
        'reedl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 60,
            'lamb1': 1.0,
            'lamb2': 0.1,
            'kl_c': 0.0,
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64],
            'kernel_dim': 5,
            'architecture': 'conv'
        },
        'daedl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 50,
            'reg': 5e-2,
            'dropout_rate': 0.5
        },
        'postnet': {
            'batch_size': 64,
            'lr': 5e-5,
            'epochs': 50,
            'latent_dim': 6,
            'density_type': 'radial_flow',
            'architecture': 'conv'
        },
        'ensemble': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
            'num_estimators': 5,
        },
        'mc_dropout': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
            'num_estimators': 50,
            'dropout_rate': 0.5,
        },
    },
    'lamost': {
        'num_classes': 2,
        'input_dims': None,   # set dynamically after data loading
        'maf': {
            'batch_size': 128,
            'lr': 1e-4,
            'epochs': 50,
            'maf_hidden_features': 64,
            'maf_num_layers': 5,
            'maf_blocks': 2,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,
            'scheduler_step_size': 10,
        },
        'edl': {
            'batch_size': 128,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 50,
            'annealing_step': 10,
        },
        'cnn': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
        },
        'redl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 60,
            'lamb1': 1.0,
            'lamb2': 0.1,
            'fisher_c': 0.0,
            'hidden_dims': [256, 256, 256],
            'kernel_dim': None,
            'architecture': 'linear',
        },
        'reedl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 60,
            'lamb1': 1.0,
            'lamb2': 0.8,
            'kl_c': 0.0,
            'fisher_c': 0.0,
            'hidden_dims': [256, 256, 256],
            'kernel_dim': None,
            'architecture': 'linear',
        },
        'daedl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 50,
            'reg': 5e-2,
            'dropout_rate': 0.5,
        },
        'postnet': {
            'batch_size': 64,
            'lr': 5e-5,
            'epochs': 50,
            'latent_dim': 6,
            'density_type': 'radial_flow',
            'architecture': 'linear',
        },
        'ensemble': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
            'num_estimators': 5,
        },
        'mc_dropout': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
            'num_estimators': 50,
            'dropout_rate': 0.5,
        },
    },
    'cifar10': {
        'num_classes': 10,
        'input_dims': (3, 32, 32),
        'maf': {
            'batch_size': 128,
            'lr': 1e-4,
            'epochs': 50,
            'maf_hidden_features': 1024,
            'maf_num_layers': 10,
            'maf_blocks': 5,
            'weight_decay': 1e-6,
            'early_stopping_patience': 10,
            'scheduler_step_size': 30,
        },
        'edl': {
            'batch_size': 128,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'epochs': 100,
            'annealing_step': 10,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
        },
        'cnn': {
            'batch_size': 128,
            'lr': 0.1,
            'epochs': 100,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
        },
        'redl': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 200,
            'lamb1': 1.0,
            'lamb2': 0.1,
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64],
            'kernel_dim': 5,
            'architecture': 'vgg'
        },
        'reedl': {
            'batch_size': 64,
            'lr': 1e-4,
            'epochs': 200,
            'lamb1': 1.0,
            'lamb2': 0.8,
            'kl_c': 0.0,
            'fisher_c': 0.0,
            'hidden_dims': [64, 64, 64],
            'kernel_dim': 5,
            'architecture': 'vgg'
        },
        'daedl': {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 100,
            'reg': 5e-2,
            'dropout_rate': 0.5
        },
        'postnet': {
            'batch_size': 64,
            'lr': 5e-4,
            'epochs': 200,
            'latent_dim': 6,
            'density_type': 'radial_flow',
            'architecture': 'conv'
        },
        'ensemble': {
            'batch_size': 128,
            'lr': 0.1,
            'epochs': 100,
            'num_estimators': 5,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
        },
        'mc_dropout': {
            'batch_size': 128,
            'lr': 0.1,
            'epochs': 100,
            'num_estimators': 50,
            'dropout_rate': 0.3,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
        },
    }
}


def main():
    parser = argparse.ArgumentParser(description="DIP-EDL Benchmarking Script.")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'lamost'], default='mnist',
                        help="ID dataset.")
    parser.add_argument('--data_dir', type=str, default='./data', help="Root data directory.")
    parser.add_argument('--model', type=str,
                        choices=['dip_edl', 'edl', 'redl', 'reedl', 'daedl', 'postnet', 'ensemble', 'mc_dropout'],
                        default='dip_edl', help="Model to train and evaluate.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument('--lamost_ood', type=str, default='star', choices=['star', 'quasar', 'galaxy'],
                        help="LAMOST class to hold out as OOD.")

    # Training flags
    parser.add_argument('--train_edl', action='store_true', help="Train the EDL model from scratch.")
    parser.add_argument('--train_maf', action='store_true', help="Train the MAF density estimator from scratch.")
    parser.add_argument('--train_cnn', action='store_true', help="Train the CNN backbone from scratch.")
    parser.add_argument('--train_redl', action='store_true', help="Train R-EDL from scratch.")
    parser.add_argument('--train_reedl', action='store_true', help="Train Re-EDL from scratch.")
    parser.add_argument('--train_daedl', action='store_true', help="Train DAEDL from scratch.")
    parser.add_argument('--train_postnet', action='store_true', help="Train PostNet from scratch.")
    parser.add_argument('--train_ensemble', action='store_true', help="Train Deep Ensemble from scratch.")
    parser.add_argument('--train_mc_dropout', action='store_true', help="Train MC Dropout from scratch.")

    # Weight paths (auto-set if not provided)
    parser.add_argument('--edl_path', type=str, default=None)
    parser.add_argument('--maf_path', type=str, default=None)
    parser.add_argument('--cnn_path', type=str, default=None)
    parser.add_argument('--redl_path', type=str, default=None)
    parser.add_argument('--reedl_path', type=str, default=None)
    parser.add_argument('--daedl_path', type=str, default=None)
    parser.add_argument('--postnet_path', type=str, default=None)
    parser.add_argument('--ensemble_path', type=str, default=None)
    parser.add_argument('--mc_dropout_path', type=str, default=None)

    args = parser.parse_args()

    # --- Seed ---
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)

    dataset_cfg = CONFIGS[args.dataset]

    # Determine model config
    if args.model == 'dip_edl':
        maf_cfg = dataset_cfg['maf']
        model_cfg = dataset_cfg['cnn']
    elif args.dataset == 'lamost' and args.model not in dataset_cfg:
        raise NotImplementedError(f"Model '{args.model}' is not configured for LAMOST.")
    else:
        model_cfg = dataset_cfg[args.model]

    NUM_CLASSES = dataset_cfg['num_classes']
    INPUT_DIMS = dataset_cfg['input_dims']
    BATCH_SIZE = model_cfg['batch_size']
    EPOCHS = model_cfg['epochs']

    # Auto-set weight paths
    save_dir = "saved_model_weights"
    os.makedirs(save_dir, exist_ok=True)
    seed_suffix = f"seed{args.seed}"

    if args.edl_path is None:
        args.edl_path = f"{save_dir}/{args.dataset}_EDL_model_{seed_suffix}.pth"
    if args.maf_path is None:
        if args.dataset == 'mnist':
            args.maf_path = f"{save_dir}/{args.dataset}_dip_EDL_MAF_model_{seed_suffix}.pth"
        elif args.dataset == 'cifar10':
            args.maf_path = f"{save_dir}/{args.dataset}_dip_EDL_GDA_{seed_suffix}.pt"
        elif args.dataset == 'lamost':
            args.maf_path = f"{save_dir}/{args.dataset}_dip_EDL_GDA_{seed_suffix}.pt"
    if args.cnn_path is None:
        args.cnn_path = f"{save_dir}/{args.dataset}_dip_EDL_CNN_model_{seed_suffix}.pth"
    if args.redl_path is None:
        args.redl_path = f"{save_dir}/{args.dataset}_R_EDL_model_{seed_suffix}"
    if args.reedl_path is None:
        args.reedl_path = f"{save_dir}/{args.dataset}_Re_EDL_model_{seed_suffix}"
    if args.daedl_path is None:
        args.daedl_path = f"{save_dir}/{args.dataset}_DAEDL_model_{seed_suffix}.pth"
    if args.postnet_path is None:
        args.postnet_path = f"{save_dir}/{args.dataset}_PostNet_model_{seed_suffix}.pth"
    if args.ensemble_path is None:
        args.ensemble_path = f"{save_dir}/{args.dataset}_Ensemble_model_{seed_suffix}"
    if args.mc_dropout_path is None:
        args.mc_dropout_path = f"{save_dir}/{args.dataset}_MCDropout_model_{seed_suffix}.pth"

    # --- Device ---
    class _FlatLoader:
        """Squeeze channel dim: (B, 1, L) → (B, L) for linear-architecture models."""
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        def __iter__(self):
            for x, y in self.loader:
                yield x.view(x.size(0), -1), y
        def __len__(self):
            return len(self.loader)

    _LINEAR_MODELS_LAMOST = {'redl', 'reedl', 'postnet'}

    device = get_best_device()
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading data...")
    _loaders = get_experiment_loaders(
        args.dataset, BATCH_SIZE, val_split=args.val_split, seed=args.seed,
        data_dir=args.data_dir, lamost_ood=args.lamost_ood,
    )
    if args.dataset == 'cifar10':
        train_loader, val_loader, test_loader_id, ood_loaders, train_loader_gda = _loaders
    else:
        train_loader, val_loader, test_loader_id, ood_loaders = _loaders
        train_loader_gda = None

    if args.dataset == 'lamost':
        sample_x, _ = next(iter(train_loader))
        INPUT_DIMS = tuple(sample_x.shape[1:])
        print(f"[LAMOST] Detected input_dims: {INPUT_DIMS}")

        if args.model in _LINEAR_MODELS_LAMOST:
            train_loader   = _FlatLoader(train_loader)
            val_loader     = _FlatLoader(val_loader)
            test_loader_id = _FlatLoader(test_loader_id)
            ood_loaders    = {k: _FlatLoader(v) for k, v in ood_loaders.items()}
            INPUT_DIMS     = (INPUT_DIMS[1],)
            print(f"[LAMOST] Flat loader applied for {args.model}. input_dims: {INPUT_DIMS}")

    # --- Initialize Model ---
    print("Initializing model...")

    if args.model == 'dip_edl':
        model = dip_EDL(
            num_classes=NUM_CLASSES,
            input_dims=INPUT_DIMS,
            maf_hidden_features=maf_cfg.get('maf_hidden_features', 256),
            maf_num_layers=maf_cfg.get('maf_num_layers', 5),
            maf_blocks=maf_cfg.get('maf_blocks', 5),
            n_train_samples=len(train_loader.dataset),
        ).to(device)

    elif args.model == 'edl':
        model = EDL(num_classes=NUM_CLASSES, input_dims=INPUT_DIMS,
                    cifar_backbone='wrn28_10').to(device)

    elif args.model == 'redl':
        redl_input_dims = INPUT_DIMS
        if args.dataset == 'mnist':
            redl_input_dims = (28, 28, 1)
        model = REDL(
            num_classes=NUM_CLASSES,
            input_dims=redl_input_dims,
            architecture=model_cfg['architecture'],
            batch_size=BATCH_SIZE,
            lr=model_cfg['lr'],
            lamb1=model_cfg['lamb1'],
            lamb2=model_cfg['lamb2'],
            fisher_c=model_cfg['fisher_c'],
            hidden_dims=model_cfg['hidden_dims'],
            kernel_dim=model_cfg['kernel_dim'],
            seed=args.seed,
        ).to(device)

    elif args.model == 'reedl':
        reedl_input_dims = INPUT_DIMS
        if args.dataset == 'mnist':
            reedl_input_dims = (28, 28, 1)
        model = ReEDL(
            num_classes=NUM_CLASSES,
            input_dims=reedl_input_dims,
            architecture=model_cfg['architecture'],
            batch_size=BATCH_SIZE,
            lr=model_cfg['lr'],
            lamb1=model_cfg['lamb1'],
            lamb2=model_cfg['lamb2'],
            kl_c=model_cfg['kl_c'],
            fisher_c=model_cfg['fisher_c'],
            hidden_dims=model_cfg['hidden_dims'],
            kernel_dim=model_cfg['kernel_dim'],
            seed=args.seed,
        ).to(device)

    elif args.model == 'daedl':
        daedl_dataset_name = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'lamost': 'LAMOST'}[args.dataset]
        model = DAEDL(
            num_classes=NUM_CLASSES,
            dataset=daedl_dataset_name,
            input_dims=INPUT_DIMS,
            dropout_rate=model_cfg['dropout_rate'],
            device=device
        ).to(device)

    elif args.model == 'postnet':
        print("Calculating class counts (N) for PostNet...")
        if hasattr(train_loader.dataset, 'targets'):
            targets = torch.tensor(train_loader.dataset.targets)
        elif hasattr(train_loader.dataset, 'tensors'):
            targets = train_loader.dataset.tensors[1]
        else:
            targets = torch.cat([y for _, y in train_loader])
        unique, counts = torch.unique(targets, return_counts=True)
        N_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        N_counts[unique.long()] = counts.long()
        print(f"Class counts: {N_counts}")
        model = PostNet(
            num_classes=NUM_CLASSES,
            N=N_counts,
            input_dims=INPUT_DIMS,
            architecture=model_cfg['architecture'],
            latent_dim=model_cfg['latent_dim'],
            density_type=model_cfg['density_type'],
            lr=model_cfg['lr'],
            regr=1e-5,
            batch_size=BATCH_SIZE
        ).to(device)

    elif args.model == 'ensemble':
        model = DeepEnsembleModel(
            num_classes=NUM_CLASSES,
            input_dims=INPUT_DIMS,
            num_estimators=model_cfg['num_estimators'],
            cifar_backbone='wrn28_10',
        ).to(device)

    elif args.model == 'mc_dropout':
        model = MCDropoutModel(
            num_classes=NUM_CLASSES,
            input_dims=INPUT_DIMS,
            num_estimators=model_cfg['num_estimators'],
            dropout_rate=model_cfg['dropout_rate'],
            cifar_backbone='wrn28_10',
        ).to(device)

    # --- Training ---
    if args.model == 'edl' and args.train_edl:
        print("--- Training EDL ---")
        train_EDL(
            model_EDL=model,
            train_loader=train_loader,
            device=device,
            EDL_mse_loss=EDL_mse_loss,
            num_classes=NUM_CLASSES,
            file_path=args.edl_path,
            epochs=EPOCHS,
            annealing_step=model_cfg['annealing_step'],
            lr=model_cfg['lr'],
            weight_decay=model_cfg['weight_decay'],
            save_and_upload_model_fn=save_model,
            dataset_name=args.dataset,
            optimizer_type=model_cfg.get('optimizer', 'adam'),
            scheduler_type=model_cfg.get('scheduler', 'step'),
        )

    elif args.model == 'dip_edl':
        # For CIFAR-10 and LAMOST: CNN must be loaded before GDA fitting.
        if not args.train_cnn and args.dataset in ('cifar10', 'lamost'):
            print("Loading pre-trained CNN weights (required before GDA fitting)...")
            model.load_cnn_weights(args.cnn_path, device)

        if args.train_cnn:
            print("--- Training DIP-EDL CNN Component ---")
            train_CNN(
                model_CNN=model.cnn,
                train_loader=train_loader,
                device=device,
                file_path=args.cnn_path,
                epochs=model_cfg['epochs'],
                lr=model_cfg['lr'],
                save_and_upload_model_fn=save_model,
                optimizer_type=model_cfg.get('optimizer', 'adam'),
                scheduler_type=model_cfg.get('scheduler', 'step'),
            )

        if args.train_maf and args.dataset == 'mnist':
            print("--- Training DIP-EDL MAF (MNIST pixel-space) ---")
            train_density_estimator(
                model_wrapper=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                file_path=args.maf_path,
                learning_rate=maf_cfg['lr'],
                weight_decay=maf_cfg['weight_decay'],
                num_epochs=maf_cfg['epochs'],
                early_stopping_patience=maf_cfg['early_stopping_patience'],
                scheduler_step_size=maf_cfg['scheduler_step_size'],
                save_and_upload_model_fn=save_model,
                dataset_name=args.dataset,
            )

    elif args.model == 'redl' and args.train_redl:
        print("--- Training R-EDL ---")
        ckpt_dir = os.path.dirname(args.redl_path) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        redl_config_dict = {
            'dataset_name': 'MNIST' if args.dataset in ('mnist', 'lamost') else 'CIFAR10',
            'model_type': 'menet',
            'batch_size': BATCH_SIZE,
            'split': [1 - args.val_split, args.val_split],
            'loss': 'MEDL'
        }
        redl_train(
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=EPOCHS,
            model_path=args.redl_path,
            full_config_dict=redl_config_dict,
            use_wandb=False,
            device=device,
            is_fisher=False,
            output_dim=NUM_CLASSES
        )
        best_path = args.redl_path + "_best"
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.model.load_state_dict(state['model_state_dict'])
            print(f"Loaded best R-EDL weights from {best_path}")
            redl_state_dict_path = args.redl_path
            if not redl_state_dict_path.endswith('.pth'):
                redl_state_dict_path += '.pth'
            save_model(model.model, redl_state_dict_path)
        else:
            print(f"Best checkpoint not found at {best_path}; using in-memory weights.")

    elif args.model == 'reedl' and args.train_reedl:
        print("--- Training Re-EDL ---")
        ckpt_dir = os.path.dirname(args.reedl_path) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
        reedl_config_dict = {
            'dataset_name': 'MNIST' if args.dataset in ('mnist', 'lamost') else 'CIFAR10',
            'model_type': 'menet',
            'batch_size': BATCH_SIZE,
            'split': [1 - args.val_split, args.val_split],
            'loss': 'MEDL',
        }
        redl_train(
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=EPOCHS,
            model_path=args.reedl_path,
            full_config_dict=reedl_config_dict,
            use_wandb=False,
            device=device,
            is_fisher=False,
            output_dim=NUM_CLASSES
        )
        best_path = args.reedl_path + "_best"
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.model.load_state_dict(state['model_state_dict'])
            print(f"Loaded best Re-EDL weights from {best_path}")
            reedl_state_dict_path = args.reedl_path
            if not reedl_state_dict_path.endswith('.pth'):
                reedl_state_dict_path += '.pth'
            save_model(model.model, reedl_state_dict_path)
        else:
            print(f"Best checkpoint not found at {best_path}; using in-memory weights.")

    elif args.model == 'daedl' and args.train_daedl:
        print("--- Training DAEDL ---")
        train_daedl(
            model=model.model,
            learning_rate=model_cfg['lr'],
            reg_param=model_cfg['reg'],
            num_epochs=EPOCHS,
            trainloader=train_loader,
            validloader=val_loader,
            num_classes=NUM_CLASSES,
            device=device
        )
        save_model(model.model, args.daedl_path)
        model.fit_density(train_loader)

    elif args.model == 'ensemble' and args.train_ensemble:
        print("--- Training Deep Ensemble ---")
        train_deep_ensemble(
            ensemble_model=model,
            train_loader=train_loader,
            device=device,
            file_path_prefix=args.ensemble_path,
            epochs=EPOCHS,
            lr=model_cfg['lr'],
            save_and_upload_model_fn=save_model,
            optimizer_type=model_cfg.get('optimizer', 'adam'),
            scheduler_type=model_cfg.get('scheduler', 'step'),
        )

    elif args.model == 'mc_dropout' and args.train_mc_dropout:
        print("--- Training MC Dropout ---")
        train_CNN(
            model_CNN=model.mc_model.core_model,
            train_loader=train_loader,
            device=device,
            file_path=args.mc_dropout_path,
            epochs=EPOCHS,
            lr=model_cfg['lr'],
            save_and_upload_model_fn=save_model,
            optimizer_type=model_cfg.get('optimizer', 'adam'),
            scheduler_type=model_cfg.get('scheduler', 'step'),
        )

    elif args.model == 'postnet' and args.train_postnet:
        print("--- Training PostNet ---")
        if not hasattr(train_loader.dataset, 'output_dim'):
            train_loader.dataset.output_dim = NUM_CLASSES
        if not hasattr(val_loader.dataset, 'output_dim'):
            val_loader.dataset.output_dim = NUM_CLASSES
        train_postnet(
            model=model.model,
            train_loader=PostNetLoaderWrapper(train_loader),
            val_loader=PostNetLoaderWrapper(val_loader),
            max_epochs=EPOCHS,
            frequency=2,
            patience=10,
            model_path=args.postnet_path,
            full_config_dict={}
        )
        save_model(model.model, args.postnet_path)

    # --- Load Weights ---
    print("--- Checking and loading weights ---")

    if args.model == 'edl' and not args.train_edl:
        print(f"[Weights] Loading EDL from {args.edl_path}")
        model.load_edl_weights(args.edl_path, device)

    elif args.model == 'dip_edl':
        # CNN weights for MNIST are loaded here; for CIFAR-10/LAMOST loaded before GDA above.
        if not args.train_cnn and args.dataset == 'mnist':
            print(f"[Weights] Loading CNN from {args.cnn_path}")
            model.load_cnn_weights(args.cnn_path, device)

        if args.dataset == 'mnist':
            if not args.train_maf:
                print(f"[Weights] Loading MAF from {args.maf_path}")
                model.load_maf_weights(args.maf_path, device)
        elif args.dataset in ('cifar10', 'lamost'):
            # GDA is always re-fitted from loaded CNN weights (closed-form, fast).
            gda_loader = train_loader_gda if args.dataset == 'cifar10' else train_loader
            print("Fitting GDA density estimator...")
            model.fit_density(gda_loader, device)

    elif args.model == 'redl' and not args.train_redl:
        redl_state_dict_path = args.redl_path
        if not redl_state_dict_path.endswith('.pth'):
            redl_state_dict_path += '.pth'
        print(f"[Weights] Loading R-EDL from {redl_state_dict_path}")
        model.load_weights(redl_state_dict_path, device)

    elif args.model == 'reedl' and not args.train_reedl:
        reedl_state_dict_path = args.reedl_path
        if not reedl_state_dict_path.endswith('.pth'):
            reedl_state_dict_path += '.pth'
        print(f"[Weights] Loading Re-EDL from {reedl_state_dict_path}")
        model.load_weights(reedl_state_dict_path, device)

    elif args.model == 'daedl' and not args.train_daedl:
        print(f"[Weights] Loading DAEDL from {args.daedl_path}")
        model.load_weights(args.daedl_path)
        model.fit_density(train_loader)

    elif args.model == 'postnet' and not args.train_postnet:
        print(f"[Weights] Loading PostNet from {args.postnet_path}")
        model.load_weights(args.postnet_path, device)

    elif args.model == 'ensemble' and not args.train_ensemble:
        print(f"[Weights] Loading Ensemble from {args.ensemble_path}")
        model.load_weights(args.ensemble_path, device)

    elif args.model == 'mc_dropout' and not args.train_mc_dropout:
        print(f"[Weights] Loading MC Dropout from {args.mc_dropout_path}")
        model.load_weights(args.mc_dropout_path, device)

    else:
        print("[Weights] Model trained from scratch — no weights to load.")

    # --- Calibrate DIP-EDL density (MNIST only) ---
    if args.model == 'dip_edl' and args.dataset == 'mnist':
        model.to(device)
        model.calibrate_density(train_loader, device)

    # --- Cast to float32 for stable evaluation ---
    model.to(device)
    model.float()
    print("Model cast to float32 for evaluation.")

    # --- Evaluation ---
    print("\n--- Running Benchmark ---")
    if args.model in ('ensemble', 'mc_dropout'):
        id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets = run_benchmark_baseline(
            model=model,
            test_loader_id=test_loader_id,
            ood_loaders=ood_loaders,
            device=device,
            num_classes=NUM_CLASSES,
        )
    else:
        min_alpha = getattr(model, 'min_alpha', None)
        id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets = run_benchmark(
            model=model,
            test_loader_id=test_loader_id,
            ood_loaders=ood_loaders,
            device=device,
            num_classes=NUM_CLASSES,
            min_alpha=min_alpha,
        )

    # --- Report ---
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Dataset: {args.dataset.upper()} | Model: {args.model.upper()}")
    print(f"  ID Accuracy:  {id_accuracy * 100:.2f}%")
    print(f"  ID Brier:     {id_brier:.4f}")
    for ood_name, metrics in ood_results.items():
        print(f"\n  OOD vs {ood_name.upper()}:")
        print(f"    AUROC:     {metrics['AUROC']:.4f}")
        print(f"    AUPR:      {metrics['AUPR']:.4f}")
        print(f"    OOD Brier: {metrics['OOD_Brier']:.4f}")
    print("------------------------------------------")

    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj

    results_payload = {
        'id_acc': id_accuracy,
        'id_brier': id_brier,
        # CIFAR-10 OOD
        'cifar100_auroc': ood_results.get('cifar100', {}).get('AUROC', 0),
        'cifar100_aupr':  ood_results.get('cifar100', {}).get('AUPR', 0),
        'cifar100_brier': ood_results.get('cifar100', {}).get('OOD_Brier', 0),
        'svhn_auroc':     ood_results.get('svhn', {}).get('AUROC', 0),
        'svhn_aupr':      ood_results.get('svhn', {}).get('AUPR', 0),
        'svhn_brier':     ood_results.get('svhn', {}).get('OOD_Brier', 0),
        # MNIST OOD
        'omniglot_auroc': ood_results.get('omniglot', {}).get('AUROC', 0),
        'omniglot_aupr':  ood_results.get('omniglot', {}).get('AUPR', 0),
        'omniglot_brier': ood_results.get('omniglot', {}).get('OOD_Brier', 0),
        'kmnist_auroc':   ood_results.get('kmnist', {}).get('AUROC', 0),
        'kmnist_aupr':    ood_results.get('kmnist', {}).get('AUPR', 0),
        'kmnist_brier':   ood_results.get('kmnist', {}).get('OOD_Brier', 0),
        # LAMOST OOD
        **{f'{k}_auroc': v.get('AUROC', 0) for k, v in ood_results.items() if k in ('star', 'quasar', 'galaxy')},
        **{f'{k}_aupr':  v.get('AUPR', 0)  for k, v in ood_results.items() if k in ('star', 'quasar', 'galaxy')},
        **{f'{k}_brier': v.get('OOD_Brier', 0) for k, v in ood_results.items() if k in ('star', 'quasar', 'galaxy')},
    }

    print(f"__JSON_START__{json.dumps(results_payload, default=make_serializable)}__JSON_END__")


if __name__ == '__main__':
    main()
