"""
Gamma Ablation Experiment for nu-EDL.

Tests the effect of scaling the density estimator output by gamma:
    evidence = N * gamma * p(x) * P(y|x)

gamma in {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}

Usage:
    python gamma_ablation.py --dataset mnist --seed 42
"""

import argparse
import os
import sys
import json
import datetime
import torch
import numpy as np

from src.dataloaders import get_experiment_loaders
from src.models.dip_edl import dip_EDL
from src.utils import get_best_device
from src.experiments import run_benchmark

# -------------------------------------------------------
# Config (mirrors main.py)
# -------------------------------------------------------
CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_dims': (1, 28, 28),
        'maf': {
            'batch_size': 128,
            'maf_hidden_features': 1024,
            'maf_num_layers': 10,
            'maf_blocks': 20,
        },
        'cnn': {'batch_size': 128},
    },
    'cifar10': {
        'num_classes': 10,
        'input_dims': (3, 32, 32),
        'maf': {
            'batch_size': 128,
            'maf_hidden_features': 1024,
            'maf_num_layers': 10,
            'maf_blocks': 20,
        },
        'cnn': {'batch_size': 128},
    },
}

GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--cnn_path', type=str, default=None)
    parser.add_argument('--maf_path', type=str, default=None)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"results/gamma_ablation_results_{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    set_seed(args.seed)

    cfg        = CONFIGS[args.dataset]
    maf_cfg    = cfg['maf']
    NUM_CLASSES = cfg['num_classes']
    INPUT_DIMS  = cfg['input_dims']
    BATCH_SIZE  = cfg['cnn']['batch_size']

    save_dir    = "saved_model_weights"
    seed_suffix = f"seed{args.seed}"

    if args.cnn_path is None:
        args.cnn_path = f"{save_dir}/{args.dataset}_dip_EDL_CNN_model_{seed_suffix}.pth"
    if args.maf_path is None:
        args.maf_path = f"{save_dir}/{args.dataset}_dip_EDL_MAF_model_{seed_suffix}.pth"

    device = get_best_device()
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading data...")
    _loaders = get_experiment_loaders(
        args.dataset, BATCH_SIZE, val_split=args.val_split, seed=args.seed
    )
    train_loader, val_loader, test_loader_id, ood_loaders = _loaders[:4]
    train_loader_gda = _loaders[4] if len(_loaders) > 4 else train_loader

    # --- Model (load once, reuse across all gammas) ---
    print("Initializing and loading model...")
    model = dip_EDL(
        num_classes=NUM_CLASSES,
        input_dims=INPUT_DIMS,
        maf_hidden_features=maf_cfg['maf_hidden_features'],
        maf_num_layers=maf_cfg['maf_num_layers'],
        maf_blocks=maf_cfg['maf_blocks'],
        n_train_samples=len(train_loader.dataset),
    ).to(device)

    model.load_cnn_weights(args.cnn_path, device)

    if args.dataset == 'mnist':
        model.load_maf_weights(args.maf_path, device)
        model.to(device)
        model.calibrate_density(train_loader, device)
    else:
        model.fit_density(train_loader_gda, device)

    model.to(device)
    model.float()
    model.eval()

    # --- Gamma sweep ---
    all_results = []
    ood_names = list(ood_loaders.keys())

    print(f"\nRunning gamma ablation over {GAMMAS}...\n")

    for gamma in GAMMAS:
        model.gamma = gamma
        print(f"  gamma = {gamma}")

        id_accuracy, id_brier, ood_results, _, _ = run_benchmark(
            model=model,
            test_loader_id=test_loader_id,
            ood_loaders=ood_loaders,
            device=device,
            num_classes=NUM_CLASSES,
            min_alpha=getattr(model, 'min_alpha', None),
        )

        row = {
            'gamma':    gamma,
            'id_acc':   id_accuracy,
            'id_brier': id_brier,
        }
        for name, metrics in ood_results.items():
            row[f'{name}_auroc'] = metrics['AUROC']
            row[f'{name}_aupr']  = metrics['AUPR']
            row[f'{name}_brier'] = metrics['OOD_Brier']

        row['seed']      = args.seed
        row['dataset']   = args.dataset
        row['timestamp'] = str(datetime.datetime.now())
        all_results.append(row)

        # Stream results to file as we go
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(row) + '\n')

    # --- Print table ---
    print("\n\n========== GAMMA ABLATION RESULTS ==========")
    print(f"Dataset: {args.dataset.upper()} | Model: DIP-EDL | Seed: {args.seed}")
    print()

    # Header
    col_gamma  = "gamma"
    col_idacc  = "ID Acc%"
    col_idbr   = "ID Brier"
    ood_cols   = []
    for name in ood_names:
        ood_cols += [f"{name} AUROC", f"{name} AUPR"]

    header = f"{'gamma':>6}  {'ID Acc%':>8}  {'ID Brier':>9}"
    for name in ood_names:
        header += f"  {name+' AUROC':>14}  {name+' AUPR':>12}  {name+' Brier':>13}"
    print(header)
    print("-" * len(header))

    for row in all_results:
        line = f"{row['gamma']:>6.1f}  {row['id_acc']*100:>7.2f}%  {row['id_brier']:>9.4f}"
        for name in ood_names:
            line += f"  {row[f'{name}_auroc']:>14.4f}  {row[f'{name}_aupr']:>12.4f}  {row[f'{name}_brier']:>13.4f}"
        print(line)

    print("=" * len(header))
    print(f"\nResults also saved to: {args.output_file}")


if __name__ == '__main__':
    main()
