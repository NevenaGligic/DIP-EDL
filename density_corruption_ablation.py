"""
Density Corruption Ablation for nu-EDL.

Simulates a degraded density estimator by injecting Gaussian noise into
the z-scored log-prob before the exp() in the forward pass:

    log_prob_noisy = log_prob_scaled + sigma * epsilon,  epsilon ~ N(0,1)

At sigma=0:   clean density estimator (original model)
At sigma=1:   noise std equals signal std (SNR ~ 1)
At sigma=inf: density replaced entirely by N(0,1) — uninformative

Multiple noise seeds are run per sigma level to produce mean ± std.

Key predictions:
  - ID Accuracy:  invariant to sigma (argmax depends only on P(y|x))
  - AUROC/AUPR:   degrades from peak toward 0.5 as sigma grows
  - ID Brier:     mild degradation as calibration worsens

Usage:
    python density_corruption_ablation.py --dataset mnist  --seed 42
    python density_corruption_ablation.py --dataset cifar10 --seed 42
"""

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

from src.dataloaders import get_experiment_loaders
from src.models.dip_edl import dip_EDL
from src.utils import get_best_device
from src.experiments import run_benchmark

# -------------------------------------------------------
# Config (mirrors main.py / gamma_ablation.py)
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

# Noise levels: 0 = clean, inf = pure random density
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0, float('inf')]
NOISE_SEEDS  = [0, 1, 2, 3, 4]   # repeated runs per noise level for error bars


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigma_label(sigma):
    return "inf (random)" if sigma == float('inf') else str(sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     type=str,  default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--cnn_path',    type=str,  default=None)
    parser.add_argument('--maf_path',    type=str,  default=None)
    parser.add_argument('--val_split',   type=float, default=0.2)
    parser.add_argument('--output_file', type=str,  default=None)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"results/density_corruption_results_{args.dataset}.jsonl"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    set_seed(args.seed)

    cfg         = CONFIGS[args.dataset]
    maf_cfg     = cfg['maf']
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

    # --- Model (load once, reuse across all conditions) ---
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

    ood_names   = list(ood_loaders.keys())
    all_results = []  # one entry per (sigma, noise_seed)

    print(f"\nRunning density corruption ablation over sigma = {[sigma_label(s) for s in NOISE_LEVELS]}...")
    print(f"Noise seeds per level: {NOISE_SEEDS}\n")

    for sigma in NOISE_LEVELS:
        # sigma=0 is deterministic — only one run needed
        seeds_to_run = [NOISE_SEEDS[0]] if sigma == 0.0 else NOISE_SEEDS

        run_metrics = {
            'id_acc': [], 'id_brier': [],
            **{f'{n}_auroc': [] for n in ood_names},
            **{f'{n}_aupr':  [] for n in ood_names},
            **{f'{n}_brier': [] for n in ood_names},
        }

        for noise_seed in seeds_to_run:
            # Set noise seed so each run gets a different noise realization
            torch.manual_seed(noise_seed)
            np.random.seed(noise_seed)

            model.density_noise_std = sigma
            print(f"  sigma = {sigma_label(sigma)}, noise_seed = {noise_seed}")

            id_accuracy, id_brier, ood_results, _, _ = run_benchmark(
                model=model,
                test_loader_id=test_loader_id,
                ood_loaders=ood_loaders,
                device=device,
                num_classes=NUM_CLASSES,
                min_alpha=getattr(model, 'min_alpha', None),
            )

            run_metrics['id_acc'].append(id_accuracy)
            run_metrics['id_brier'].append(id_brier)
            for name, metrics in ood_results.items():
                run_metrics[f'{name}_auroc'].append(metrics['AUROC'])
                run_metrics[f'{name}_aupr'].append(metrics['AUPR'])
                run_metrics[f'{name}_brier'].append(metrics['OOD_Brier'])

            # Stream raw run to file
            raw_row = {
                'sigma': sigma_label(sigma),
                'noise_seed': noise_seed,
                'id_acc': id_accuracy,
                'id_brier': id_brier,
            }
            for name, metrics in ood_results.items():
                raw_row[f'{name}_auroc'] = metrics['AUROC']
                raw_row[f'{name}_aupr']  = metrics['AUPR']
                raw_row[f'{name}_brier'] = metrics['OOD_Brier']
            raw_row['seed']      = args.seed
            raw_row['dataset']   = args.dataset
            raw_row['timestamp'] = str(datetime.datetime.now())
            with open(args.output_file, 'a') as f:
                f.write(json.dumps(raw_row) + '\n')

        # Aggregate: mean ± std across noise seeds
        agg = {'sigma': sigma_label(sigma), 'n_seeds': len(seeds_to_run)}
        for key, vals in run_metrics.items():
            agg[key + '_mean'] = float(np.mean(vals))
            agg[key + '_std']  = float(np.std(vals)) if len(vals) > 1 else 0.0
        all_results.append(agg)

    # Reset model to clean state
    model.density_noise_std = 0.0

    # --- Print summary table ---
    print("\n\n========== DENSITY CORRUPTION ABLATION RESULTS ==========")
    print(f"Dataset: {args.dataset.upper()} | Model: nu-EDL | Seed: {args.seed}")
    print(f"Values shown as mean ± std over {len(NOISE_SEEDS)} noise seeds (sigma=0 is deterministic)\n")

    header = f"{'sigma':>12}  {'ID Acc%':>14}  {'ID Brier':>14}"
    for name in ood_names:
        header += f"  {name+' AUROC':>20}  {name+' AUPR':>18}  {name+' OOD Brier':>20}"
    print(header)
    print("-" * len(header))

    for row in all_results:
        def fmt(key):
            m, s = row[key + '_mean'], row[key + '_std']
            return f"{m:.4f}±{s:.4f}"

        line = f"{row['sigma']:>12}  {row['id_acc_mean']*100:>6.2f}%±{row['id_acc_std']*100:.2f}%  {fmt('id_brier'):>14}"
        for name in ood_names:
            line += f"  {fmt(name+'_auroc'):>20}  {fmt(name+'_aupr'):>18}  {fmt(name+'_brier'):>20}"
        print(line)

    print("=" * len(header))
    print(f"\nRaw per-run results saved to: {args.output_file}")
    print("\nKey result to highlight for reviewers:")
    clean = all_results[0]
    random = all_results[-1]
    print(f"  Clean density  (sigma=0):   ID Acc = {clean['id_acc_mean']*100:.2f}%")
    print(f"  Random density (sigma=inf): ID Acc = {random['id_acc_mean']*100:.2f}%  (should be identical)")
    for name in ood_names:
        print(f"  AUROC degradation on {name}: {clean[name+'_auroc_mean']:.4f} -> {random[name+'_auroc_mean']:.4f}±{random[name+'_auroc_std']:.4f}")


if __name__ == '__main__':
    main()
