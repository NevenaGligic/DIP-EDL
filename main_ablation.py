import argparse
import os
import torch
import numpy as np
import json

from src.dataloaders import get_experiment_loaders
from src.train import train_CNN, train_density_estimator
from src.models.dip_edl_ablation import dip_EDL
from src.utils import get_best_device, save_model
from src.experiments import run_benchmark


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
        # StandardCNN (ResNet-18 adapted): Adam + step scheduler
        'cnn': {
            'batch_size': 128,
            'lr': 1e-3,
            'epochs': 50,
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
        # WideResNet-28-10: SGD + cosine (standard WRN training recipe)
        'cnn': {
            'batch_size': 128,
            'lr': 0.1,
            'epochs': 100,
            'optimizer': 'sgd',
            'scheduler': 'cosine',
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Ablation study for DIP-EDL components.")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--task', type=str, default='3',
                        choices=['1a', '1b', '1c', '2a', '2b', '2c', '3'],
                        help="Which component combination to evaluate:\n"
                             "  1a: N only | 1b: p(x) only | 1c: P(y|x) only\n"
                             "  2a: N×p(x) | 2b: N×P(y|x) | 2c: p(x)×P(y|x)\n"
                             "  3:  full model (default)")
    parser.add_argument('--results_file', type=str, default=None,
                        help="JSONL file to append results to (default: ablation_results.jsonl).")
    parser.add_argument('--val_split', type=float, default=0.2)

    parser.add_argument('--train_cnn', action='store_true', help="Train the CNN backbone from scratch.")
    parser.add_argument('--train_maf', action='store_true', help="Train the MAF density estimator from scratch (MNIST only).")

    parser.add_argument('--cnn_path', type=str, default=None)
    parser.add_argument('--maf_path', type=str, default=None)

    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)

    dataset_cfg = CONFIGS[args.dataset]
    maf_cfg     = dataset_cfg['maf']
    cnn_cfg     = dataset_cfg['cnn']

    NUM_CLASSES = dataset_cfg['num_classes']
    INPUT_DIMS  = dataset_cfg['input_dims']
    BATCH_SIZE  = cnn_cfg['batch_size']

    save_dir = "saved_model_weights"
    os.makedirs(save_dir, exist_ok=True)
    seed_suffix = f"seed{args.seed}"

    if args.cnn_path is None:
        args.cnn_path = f"{save_dir}/{args.dataset}_dip_EDL_CNN_model_{seed_suffix}.pth"
    if args.maf_path is None:
        args.maf_path = f"{save_dir}/{args.dataset}_dip_EDL_MAF_model_{seed_suffix}.pth"

    device = get_best_device()
    print(f"Using device: {device}")

    print("Loading data...")
    _loaders = get_experiment_loaders(
        args.dataset, BATCH_SIZE, val_split=args.val_split, seed=args.seed
    )
    train_loader, val_loader, test_loader_id, ood_loaders = _loaders[:4]
    train_loader_gda = _loaders[4] if len(_loaders) > 4 else train_loader

    # --- Initialize Model ---
    print("Initializing model...")
    model = dip_EDL(
        num_classes=NUM_CLASSES,
        input_dims=INPUT_DIMS,
        maf_hidden_features=maf_cfg['maf_hidden_features'],
        maf_num_layers=maf_cfg['maf_num_layers'],
        maf_blocks=maf_cfg['maf_blocks'],
        n_train_samples=len(train_loader.dataset),
    ).to(device)

    # --- Training Phase ---
    # CIFAR-10: CNN must be available before GDA fitting — load it now if not training.
    if not args.train_cnn and args.dataset == 'cifar10':
        print(f"[Weights] Loading CNN from {args.cnn_path}")
        model.load_cnn_weights(args.cnn_path, device)

    if args.train_cnn:
        print("--- Training DIP-EDL CNN ---")
        # MNIST → StandardCNN (Adam+step); CIFAR-10 → WideResNet (SGD+cosine)
        train_CNN(
            model_CNN=model.cnn,
            train_loader=train_loader,
            device=device,
            file_path=args.cnn_path,
            epochs=cnn_cfg['epochs'],
            lr=cnn_cfg['lr'],
            save_and_upload_model_fn=save_model,
            optimizer_type=cnn_cfg.get('optimizer', 'adam'),
            scheduler_type=cnn_cfg.get('scheduler', 'step'),
        )

    if args.train_maf and args.dataset == 'mnist':
        print("--- Training DIP-EDL MAF (MNIST pixel-space density) ---")
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

    # --- Load Weights ---
    print("--- Checking and loading weights ---")

    # MNIST: CNN loaded here. CIFAR-10: already loaded above before GDA fitting.
    if not args.train_cnn and args.dataset == 'mnist':
        print(f"[Weights] Loading CNN from {args.cnn_path}")
        model.load_cnn_weights(args.cnn_path, device)

    if args.dataset == 'mnist':
        if not args.train_maf:
            print(f"[Weights] Loading MAF from {args.maf_path}")
            model.load_maf_weights(args.maf_path, device)
    elif args.dataset == 'cifar10':
        # GDA is always re-fitted (closed-form, fast); CNN weights already in model.
        print("Fitting GDA density estimator...")
        model.fit_density(train_loader_gda, device)

    # --- Ablation task + calibration ---
    model.task = args.task

    if args.dataset == 'mnist':
        model.to(device)
        model.calibrate_density(train_loader, device)

    model.to(device)
    model.float()
    print("Model cast to float32 for stable evaluation.")

    # --- Evaluation ---
    print(f"\n--- Running Benchmark (task={args.task}) ---")
    id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets = run_benchmark(
        model=model,
        test_loader_id=test_loader_id,
        ood_loaders=ood_loaders,
        device=device,
        num_classes=NUM_CLASSES,
        min_alpha=getattr(model, 'min_alpha', None),
    )

    # --- Report ---
    print("\n\n--- EVALUATION SUMMARY ---")
    print(f"Dataset: {args.dataset.upper()} | Ablation task: {args.task}")
    print(f"  ID Accuracy: {id_accuracy * 100:.2f}%")
    print(f"  ID Brier:    {id_brier:.4f}")
    for ood_name, metrics in ood_results.items():
        print(f"\n  OOD vs {ood_name.upper()}:")
        print(f"    AUROC: {metrics['AUROC']:.4f}  AUPR: {metrics['AUPR']:.4f}  Brier: {metrics['OOD_Brier']:.4f}")
    print("------------------------------------------")

    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        return obj

    results_payload = {
        'ablation_task': args.task,
        'id_acc':    id_accuracy,
        'id_brier':  id_brier,
        'cifar100_auroc': ood_results.get('cifar100', {}).get('AUROC', 0),
        'cifar100_aupr':  ood_results.get('cifar100', {}).get('AUPR', 0),
        'cifar100_brier': ood_results.get('cifar100', {}).get('OOD_Brier', 0),
        'svhn_auroc':     ood_results.get('svhn', {}).get('AUROC', 0),
        'svhn_aupr':      ood_results.get('svhn', {}).get('AUPR', 0),
        'svhn_brier':     ood_results.get('svhn', {}).get('OOD_Brier', 0),
        'omniglot_auroc': ood_results.get('omniglot', {}).get('AUROC', 0),
        'omniglot_aupr':  ood_results.get('omniglot', {}).get('AUPR', 0),
        'omniglot_brier': ood_results.get('omniglot', {}).get('OOD_Brier', 0),
        'kmnist_auroc':   ood_results.get('kmnist', {}).get('AUROC', 0),
        'kmnist_aupr':    ood_results.get('kmnist', {}).get('AUPR', 0),
        'kmnist_brier':   ood_results.get('kmnist', {}).get('OOD_Brier', 0),
    }

    print(f"__JSON_START__{json.dumps(results_payload, default=make_serializable)}__JSON_END__")

    import datetime
    record = {
        **{k: make_serializable(v) for k, v in results_payload.items()},
        'model':     'dip_edl',
        'dataset':   args.dataset,
        'seed':      args.seed,
        'timestamp': str(datetime.datetime.now()),
    }
    out_file = args.results_file or 'results/ablation_results.jsonl'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'a') as fh:
        fh.write(json.dumps(record) + '\n')
    print(f"[Saved result to {out_file}]")


if __name__ == '__main__':
    main()
