import torch
import numpy as np
from tqdm import tqdm
from src.metrics import calculate_ood_auroc, get_edl_total_uncertainty, calculate_classification_accuracy, calculate_brier_score
from src.utils import stabilize_alpha


def run_benchmark(model: torch.nn.Module, test_loader_id, ood_loaders: dict, device: torch.device, num_classes: int, min_alpha: float = None):
    """
    Runs unified OOD evaluation for one model and returns benchmark metrics.

    Returns:
        id_accuracy, id_brier, ood_results (dict), all_id_alphas, all_id_targets
    """
    model.eval()

    all_id_alphas = []
    all_id_probs = []
    all_id_targets = []

    print("Collecting ID data...")
    with torch.no_grad():
        for data, target in tqdm(test_loader_id, desc="ID Inference"):
            alpha = model(data.to(device)).cpu().numpy()
            if min_alpha is not None:
                alpha = stabilize_alpha(alpha, min_alpha)
            S = np.sum(alpha, axis=1, keepdims=True)
            probs = alpha / S
            all_id_alphas.append(alpha)
            all_id_probs.append(probs)
            all_id_targets.append(target.numpy())

    all_id_alphas = np.concatenate(all_id_alphas, axis=0)
    all_id_probs = np.concatenate(all_id_probs, axis=0)
    all_id_targets = np.concatenate(all_id_targets, axis=0)

    id_accuracy = calculate_classification_accuracy(all_id_probs, all_id_targets)
    print(f"Accuracy ID: {id_accuracy}")
    id_brier = calculate_brier_score(all_id_probs, all_id_targets, num_classes, is_ood=False)
    print(f"Brier ID: {id_brier}")

    id_uncertainty_scores = get_edl_total_uncertainty(all_id_alphas)
    problematic_mask = (id_uncertainty_scores == 0) | np.isnan(id_uncertainty_scores)
    if np.any(problematic_mask):
        print(f"Warning: {np.count_nonzero(problematic_mask)} zero/NaN scores in ID uncertainty.")

    ood_results = {}
    for ood_name, ood_loader in ood_loaders.items():
        all_ood_alphas = []
        all_ood_probs = []

        print(f"Collecting OOD data for {ood_name}...")
        with torch.no_grad():
            for data, _ in tqdm(ood_loader, desc=f"OOD Inference ({ood_name})"):
                alpha = model(data.to(device)).cpu().numpy()
                if min_alpha is not None:
                    alpha = stabilize_alpha(alpha, min_alpha)
                S = np.sum(alpha, axis=1, keepdims=True)
                probs = alpha / S
                all_ood_alphas.append(alpha)
                all_ood_probs.append(probs)

        all_ood_alphas = np.concatenate(all_ood_alphas, axis=0)
        all_ood_probs = np.concatenate(all_ood_probs, axis=0)

        ood_uncertainty_scores = get_edl_total_uncertainty(all_ood_alphas)
        auroc, aupr = calculate_ood_auroc(id_uncertainty_scores, ood_uncertainty_scores)
        dummy_targets = np.zeros(all_ood_probs.shape[0])
        ood_brier = calculate_brier_score(all_ood_probs, dummy_targets, num_classes, is_ood=True)

        ood_results[ood_name] = {'AUROC': auroc, 'AUPR': aupr, 'OOD_Brier': ood_brier}

    return id_accuracy, id_brier, ood_results, all_id_alphas, all_id_targets
