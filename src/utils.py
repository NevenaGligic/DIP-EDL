import numpy as np
import torch
import os


def get_weight_path(filename):
    """
    Locates the weight file on local disk.
    Returns the absolute file path, or raises FileNotFoundError with instructions.
    """
    path = os.path.abspath(filename)
    if os.path.exists(path):
        print(f"[Weights] Loaded from: {filename}")
        return path
    raise FileNotFoundError(
        f"Weight file not found: {filename}\n"
        "Run training first (e.g. --train_edl, --train_cnn) to generate weights."
    )


def save_model(model, model_path):
    """Save model weights to disk."""
    parent_dir = os.path.dirname(model_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[Weights] Saved to: {model_path}")


def load_model_smart(model_class, filename, device, **kwargs):
    path = get_weight_path(filename)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ----------------------------------------
# GPU Selection
# ----------------------------------------
def get_best_device():
    """Selects the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        print("NVIDIA CUDA GPU found.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            print("Using single CUDA GPU: cuda:0")
            return torch.device("cuda:0")

        max_free_memory = 0
        best_gpu_index = 0
        for i in range(num_gpus):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total")
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_gpu_index = i

        print(f"Selected GPU {best_gpu_index} with the most free memory.")
        return torch.device(f"cuda:{best_gpu_index}")

    elif torch.backends.mps.is_available():
        print("Apple Metal (MPS) GPU found. Using mps.")
        return torch.device("mps")

    else:
        print("No GPU acceleration available. Using CPU.")
        return torch.device("cpu")


# ----------------------------------------
# MAF helper
# ----------------------------------------
def transform_for_maf(x: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
    """
    Applies pre-processing transforms for the MAF density estimator.

    MNIST: dequantize -> logit transform -> flatten.
    CIFAR-10 / LAMOST: flatten only (features are already continuous).
    """
    if dataset_name is None:
        flat_dim = x.numel() // x.size(0)
        is_cifar = (flat_dim != 784)
        lam = None if is_cifar else 1e-6
    else:
        is_cifar = (dataset_name.lower() in ('cifar10', 'lamost'))
        lam = None if is_cifar else 1e-6

    if is_cifar:
        return x.view(x.shape[0], -1)

    noise = torch.rand_like(x)
    x = (x * 255.0 + noise) / 256.0
    x = lam + (1 - 2 * lam) * x
    x_logit = torch.log(x / (1.0 - x))
    return x_logit.view(x_logit.size(0), -1)


def stabilize_alpha(alpha: np.ndarray, min_alpha: float = 1.0) -> np.ndarray:
    """
    Replaces corrupted alpha values (NaN, Inf, or below the model's theoretical
    minimum) with that minimum.

    min_alpha per model:
      - EDL, PostNet, DIP-EDL : 1.0   (alpha = evidence + 1)
      - R-EDL                 : lamb2 (typically 0.1)
      - Re-EDL                : lamb2 (typically 0.8 for CIFAR-10, 0.1 for MNIST)
      - DAEDL                 : 1e-6
    """
    is_corrupted = ~np.isfinite(alpha) | (alpha < min_alpha)
    if np.any(is_corrupted):
        alpha[is_corrupted] = min_alpha
    return alpha
