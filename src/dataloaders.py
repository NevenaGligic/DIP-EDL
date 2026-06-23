import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np

# --- Configuration ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
# ---------------------

def get_base_transforms(dataset='mnist', train=False):
    """
    Returns the appropriate transforms.
    Adds augmentation (Crop/Flip) only for CIFAR-10 training.
    """
    if dataset == 'mnist':
        return transforms.Compose([transforms.ToTensor()])
        
    elif dataset == 'cifar10':
        if train:
            # Standard CIFAR-10 Augmentation
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                ])
            
    return transforms.Compose([transforms.ToTensor()])

# ─── LAMOST ────────────────────────────────────────────────────────────────────
# Data: LAMOST DR9, 100k spectra in 10 FITS files (10k each).
# Classes in the FITS label column: Galaxy=0, Quasar=1, Star=2.
# ID setup:  Galaxy (relabelled 0) + Star (relabelled 1) — 2 classes.
# OOD setup: Quasar (original label 1) held out entirely.
#
# Data download:
#   https://www.dropbox.com/scl/fi/tp81mfopdqbep50vwhhhb/spectra_training_data.tar
#   Extract into  <data_dir>/lamost/  so that  *.fits  files are present.
#
# Preprocessing / feature extraction repo (must be cloned alongside this project):
#   git clone https://github.com/superdreamliner/LAMOST-Spectra-Classifier
#   Expected path: Code/EDL/LAMOST-Spectra-Classifier
#
# Feature pipeline (mirrors spectra_classify.py in the LAMOST repo):
#   1. load_fits_data + energy_normalization + spectra_smooth  (per file)
#   2. PCA(n_components=500)  ← fit on training spectra ONLY
#   3. extract_spectral_features  (6 known emission/absorption lines)
#   4. extract_stat_features      (statistical summary per block, n=50)
#   5. Concatenate → (N, n_features) — fed as (N, 1, n_features) to models
#
# Fitting is done on the training portion only (no leakage into val/test/OOD).

import sys as _sys

# Path to the cloned LAMOST-Spectra-Classifier repo
_LAMOST_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "LAMOST-Spectra-Classifier")
)

# Emission/absorption lines used for spectral feature extraction (in Angstroms).
_LAMOST_LINE_LIST = [6562.81, 4861.34, 4340.47, 4101.75, 5183.62, 5889.95]


def _import_lamost_functions():
    """
    Load functions.py from the LAMOST repo directly via importlib, bypassing
    utils/__init__.py which imports CNN_model.py and requires TensorFlow.
    """
    import importlib.util

    functions_path = os.path.join(_LAMOST_REPO_ROOT, 'utils', 'functions.py')
    if not os.path.isfile(functions_path):
        raise FileNotFoundError(
            f"LAMOST-Spectra-Classifier repo not found at:\n  {_LAMOST_REPO_ROOT}\n"
            "Clone it with:\n"
            "  git clone https://github.com/superdreamliner/LAMOST-Spectra-Classifier\n"
            f"into  {os.path.dirname(_LAMOST_REPO_ROOT)}"
        )

    spec = importlib.util.spec_from_file_location('lamost_functions', functions_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return (module.load_fits_data, module.energy_normalization, module.spectra_smooth,
            module.extract_spectral_features, module.extract_stat_features)


class LAMOSTDataset(Dataset):
    """
    Wraps pre-extracted LAMOST feature vectors.

    Each sample is a (1, n_features) float32 tensor, where n_features is
    determined by the PCA + spectral + statistical feature pipeline.
    No further normalisation is applied here — it is handled upstream.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # features: (N, n_features)   labels: (N,)
        self.labels = torch.tensor(labels, dtype=torch.long)
        # Shape (N, 1, n_features) — treat feature vector as 1-channel 1-D signal
        self.data = torch.tensor(features[:, None, :], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _load_lamost_fits(data_dir: str):
    """
    Load all FITS files, apply energy_normalization + spectra_smooth.
    Returns (spectra, labels) with shapes (N, 3000) and (N,).
    """
    (load_fits_data, energy_normalization, spectra_smooth,
     _, _) = _import_lamost_functions()

    lamost_dir = os.path.join(data_dir, 'lamost')
    if not os.path.isdir(lamost_dir):
        raise FileNotFoundError(
            f"LAMOST directory not found: {lamost_dir}\n"
            "Download the data from:\n"
            "  https://www.dropbox.com/scl/fi/tp81mfopdqbep50vwhhhb/"
            "spectra_training_data.tar\n"
            f"and extract into  {lamost_dir}/  so that  *.fits  files are present."
        )

    fits_files = sorted(
        os.path.join(lamost_dir, f)
        for f in os.listdir(lamost_dir)
        if f.endswith('.fits')
    )
    if not fits_files:
        raise FileNotFoundError(f"No .fits files found in {lamost_dir}")

    all_spectra, all_labels = [], []
    for path in fits_files:
        # load_fits_data → (N, 3001): flux cols 0..2999, label col 3000
        data_array = load_fits_data(path)
        data_array = energy_normalization(data_array)
        data_array = spectra_smooth(data_array, window_length=10, polyorder=3)
        all_spectra.append(data_array[:, :-1].astype(np.float32))  # (N, 3000)
        all_labels.append(data_array[:,  -1].astype(np.int64))     # (N,)

    return (np.concatenate(all_spectra, axis=0),
            np.concatenate(all_labels,  axis=0))


def _fit_lamost_features(spectra: np.ndarray):
    """
    Fit PCA + extract spectral/statistical features on training spectra.

    Parameters
    ----------
    spectra : (N_train, 3000) preprocessed flux arrays

    Returns
    -------
    transformers : dict  — fitted PCA object + line list (needed for inference)
    features     : (N_train, n_features) float32 feature matrix
    """
    from sklearn.decomposition import PCA  # type: ignore
    (_, _, _, extract_spectral_features,
     extract_stat_features) = _import_lamost_functions()

    # Their feature functions expect (N, 3001) with a dummy label column
    dummy = np.zeros((len(spectra), 1), dtype=np.float32)
    data_with_dummy = np.hstack([spectra, dummy])

    spectral_feats = extract_spectral_features(
        data_with_dummy, _LAMOST_LINE_LIST, window=3)           # (N, 6)
    stat_feats = extract_stat_features(data_with_dummy, n=50)   # (N, ?)

    pca = PCA(n_components=500, copy=True)
    pca_feats = pca.fit_transform(spectra)                      # (N, 500)

    features = np.hstack([pca_feats, spectral_feats, stat_feats]).astype(np.float32)
    print(f"[LAMOST] Feature dims — PCA: {pca_feats.shape[1]}, "
          f"spectral: {spectral_feats.shape[1]}, "
          f"stat: {stat_feats.shape[1]}, total: {features.shape[1]}")

    transformers = {'pca': pca, 'line_list': _LAMOST_LINE_LIST}
    return transformers, features


def _apply_lamost_features(spectra: np.ndarray, transformers: dict) -> np.ndarray:
    """Apply fitted transformers to a new batch of preprocessed spectra."""
    (_, _, _, extract_spectral_features,
     extract_stat_features) = _import_lamost_functions()

    dummy = np.zeros((len(spectra), 1), dtype=np.float32)
    data_with_dummy = np.hstack([spectra, dummy])

    spectral_feats = extract_spectral_features(
        data_with_dummy, transformers['line_list'], window=3)
    stat_feats = extract_stat_features(data_with_dummy, n=50)
    pca_feats  = transformers['pca'].transform(spectra)

    return np.hstack([pca_feats, spectral_feats, stat_feats]).astype(np.float32)


_LAMOST_CLASS_RAW = {'galaxy': 0, 'quasar': 1, 'star': 2}


def _get_lamost_loaders(batch_size: int, val_split: float, seed: int, data_dir: str,
                        ood_class: str = 'star'):
    """
    Build LAMOST loaders using the full LAMOST-repo feature pipeline.

    ood_class : one of 'star', 'quasar', 'galaxy' — held out as OOD.
                The remaining two classes become the 2-class ID task,
                relabelled 0 and 1 in sorted order of their raw labels.

    Split order:
      1. Load + preprocess all spectra
      2. Separate ID (two classes) from OOD (one class)
      3. Split ID into train / val / test  (60 / 20 / 20)
      4. Fit PCA + feature extractors on TRAINING spectra only
      5. Apply to all splits — no leakage
    """
    if ood_class not in _LAMOST_CLASS_RAW:
        raise ValueError(f"ood_class must be one of {list(_LAMOST_CLASS_RAW)}; got '{ood_class}'")

    spectra, labels = _load_lamost_fits(data_dir)

    # ── separate classes ──────────────────────────────────────────────────────
    ood_raw  = _LAMOST_CLASS_RAW[ood_class]
    id_raws  = sorted(v for k, v in _LAMOST_CLASS_RAW.items() if k != ood_class)

    id_mask  = np.isin(labels, id_raws)
    ood_mask = (labels == ood_raw)

    id_spectra     = spectra[id_mask]
    id_labels_orig = labels[id_mask]
    # Relabel: smaller raw label → 0, larger → 1
    id_labels = (id_labels_orig == id_raws[1]).astype(np.int64)
    ood_spectra = spectra[ood_mask]

    id_names = {v: k for k, v in _LAMOST_CLASS_RAW.items()}
    print(f"[LAMOST] ID: {id_names[id_raws[0]]}={id_mask.sum() - (id_labels_orig == id_raws[1]).sum()}, "
          f"{id_names[id_raws[1]]}={(id_labels_orig == id_raws[1]).sum()}  |  "
          f"OOD ({ood_class}): {ood_mask.sum()}")

    # ── balance ID classes by undersampling the majority ─────────────────────
    # Skipped for star OOD (Galaxy vs Quasar) which is already balanced.
    if ood_class != 'star':
        rng_balance = np.random.default_rng(seed)
        cls0_idx = np.where(id_labels == 0)[0]
        cls1_idx = np.where(id_labels == 1)[0]
        n_min = min(len(cls0_idx), len(cls1_idx))
        if len(cls0_idx) > n_min:
            cls0_idx = rng_balance.choice(cls0_idx, n_min, replace=False)
        elif len(cls1_idx) > n_min:
            cls1_idx = rng_balance.choice(cls1_idx, n_min, replace=False)
        balanced_idx = np.sort(np.concatenate([cls0_idx, cls1_idx]))
        id_spectra = id_spectra[balanced_idx]
        id_labels  = id_labels[balanced_idx]
        print(f"[LAMOST] After balancing: {n_min} samples per class ({2*n_min} total ID)")

    # ── split ID  (60 / 20 / 20) ─────────────────────────────────────────────
    rng     = np.random.default_rng(seed)
    n_id    = len(id_labels)
    idx     = rng.permutation(n_id)
    n_test  = int(n_id * val_split)
    n_val   = int(n_id * val_split)
    n_train = n_id - n_val - n_test

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    # ── subsample OOD to match ID test set size (quasar/galaxy only) ────────
    if ood_class != 'star':
        n_test_id = len(test_idx)
        if len(ood_spectra) > n_test_id:
            rng_ood = np.random.default_rng(seed + 1)
            ood_keep = rng_ood.choice(len(ood_spectra), n_test_id, replace=False)
            ood_spectra = ood_spectra[ood_keep]
            print(f"[LAMOST] OOD subsampled to {n_test_id} samples (= ID test size)")

    # ── feature extraction (PCA fit on train only) ────────────────────────────
    print("[LAMOST] Fitting PCA and extracting features on training split …")
    transformers, train_feats = _fit_lamost_features(id_spectra[train_idx])
    val_feats  = _apply_lamost_features(id_spectra[val_idx],  transformers)
    test_feats = _apply_lamost_features(id_spectra[test_idx], transformers)
    ood_feats  = _apply_lamost_features(ood_spectra,          transformers)

    # ── assemble loaders ──────────────────────────────────────────────────────
    ood_dummy = np.zeros(len(ood_spectra), dtype=np.int64)

    train_loader   = DataLoader(LAMOSTDataset(train_feats, id_labels[train_idx]),
                                batch_size=batch_size, shuffle=True)
    val_loader     = DataLoader(LAMOSTDataset(val_feats,   id_labels[val_idx]),
                                batch_size=batch_size, shuffle=False)
    test_loader_id = DataLoader(LAMOSTDataset(test_feats,  id_labels[test_idx]),
                                batch_size=batch_size, shuffle=False)
    ood_loaders    = {
        ood_class: DataLoader(LAMOSTDataset(ood_feats, ood_dummy),
                              batch_size=batch_size, shuffle=False)
    }

    return train_loader, val_loader, test_loader_id, ood_loaders


# ───────────────────────────────────────────────────────────────────────────────


def get_experiment_loaders(id_name: str, batch_size: int, val_split: float = 0.2, seed: int = 42, data_dir: str = './data', lamost_ood: str = 'star'):
    """
    Returns the ID training/validation/testing loaders and a dictionary of OOD test loaders
    for the specified experimental track.
    """
    
    if id_name == 'mnist':
        id_transform = get_base_transforms(dataset='mnist')
        # Load ID Datasets (MNIST)
        train_set = datasets.MNIST('./data', train=True, download=True, transform=id_transform)
        test_set_id = datasets.MNIST('./data', train=False, download=True, transform=id_transform)

        # OOD Transforms
        ood_transform_omni = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            id_transform
        ])
        
        # OOD Datasets (All 28x28, 1-channel compatible)
        ood_set_kmnist = datasets.KMNIST('./data', train=False, download=True, transform=id_transform)
        ood_set_omni = datasets.Omniglot('./data', background=False, download=True, transform=ood_transform_omni)

        ood_loaders = {
            'kmnist': DataLoader(ood_set_kmnist, batch_size=batch_size, shuffle=False),
            'omniglot': DataLoader(ood_set_omni, batch_size=batch_size, shuffle=False)
        }

    elif id_name == 'cifar10':
        # Load ID Datasets (CIFAR-10)
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=get_base_transforms(dataset='cifar10', train=True))
        val_set = datasets.CIFAR10('./data', train=True, download=True, transform=get_base_transforms(dataset='cifar10', train=False))
        test_set_id = datasets.CIFAR10('./data', train=False, download=True, transform=get_base_transforms(dataset='cifar10', train=False))

        # OOD 1: SVHN (Far OOD)
        ood_set_svhn = datasets.SVHN('./data', split='test', download=True, transform=get_base_transforms(dataset='cifar10', train=False))
        
        # OOD 2: CIFAR-100 (Near OOD)
        ood_set_cifar100 = datasets.CIFAR100('./data', train=False, download=True, transform=get_base_transforms(dataset='cifar10', train=False))

        ood_loaders = {
            'svhn': DataLoader(ood_set_svhn, batch_size=batch_size, shuffle=False),
            'cifar100': DataLoader(ood_set_cifar100, batch_size=batch_size, shuffle=False)
        }

    elif id_name == 'imagenet':
        # ── ImageNet-1k ──────────────────────────────────────────────────────
        # Expects pre-organised directories:
        #   data_dir/train/<class_folder>/...
        #   data_dir/val/<class_folder>/...
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)

        # BICUBIC is the standard interpolation used during DINOv2 pre-training
        bicubic = transforms.InterpolationMode.BICUBIC
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=bicubic),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=bicubic),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        train_dir = os.path.join(data_dir, 'train')
        val_dir   = os.path.join(data_dir, 'val')
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet train directory not found: {train_dir}\n"
                "Pass the correct path via --data_dir."
            )
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"ImageNet val directory not found: {val_dir}\n"
                "Pass the correct path via --data_dir."
            )

        # Augmented loader for linear-head training
        train_set_aug = datasets.ImageFolder(train_dir, transform=train_transform)
        # Deterministic loader for GDA fitting — must NOT use random augmentations.
        # Random crops / flips shift embeddings, inflating within-class variance
        # and ruining the GDA's ability to detect OOD samples.
        train_set_det = datasets.ImageFolder(train_dir, transform=val_transform)
        val_set       = datasets.ImageFolder(val_dir,   transform=val_transform)
        test_set_id   = datasets.ImageFolder(val_dir,   transform=val_transform)

        num_workers = min(8, os.cpu_count() or 1)

        # ── OOD 1: Textures (DTD) ────────────────────────────────────────────
        # Must be pre-downloaded on the login node (compute nodes have no internet).
        # On a login node:  python -c "from torchvision import datasets; datasets.DTD('<root>', download=True)"
        dtd_root = os.path.join(os.path.dirname(data_dir), 'dtd')
        ood_dtd  = datasets.DTD(
            dtd_root, split='test', transform=val_transform, download=False
        )

        # ── OOD 2: Places365 → fallback SUN397 ──────────────────────────────
        # Both datasets are large (>20 GB) and must be pre-downloaded.
        # Places365 is tried first; if unavailable, SUN397 is used instead.
        ood_scene_name    = None
        ood_scene_dataset = None

        places_root = os.path.join(os.path.dirname(data_dir), 'places365')
        try:
            ood_scene_dataset = datasets.Places365(
                places_root, split='val', small=True,
                transform=val_transform, download=False
            )
            ood_scene_name = 'places365'
            print(f"[dataloaders] Places365 loaded ({len(ood_scene_dataset):,} samples).")
        except Exception as e_places:
            print(
                f"[dataloaders] Places365 not available ({e_places}).\n"
                "  Trying SUN397 as alternative scene-OOD dataset …"
            )
            sun_root = os.path.join(os.path.dirname(data_dir), 'sun397')
            try:
                ood_scene_dataset = datasets.SUN397(
                    sun_root, transform=val_transform, download=False
                )
                ood_scene_name = 'sun397'
                print(f"[dataloaders] SUN397 loaded ({len(ood_scene_dataset):,} samples).")
            except Exception as e_sun:
                print(
                    f"[dataloaders] SUN397 also not available ({e_sun}).\n"
                    "  OOD evaluation will use Textures (DTD) only."
                )

        # Build OOD loader dict
        ood_loaders = {
            'textures': DataLoader(
                ood_dtd, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            ),
        }
        if ood_scene_name is not None:
            ood_loaders[ood_scene_name] = DataLoader(
                ood_scene_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )

        # ImageNet already provides separate train / val splits — no manual split.
        # train_loader     : augmented (for linear-head training)
        # train_loader_gda : deterministic (for GDA fitting — must be consistent)
        train_loader     = DataLoader(
            train_set_aug, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        train_loader_gda = DataLoader(
            train_set_det, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        val_loader     = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader_id = DataLoader(
            test_set_id, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        # Returns 5-tuple for imagenet (extra train_loader_gda)
        return train_loader, val_loader, test_loader_id, ood_loaders, train_loader_gda

    elif id_name == 'lamost':
        train_loader, val_loader, test_loader_id, ood_loaders = _get_lamost_loaders(
            batch_size=batch_size, val_split=val_split, seed=seed, data_dir=data_dir,
            ood_class=lamost_ood,
        )
        return train_loader, val_loader, test_loader_id, ood_loaders

    else:
        raise NotImplementedError(f"Dataset {id_name} not configured.")

    # Create train/val split for ID data
    # We generate indices once to ensure no overlap
    num_train_full = len(train_set)
    val_len = int(num_train_full * val_split)
    train_len = num_train_full - val_len
    
    # Generate shuffled indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_train_full)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]

    # Create Subsets
    if id_name == 'cifar10':
        # Augmented transforms for training, clean transforms for val and GDA fitting.
        # train_set uses augmented transforms; val_set uses clean transforms (same underlying data).
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)
        # Clean-transform view of the training indices: used for GDA fitting so that
        # random crops/flips don't inflate within-class covariance and degrade OOD detection.
        train_subset_clean = Subset(val_set, train_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader_id = DataLoader(test_set_id, batch_size=batch_size, shuffle=False)
        train_loader_gda = DataLoader(train_subset_clean, batch_size=batch_size, shuffle=False)

        # Returns 5-tuple for CIFAR-10 (matching ImageNet convention)
        return train_loader, val_loader, test_loader_id, ood_loaders, train_loader_gda
    else:
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(train_set, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader_id = DataLoader(test_set_id, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader_id, ood_loaders