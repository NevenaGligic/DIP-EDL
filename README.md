# Density-Informed Pseudo-Counts for Calibrated Evidential Deep Learning (DIP-EDL)

**Pietro Carlotti\*, Nevena Gligić\*, Arya Farahi**

\* Equal contribution

TEST

*STAI-X 2026 (Full Paper Track)*

> Evidential Deep Learning (EDL) is a popular framework for uncertainty-aware classification that models predictive uncertainty via Dirichlet distributions parameterized by neural networks. Despite its popularity, its theoretical foundations and behavior under distributional shift remain poorly understood. In this work, we provide a principled statistical interpretation by proving that EDL training corresponds to amortized variational inference in a hierarchical Bayesian model with a tempered pseudo-likelihood. This perspective reveals a major drawback: standard EDL conflates epistemic and aleatoric uncertainty, leading to systematic overconfidence on out-of-distribution inputs. To address this, we introduce Density-Informed Pseudo-count EDL, a new parametrization that decouples class prediction from uncertainty quantification by separately estimating the conditional label distribution and the marginal covariate density. This separation preserves evidence in high-density regions while shrinking predictions toward a uniform prior for out-of-distribution data. Theoretically, we prove that our method achieves asymptotic concentration. Empirically, we show our method enhances interpretability and improves robustness and uncertainty calibration under distributional shift.

Paper: https://arxiv.org/abs/2602.01477

---

## Installation

```bash
pip install -r requirements.txt
```

The following external repositories are bundled in this repo and require no separate installation:
- `DAEDL/` — modified to add LAMOST support
- `ICLR2024-REDL/` — unchanged from original
- `Re-EDL/` — unchanged from original
- `Posterior-Network/` — training scripts reorganised
- `LAMOST-Spectra-Classifier/` — unchanged from original

---

## Data

**MNIST and CIFAR-10** download automatically via `torchvision` on first run.

**LAMOST** spectra must be downloaded separately. The dataset is available at [LAMOST DR](http://www.lamost.org/dr9/). Place the FITS files in `data/lamost/` following the structure expected by `LAMOST-Spectra-Classifier/`. Set `--lamost_dir` when running if your path differs.

---

## Reproducing paper results

Run all models across seeds 10, 20, 30, 40 and collect results into a JSONL file:

```bash
# MNIST
python run_experiments.py --dataset mnist

# CIFAR-10
python run_experiments.py --dataset cifar10

# LAMOST (galaxy as OOD, for example)
python run_experiments.py --dataset lamost --lamost_ood galaxy

# Print results table (mean ± std across seeds)
python analyze_results.py
```

To run a single model/seed manually:

```bash
# Train DIP-EDL on MNIST from scratch
python main.py --dataset mnist --model dip_edl --seed 10 --train_cnn --train_maf

# Train DIP-EDL on CIFAR-10 from scratch
python main.py --dataset cifar10 --model dip_edl --seed 10 --train_cnn

# Load pre-trained DIP-EDL weights and evaluate (no training flags)
python main.py --dataset cifar10 --model dip_edl --seed 10

# Train baseline EDL
python main.py --dataset mnist --model edl --seed 10 --train_edl
```

Weights are saved to `saved_model_weights/` by default. Pass `--train_cnn`, `--train_maf`, `--train_edl`, etc. to train from scratch. If no weights are found and no training flag is passed, the script will raise an error.

---

## Ablation studies

### Component ablation (Table in paper)

Evaluates individual and pairwise contributions of training set size, the density estimator, and the discriminative classifier:

```bash
# MNIST (omit --train_cnn --train_maf if weights are already in saved_model_weights/)
for task in 1a 1b 1c 2a 2b 2c 3; do
    python main_ablation.py --dataset mnist --task $task --seed 10 --train_cnn --train_maf
done

# CIFAR-10
for task in 1a 1b 1c 2a 2b 2c 3; do
    python main_ablation.py --dataset cifar10 --task $task --seed 10
done
```

Results append to `results/ablation_results.jsonl`.

### Gamma ablation

Effect of the density scaling factor gamma:

```bash
python gamma_ablation.py --dataset mnist  --seed 10
python gamma_ablation.py --dataset cifar10 --seed 10
```

Results saved to `results/`.

### Density corruption ablation

Effect of corrupting the covariates with Gaussian noise:

```bash
python density_corruption_ablation.py --dataset mnist  --seed 10
python density_corruption_ablation.py --dataset cifar10 --seed 10
```

Results saved to `results/`.

Print LaTeX tables:

```bash
python analyze_ablations.py
```

### Toy experiments

2-D visualisation of uncertainty decomposition:

```bash
cd toy_experiments
python toy_main.py --model_type both   # side-by-side EDL vs DIP-EDL
python toy_main.py --model_type dip_edl

# Vacuity under perfect interpolation (varies EDL regularisation strength nu)
python vacuity_perfect_interpolation.py
```

Figures and tables are saved to `figures/` and `tables/` (gitignored).

---

## Repository structure

```
DIP-EDL-public/
├── main.py                        # Main benchmark script
├── main_ablation.py               # Component ablation (DIP-EDL only)
├── gamma_ablation.py              # Gamma scaling sweep
├── density_corruption_ablation.py # Density noise sweep
├── run_experiments.py             # Runs all models across seeds
├── analyze_results.py             # Prints results table
├── analyze_ablations.py           # LaTeX table for ablation results
├── requirements.txt
├── results/                       # Output JSONL files (gitignored)
├── src/
│   ├── dataloaders.py
│   ├── train.py
│   ├── experiments.py
│   ├── metrics.py
│   └── models/
│       ├── dip_edl.py             # DIP-EDL (main model)
│       ├── dip_edl_ablation.py    # Ablation variant
│       ├── EDL.py                 # EDL baseline
│       ├── R_EDL.py               # R-EDL wrapper
│       ├── Re_EDL.py              # Re-EDL wrapper
│       ├── DAEDL.py               # DA-EDL wrapper
│       ├── PostNet.py             # Posterior Network wrapper
│       ├── Baselines.py           # Deep Ensembles, MC Dropout
│       └── spectral_backbone.py   # 1D Conv backbone for LAMOST
├── toy_experiments/
│   ├── toy_main.py
│   └── vacuity_perfect_interpolation.py
├── DAEDL/                         # Bundled (modified: added LAMOST support)
├── ICLR2024-REDL/                 # Bundled (unchanged)
├── Re-EDL/                        # Bundled (unchanged)
├── Posterior-Network/             # Bundled (training scripts reorganised)
└── LAMOST-Spectra-Classifier/     # Bundled (unchanged)
```

---

## Citation

```bibtex
@article{carlotti2026density,
  title={Density-Informed Pseudo-Counts for Calibrated Evidential Deep Learning},
  author={Carlotti, Pietro and Gligi{\'c}, Nevena and Farahi, Arya},
  journal={arXiv preprint arXiv:2602.01477},
  year={2026}
}
```
