import json
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lamost_ood', type=str, default='star',
                    choices=['star', 'quasar', 'galaxy'],
                    help='Which LAMOST OOD class to analyse (default: star)')
cli = parser.parse_args()

DATASETS = ["mnist", "cifar10", "lamost"]
MODELS   = ["edl", "redl", "reedl", "daedl", "postnet", "mc_dropout", "ensemble", "dip_edl"]

# Pretty names for the table
MODEL_NAMES = {
    "edl":        "EDL",
    "dip_edl":     "DIP-EDL",
    "redl":       "R-EDL",
    "reedl":      "Re-EDL",
    "daedl":      "DAEDL",
    "postnet":    "PostNet",
    "mc_dropout": "MC Dropout",
    "ensemble":   "Deep Ensemble",
}

_LAMOST_OOD_LABEL = {"star": "Star", "quasar": "Quasar", "galaxy": "Galaxy"}

OOD_DATASETS = {
    "mnist":   [("kmnist", "KMNIST"), ("omniglot", "Omniglot")],
    "cifar10": [("cifar100", "CIFAR-100"), ("svhn", "SVHN")],
    "lamost":  [(cli.lamost_ood, _LAMOST_OOD_LABEL[cli.lamost_ood])],
}

db = {}
result_files = glob.glob(os.path.join("results", "results_database*.jsonl"))
if not result_files:
    print("No results files found.")
    exit()

for path in result_files:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # For LAMOST, include lamost_ood in the key so different OOD
                # combinations don't overwrite each other.
                lamost_ood = entry.get("lamost_ood", "star") if entry.get("dataset") == "lamost" else None
                key = (entry["model"], entry["dataset"], entry["seed"], lamost_ood)
                new_ts = entry.get("timestamp", "")
                old_ts = db[key].get("timestamp", "") if key in db else ""
                if new_ts >= old_ts:
                    db[key] = entry
            except (json.JSONDecodeError, KeyError):
                pass

VALID_SEEDS = {10, 20, 30, 40}
runs = [v for v in db.values() if v.get("seed") in VALID_SEEDS]
print(f"Loaded {len(runs)} unique runs from {len(result_files)} file(s) (seeds {sorted(VALID_SEEDS)} only).\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(values):
    """Return 'mean ± std' string, or 'N/A' if no data."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return "---"
    mean, std = np.mean(vals), np.std(vals)
    return f"${mean:.4f} \\pm {std:.4f}$"

def get_metric(run_list, key):
    return [r.get(key) for r in run_list]

# ── Build table ───────────────────────────────────────────────────────────────
# Column layout (per dataset block):
#   Model | ID Acc | ID Brier | near_AUROC | near_AUPR | near_Brier | far_AUROC | far_AUPR | far_Brier

SEP = " & "

print("=" * 100)
print("COMBINED RESULTS TABLE (mean ± std across seeds)")
print("=" * 100)

def _ood_console_cols(model_runs, ood_pairs):
    """Ordered metric strings for the plain-text table: AUROC/AUPR/Brier per OOD."""
    cols = []
    for key, _ in ood_pairs:
        cols += [
            fmt(get_metric(model_runs, key + '_auroc')),
            fmt(get_metric(model_runs, key + '_aupr')),
            fmt(get_metric(model_runs, key + '_brier')),
        ]
    return cols

def _ood_latex_cols(model_runs, ood_pairs):
    """Ordered metric strings for the LaTeX table.

    Column order matches the header: all AUROCs, then all AUPRs, then all Briers.
    For a single OOD dataset this collapses to AUROC | AUPR | Brier.
    """
    cols = []
    for metric in ('auroc', 'aupr', 'brier'):
        for key, _ in ood_pairs:
            cols.append(fmt(get_metric(model_runs, key + '_' + metric)))
    return cols

for dataset in DATASETS:
    ood_pairs = OOD_DATASETS[dataset]

    print(f"\n\n{'─'*100}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'─'*100}")

    # Header — one AUROC/AUPR/Brier triple per OOD dataset
    header_parts = [f"{'Model':<20}", f"{'ID Acc':>18}", f"{'ID Brier':>18}"]
    for _, label in ood_pairs:
        header_parts += [f"{label+' AUROC':>22}", f"{label+' AUPR':>22}", f"{label+' Brier':>22}"]
    header = SEP.join(header_parts)
    print(header)
    print("-" * len(header))

    for model in MODELS:
        model_runs  = [r for r in runs if r["model"] == model and r["dataset"] == dataset
                       and (dataset != "lamost" or r.get("lamost_ood", "star") == cli.lamost_ood)]
        seeds_found = sorted(set(r["seed"] for r in model_runs))
        status      = f"({len(model_runs)} seed(s): {seeds_found})" if model_runs else "(no results)"
        name        = MODEL_NAMES.get(model, model)

        row_parts = [
            f"{name:<20}",
            f"{fmt(get_metric(model_runs, 'id_acc')):>18}",
            f"{fmt(get_metric(model_runs, 'id_brier')):>18}",
        ] + [f"{v:>22}" for v in _ood_console_cols(model_runs, ood_pairs)]
        print(SEP.join(row_parts) + f"   % {status}")

# ── LaTeX version ─────────────────────────────────────────────────────────────
# Column order: Acc | BS | AUROC(all OODs) | AUPR(all OODs) | OOD BS(all OODs)
print("\n\n" + "=" * 100)
print("LATEX TABLE")
print("=" * 100)

for dataset in DATASETS:
    ood_pairs = OOD_DATASETS[dataset]
    n_ood     = len(ood_pairs)
    n_ood_cols = n_ood * 3                           # AUROC + AUPR + Brier per OOD
    n_total    = 2 + n_ood_cols                      # 2 ID cols + OOD cols
    ds_label   = dataset.upper().replace("10", "-10")

    # Column spec: l | c c | <n_ood_cols × c>
    col_spec = "@{} l | c c | " + " ".join(["c"] * n_ood_cols) + " @{}"

    # cmidrule ranges (1-indexed, column 1 = model name)
    id_range  = "2-3"
    ood_start = 4
    ood_end   = 3 + n_ood_cols

    print(f"\n% ── {dataset.upper()} ──────────────────────────────────────────")
    print(r"\begin{table*}[h]")
    print(r"    \centering")
    print(f"    \\caption{{{ds_label} Results}}")
    print(r"    \setlength{\tabcolsep}{3pt}")
    print(r"    \resizebox{\textwidth}{!}{")
    print(f"        \\begin{{tabular}}{{{col_spec}}}")
    print(r"            \toprule")
    print(
        f"            \\textbf{{Model}} & "
        f"\\multicolumn{{2}}{{c|}}{{\\textbf{{ID Performance}}}} & "
        f"\\multicolumn{{{n_ood_cols}}}{{c}}{{\\textbf{{OOD Performance Metrics}}}} \\\\"
    )
    print(f"            \\cmidrule(lr){{{id_range}}} \\cmidrule(lr){{{ood_start}-{ood_end}}}")
    print(r"            & \multicolumn{1}{c}{Acc. ($\uparrow$)} & \multicolumn{1}{c|}{BS ($\downarrow$)}")

    # Metric group headers (AUROC / AUPR / OOD BS), each spanning n_ood columns
    metric_headers = [
        f"\\multicolumn{{{n_ood}}}{{c}}{{AUROC ($\\uparrow$)}}",
        f"\\multicolumn{{{n_ood}}}{{c}}{{AUPR ($\\uparrow$)}}",
        f"\\multicolumn{{{n_ood}}}{{c}}{{OOD BS ($\\downarrow$)}}",
    ]
    print("            & " + " & ".join(metric_headers) + r" \\")

    # cmidrules under each metric group
    cmidrules = []
    for i in range(3):
        s = ood_start + i * n_ood
        e = s + n_ood - 1
        cmidrules.append(f"\\cmidrule(lr){{{s}-{e}}}")
    print("            " + " ".join(cmidrules))

    # OOD dataset name sub-headers
    ood_label_cols = []
    for metric_idx in range(3):
        for _, label in ood_pairs:
            ood_label_cols.append(f"\\multicolumn{{1}}{{c}}{{\\textbf{{{label}}}}}")
    print(
        f"            & \\multicolumn{{2}}{{c|}}{{\\textbf{{{ds_label}}}}} "
        "& " + " & ".join(ood_label_cols) + r" \\"
    )
    print(r"            \midrule")

    for model in MODELS:
        model_runs  = [r for r in runs if r["model"] == model and r["dataset"] == dataset
                       and (dataset != "lamost" or r.get("lamost_ood", "star") == cli.lamost_ood)]
        name        = MODEL_NAMES.get(model, model)
        seeds_found = sorted(set(r["seed"] for r in model_runs))
        n_seeds     = len(seeds_found)
        warn        = (f"  % WARNING: only {n_seeds} seed(s) {seeds_found}"
                       if n_seeds < 4 else f"  % {n_seeds} seeds {seeds_found}")

        data_cols = [
            fmt(get_metric(model_runs, 'id_acc')),
            fmt(get_metric(model_runs, 'id_brier')),
        ] + _ood_latex_cols(model_runs, ood_pairs)

        print(f"            \\textbf{{{name}}} & " + " & ".join(data_cols) + r" \\" + warn)

    print(r"            \bottomrule")
    print(r"        \end{tabular}")
    print(r"    }")
    print(r"\end{table*}")
