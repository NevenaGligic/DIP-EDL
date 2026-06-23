"""
Print LaTeX tables for DIP-EDL ablation experiments (MNIST and CIFAR-10):
  1. Component ablation  (main_ablation.py               → ablation_results.jsonl)
  2. Gamma sweep         (gamma_ablation.py               → gamma_ablation_results_{dataset}.jsonl)
  3. Density corruption  (density_corruption_ablation.py  → density_corruption_results_{dataset}.jsonl)

Usage:
    python analyze_ablations.py
"""

import json, glob, numpy as np
from collections import defaultdict

# OOD dataset keys (as they appear in the JSONL) and display labels per dataset
OOD_KEYS = {
    'mnist':   ['kmnist', 'omniglot'],
    'cifar10': ['cifar100', 'svhn'],
}
OOD_LABELS = {
    'kmnist':   'KMNIST',
    'omniglot': 'Omniglot',
    'cifar100': 'CIFAR-100',
    'svhn':     'SVHN',
}

DATASETS = ['cifar10', 'mnist']

TASK_SYMBOLS = {
    "1a": (r"\checkmark", r"$\times$",     r"$\times$"),
    "1b": (r"$\times$",   r"\checkmark",   r"$\times$"),
    "1c": (r"$\times$",   r"$\times$",     r"\checkmark"),
    "2a": (r"\checkmark", r"\checkmark",   r"$\times$"),
    "2b": (r"\checkmark", r"$\times$",     r"\checkmark"),
    "2c": (r"$\times$",   r"\checkmark",   r"\checkmark"),
    "3":  (r"\checkmark", r"\checkmark",   r"\checkmark"),
}

TASK_COMMENT = {
    "1a": "N only",
    "1b": "density only",
    "1c": "classifier only",
    "2a": "N + density",
    "2b": "N + classifier",
    "2c": "density + classifier",
    "3":  "full model (DIP-EDL)",
}


def load_jsonl(pattern, key_fn=None):
    if key_fn is None:
        rows = []
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return rows

    db = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry  = json.loads(line)
                    key    = key_fn(entry)
                    new_ts = entry.get('timestamp', '')
                    old_ts = db[key].get('timestamp', '') if key in db else ''
                    if new_ts >= old_ts:
                        db[key] = entry
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
    return list(db.values())


def fmt(mean, std=None, decimals=4):
    if std is None or std < 1e-9:
        return f"${mean:.{decimals}f}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def agg(rows, key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return float('nan'), 0.0
    return float(np.mean(vals)), float(np.std(vals))


def sigma_sort_key(s):
    return float('inf') if 'inf' in str(s) else float(s)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPONENT ABLATION
# ─────────────────────────────────────────────────────────────────────────────
abl_rows = load_jsonl("results/ablation_results*.jsonl",
                      key_fn=lambda r: (r.get('ablation_task'), r.get('seed'), r.get('dataset')))

for dataset in DATASETS:
    ood_keys = OOD_KEYS[dataset]
    ds_label = dataset.upper().replace("10", "-10")

    by_task = defaultdict(list)
    for r in abl_rows:
        if r.get('dataset') == dataset:
            by_task[r['ablation_task']].append(r)

    def abl_data_row(task):
        rs = by_task.get(task, [])
        if not rs:
            return "% NO DATA FOR TASK " + task
        m = lambda k: agg(rs, k)[0]
        ood_cols = " & ".join(
            fmt(m(f"{k}_{metric}"))
            for metric in ['auroc', 'aupr', 'brier']
            for k in ood_keys
        )
        return f"& {fmt(m('id_acc'))} & {fmt(m('id_brier'))} & {ood_cols} \\\\"

    def abl_full_row(task):
        n, de, nn = TASK_SYMBOLS[task]
        data = abl_data_row(task)
        return f"    {n} & {de} & {nn} {data} % {TASK_COMMENT[task]}"

    ood_header_cols = " & ".join(
        f"\\multicolumn{{1}}{{c}}{{\\textbf{{{OOD_LABELS[k]}}}}}" for k in ood_keys * 3
    )
    n_ood_cols = len(ood_keys) * 3  # AUROC + AUPR + Brier per OOD key

    print(f"% ═══════════════════════════════════════════════════════════════")
    print(f"% TABLE: Component ablation — {ds_label}")
    print(f"% ═══════════════════════════════════════════════════════════════")
    print(r"""\begin{table*}[ht]
    \centering""")
    print(f"    \\caption{{\\textbf{{Component ablation on {ds_label}.}}}}")
    print(r"""    \setlength{\tabcolsep}{3pt}
    \resizebox{\textwidth}{!}{""")

    col_spec = "@{} ccc | cc | " + " ".join(["c"] * n_ood_cols) + " @{}"
    print(f"        \\begin{{tabular}}{{{col_spec}}}")
    print(r"            \toprule")
    print(r"            \multicolumn{3}{c|}{\textbf{Components}} & \multicolumn{2}{c|}{\textbf{ID Performance}} & \multicolumn{" + str(n_ood_cols) + r"}{c}{\textbf{OOD Performance Metrics}} \\")
    print(r"            \cmidrule(r){1-3} \cmidrule(lr){4-5} \cmidrule(l){6-" + str(5 + n_ood_cols) + r"}")
    print(r"            $n$ & $\mathrm{DE}^{\psi}_{X_{i}}$ & $\mathrm{NN}^{\phi}_{X_{i}}$")
    print(r"            & \multicolumn{1}{c}{Acc.\ ($\uparrow$)} & \multicolumn{1}{c|}{BS ($\downarrow$)}")

    # AUROC / AUPR / OOD BS group headers
    metric_headers = " & ".join(
        f"\\multicolumn{{{len(ood_keys)}}}{{c}}{{{label} ($\\{'uparrow' if label != 'OOD BS' else 'downarrow'}$)}}"
        for label in ['AUROC', 'AUPR', 'OOD BS']
    )
    print(f"            & {metric_headers} \\\\")

    # cmidrules for each metric group
    parts = []
    for i in range(3):
        s = 6 + i * len(ood_keys)
        e = s + len(ood_keys) - 1
        parts.append(f"\\cmidrule(lr){{{s}-{e}}}")
    print("            " + " ".join(parts))

    # OOD dataset sub-headers (repeated across metric groups)
    ood_sub = " & ".join(
        f"\\multicolumn{{1}}{{c}}{{\\textbf{{{OOD_LABELS[k]}}}}}"
        for _ in range(3) for k in ood_keys
    )
    print(f"            & & & & & {ood_sub} \\\\")
    print(r"            \midrule")
    print(f"            % --- PAIR COMBINATIONS ---")
    for task in ["2a", "2b", "2c"]:
        print(abl_full_row(task))
    print(r"            \midrule")
    print(f"            % --- FULL MODEL ---")
    print(abl_full_row("3"))
    print(r"            \midrule")
    print(f"            % --- SINGLE COMPONENTS ---")
    for task in ["1a", "1b", "1c"]:
        print(abl_full_row(task))
    print(r"            \bottomrule")
    print(r"        \end{tabular}")
    print(r"    }")
    print(f"    \\label{{tab:ablation_{dataset}_appdx}}")
    print(r"\end{table*}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAMMA ABLATION
# ─────────────────────────────────────────────────────────────────────────────
gamma_rows = load_jsonl("results/gamma_ablation_results*.jsonl",
                        key_fn=lambda r: (r.get('gamma'), r.get('seed'), r.get('dataset')))

GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

for dataset in DATASETS:
    ood_keys = OOD_KEYS[dataset]
    ds_label = dataset.upper().replace("10", "-10")

    by_gamma = defaultdict(list)
    for r in gamma_rows:
        if r.get('dataset') == dataset:
            by_gamma[r['gamma']].append(r)

    def gamma_row(gamma):
        rs = by_gamma.get(gamma, [])
        if not rs:
            return f"    % NO DATA for gamma={gamma}"
        n = len(rs)
        s = lambda k: fmt(*agg(rs, k), decimals=4) if n > 1 else fmt(agg(rs, k)[0])
        label = f"\\textbf{{{gamma}}}"
        if abs(gamma - 1.0) < 1e-6:
            label += r" {\small(DIP-EDL)}"
        ood_cols = " & ".join(
            s(f"{k}_{metric}")
            for metric in ['auroc', 'aupr', 'brier']
            for k in ood_keys
        )
        return f"    {label} & {s('id_acc')} & {s('id_brier')} & {ood_cols} \\\\"

    n_ood_cols = len(ood_keys) * 3
    ood_col_spec = " ".join(["c"] * n_ood_cols)

    print(f"% ═══════════════════════════════════════════════════════════════")
    print(f"% TABLE: Gamma sweep — {ds_label}")
    print(f"% ═══════════════════════════════════════════════════════════════")
    print(r"""\begin{table*}[ht]
    \centering""")
    print(f"    \\caption{{\\textbf{{Likelihood scaling ($\\gamma$) on {ds_label}.}}}}")
    print(r"""    \setlength{\tabcolsep}{3pt}
    \resizebox{\textwidth}{!}{""")
    col_spec = "@{} c | c c | " + ood_col_spec + " @{}"
    print(f"        \\begin{{tabular}}{{{col_spec}}}")
    print(r"            \toprule")
    print(r"            \textbf{$\gamma$} & \multicolumn{2}{c|}{\textbf{ID Performance}} & \multicolumn{" + str(n_ood_cols) + r"}{c}{\textbf{OOD Performance Metrics}} \\")
    print(r"            \cmidrule(lr){2-3} \cmidrule(lr){4-" + str(3 + n_ood_cols) + r"}")
    print(r"            & \multicolumn{1}{c}{Acc.\ ($\uparrow$)} & \multicolumn{1}{c|}{BS ($\downarrow$)}")
    metric_hdrs = " & ".join(
        f"\\multicolumn{{{len(ood_keys)}}}{{c}}{{{lbl} ($\\{'uparrow' if lbl != 'OOD BS' else 'downarrow'}$)}}"
        for lbl in ['AUROC', 'AUPR', 'OOD BS']
    )
    print(f"            & {metric_hdrs} \\\\")
    parts = []
    for i in range(3):
        s = 4 + i * len(ood_keys)
        e = s + len(ood_keys) - 1
        parts.append(f"\\cmidrule(lr){{{s}-{e}}}")
    print("            " + " ".join(parts))
    ood_sub = " & ".join(
        f"\\multicolumn{{1}}{{c}}{{\\textbf{{{OOD_LABELS[k]}}}}}"
        for _ in range(3) for k in ood_keys
    )
    print(f"            & \\multicolumn{{2}}{{c|}}{{\\textbf{{{ds_label}}}}} & {ood_sub} \\\\")
    print(r"            \midrule")
    for g in GAMMAS:
        print(gamma_row(g))
    print(r"            \bottomrule")
    print(r"        \end{tabular}")
    print(r"    }")
    print(f"    \\label{{tab:dip_edl_gamma_{dataset}_appdx}}")
    print(r"\end{table*}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DENSITY CORRUPTION
# ─────────────────────────────────────────────────────────────────────────────
dc_rows = load_jsonl("results/density_corruption_results*.jsonl",
                     key_fn=lambda r: (r.get('sigma'), r.get('noise_seed'), r.get('seed'), r.get('dataset')))

SIGMA_ORDER = ["0.0", "0.5", "1.0", "2.0", "5.0", "inf (random)"]

for dataset in DATASETS:
    ood_keys = OOD_KEYS[dataset]
    ds_label = dataset.upper().replace("10", "-10")

    by_sigma = defaultdict(list)
    for r in dc_rows:
        if r.get('dataset') == dataset:
            by_sigma[r['sigma']].append(r)

    def dc_row(sigma):
        rs = by_sigma.get(sigma, [])
        if not rs:
            return f"    % NO DATA for sigma={sigma}"
        is_clean = (sigma_sort_key(sigma) == 0.0)
        def s(k):
            m, sd = agg(rs, k)
            return fmt(m, None if is_clean else sd)
        if 'inf' in str(sigma):
            label = r"\textbf{$\infty$}"
        elif is_clean:
            label = r"\textbf{$0$} {\small(clean)}"
        else:
            sv = sigma.rstrip('0').rstrip('.')
            label = f"\\textbf{{${sv}$}}"
        ood_cols = " & ".join(
            s(f"{k}_{metric}")
            for metric in ['auroc', 'aupr', 'brier']
            for k in ood_keys
        )
        return f"    {label} & {s('id_acc')} & {s('id_brier')} & {ood_cols} \\\\"

    n_ood_cols = len(ood_keys) * 3
    ood_col_spec = " ".join(["c"] * n_ood_cols)
    sigmas_in_data = sorted(by_sigma.keys(), key=sigma_sort_key) if by_sigma else []
    order = [s for s in SIGMA_ORDER if s in by_sigma] or sigmas_in_data

    print(f"% ═══════════════════════════════════════════════════════════════")
    print(f"% TABLE: Density corruption — {ds_label}")
    print(f"% ═══════════════════════════════════════════════════════════════")
    print(r"""\begin{table*}[ht]
    \centering""")
    print(f"    \\caption{{\\textbf{{Density corruption ($\\sigma$) on {ds_label}.}}}}")
    print(r"""    \setlength{\tabcolsep}{3pt}
    \resizebox{\textwidth}{!}{""")
    col_spec = "@{} c | c c | " + ood_col_spec + " @{}"
    print(f"        \\begin{{tabular}}{{{col_spec}}}")
    print(r"            \toprule")
    print(r"            \textbf{$\sigma$} & \multicolumn{2}{c|}{\textbf{ID Performance}} & \multicolumn{" + str(n_ood_cols) + r"}{c}{\textbf{OOD Performance Metrics}} \\")
    print(r"            \cmidrule(lr){2-3} \cmidrule(lr){4-" + str(3 + n_ood_cols) + r"}")
    print(r"            & \multicolumn{1}{c}{Acc.\ ($\uparrow$)} & \multicolumn{1}{c|}{BS ($\downarrow$)}")
    metric_hdrs = " & ".join(
        f"\\multicolumn{{{len(ood_keys)}}}{{c}}{{{lbl} ($\\{'uparrow' if lbl != 'OOD BS' else 'downarrow'}$)}}"
        for lbl in ['AUROC', 'AUPR', 'OOD BS']
    )
    print(f"            & {metric_hdrs} \\\\")
    parts = []
    for i in range(3):
        s2 = 4 + i * len(ood_keys)
        e2 = s2 + len(ood_keys) - 1
        parts.append(f"\\cmidrule(lr){{{s2}-{e2}}}")
    print("            " + " ".join(parts))
    ood_sub = " & ".join(
        f"\\multicolumn{{1}}{{c}}{{\\textbf{{{OOD_LABELS[k]}}}}}"
        for _ in range(3) for k in ood_keys
    )
    print(f"            & \\multicolumn{{2}}{{c|}}{{\\textbf{{{ds_label}}}}} & {ood_sub} \\\\")
    print(r"            \midrule")
    for sigma in order:
        print(dc_row(sigma))
    print(r"            \bottomrule")
    print(r"        \end{tabular}")
    print(r"    }")
    print(f"    \\label{{tab:density_corruption_{dataset}_appdx}}")
    print(r"\end{table*}")
    print()
