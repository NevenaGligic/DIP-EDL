import subprocess
import json
import numpy as np
import sys
import os
import time
import datetime
import argparse
import platform

# PTY gives the child a real terminal so tqdm uses \r (single-line bars).
# Only available on Unix/macOS; Windows falls back to plain pipes.
_USE_PTY = platform.system() != 'Windows'
if _USE_PTY:
    import pty
    import select

# --- CLI ARGS (for SLURM array jobs) ---
parser = argparse.ArgumentParser()
parser.add_argument('--seed',         type=int, default=None,
                    help='Single seed to run (overrides default SEEDS list)')
parser.add_argument('--dataset',      type=str, default=None,
                    help='Single dataset to run: cifar10, mnist, or lamost')
parser.add_argument('--model',        type=str, default=None,
                    help='Single model to run (overrides full MODEL_FLAGS loop)')
parser.add_argument('--results_file', type=str, default=None,
                    help='Output .jsonl file (default: results_database.jsonl)')
parser.add_argument('--lamost_ood',   type=str, default='star',
                    choices=['star', 'quasar', 'galaxy'],
                    help='LAMOST OOD class (only used when --dataset lamost)')
parser.add_argument('--data_dir',     type=str, default='./data',
                    help='Root data directory (used for LAMOST; default: ./data)')
args = parser.parse_args()

# --- CONFIGURATION ---
PYTHON_EXEC = sys.executable
SCRIPT_NAME = "main.py"
SEEDS        = [args.seed]    if args.seed    is not None else [10, 20, 30, 40]
DATASETS     = [args.dataset] if args.dataset is not None else ["cifar10", "mnist"]
RESULTS_FILE = args.results_file if args.results_file is not None else "results/results_database.jsonl"

MODEL_FLAGS_DEFAULT = {
    "edl":        ["--train_edl"],
    "dip_edl":    ["--train_cnn", "--train_maf"],
    "mc_dropout": ["--train_mc_dropout"],
    "ensemble":   ["--train_ensemble"],
    "redl":       ["--train_redl"],
    "reedl":      ["--train_reedl"],
    "daedl":      ["--train_daedl"],
    "postnet":    ["--train_postnet"],
}

# LAMOST: no pre-trained weights exist, every model is trained from scratch.
# dip_edl: --train_cnn trains the backbone; GDA is fitted automatically in the weight-loading section.
MODEL_FLAGS_LAMOST = {
    "edl":        ["--train_edl"],
    "dip_edl":     ["--train_cnn"],
    "redl":       ["--train_redl"],
    "reedl":      ["--train_reedl"],
    "daedl":      ["--train_daedl"],
    "postnet":    ["--train_postnet"],
    "ensemble":   ["--train_ensemble"],
    "mc_dropout": ["--train_mc_dropout"],
}

_all_flags = MODEL_FLAGS_LAMOST if (args.dataset == "lamost") else MODEL_FLAGS_DEFAULT
# If --model is given, restrict to that single model
if args.model is not None:
    if args.model not in _all_flags:
        raise ValueError(f"Unknown model '{args.model}' for dataset '{args.dataset}'")
    MODEL_FLAGS = {args.model: _all_flags[args.model]}
else:
    MODEL_FLAGS = _all_flags

def save_result(data):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
    print(f"  [Saved result to {RESULTS_FILE}]")

def run_trial(dataset, model, flags, seed, lamost_ood='star', data_dir='./data'):

    cmd = [PYTHON_EXEC, "-u", SCRIPT_NAME,
           "--dataset", dataset,
           "--model", model,
           "--seed", str(seed)] + flags

    if dataset == 'lamost':
        cmd += ["--lamost_ood", lamost_ood, "--data_dir", data_dir]

    start_time = time.time()

    if _USE_PTY:
        # PTY lets tqdm use \r so progress bars overwrite in place
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(cmd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
        os.close(slave_fd)

        chunks = []
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 0.1)
            except (ValueError, select.error):
                break
            if r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                text = data.decode('utf-8', errors='replace')
                sys.stdout.write(text)
                sys.stdout.flush()
                chunks.append(text)
            elif process.poll() is not None:
                try:
                    while True:
                        r2, _, _ = select.select([master_fd], [], [], 0.05)
                        if not r2:
                            break
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        text = data.decode('utf-8', errors='replace')
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        chunks.append(text)
                except OSError:
                    pass
                break

        try:
            os.close(master_fd)
        except OSError:
            pass
        process.wait()
        full_output_str = ''.join(chunks)

    else:
        # Windows fallback: plain pipe, tqdm will print on multiple lines
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        full_output = []
        for line in process.stdout:
            print(line, end='')
            full_output.append(line)
        process.wait()
        full_output_str = ''.join(full_output)

    duration_min = (time.time() - start_time) / 60.0

    if process.returncode != 0:
        print(f"\n\nCRASH: Seed {seed} failed after {duration_min:.1f} min")
        with open("crash_log.txt", "a") as f:
            f.write(f"[{datetime.datetime.now()}] Crash {model} {dataset} {seed}\n")
            f.write(full_output_str[-1000:] + "\n\n")
        return

    # Parse JSON output from the accumulated string
    try:
        start_tag = "__JSON_START__"
        end_tag = "__JSON_END__"
        idx_start = full_output_str.find(start_tag) + len(start_tag)
        idx_end = full_output_str.find(end_tag)
        
        if idx_start != -1 and idx_end != -1:
            raw_data = json.loads(full_output_str[idx_start:idx_end])
            
            raw_data['model'] = model
            raw_data['dataset'] = dataset
            raw_data['seed'] = seed
            raw_data['timestamp'] = str(datetime.datetime.now())
            if dataset == 'lamost':
                raw_data['lamost_ood'] = lamost_ood
            
            save_result(raw_data)
        else:
            print("\n  Error: Could not find JSON tag in output.")

    except Exception as e:
        print(f"\n  Error parsing output: {e}")

    print(f"  Completed in {duration_min:.1f} min")

# --- MAIN LOOP ---
LAMOST_OOD_CLASSES = [args.lamost_ood] if args.dataset == 'lamost' else ['star']

total_jobs = len(DATASETS) * len(MODEL_FLAGS) * len(SEEDS) * len(LAMOST_OOD_CLASSES)
current_job = 0

print(f"Starting experiments. Results saving to {RESULTS_FILE}")
print(f"Total Jobs Scheduled: {total_jobs}")

for dataset in DATASETS:
    ood_classes = LAMOST_OOD_CLASSES if dataset == 'lamost' else ['star']
    for ood in ood_classes:
        for model, flags in MODEL_FLAGS.items():
            for seed in SEEDS:
                current_job += 1

                print(f"\n{'='*60}")
                print(f">> PROGRESS: [{current_job}/{total_jobs}]")
                ood_tag = f" | OOD {ood.upper()}" if dataset == 'lamost' else ""
                print(f">> STARTING: {dataset.upper()}{ood_tag} | {model.upper()} | Seed {seed}")
                print(f"{'='*60}\n")

                run_trial(dataset, model, flags, seed, lamost_ood=ood, data_dir=args.data_dir)