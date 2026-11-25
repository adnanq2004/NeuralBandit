"""Run NeuralBandit experiments for several datasets and seeds, in parallel.

Usage (from the folder containing neuralbandit3_improvingactivation.py and datasets.py):
    python run_experiments.py                    # every dataset listed in DATASETS_TO_RUN
    python run_experiments.py letter mnist       # only these datasets

Each run writes results/<dataset>/<condition>_seed<seed>.csv. Finished runs are skipped,
so you can stop and restart the script safely.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT = "neuralbandit3_improvingactivation.py"
STEPS = 100_000
DRIFT_EVERY = 5_000
SEEDS = [123, 124, 125, 126, 127]   # 123 = the seed of your earlier single runs
MODELS = "small"                    # must match the grid used in your earlier runs
WORKERS = max(1, (os.cpu_count() or 2) // 2)
OUT = Path("results")

# (condition name, activation, gamma, gamma_model)
FULL_CONDITIONS = [
    ("sigmoid_default", "sigmoid",    0.005, 0.10),
    ("relu",            "relu",       0.005, 0.10),
    ("tanh",            "tanh",       0.005, 0.10),
    ("leaky_relu",      "leaky_relu", 0.005, 0.10),
    ("gamma_0.01",      "sigmoid",    0.01,  0.10),
    ("gamma_0.05",      "sigmoid",    0.05,  0.10),
    ("gamma_0.10",      "sigmoid",    0.10,  0.10),
    ("gm_0.01",         "sigmoid",    0.005, 0.01),
    ("gm_0.05",         "sigmoid",    0.005, 0.05),
    ("gm_0.20",         "sigmoid",    0.005, 0.20),
    ("gm_0.30",         "sigmoid",    0.005, 0.30),
    ("tanh_gm_0.30",    "tanh",       0.005, 0.30),
    ("leaky_gm_0.30",   "leaky_relu", 0.005, 0.30),
]
# On the new datasets only the key comparison is repeated: it tests whether the activation
# effect (and the best combination) carries over, at a fraction of the cost.
KEY_CONDITIONS = [c for c in FULL_CONDITIONS
                  if c[0] in ("sigmoid_default", "relu", "tanh", "leaky_relu", "tanh_gm_0.30")]

# dataset -> conditions to run. "covertype" is the dataset you already ran.
DATASETS_TO_RUN = {
    "covertype": FULL_CONDITIONS,
    "mnist":     KEY_CONDITIONS,
    "letter":    KEY_CONDITIONS,
    "shuttle":   KEY_CONDITIONS,
}
# Datasets with fewer rows than STEPS replay their rows; add "--reshuffle" for those
# (letter has 20,000 rows, so a 100,000-step run makes 5 passes).
RESHUFFLE = {"letter"}


def run_one(job):
    dataset, name, activation, gamma, gamma_model, seed = job
    folder = OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{name}_seed{seed}.csv"
    if output.exists():
        return f"skip  {dataset}/{output.name}"
    tmp = output.with_suffix(".partial")
    cmd = [
        sys.executable, SCRIPT,
        "--dataset", dataset,
        "--steps", str(STEPS),
        "--drift-every", str(DRIFT_EVERY),
        "--models", MODELS,
        "--activation", activation,
        "--gamma", str(gamma),
        "--gamma-model", str(gamma_model),
        "--seed", str(seed),
        "--output", str(tmp),
    ]
    if dataset in RESHUFFLE:
        cmd.append("--reshuffle")
    # One CPU thread per run so parallel runs do not fight each other.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL  {dataset}/{output.name}\n{result.stderr[-500:]}"
    tmp.rename(output)  # only finished runs get the final name
    return f"done  {dataset}/{output.name}"


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(DATASETS_TO_RUN)
    unknown = [d for d in wanted if d not in DATASETS_TO_RUN]
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}. Choose from {list(DATASETS_TO_RUN)}")
    jobs = [(d, n, a, g, gm, s)
            for d in wanted
            for (n, a, g, gm) in DATASETS_TO_RUN[d]
            for s in SEEDS]
    print(f"{len(jobs)} runs, {WORKERS} in parallel")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for message in pool.map(run_one, jobs):
            print(message, flush=True)
