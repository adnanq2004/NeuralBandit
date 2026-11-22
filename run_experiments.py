"""Run every NeuralBandit experiment for several seeds, in parallel.

Usage (from the folder containing neuralbandit3_improvingactivation.py):
    python run_experiments.py

Each run writes results/<condition>_seed<seed>.csv. Runs whose CSV already exists
are skipped, so you can stop and restart the script safely.
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
# The default run (sigmoid, 0.005, 0.10) doubles as gamma=0.005 and gamma_model=0.10.
CONDITIONS = [
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


def run_one(job):
    name, activation, gamma, gamma_model, seed = job
    output = OUT / f"{name}_seed{seed}.csv"
    if output.exists():
        return f"skip  {output.name}"
    cmd = [
        sys.executable, SCRIPT,
        "--steps", str(STEPS),
        "--drift-every", str(DRIFT_EVERY),
        "--models", MODELS,
        "--activation", activation,
        "--gamma", str(gamma),
        "--gamma-model", str(gamma_model),
        "--seed", str(seed),
        "--output", str(output),
    ]
    # One CPU thread per run so parallel runs do not fight each other.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    tmp = output.with_suffix(".partial")
    result = subprocess.run(cmd[:-1] + [str(tmp)], env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL  {output.name}\n{result.stderr[-500:]}"
    tmp.rename(output)  # only finished runs get the final name
    return f"done  {output.name}"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    jobs = [(n, a, g, gm, s) for (n, a, g, gm) in CONDITIONS for s in SEEDS]
    print(f"{len(jobs)} runs, {WORKERS} in parallel")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for message in pool.map(run_one, jobs):
            print(message, flush=True)
