"""Datasets for the NeuralBandit drift experiments.

Every dataset is a multi-class classification problem turned into a contextual bandit:
the rows are the contexts, the classes are the actions, and the reward is 1 when the
chosen action equals the (drifted) true class.

Usage from the model script:   X, y, K = load_dataset("letter", seed)
Overview of all datasets:      python datasets.py          (needs internet the first time)

OpenML datasets are downloaded once by scikit-learn and cached in ~/scikit_learn_data.
If a name is not found, look up the exact name and version on openml.org and edit
DATASETS below.
"""
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import KBinsDiscretizer

# prep = "bin"   : every column is cut into 5 equal-frequency bins and one-hot encoded
#                  (what the covertype loader does for its continuous columns)
# prep = "scale" : pixel-like features are divided by `scale` so they lie in [0, 1]
DATASETS = {
    "digits":        dict(source="digits", prep="scale", scale=16.0),  # 1,797 rows: quick tests only
    "pendigits":     dict(source="openml", name="pendigits", version=1, prep="bin"),
    "satimage":      dict(source="openml", name="satimage", version=1, prep="bin"),
    "letter":        dict(source="openml", name="letter", version=1, prep="bin"),
    "shuttle":       dict(source="openml", name="shuttle", version=1, prep="bin"),
    "mnist":         dict(source="openml", name="mnist_784", version=1, prep="scale", scale=255.0),
    "fashion_mnist": dict(source="openml", name="Fashion-MNIST", version=1, prep="scale", scale=255.0),
}
# "covertype" is handled by load_forest_cover() in the original script, so that your
# existing results stay exactly reproducible.


def _fetch(spec):
    if spec["source"] == "digits":
        bunch = load_digits()
        return np.asarray(bunch.data, dtype=np.float32), np.asarray(bunch.target)
    bunch = fetch_openml(name=spec["name"], version=spec["version"],
                         as_frame=False, parser="auto")
    return np.asarray(bunch.data, dtype=np.float32), np.asarray(bunch.target)


def load_dataset(name, seed=123):
    """Return (X, y, K): float32 context matrix, integer labels 0..K-1, number of actions."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    raw_x, raw_y = _fetch(spec)

    # labels -> 0..K-1 (works for numbers and for strings such as letters)
    classes, y = np.unique(raw_y, return_inverse=True)
    y = y.astype(np.int64)
    K = len(classes)

    if spec["prep"] == "bin":
        x = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile",
                             subsample=None).fit_transform(raw_x).astype(np.float32)
    else:
        x = (raw_x / spec["scale"]).astype(np.float32)

    # same fixed shuffle per seed as the covertype loader
    order = np.random.default_rng(seed).permutation(len(y))
    return x[order], y[order], K


if __name__ == "__main__":
    # Table for the paper: size, features, actions, random level and majority-class share.
    print(f"{'dataset':14s} {'rows':>8s} {'features':>9s} {'K':>3s} {'random':>7s} {'majority':>9s}")
    for name in DATASETS:
        try:
            x, y, K = load_dataset(name)
        except Exception as error:  # report and continue with the next dataset
            print(f"{name:14s} FAILED: {error}")
            continue
        majority = np.bincount(y).max() / len(y)
        print(f"{name:14s} {len(y):8d} {x.shape[1]:9d} {K:3d} {1 / K:7.3f} {majority:9.3f}")
