"""Builds the training and validation sets from data/success.csv and data/fail.csv.

Run from the project folder:   python make_dataset.py
Output: dataset.pt  (training and validation tensors)
"""
import re

import numpy as np
import pandas as pd
import torch
from periodictable import elements

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SUCCESS_CSV = "data/success.csv"     # superconductors (Tc > 0)
FAIL_CSV = "data/fail.csv"           # non-superconductors (Tc = 0)
OUTPUT_FILE = "data/dataset.pt"
VAL_FRACTION = 0.2
SEED = 42                            # makes the split reproducible
N_FEATURES = 120                     # vector index = atomic number (index 0 is never used)

# symbol -> atomic number, e.g. {"H": 1, "He": 2, ...}; D and T (hydrogen isotopes) count as H
SYMBOLS = {e.symbol: e.number for e in elements if e.number > 0}
SYMBOLS.update({"D": 1, "T": 1})

# matches one element symbol (capital + optional lowercase) followed by an optional number
TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


# ----------------------------------------------------------------------------
# Step 1: from a formula string to a dictionary {atomic number: fraction}
# ----------------------------------------------------------------------------
def parse_formula(formula):
    """'Y1Ba2Cu3O7' -> {39: 0.083, 56: 0.167, 29: 0.25, 8: 0.583}.
    Returns None if the formula is not valid (unknown symbol, +, -, parentheses, ...)."""
    formula = formula.replace(" ", "")
    if re.search(r"[^A-Za-z0-9.]", formula):         # any character that is not a letter, digit or "."
        return None

    amounts = {}
    for symbol, number in TOKEN.findall(formula):
        if symbol not in SYMBOLS:
            return None
        z = SYMBOLS[symbol]
        amount = float(number) if number else 1.0     # "Cu" without a number means 1
        amounts[z] = amounts.get(z, 0.0) + amount     # repeated elements are summed

    total = sum(amounts.values())
    if total <= 0:
        return None
    return {z: a / total for z, a in amounts.items()}  # normalise: fractions sum to 1


# ----------------------------------------------------------------------------
# Step 2: from a CSV file to arrays
# ----------------------------------------------------------------------------
def load_csv(path, discarded):
    """Returns X (n, 120) with the compositions and tc (n,) with the critical temperatures.
    Rows that cannot be parsed are skipped and their formula is added to `discarded`."""
    df = pd.read_csv(path)
    X, tc = [], []
    for formula, t in zip(df["Compound"], df["Tc"]):
        composition = None if (pd.isna(formula) or pd.isna(t)) else parse_formula(formula)
        if composition is None:
            discarded.append(formula)
            continue
        vector = np.zeros(N_FEATURES)
        for z, fraction in composition.items():
            vector[z] = fraction
        X.append(vector)
        tc.append(t)
    return np.array(X), np.array(tc)


discarded = []
X_pos, tc_pos = load_csv(SUCCESS_CSV, discarded)
X_neg, tc_neg = load_csv(FAIL_CSV, discarded)
print("Discarded formulas:", discarded)

# ----------------------------------------------------------------------------
# Step 3: put the two files together and remove duplicates
# ----------------------------------------------------------------------------
X = np.vstack([X_pos, X_neg])
tc = np.concatenate([tc_pos, tc_neg])
is_sc = (tc > 0).astype(np.int64)                    # 1 = superconductor, 0 = not

# two rows with the same composition: keep only the first one
_, first_idx = np.unique(X.round(6), axis=0, return_index=True)
keep = np.sort(first_idx)
print("Duplicates removed:", len(X) - len(keep))
X, tc, is_sc = X[keep], tc[keep], is_sc[keep]

# ----------------------------------------------------------------------------
# Step 4: random split, with the same fraction of superconductors in both sets
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
is_val = np.zeros(len(X), dtype=bool)
for c in (0, 1):                                     # split each class separately
    indices = rng.permutation(np.where(is_sc == c)[0])
    n_val = int(VAL_FRACTION * len(indices))
    is_val[indices[:n_val]] = True

print(f"Training: {(~is_val).sum()}   Validation: {is_val.sum()}")
print(f"Fraction of superconductors: train {is_sc[~is_val].mean():.3f}, val {is_sc[is_val].mean():.3f}")

# ----------------------------------------------------------------------------
# Step 5: convert to tensors and save
# ----------------------------------------------------------------------------
def to_tensors(mask):
    # input: composition * 100, shaped like a 1-channel 10x12 image
    x = torch.tensor(X[mask] * 100, dtype=torch.float32).reshape(-1, 1, 10, 12)
    # labels: columns [is NOT superconductor, is superconductor, Tc]
    y = torch.tensor(np.stack([1 - is_sc[mask], is_sc[mask], tc[mask]], axis=1), dtype=torch.float32)
    return x, y


x_train, y_train = to_tensors(~is_val)
x_val, y_val = to_tensors(is_val)
print("x_train:", tuple(x_train.shape), " y_train:", tuple(y_train.shape))
print("x_val:  ", tuple(x_val.shape), " y_val:  ", tuple(y_val.shape))

torch.save({"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val}, OUTPUT_FILE)
print("Saved", OUTPUT_FILE)