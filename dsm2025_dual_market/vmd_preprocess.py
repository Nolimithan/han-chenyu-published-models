"""VMD of every numeric column.

Equation (8) normalizes each column with a 1e-5 floor. Each column is then
decomposed into K = 11 modes (Section 5.1). Modes are concatenated in the
original column order, giving 21 x 11 = 231 features when the input has the
21 series used in the paper.
"""

import numpy as np
import pandas as pd
from vmdpy import VMD

from .config import VMD as VMD_CFG


def minmax_scale(data):
    data = np.asarray(data, dtype=float)
    numerator = data - np.min(data, axis=0)
    denominator = np.max(data, axis=0) - np.min(data, axis=0)
    return numerator / (denominator + 1e-5)


def load_numeric_frame(path):
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame = frame.drop(columns=["date"])
    numeric = frame.select_dtypes(include=[np.number])
    numeric = numeric.fillna(numeric.mean())
    return numeric


def decompose_frame(frame, alpha=None, k=None):
    cfg = dict(VMD_CFG)
    if alpha is not None:
        cfg["alpha"] = alpha
    if k is not None:
        cfg["K"] = k
    scaled = minmax_scale(frame.to_numpy(dtype=float))
    blocks = []
    for column in range(scaled.shape[1]):
        modes, _, _ = VMD(
            scaled[:, column],
            cfg["alpha"],
            cfg["tau"],
            cfg["K"],
            cfg["DC"],
            cfg["init"],
            cfg["tol"],
        )
        blocks.append(np.asarray(modes).T)
    return np.concatenate(blocks, axis=1)


def decompose_file(path, output_path):
    features = decompose_frame(load_numeric_frame(path))
    pd.DataFrame(features).to_csv(output_path, index=False)
    return features
