"""Min-max scaling (Equation 14) and VMD with K = 11."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

from .config import VMD as VMD_CFG


def load_numeric_frame(path):
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.drop(columns=["date"])
    numeric = frame.select_dtypes(include=[np.number]).fillna(frame.mean(numeric_only=True))
    return numeric


def decompose_frame(frame):
    scaled = MinMaxScaler().fit_transform(frame.to_numpy(dtype=float))
    blocks = []
    for column in range(scaled.shape[1]):
        modes, _, _ = VMD(
            scaled[:, column],
            VMD_CFG["alpha"],
            VMD_CFG["tau"],
            VMD_CFG["K"],
            VMD_CFG["DC"],
            VMD_CFG["init"],
            VMD_CFG["tol"],
        )
        blocks.append(np.asarray(modes).T)
    return np.concatenate(blocks, axis=1)


def decompose_file(path, output_path):
    features = decompose_frame(load_numeric_frame(path))
    pd.DataFrame(features).to_csv(output_path, index=False)
    return features
