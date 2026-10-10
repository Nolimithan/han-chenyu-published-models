"""Equation (6): drop IMF 1 and form a 10 by 20 matrix at each date.

Columns of the VMD matrix are grouped in blocks of 11. Index 0 of each block
is the low-frequency mode and is removed. The remaining 10 modes become the
rows; the 20 indicators become the columns.
"""

import numpy as np
import pandas as pd

from .config import N_INDICATORS, N_MODES_KEPT


def drop_index_column(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] % 11 == 1:
        return matrix[:, 1:]
    return matrix


def to_image_tensor(vmd_matrix, n_indicators=N_INDICATORS):
    matrix = drop_index_column(vmd_matrix)
    if matrix.shape[1] != n_indicators * 11:
        raise ValueError(
            "expected %d columns (20 indicators x 11 modes), got %d"
            % (n_indicators * 11, matrix.shape[1])
        )
    images = []
    for row in matrix:
        kept = []
        for indicator in range(n_indicators):
            start = indicator * 11
            kept.extend(row[start + 1 : start + 11])
        images.append(np.reshape(kept, (n_indicators, N_MODES_KEPT)).T)
    return np.asarray(images, dtype=np.float32)


def flatten_images(images):
    return images.reshape(images.shape[0], -1)


def save_images(vmd_csv, output_csv):
    matrix = pd.read_csv(vmd_csv).to_numpy(dtype=float)
    images = to_image_tensor(matrix)
    flat = flatten_images(images)
    pd.DataFrame(flat).to_csv(output_csv, header=False, index=False)
    return images
