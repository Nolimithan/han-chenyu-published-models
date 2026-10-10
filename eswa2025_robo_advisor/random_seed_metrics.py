"""Figure 10 metrics for capital paths stored by random seed.

The archived post-processing script reads a wide table: six assets, five
seeds each, in the order DXU4, CLU4, SSE50ETF, GCM4, SSEC, NDX and seeds
111, 222, 369, 149, 800. The first row of that table is skipped.
"""

import numpy as np
import pandas as pd

from .config import ASSETS, RANDOM_SEEDS
from .metrics import risk_metrics


def metrics_from_wide_table(path):
    frame = pd.read_csv(path)
    rows = []
    asset_names = list(ASSETS)
    for i, asset in enumerate(asset_names):
        columns = frame.columns[i * len(RANDOM_SEEDS) : (i + 1) * len(RANDOM_SEEDS)]
        for seed, column in zip(RANDOM_SEEDS, columns):
            capital = frame[column].to_numpy(dtype=float)[1:]
            capital = capital[np.isfinite(capital)]
            stats = risk_metrics(capital)
            stats["Asset"] = asset
            stats["Seed"] = seed
            rows.append(stats)
    return pd.DataFrame(rows)
