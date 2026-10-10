"""Three-class labels from a three-day sliding window.

Equation (2): the middle price is sell (0) at a local maximum, buy (2) at a
local minimum, and hold (1) otherwise. Returned one-hot columns are
sell, hold, buy.
"""

import numpy as np
import pandas as pd

from .config import WINDOW


def ternary_labels(close, window_size=WINDOW):
    close = np.asarray(close, dtype=float).reshape(-1)
    rows = []
    for count_row in range(0, len(close) + 1):
        if count_row >= window_size:
            begin = count_row - window_size
            end = begin + window_size - 1
            mid = int((begin + end) / 2)
            window = close[begin : end + 1]
            if close[mid] == np.max(window):
                rows.append((1, 0, 0))
            elif close[mid] == np.min(window):
                rows.append((0, 0, 1))
            else:
                rows.append((0, 1, 0))
    return pd.DataFrame(rows, columns=["sell", "hold", "buy"])
