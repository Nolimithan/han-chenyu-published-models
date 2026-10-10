"""Binary action labels for the T+0 strategy.

Section 4.2 marks a day as action when the close is above the open, and
changes that mark to inaction when commission and stamp duty exceed the
intraday gain. The archived labeling cell implements the cost screen as
``close * 0.9955 > open``.
"""

import numpy as np
import pandas as pd

from .config import COST_FACTOR


def binary_action_labels(open_price, close_price, cost_factor=COST_FACTOR):
    """Return one-hot labels with columns (action, inaction)."""
    open_price = np.asarray(open_price, dtype=float).reshape(-1)
    close_price = np.asarray(close_price, dtype=float).reshape(-1)
    if open_price.shape != close_price.shape:
        raise ValueError("open and close must have the same length")
    labels = []
    for open_i, close_i in zip(open_price, close_price):
        if close_i >= open_i and close_i * cost_factor > open_i:
            labels.append((1, 0))
        else:
            labels.append((0, 1))
    return pd.DataFrame(labels, columns=["action", "inaction"])
