"""Asset settings that match the archived experiment notebooks.

The published Table 2 lists one hidden-unit value of 231. In the notebooks
that value is the input width ``data_dim`` (21 indicators times 11 VMD modes).
The recurrent layer width is ``hidden_dim = 200``.
"""

ASSETS = {
    "SSEC": {
        "market": "CN",
        "initial_capital": 210_000.0,
        "currency": "CNY",
        "learning_rate": 0.005,
        "train_ratio": 0.65,
    },
    "IXIC": {
        "market": "US",
        "initial_capital": 30_000.0,
        "currency": "USD",
        "learning_rate": 0.005,
        "train_ratio": 0.65,
    },
    "SZI": {
        "market": "CN",
        "initial_capital": 350_000.0,
        "currency": "CNY",
        # Archived robustness notebook (saved under the filename NDX训练模型)
        # uses 0.01. The paper table states 0.005 for the prediction model.
        "learning_rate": 0.01,
        "train_ratio": 0.65,
    },
    "DJI": {
        "market": "US",
        "initial_capital": 50_000.0,
        "currency": "USD",
        "learning_rate": 0.005,
        "train_ratio": 0.65,
    },
}

DUAL_PAIRS = {
    "SSEC-IXIC": {"cn": "SSEC", "us": "IXIC", "initial_capital": 210_000.0},
    "SZI-DJI": {"cn": "SZI", "us": "DJI", "initial_capital": 350_000.0},
}

VMD = {
    "alpha": 4000,
    "tau": 0.0,
    "K": 11,
    "DC": 0,
    "init": 1,
    "tol": 1e-7,
}

# Table 2 and the training notebooks.
SEQ_LENGTH = 3
DATA_DIM = 231
HIDDEN_DIM = 200
OUTPUT_DIM = 2
PERIOD = 120
STOP_LOSS = 0.05
SEED = 777

# Round-trip cost 2 per mille + 2 per mille + 0.5 per mille.
BUY_RATE = 0.002
SELL_RATE = 0.002
STAMP_RATE = 0.0005
MIN_FEE = 5.0
# close * COST_FACTOR > open keeps a day only when the move covers that cost.
COST_FACTOR = 0.9955
