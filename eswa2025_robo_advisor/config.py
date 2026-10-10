"""Settings for the six assets in Tables 1-4.

Training ratios are the fractions written in the asset-specific BiGRU
notebooks. They implement the nine-year / tenth-year split on each series.
The local crude-oil files use the prefix CLQ4; the paper names the contract
CLU4. SSE 50 ETF files use the prefix sz50etf.
"""

ASSETS = {
    "DXU4": {"initial_capital": 10_000.0, "currency": "USD", "train_ratio": 0.8878, "file_stems": ["DXU4"]},
    "CLU4": {"initial_capital": 10_000.0, "currency": "USD", "train_ratio": 0.889, "file_stems": ["CLU4", "CLQ4"]},
    "SSE50ETF": {"initial_capital": 10_000.0, "currency": "CNY", "train_ratio": 0.888, "file_stems": ["SSE50ETF", "sz50etf"]},
    "GCM4": {"initial_capital": 200_000.0, "currency": "USD", "train_ratio": 0.8872, "file_stems": ["GCM4"]},
    "SSEC": {"initial_capital": 100_000.0, "currency": "CNY", "train_ratio": 0.8881, "file_stems": ["SSEC"]},
    "NDX": {"initial_capital": 100_000.0, "currency": "USD", "train_ratio": 0.8881, "file_stems": ["NDX"]},
}

VMD = {"alpha": 4000, "tau": 0.0, "K": 11, "DC": 0, "init": 1, "tol": 1e-7}

N_INDICATORS = 20
N_MODES_KEPT = 10
IMAGE_ROWS = 10
IMAGE_COLS = 20

DCNN = {
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel": 3,
    "pool": 2,
    "feature_dim": 200,
    "learning_rate": 0.005,
    "keep_prob": 0.5,
    "stop_loss": 0.01,
    "leak": 0.1,
    "seed": 800,
}

BIGRU = {
    "hidden_dim": 200,
    "seq_length": 3,
    "output_dim": 3,
    "learning_rate": 0.005,
    "period": 90,
    "stop_loss": 0.05,
    "seed": 800,
}

BUY_RATE = 0.003
SELL_RATE = 0.003
STAMP_RATE = 0.0005
MIN_FEE = 5.0
WINDOW = 3
RANDOM_SEEDS = (111, 222, 369, 149, 800)
