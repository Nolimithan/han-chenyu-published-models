"""Table 1 statistics and the risk columns of Table 4."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import bds


def descriptive_statistics(close):
    close = np.asarray(close, dtype=float)
    close = close[np.isfinite(close)]
    shapiro_stat, shapiro_p = stats.shapiro(close)
    bds_stat, bds_p = bds(close, max_dim=5)
    return {
        "sample_size": int(len(close)),
        "mean": float(np.mean(close)),
        "std": float(np.std(close, ddof=1)),
        "skewness": float(stats.skew(close)),
        "kurtosis": float(stats.kurtosis(close)),
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "bds_stat": float(bds_stat[0]),
        "bds_p": float(bds_p[0]),
    }


def risk_metrics(capital, risk_free_rate=0.04):
    capital = np.asarray(capital, dtype=float)
    daily_returns = np.diff(capital) / capital[:-1]
    total_return = (capital[-1] - capital[0]) / capital[0]
    trading_days = len(capital)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    running_max = np.maximum.accumulate(capital / capital[0])
    drawdowns = (running_max - capital / capital[0]) / running_max
    mdd = float(np.max(drawdowns))
    daily_std = float(np.std(daily_returns, ddof=1))
    annual_volatility = daily_std * np.sqrt(252)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    sharpe = 0.0 if daily_std == 0 else float(np.mean(excess) / daily_std * np.sqrt(252))
    if abs(annual_return) < 0.0001:
        risk_return = 999.99
    else:
        risk_return = min(annual_volatility / abs(annual_return), 999.99)
    return {
        "Annual Return(%)": annual_return * 100,
        "MDD(%)": mdd * 100,
        "Sharpe Ratio": sharpe,
        "Risk/Return": risk_return,
    }


def summarize_close_files(paths):
    rows = []
    for name, path in paths:
        frame = pd.read_csv(path)
        close = frame["close"] if "close" in frame.columns else frame.iloc[:, 4]
        row = descriptive_statistics(close)
        row["asset"] = name
        rows.append(row)
    return pd.DataFrame(rows)
