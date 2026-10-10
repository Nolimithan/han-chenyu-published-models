"""China-US dual-market T+0 backtest.

Section 3.2 defines four label combinations. Cash moves from CNY to CNH at
the CNH close, then to USD at the USDCNH close, and returns on the next
day's USDCNH and CNYCNH opens.

Two archived cells implement this loop. They are not identical: the
prediction cell debits the US buy fee before the five-unit floor and does
not restore CNY cash when the US leg loses money; the observation cell
applies the floor first and restores the post-China cash. Table 4 reports
both the model path and the realized-label path, so both variants are kept.
"""

import numpy as np
import pandas as pd


def _as_float(series):
    return np.asarray(series, dtype=float)


def simulate_dual_market(frame, label_cn, label_us, initial_money, variant="prediction"):
    """Run one dual-market path.

    ``variant='prediction'`` follows the model-label cell.
    ``variant='realized'`` follows the observation-label cell.
    Label value 1 is inaction. Any other value is action.
    """
    if variant not in {"prediction", "realized"}:
        raise ValueError("variant must be 'prediction' or 'realized'")
    close_1 = _as_float(frame["close_1"])
    open_1 = _as_float(frame["open_1"])
    close_2 = _as_float(frame["close_2"])
    open_2 = _as_float(frame["open_2"])
    labels_1 = np.asarray(frame[label_cn])
    labels_2 = np.asarray(frame[label_us])
    cnh_close = _as_float(frame["CNYCNH_close"])
    cnh_open = _as_float(frame["CNYCNH_open"])
    usd_close = _as_float(frame["USDCNH_close"])
    usd_open = _as_float(frame["USDCNH_open"])

    money = float(initial_money)
    capital_series = []
    profit_series = []
    total_trade = 0
    i = 0
    n = len(labels_1)
    while i <= n - 1:
        cn_acts = labels_1[i] != 1
        us_acts = labels_2[i] != 1
        if cn_acts and us_acts:
            volume_1 = int(money / open_1[i])
            stock_value_1 = volume_1 * open_1[i]
            buy_brokerage_1 = stock_value_1 * 0.002
            money = money - stock_value_1 - buy_brokerage_1
            if buy_brokerage_1 < 5:
                buy_brokerage_1 = 5
            stock_value_1 = volume_1 * close_1[i]
            stamp_tax_1 = close_1[i] * volume_1 * 0.0005
            sell_brokerage_1 = close_1[i] * volume_1 * 0.002
            if sell_brokerage_1 < 5:
                sell_brokerage_1 = 5
            sell_cost_1 = sell_brokerage_1 + stamp_tax_1
            capital = money + stock_value_1 - sell_cost_1
            money = capital
            capital_1 = money
            profit_1 = (close_1[i] - open_1[i]) * volume_1 - buy_brokerage_1 - sell_cost_1

            initial_capital = money
            money = money * cnh_close[i]
            money = money / usd_close[i]
            volume_2 = int(money / open_2[i])
            stock_value_2 = volume_2 * open_2[i]
            buy_brokerage_2 = stock_value_2 * 0.002
            if variant == "realized" and buy_brokerage_2 < 5:
                buy_brokerage_2 = 5
            money = money - stock_value_2 - buy_brokerage_2
            if variant == "prediction" and buy_brokerage_2 < 5:
                buy_brokerage_2 = 5
            stock_value_2 = volume_2 * close_2[i]
            stamp_tax_2 = close_2[i] * volume_2 * 0.0005
            sell_brokerage_2 = close_2[i] * volume_2 * 0.002
            if sell_brokerage_2 < 5:
                sell_brokerage_2 = 5
            sell_cost_2 = sell_brokerage_2 + stamp_tax_2
            capital = money + stock_value_2 - sell_cost_2
            money = capital
            profit = money - initial_capital
            total_trade += 1
            i += 1
            money = money * usd_open[i]
            money = money / cnh_open[i]
            capital = money
            profit_series.append(profit)
            if money - initial_capital > 0:
                capital_series.append(capital)
            else:
                capital = capital_1
                profit = profit_1
                capital_series.append(capital)
                profit_series.append(profit)
                if variant == "realized":
                    money = capital_1
        elif cn_acts and not us_acts:
            volume_1 = int(money / open_1[i])
            stock_value_1 = volume_1 * open_1[i]
            buy_brokerage_1 = stock_value_1 * 0.002
            money = money - stock_value_1 - buy_brokerage_1
            if buy_brokerage_1 < 5:
                buy_brokerage_1 = 5
            stock_value_1 = volume_1 * close_1[i]
            stamp_tax_1 = close_1[i] * volume_1 * 0.0005
            sell_brokerage_1 = close_1[i] * volume_1 * 0.002
            if sell_brokerage_1 < 5:
                sell_brokerage_1 = 5
            sell_cost_1 = sell_brokerage_1 + stamp_tax_1
            capital = money + stock_value_1 - sell_cost_1
            money = capital
            profit = (close_1[i] - open_1[i]) * volume_1 - buy_brokerage_1 - sell_cost_1
            capital_series.append(capital)
            profit_series.append(profit)
            total_trade += 1
            i += 1
        elif (not cn_acts) and us_acts:
            initial_capital = money
            money = money * cnh_close[i]
            money = money / usd_close[i]
            volume_2 = int(money / open_2[i])
            stock_value_2 = volume_2 * open_2[i]
            buy_brokerage_2 = stock_value_2 * 0.002
            money = money - stock_value_2 - buy_brokerage_2
            if buy_brokerage_2 < 5:
                buy_brokerage_2 = 5
            stock_value_2 = volume_2 * close_2[i]
            stamp_tax_2 = close_2[i] * volume_2 * 0.0005
            sell_brokerage_2 = close_2[i] * volume_2 * 0.002
            if sell_brokerage_2 < 5:
                sell_brokerage_2 = 5
            sell_cost_2 = sell_brokerage_2 + stamp_tax_2
            capital = money + stock_value_2 - sell_cost_2
            money = capital
            profit = money - initial_capital
            total_trade += 1
            i += 1
            money = money * usd_open[i]
            money = money / cnh_open[i]
            capital = money
            profit_series.append(profit)
            if money - initial_capital > 0:
                capital_series.append(capital)
            else:
                capital = initial_capital
                money = initial_capital
                profit = 0
                capital_series.append(capital)
                profit_series.append(profit)
        else:
            capital_series.append(money)
            profit_series.append(0.0)
            i += 1
    return {
        "capital": capital_series,
        "profit": profit_series,
        "trades": total_trade,
        "final_capital": capital_series[-1] if capital_series else money,
    }


def load_aligned_markets(path):
    return pd.read_csv(path)
