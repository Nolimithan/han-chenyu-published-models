"""Single-market T+0 backtest.

A predicted class other than inaction (index 1) buys at the open and sells
at the close on the same day. Rates follow Section 3.2: buy 2 per mille,
sell 2 per mille, stamp duty 0.5 per mille, with a five-currency minimum.

The minimum-fee assignment is applied after the cash debit, which is the
order in the archived notebook. Later days therefore use the uncapped debit.
"""

import numpy as np


def simulate_single_market(
    open_price,
    close_price,
    labels,
    initial_money,
    buy_rate=0.002,
    sell_rate=0.002,
    stamp_rate=0.0005,
    min_fee=5.0,
):
    open_price = np.asarray(open_price, dtype=float)
    close_price = np.asarray(close_price, dtype=float)
    labels = np.asarray(labels)
    money = float(initial_money)
    capital_series = []
    profit_series = []
    total_trade = 0
    i = 0
    while i <= (len(labels) - 1):
        if labels[i] != 1:
            volume = int(money / open_price[i])
            stock_value = volume * open_price[i]
            buy_brokerage = stock_value * buy_rate
            money = money - stock_value - buy_brokerage
            if buy_brokerage < min_fee:
                buy_brokerage = min_fee
            stock_value = volume * close_price[i]
            stamp_tax = close_price[i] * volume * stamp_rate
            sell_brokerage = close_price[i] * volume * sell_rate
            if sell_brokerage < min_fee:
                sell_brokerage = min_fee
            sell_cost = sell_brokerage + stamp_tax
            capital = money + stock_value - sell_cost
            money = capital
            profit = (close_price[i] - open_price[i]) * volume - buy_brokerage - sell_cost
            capital_series.append(capital)
            profit_series.append(profit)
            total_trade += 1
            i += 1
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
