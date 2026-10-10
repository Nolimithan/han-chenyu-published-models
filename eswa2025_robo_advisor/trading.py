"""Algorithm 1, in the control flow of the archived backtest.

Buy when the predicted class is 2 and sell when it is 0. Commission is
0.3 percent with a five-currency floor. Stamp duty is 0.05 percent.
The sale is kept only when the price move exceeds buy cost plus sell cost.
The five-unit sell floor is applied after that comparison, as in the notebook;
Algorithm 1 in the paper applies the floor before the comparison.
"""

import numpy as np


def simulate_trading(close, labels, initial_money, buy_rate=0.003, sell_rate=0.003, stamp_rate=0.0005, min_fee=5.0):
    close = np.asarray(close, dtype=float)
    labels = np.asarray(labels)
    money = float(initial_money)
    volume = 0
    capital_series = []
    profit_series = []
    total_trade = 0
    i = 0
    while i <= (len(labels) - 2):
        if labels[i] != 2:
            stock_value = volume * close[i]
            capital = money + stock_value
            i += 1
            capital_series.append(capital)
            profit_series.append(0.0)
        else:
            volume = int(money / close[i])
            stock_value = close[i] * volume
            buy_brokerage = stock_value * buy_rate
            money = money - stock_value
            if buy_brokerage < min_fee:
                buy_brokerage = min_fee
            buy_stamp_tax = stock_value * stamp_rate
            buy_cost = buy_brokerage + buy_stamp_tax
            capital = money + stock_value - buy_cost
            capital_series.append(capital)
            profit_series.append(0.0)
            j = i + 1
            active = True
            while active:
                if j == (len(labels) - 1):
                    i = j
                    active = False
                    break
                if labels[j] != 0:
                    stock_value = volume * close[j]
                    capital = money + stock_value
                    capital_series.append(capital)
                    profit_series.append(0.0)
                    j += 1
                    if j == (len(labels) - 1):
                        i = j
                        active = False
                else:
                    potential_profit = (close[j] - close[i]) * volume
                    potential_sell_brokerage = close[j] * volume * sell_rate
                    potential_sell_stamp_tax = close[j] * volume * stamp_rate
                    potential_sell_cost = potential_sell_brokerage + potential_sell_stamp_tax
                    if potential_profit > (buy_cost + potential_sell_cost):
                        total_trade += 1
                        sell_brokerage = potential_sell_brokerage
                        if sell_brokerage < min_fee:
                            sell_brokerage = min_fee
                        sell_cost = sell_brokerage + potential_sell_stamp_tax
                        profit = potential_profit
                        stock_value = close[j] * volume
                        capital = money + stock_value - sell_cost
                        money = capital
                        capital_series.append(capital)
                        profit_series.append(profit)
                    else:
                        stock_value = volume * close[j]
                        capital = money + stock_value
                        capital_series.append(capital)
                        profit_series.append(0.0)
                    i = j + 1
                    volume = 0
                    money = capital
                    active = False
    return {
        "capital": capital_series,
        "profit": profit_series,
        "trades": total_trade,
        "final_capital": capital_series[-1] if capital_series else money,
    }
