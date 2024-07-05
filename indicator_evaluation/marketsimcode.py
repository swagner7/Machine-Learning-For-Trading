import pandas as pd
import numpy as np
from util import get_data

def author():
    return 'swagner38'

def compute_portvals(orders, start_val=1000000, commission=0.0, impact=0.0):

    # Get start and end dates
    start_date = pd.to_datetime(min(list(orders.index.values)), format='%Y-%m-%d')
    end_date = pd.to_datetime(max(list(orders.index.values)), format='%Y-%m-%d')

    # Get unique symbols
    symbols = list(set(orders['Symbol']))

    # Get price data
    prices = get_data(symbols, pd.date_range(start_date, end_date)).ffill().bfill()
    prices["Cash"] = 1

    # Initialize trades df using prices df structure
    trades = pd.DataFrame(data=0.00000, columns=prices.columns.values, index=prices.index.values)

    # Loop over orders
    for i in range(orders.shape[0]):
        order_row = orders.iloc[[i]]
        date = order_row.index.values[0]

        # Determine if buy or sell
        if order_row.Order[0] == "BUY":
            trades.at[date, order_row.Symbol[0]] += order_row.Shares[0]  # increase shares for symbol
            cft = -1 * prices.at[date, order_row.Symbol[0]] * order_row.Shares[0]

        elif order_row.Order[0] == "SELL":
            trades.at[date, order_row.Symbol[0]] -= order_row.Shares[0]  # decrease shares
            cft = prices.at[date, order_row.Symbol[0]] * order_row.Shares[0]

        mark_imp = impact * abs(cft)
        trades.at[date, "Cash"] += cft - mark_imp - commission  #

    # Setup holdings df
    holdings = pd.DataFrame(data=0.00000, columns=trades.columns.values, index=trades.index.values)
    holdings.iloc[[0]] = trades.iloc[[0]]
    holdings.Cash.iat[0] += float(start_val)

    for i in range(1, holdings.shape[0]):
        holdings.iloc[[i]] = holdings.iloc[[i - 1]].values + trades.iloc[[i]]

    values = prices * holdings
    portvals = values.sum(axis=1)

    return portvals
