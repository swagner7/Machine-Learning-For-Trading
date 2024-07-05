import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


def compute_portvals(symbol, orders, start_val=1000000, commission=0.0, impact=0.0):

    orders['Symbol'] = symbol
    orders['Order'] = np.where(orders['Shares'] > 0, "BUY", 'SELL')

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
        if order_row.Shares[0] > 0:
            trades.at[date, order_row.Symbol[0]] += order_row.Shares[0]  # increase shares for symbol
            cft = -1 * prices.at[date, order_row.Symbol[0]] * order_row.Shares[0]

        elif order_row.Shares[0] < 0:
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

def test_code():
    of = "./orders/orders-11.csv"
    sv = 1000000

    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    daily_returns = portvals.pct_change()
    cum_ret = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252) * (avg_daily_ret / std_daily_ret)

    spy_prices = get_data(['SPY'], pd.date_range(start_date, end_date)).ffill().bfill()['SPY']
    spy_daily_returns = spy_prices.pct_change()
    cum_ret_SPY = (spy_prices.iloc[-1] / spy_prices.iloc[0]) - 1
    avg_daily_ret_SPY = spy_daily_returns.mean()
    std_daily_ret_SPY = spy_daily_returns.std()
    sharpe_ratio_SPY = np.sqrt(252) * (avg_daily_ret_SPY / std_daily_ret_SPY)

    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

def author():
    return 'swagner38'

if __name__ == "__main__":
    test_code()
