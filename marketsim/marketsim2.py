import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=0.00,
    impact=0.00,
):
    # --------------- Orders Data --------------- #
    # Organize data into df
    orders = pd.read_csv(orders_file, header=0, index_col=0)

    # Convert index to datetime
    orders.index = pd.to_datetime(list(orders.index.values))

    # --------------- Prices Data --------------- #
    # grab start date / end date from orders
    start_date = pd.to_datetime(min(list(orders.index.values)), format='%Y-%m-%d')
    end_date = pd.to_datetime(max(list(orders.index.values)), format='%Y-%m-%d')

    # Get list of all stocks used
    unique_stocks = list(set(orders["Symbol"]))

    # get stock data for date range
    prices = get_data(unique_stocks, pd.date_range(start_date, end_date)).drop(columns=["SPY"])
    prices["Cash"] = 1  # column should be all ones

    # --------------- Trades Data --------------- #
    # Setup position trades df with zeros, copy column and row indexes from prices
    ## trades data represents the net impact to positions and cash
    trades = pd.DataFrame(data=0.000, columns=prices.columns.values, index=prices.index.values)

    # Populate trades data by tracing orders data
    for i in range(orders.shape[0]):
        # Get data for orders
        order_row = orders.iloc[[i]]  # 0 = symbol, 1 = Position, 2 = Num shares
        date = order_row.index.values[0]

        # Determine if buy or sell
        if order_row.Order[0] == "BUY":
            position_factor = 1
        else:
            position_factor = -1

        # Update trades with position, add to existing trade data if already there
        trades.at[date, order_row.Symbol[0]] += order_row.Shares[0] * position_factor

        # -- Cash Impacts -- #
        cash_for_trade_impact = (-1 * position_factor) * prices.at[date, order_row.Symbol[0]] * order_row.Shares[0]
        # Transaction costs
        market_impact_fee = impact * abs(cash_for_trade_impact)

        # Update Cash
        trades.at[date, "Cash"] += cash_for_trade_impact - market_impact_fee - commission

    # --------------- Calculate holdings --------------- #
    # Setup holdings df
    holdings = pd.DataFrame(data=0.000, columns=trades.columns.values, index=trades.index.values)
    holdings.iloc[[0]] = trades.iloc[[0]]
    holdings.Cash.iat[0] += float(start_val)  # add in starting cash

    for i in range(1, holdings.shape[0]):
        # Carry over holdings position values from prior trading day, add trades data
        holdings.iloc[[i]] = holdings.iloc[[i - 1]].values + trades.iloc[[i]]

    values = prices * holdings
    port_values = values.sum(axis=1)  # used for debugging

    return port_values


def test_code():
    of = "./orders/orders-01.csv"
    sv = 1000000

    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    daily_returns = (portvals / portvals.shift(1)) - 1
    daily_returns = daily_returns[1:]
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
