import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
import indicators
from marketsimcode import compute_portvals
import warnings
warnings.simplefilter(action='ignore')


def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # Get price data
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[symbol]

    # Compute indicators
    rsi = indicators.compute_rsi(prices)
    sma = indicators.compute_sma(prices)
    momentum = indicators.compute_momentum(prices)

    # Create trades dataframe
    trades = pd.DataFrame(index=prices.index, columns=['Symbol', 'Order', 'Shares'])
    trades['Symbol'] = symbol
    trades['Order'] = ''
    trades['Shares'] = 0

    current_holdings = 0
    for i in range(1, len(prices) - 1):

        # Example logic incorporating RSI, Momentum, and SMA
        if (rsi.iloc[i] < 30 and rsi.iloc[i + 1] >= 30) or \
                (momentum.iloc[i] > 0 and momentum.iloc[i - 1] <= 0) or \
                (prices.iloc[i] > sma.iloc[i] and prices.iloc[i - 1] <= sma.iloc[i - 1]):
            trades.iloc[i] = [symbol, 'BUY', abs(1000 - current_holdings)]
            current_holdings += abs(1000 - current_holdings)
        elif (rsi.iloc[i] > 70 and rsi.iloc[i + 1] <= 70) or \
                (momentum.iloc[i] < 0 and momentum.iloc[i - 1] >= 0) or \
                (prices.iloc[i] < sma.iloc[i] and prices.iloc[i - 1] >= sma.iloc[i - 1]):
            trades.iloc[i] = [symbol, 'SELL', abs(-1000 - current_holdings)]
            current_holdings -= abs(-1000 - current_holdings)

    # Remove rows where no order was placed
    trades = trades[(trades['Order'] != '') & (trades['Shares'] != 0)]

    trades['Shares'] = np.where(trades['Order'] == 'SELL', trades['Shares']*-1, trades['Shares'])

    return trades[['Shares']]

def generate_results(symbol = "JPM", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
    df_trades = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)

    # -------------- Port Stats --------------- #
    port_value = compute_portvals(symbol, df_trades, start_val=sv, commission=9.95, impact=0.005)
    port_value = port_value / port_value.iloc[0]

    # --------------- Bench Mark --------------- #
    data = get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns="SPY")
    data["Benchmark"] = data / data.iloc[0, 0]
    data["ManualStrategy"] = port_value

    # ------------- MS Comparison Chart ---------- #
    fig = plt.figure()
    plt.plot(data.Benchmark, 'tab:purple')
    plt.plot(data.ManualStrategy, 'r')

    # Plot long and short entry points
    ymin, ymax = plt.ylim()
    for index, row in df_trades.iterrows():
        if row['Order'] == 'BUY':
            plt.bar(x=index, height=ymax - 1, bottom=1, width=1.5, color='blue')
        elif row['Order'] == 'SELL':
            plt.bar(x=index, height=1 - ymin, bottom=ymin, width=1.5, color='black')

    plt.legend(["Bechmark", "Portfolio"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Manual Strategy Portfolio vs Benchmark")
    plt.grid(True)
    fig.autofmt_xdate()
    plt.savefig("Manual Strategy Portfolio vs Benchmark.png")

    # ------------- Tables ------------- #
    # Stats
    daily_rets = data.diff().dropna()
    # Bench
    bench_std = round(daily_rets.Benchmark.std(), 6)
    bench_cum_rets = round(daily_rets.Benchmark.sum(), 6)
    bench_avg_rets = round(daily_rets.Benchmark.mean(), 6)
    # port
    port_std = round(daily_rets.ManualStrategy.std(), 6)
    port_cum_rets = round(daily_rets.ManualStrategy.sum(), 6)
    port_avg_rets = round(daily_rets.ManualStrategy.mean(), 6)

    # Output
    headers = ['Portfolio', 'STD', 'Cummulative Returns', 'Average Daily Returns']
    rows = [['Benchmark', bench_std, bench_cum_rets, bench_avg_rets], ['Manual Strategy', port_std, port_cum_rets, port_avg_rets]]
    df = pd.DataFrame(data=rows, columns=headers)
    print(df)


    return




