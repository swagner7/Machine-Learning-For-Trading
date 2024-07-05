import pandas as pd
import datetime as dt
import indicators
import marketsimcode as ms
import TheoreticallyOptimalStrategy as tos
from util import get_data
import matplotlib.pyplot as plt

def author():
    return 'swagner38'

def run_test_project():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    # Run indicators
    indicators.run()

    # Run theoretically optimal strategy
    df_trades = tos.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    print(df_trades)

    # -------------- Port Stats --------------- #
    port_value = ms.compute_portvals(df_trades, start_val=sv)
    port_value = port_value / port_value.iloc[0]

    # --------------- Bench Mark --------------- #
    data = get_data(["JPM"], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns="SPY")
    data["JPM"] = data / data.iloc[0, 0]
    data["TOS"] = port_value

    # ------------- TOS Comparison Chart ---------- #
    fig = plt.figure()
    plt.plot(data.JPM, 'tab:purple')
    plt.plot(data.TOS, 'r')
    plt.legend(["JPM", "Portfolio"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("TOS Portfolio vs JPM Benchmark")
    plt.grid(True)
    fig.autofmt_xdate()
    plt.savefig("TOS Portfolio vs JPM Benchmark.png")

    # ------------- Tables ------------- #
    # Stats
    daily_rets = data.diff().dropna()
    # Bench
    bench_std = round(daily_rets.JPM.std(), 6)
    bench_cum_rets = round(daily_rets.JPM.sum(), 6)
    bench_avg_rets = round(daily_rets.JPM.mean(), 6)
    # port
    port_std = round(daily_rets.TOS.std(), 6)
    port_cum_rets = round(daily_rets.TOS.sum(), 6)
    port_avg_rets = round(daily_rets.TOS.mean(), 6)

    # Output
    headers = ['TOS', 'STD', 'Cummulative Rets', 'Average Rets']
    rows = [['JPM', bench_std, bench_cum_rets, bench_avg_rets], ['Portfolio', port_std, port_cum_rets, port_avg_rets]]
    df = pd.DataFrame(data=rows, columns=headers)
    print(df)

if __name__ == "__main__":
    run_test_project()
