import pandas as pd
import numpy as np
import datetime as dt
import StrategyLearner as sl
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt


def run_experiment2(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):

    impacts = []
    avg_daily_returns = []
    sharpe_ratios = []
    for impact in range(0, 100, 10):
        impact = impact/100
        learner = sl.StrategyLearner(verbose=True, commission=0, impact=impact)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sl_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)


        sl_port_value = compute_portvals(symbol, sl_trades, start_val=sv, commission=0, impact=impact)
        sl_port_value = sl_port_value / sl_port_value.iloc[0]

        cumul_return = float((sl_port_value[-1] / sl_port_value[0]) - 1)
        mean_ret = sl_port_value.mean()
        std_daily_ret = sl_port_value.std()
        sharpe_ratio = np.sqrt(252) * (mean_ret / std_daily_ret)

        impacts.append(impact)
        avg_daily_returns.append(mean_ret)
        sharpe_ratios.append(sharpe_ratio)

    fig, axs = plt.subplots(2, figsize=(10, 8))

    axs[0].plot(impacts, avg_daily_returns, marker='o', linestyle='-')
    axs[0].set_title('Average Daily Returns by Impact')
    axs[0].set_xlabel('Impact')
    axs[0].set_ylabel('Average Daily Returns')

    axs[1].plot(impacts, sharpe_ratios, marker='o', linestyle='-')
    axs[1].set_title('Sharpe Ratios by Impact')
    axs[1].set_xlabel('Impact')
    axs[1].set_ylabel('Sharpe Ratio')

    plt.tight_layout()
    plt.savefig('Experiment 2.png')

    return

def author():
  return 'swagner38'