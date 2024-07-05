import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals
from util import get_data

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd



def call_strategies(symbol, sd, ed, sv):
    ms_trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner = sl.StrategyLearner(verbose=True, commission=9.95, impact=0.005)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    sl_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

    ms_port_value = compute_portvals(symbol, ms_trades, start_val=sv, commission=9.95, impact=0.005)
    ms_port_value = ms_port_value / ms_port_value.iloc[0]

    sl_port_value = compute_portvals(symbol, sl_trades, start_val=sv, commission=9.95, impact=0.005)
    sl_port_value = sl_port_value / sl_port_value.iloc[0]

    return ms_port_value, sl_port_value



def run_experiment1(symbol, sd, ed, sv):

    ms_port_value, sl_port_value = call_strategies(symbol = symbol, sd = sd, ed = ed, sv = sv)

    fig = plt.figure()
    plt.plot(ms_port_value, 'tab:purple')
    plt.plot(sl_port_value, 'r')
    plt.legend(["Manual Strategy", "Strategy Learner"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Manual Strategy Portfolio vs Strategy Learner")
    plt.grid(True)
    fig.autofmt_xdate()
    plt.savefig("Experiment 1.png")

    return



def author():
    return 'swagner38'