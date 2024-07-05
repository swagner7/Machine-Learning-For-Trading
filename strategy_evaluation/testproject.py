import ManualStrategy
from StrategyLearner import *
import experiment1
import experiment2

# set parameters
sd = dt.datetime(2010, 1, 1)
ed = dt.datetime(2011, 12, 31)
sv = 100000
symbol = "JPM"

# # run manual strategy
# ManualStrategy.generate_results(symbol=symbol, sd=sd, ed=ed, sv=sv)
#
# run Strategy Learner
# learner = StrategyLearner(commission=9.95, impact=0.005)
# learner.generate_results(symbol=symbol, sd=sd, ed=ed, sv=sv)

# run Experiment 1
experiment1.run_experiment1(symbol=symbol, sd=sd, ed=ed, sv=sv)

# # run Experiment 2
# experiment2.run_experiment2(symbol=symbol, sd=sd, ed=ed, sv=sv)