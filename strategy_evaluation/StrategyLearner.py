""""""
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""

import datetime as dt
import numpy as np
import random
import pandas as pd
import util as ut
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class StrategyLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol,
        sd,
        ed,
        sv,
    ):
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """
        from RTLearner import RTLearner
        import indicators

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]

        sma = indicators.compute_sma(prices, window=14)
        rsi = indicators.compute_rsi(prices, window=14)
        momentum = indicators.compute_momentum(prices, window=14)

        # Combine indicators to form features
        X = np.column_stack((sma.values, momentum.values, rsi.values))
        # X = sma.values.reshape(-1,1)

        # Generating the corresponding Y values (trading signals)
        Y = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if prices.iloc[i].values > prices.iloc[i - 1].values:
                Y[i] = 1 - self.impact  # Buy
            else:
                Y[i] = -1 - self.impact # Sell

        # Train the learner
        self.learner = RTLearner(leaf_size=5)
        self.learner.add_evidence(X, Y)

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol,
        sd,
        ed,
        sv,
    ):
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """
        import indicators

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol]

        sma = indicators.compute_sma(prices, window=14)
        rsi = indicators.compute_rsi(prices, window=14)
        momentum = indicators.compute_momentum(prices, window=14)

        X_test = np.column_stack((sma.values, momentum.values, rsi.values))
        # X_test = sma.values.reshape(-1,1)

        # Generate trading signals using the trained learner
        signals = self.learner.query(X_test)

        # Generate trades based on signals
        trades = pd.DataFrame(index=prices.index)
        trades['Symbol'] = symbol
        trades['Order'] = ''
        trades['Shares'] = 0

        holdings = 0  # Current holdings, initially 0
        for i in range(1, len(prices)):
            if signals[i] > 0 and holdings <= 0:
                trades.loc[prices.index[i], 'Order'] = 'BUY'
                trades.loc[prices.index[i], 'Shares'] = 1000
                holdings += 1000
            elif signals[i] < 0 and holdings >= 0:
                trades.loc[prices.index[i], 'Order'] = 'SELL'
                trades.loc[prices.index[i], 'Shares'] = -1000
                holdings -= 1000

        trades = trades[(trades['Order'] != '') & (trades['Shares'] != 0)]
        return trades[['Shares']]

    def generate_results(self, symbol = "JPM", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2009, 12, 31), sv = 100000):
        from marketsimcode import compute_portvals
        import util as ut
        import matplotlib.pyplot as plt

        self.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        df_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

        # -------------- Port Stats --------------- #
        port_value = compute_portvals(symbol, df_trades, start_val=sv, commission=9.95, impact=0.005)
        port_value = port_value / port_value.iloc[0]

        # --------------- Bench Mark --------------- #
        data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns="SPY")
        data["Benchmark"] = data / data.iloc[0, 0]
        data["StrategyLearner"] = port_value

        # ------------- MS Comparison Chart ---------- #
        fig = plt.figure()
        plt.plot(data.Benchmark, 'tab:purple')
        plt.plot(data.StrategyLearner, 'r')

        # Plot long and short entry points
        ymin, ymax = plt.ylim()
        for index, row in df_trades.iterrows():
            if row['Order'] == 'BUY':
                plt.bar(x=index, height=ymax - 1, bottom=1, width=1.5, color='blue')
            elif row['Order'] == 'SELL':
                plt.bar(x=index, height=1 - ymin, bottom=ymin, width=1.5, color='black')

        plt.legend(["Benchmark", "Portfolio"])
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.title("Strategy Learner Portfolio vs Benchmark")
        plt.grid(True)
        fig.autofmt_xdate()
        plt.savefig("Strategy Learner Portfolio vs Benchmark.png")

        # ------------- Tables ------------- #
        # Stats
        daily_rets = data.diff().dropna()
        # Bench
        bench_std = round(daily_rets.Benchmark.std(), 6)
        bench_cum_rets = round(daily_rets.Benchmark.sum(), 6)
        bench_avg_rets = round(daily_rets.Benchmark.mean(), 6)
        # port
        port_std = round(daily_rets.StrategyLearner.std(), 6)
        port_cum_rets = round(daily_rets.StrategyLearner.sum(), 6)
        port_avg_rets = round(daily_rets.StrategyLearner.mean(), 6)

        # Output
        headers = ['Portfolio', 'STD', 'Cummulative Returns', 'Average Daily Returns']
        rows = [['Benchmark', bench_std, bench_cum_rets, bench_avg_rets],
                ['Strategy Learner', port_std, port_cum_rets, port_avg_rets]]
        df = pd.DataFrame(data=rows, columns=headers)
        print(df)

if __name__ == "__main__":
    print("One does not simply think up a strategy")
