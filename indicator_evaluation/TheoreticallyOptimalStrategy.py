import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def author():
    return 'swagner38'

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000):
    # Get price data
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)[symbol]

    # Create trades dataframe
    trades = pd.DataFrame(index=prices.index, columns=['Symbol', 'Order', 'Shares'])
    trades['Symbol'] = symbol
    trades['Order'] = ''
    trades['Shares'] = 0

    current_holdings = 0
    for i in range(1, len(prices)-1):

        # Buy if tomorrow's price is higher than today's price
        if prices.iloc[i] < prices.iloc[i + 1]:
            trades.iloc[i] = ['JPM', 'BUY', abs(1000-current_holdings)]
            current_holdings += abs(1000-current_holdings)

        # Sell if tomorrow's price is lower than today's price
        elif prices.iloc[i] > prices.iloc[i + 1]:
            trades.iloc[i] = ['JPM', 'SELL', abs(-1000-current_holdings)]
            current_holdings -= abs(-1000-current_holdings)

    # Remove rows where no order was placed
    trades = trades[(trades['Order'] != '') & (trades['Shares'] != 0)]

    return trades
