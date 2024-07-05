""""""  		  	   		 	   			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
  		  	   		 	   			  		 			     			  	 
Student Name: Samuel Wagner (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: swagner38 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903756749 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
import scipy.optimize as spo
import matplotlib.pyplot as plt  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	   			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	   			  		 			     			  	 
def optimize_portfolio(  		  	   		 	   			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   			  		 			     			  	 
    gen_plot=False,  		  	   		 	   			  		 			     			  	 
):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   			  		 			     			  	 
    statistics.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
    :type sd: datetime  		  	   		 	   			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
    :type ed: datetime  		  	   		 	   			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	   			  		 			     			  	 
        symbol in the data directory)  		  	   		 	   			  		 			     			  	 
    :type syms: list  		  	   		 	   			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	   			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	   			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	   			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	   			  		 			     			  	 
    :rtype: tuple  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	   			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	   			  		 			     			  	 

    # Define the objective function to minimize - negative Sharpe Ratio
    def neg_sharpe_ratio(allocations, prices):

        port_val = (prices / prices.iloc[0] * allocations).sum(axis=1)
        daily_returns = port_val.pct_change()[1:]
        sr = -np.sqrt(252) * (daily_returns.mean() / daily_returns.std())

        return sr

    # Define constraints for optimization - allocations sum to 1
    constraints = ({'type': 'eq', 'fun': lambda allocations: 1 - np.sum(allocations)})

    # Set bounds for allocations (0 to 1)
    bounds = tuple((0, 1) for _ in range(len(syms)))

    # Initial guess for allocations (equal weight)
    initial_allocations = np.ones(len(syms)) / len(syms)

    # Perform optimization
    result = spo.minimize(neg_sharpe_ratio, initial_allocations, args=(prices,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract optimal allocations
    allocs = result.x

    # Calculate statistics using optimal allocations
    port_val = (prices / prices.iloc[0] * allocs).sum(axis=1)
    daily_returns = port_val.pct_change()[1:]
    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = -neg_sharpe_ratio(allocs, prices)
  		  	   		 	   			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   			  		 			     			  	 
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)

        # Calculate actual portfolio values with optimized allocations
        port_val_actual = (prices / prices.iloc[0] * allocs).sum(axis=1)

        # Normalize SPY values
        df_temp["SPY"] = df_temp["SPY"] / df_temp["SPY"].iloc[0]

        # Plot actual portfolio values and normalized SPY on the same plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_temp.index, port_val_actual, label="Portfolio")
        plt.plot(df_temp.index, df_temp["SPY"], label="SPY", color="black")
        plt.title("Optimal Portfolio vs. Normalized SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True)
        plt.savefig('Fig1.png')
  		  	   		 	   			  		 			     			  	 
    return allocs, cr, adr, sddr, sr  		  	   		 	   			  		 			     			  	 


def test_code():  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	   			  		 			     			  	 
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	   			  		 			     			  	 
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Assess the portfolio  		  	   		 	   			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	   			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # Print statistics  		  	   		 	   			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	   			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	   			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	   			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	   			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	   			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	   			  		 			     			  	 
    test_code()  		  	   		 	   			  		 			     			  	 
