import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data

def author():
    return 'swagner38'

def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window, min_periods=1).mean()
    rolling_std = data.rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_ema(data, window=20):
    return data.ewm(span=window, min_periods=1, adjust=False).mean()

def compute_sma(data, window=20):
    return data.rolling(window=window, min_periods=1).mean()

def compute_momentum(data, window=14):
    return (data / data.shift(window)) - 1

def run():
    # Load data
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    data = get_data([symbol], pd.date_range(start_date, end_date))

    # Compute indicators
    rsi = compute_rsi(data[symbol])
    macd, signal = compute_macd(data[symbol])
    upper_band, lower_band = compute_bollinger_bands(data[symbol])
    ema = compute_ema(data[symbol])
    sma = compute_sma(data[symbol])
    momentum = compute_momentum(data[symbol])

    # Plot indicators
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data[symbol], label='Price')
    plt.title('Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rsi, label='RSI', color='black')
    plt.title('RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.grid(True)
    plt.axhline(y=30, color='r', linestyle='--')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.text(data.index[-1], 28, 'Oversold', color='r', fontsize=10, ha='right', va='center')
    plt.text(data.index[-1], 72, 'Overbought', color='r', fontsize=10, ha='right', va='center')
    plt.tight_layout()
    plt.savefig("RSI Indicator Evaluation.png")


    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data[symbol], label='Price')
    plt.title('Price')
    plt.ylabel('Price')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(macd, label='MACD', color='blue')
    plt.plot(signal, label='Signal', color='red')
    plt.legend()
    plt.title('MACD Indicator')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("MACD Indicator Evaluation.png")


    plt.figure(figsize=(12, 8))
    plt.plot(data[symbol], label='Price', color='black')
    plt.plot(upper_band, label='Upper Band', color='red')
    plt.plot(lower_band, label='Lower Band', color='green')
    plt.plot(sma, label='SMA', color='blue')
    plt.legend()
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig("Bollinger Bands Indicator Evaluation.png")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data[symbol], label='Price')
    plt.title('Price')
    plt.ylabel('Price')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(momentum, label='Momentum', color='black')
    plt.title('Momentum')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.text(data.index[-1], -0.3, 'Downward Momentum (<0)', color='r', fontsize=10, ha='right', va='center')
    plt.text(data.index[-1], 0.5, 'Upward Momentum (>0)', color='r', fontsize=10, ha='right', va='center')
    plt.savefig("Momentum Indicator Evaluation.png")

    plt.figure(figsize=(12, 8))
    plt.plot(data[symbol], label='JPM Price')
    plt.plot(ema, label='EMA', color='orange')
    plt.legend()
    plt.title('Exponential Moving Average (EMA)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig("Exponential Moving Average Indicator Evaluation.png")

if __name__ == "__main__":
    run()
