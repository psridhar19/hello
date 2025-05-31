import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

def calculate_rsi(data, window=14):
    delta = data.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Example usage with historical stock data
ticker_symbol = 'INFY.NS'
start_date = '2023-04-01'
current_date = datetime.now().date()

# Generate example data (replace with actual data loading)
stock_data = yf.download(ticker_symbol, start=start_date, end=current_date)

# Calculate RSI
stock_data['RSI'] = calculate_rsi(stock_data['Close'])

# Plotting RSI
plt.figure(figsize=(14, 7))

# Plot RSI line
plt.plot(stock_data.index, stock_data['RSI'], label='RSI (14-day)', color='blue')

# Plotting overbought and oversold lines
plt.axhline(70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='g', linestyle='--', label='Oversold (30)')

# Highlighting overbought and oversold regions
plt.fill_between(stock_data.index, y1=70, y2=100, color='red', alpha=0.1)
plt.fill_between(stock_data.index, y1=0, y2=30, color='green', alpha=0.1)

# Adding markers for key RSI levels
plt.text(stock_data.index[-1], 70, ' Overbought', fontsize=10, va='center', ha='right', color='red')
plt.text(stock_data.index[-1], 30, ' Oversold', fontsize=10, va='center', ha='right', color='green')

# Adding plot labels and title
plt.title(f'Advanced Relative Strength Index (RSI) for {ticker_symbol}')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
