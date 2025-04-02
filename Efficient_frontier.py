import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf

# Function to fetch and process stock data
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)['Adj Close']

    # Ensure data is not empty
    if stock_data is None or stock_data.empty:
        raise ValueError("No data retrieved. Check ticker symbols or API connectivity.")

    # Compute log returns
    log_returns = np.log(stock_data.pct_change() + 1).dropna()

    return log_returns

# Define stock tickers
tickers = ['AAPL', 'MSFT', 'GOOG']

# Define start and end dates
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)

# Fetch stock data and log returns
try:
    log_returns = get_data(tickers, start=start_date, end=end_date)
    print(log_returns.head())  # Show first 5 rows
except ValueError as e:
    print(e)