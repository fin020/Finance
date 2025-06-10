"Value at risk and Conditional Value at Risk using the Historical Method"

import pandas as pd 
import numpy as np
import datetime as dt
import yfinance as yf

#borrowing from Efficient Fronteir we import data:
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)
    if stock_data is None:
        raise ValueError("Failed to fetch stock data. Please check the stock symbols or network connection.")
    stock_data = stock_data['Close']
    r = stock_data.pct_change().dropna()    
    r_mean = r.mean()
    cov_matrix = r.cov()  # Computes the covariance matrix for the DataFrame (no argument needed)
    return r, r_mean, cov_matrix 

def portfolio_performance(w, r_mean, cov_matrix, Time):
    r_p = np.sum(r_mean * w) * Time 
    std_p = np.sqrt(np.dot(w.T,np.dot(cov_matrix, w)) ) * np.sqrt (Time)
    return r_p, std_p 


tickers = ['TLT', 'SPY', 'QQQ', 'GLD', 'CL=F']
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=800)
w = np.random.random(len(tickers))

Time = 1 #daily

# Fetch data
r, r_mean, cov_matrix = get_data(tickers, start_date, end_date)
r_p, std_p = portfolio_performance(w, r_mean, cov_matrix, Time)

r['Portfolio'] = r.dot(w)
print(r)

def HistoricalVaR(r, alpha=5):
    if isinstance(r, pd.Series):
        return np.percentile(r, alpha)
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(HistoricalVaR, alpha=5)
    
    else:
        raise TypeError("expected returns to be dataframe or series")


def HistoricalCVaR(r, alpha=5):
    if isinstance(r, pd.Series):
        belowVaR = r <- HistoricalVaR(r, alpha = alpha)
        return r[belowVaR].mean()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(HistoricalCVaR, alpha=5)
    
    else:
        raise TypeError("expected returns to be dataframe or series")
    

VaR = -HistoricalVaR(r['Portfolio'], alpha=5)*np.sqrt(Time)
CVaR = -HistoricalCVaR(r['Portfolio'], alpha=5)*np.sqrt(Time)

InitialInvestment = 1000000
print('Expected Portfolio Return:                      ', round(InitialInvestment*r_p,2))
print('Value at Risk at the 95% Confidence:            ',round(InitialInvestment*VaR,2))
print('Conditional Value at Risk at the 95% Confidence:',round(InitialInvestment*CVaR,2))

