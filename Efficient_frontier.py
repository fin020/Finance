from unittest import result
import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import scipy.optimize as sp
r_f = 0.046
# Function to fetch and process stock data
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)
    if stock_data is None:
        raise ValueError("Failed to fetch stock data. Please check the stock symbols or network connection.")
    stock_data = stock_data['Close']

    r = stock_data.pct_change().dropna()    
    r_mean = r.mean()
    cov_matrix = r.cov()  # Computes the covariance matrix for the DataFrame (no argument needed)
    return r_mean, cov_matrix

def portfolio_performance(w, r_mean, cov_matrix):
    r_p = np.sum(r_mean * w) * 250 
    std_p = np.sqrt(np.dot(w.T,np.dot(cov_matrix, w))) * np.sqrt (250)
    return r_p, std_p 



tickers = ['AAPL', 'MSFT', 'GOOG']


end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)





def negSR(w, r_mean, cov_matrix, r_f=r_f):
    r_p, std_p = portfolio_performance(w, r_mean, cov_matrix)
    return -(r_p-r_f)/std_p

def maxSR(r_mean, cov_matrix, r_f=r_f, w_constraint = (0,0.5)):
    "minimise the negative Sharpe ratio by changing the weights"
    n_Assets = len(r_mean)
    args = (r_mean, cov_matrix, r_f)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bound = w_constraint
    bounds = tuple(bound for asset in range(n_Assets))
    result = sp.minimize(negSR,n_Assets*[1./n_Assets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def Var_p(w, r_mean, covmatrix):
    return portfolio_performance(w, r_mean, cov_matrix)

def minVar_p(r_mean, cov_matrix, r_f = r_f, w_constraint = (0,0.5)):
    "minimise the portfolio variance by changing the asset allocation of the portfolio"
    n_Assets = len(r_mean) 
    args = (r_mean, cov_matrix, r_f)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bound = w_constraint
    bounds = tuple(bound for asset in range(n_Assets))
    result = sp.minimize(Var_p,n_Assets*[1./n_Assets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

w = np.array([0.8, 0.1, 0.1])
w /= np.sum(w)
# Fetch data
r_mean, cov_matrix = get_data(tickers, start_date, end_date)
r_p, std_p = portfolio_performance(w, r_mean, cov_matrix)

print(round(r_p*100,2), round(std_p*100,2))

result = maxSR(r_mean, cov_matrix)
maxSR, maxW = result['fun'],result['x']
print(maxSR, maxW)

minVarResult = minVar_p(r_mean, cov_matrix)
minVar_p, minVarw = minVarResult['fun'],minVarResult['x']
print(minVar_p, maxW)
