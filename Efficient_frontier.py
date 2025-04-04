from unittest import result
import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
import scipy.optimize as sp
r_f = 0.
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
    std_p = np.sqrt(np.dot(w.T,np.dot(cov_matrix, w)) ) * np.sqrt (250)
    return r_p, std_p 



tickers = ['AAPL', 'MSFT', 'GOOG']


end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)

w = np.array([0.8, 0.1, 0.1])
w /= np.sum(w)
# Fetch data
r_mean, cov_matrix = get_data(tickers, start_date, end_date)
r_p, std_p = portfolio_performance(w, r_mean, cov_matrix)


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
    return portfolio_performance(w, r_mean, cov_matrix)[1]

def minVar_p(r_mean, cov_matrix, w_constraint = (0,0.5)):
    "minimise the portfolio variance by changing the asset allocation of the portfolio"
    n_Assets = len(r_mean) 
    args = (r_mean, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bound = w_constraint
    bounds = tuple(bound for asset in range(n_Assets))
    result = sp.minimize(Var_p,n_Assets*[1./n_Assets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result



def EfficientOpt(r_mean, cov_matrix, r_target, w_constraint = (0,0.5)):
    # for each target we want to optimise portfolio for min variance
     n_Assets = len(r_mean) 
     args = (r_mean, cov_matrix)
     def r_portfolio(w):
        return portfolio_performance(w, r_mean, cov_matrix)[0]

     constraints = ({'type': 'eq', 'fun': lambda x: r_portfolio(x) - r_target},
                    {'type': 'eq', 'fun': lambda x: np.sum(x)-1})
     bound = w_constraint
     bounds = tuple(bound for asset in range(n_Assets))
     result = sp.minimize(Var_p,n_Assets*[1./n_Assets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
     return result     

def calculatedResults(r_mean, cov_matrix, r_f=r_f, w_constraint = (0,0.5)):
    #for max sharpe ratio portfolio
    MaxSR_p = maxSR(r_mean, cov_matrix)
    portfolio_performance(MaxSR_p['x'], r_mean, cov_matrix)
    maxSR_r, maxSR_std = portfolio_performance(MaxSR_p['x'], r_mean, cov_matrix) 
    maxSR_r = round(maxSR_r*100,2)
    maxSR_std = round(maxSR_std*100,2)
    maxSR_allocation = pd.DataFrame(MaxSR_p['x'], index=r_mean.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,1) for i in maxSR_allocation.allocation]
    #for min variance portfolio
    MinVol_p = minVar_p(r_mean, cov_matrix)
    portfolio_performance(MinVol_p['x'], r_mean, cov_matrix)
    MinVol_r, MinVol_std = portfolio_performance(MinVol_p['x'], r_mean, cov_matrix) 
    MinVol__r = round(MinVol_r*100,2)
    MinVol_std = round(MinVol_std*100,2)
    MinVol_allocation = pd.DataFrame(MinVol_p['x'], index=r_mean.index, columns=['allocation'])
    MinVol_allocation.allocation = [round(i*100,1) for i in MinVol_allocation.allocation]
    
    efficientList = []
    targetReturns = np.linspace(MinVol__r, maxSR_r, )
    for target in targetReturns: 
        efficientList.append(EfficientOpt(r_mean, cov_matrix, target)['fun'])
    efficientList = [float(i) for i in efficientList]
    return float(MinVol_r), float(MinVol_std), MinVol_allocation, float(maxSR_r), float(maxSR_std), maxSR_allocation, efficientList

print(calculatedResults(r_mean, cov_matrix))
# print(EfficientOpt(r_mean, cov_matrix, 0.07))
