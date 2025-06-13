import yfinance as yf 
import pandas as pd 
import numpy as np
import datetime as dt

end = dt.datetime.today()
start = end - dt.timedelta(1250)
Time = 1


Portfolio = [('TLT', 0.2),('GLD', 0.2), ('SPY', 0.2), ('QQQ',0.2), ('CL=F',0.2)]
Tickers = [Portfolio[i][0] for i in range(len(Portfolio))]
weights = [Portfolio[i][1] for i in range(len(Portfolio))]
weights = np.array(weights)
if np.sum(weights)>1:
    raise TypeError('Weights exceed value of 100%')


def Getdata(start, end):
    stocks = yf.download(Tickers,start,end)
    if stocks is None:
        raise TypeError('Stock information has not been retrieved')
    stocks = stocks['Adj CLose']
    R = stocks.pct_change().dropna()
    R_mean = R.mean()
    cov = R.cov()
    return R_mean, cov

R_mean, cov = Getdata(start, end)

def PortfolioPerformance(r_mean, cov):
    R_p = R_mean.dot(weights)
    std_p = weights.T.dot(cov.dot(weights))
    return R_p, std_p

R_p, std_p = PortfolioPerformance(R_mean, cov)


    
