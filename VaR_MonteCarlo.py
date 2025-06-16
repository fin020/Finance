import yfinance as yf 
import pandas as pd 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 

end = dt.datetime.today()
start = end - dt.timedelta(1250)
Time = 1
alpha = 1 
InitialInvestment = 100000
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
    stocks = stocks['Close']
    R = stocks.pct_change(fill_method=None).dropna()
    R_mean = R.mean()
    cov = R.cov()
    return R_mean, cov

R_mean, cov = Getdata(start, end)

def PortfolioPerformance(R_mean, cov):
    R_p = R_mean.dot(weights)
    std_p = weights.T.dot(cov.dot(weights))
    return R_p, std_p

R_p, std_p = PortfolioPerformance(R_mean, cov)


mc_sims = 1000
T = Time

meanM = np.full(shape=(T, len(weights)), fill_value=R_mean)
meanM= meanM

Portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)


for m in range(mc_sims):
    z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(cov)
    R_daily = meanM + z.dot(L)
    Portfolio_sims[:,m] = np.cumprod(weights.dot(R_daily.T)+1)* InitialInvestment
    


def VaR_mc(R, alpha):
    if isinstance(R, pd.Series):
        return np.percentile(R, alpha)
    else:
        raise TypeError('Expected a pandas data series')
    
def CVaR_mc(R, alpha):
    if isinstance(R, pd.Series):
        belowVaR = R <= VaR_mc(R, alpha=alpha)
        return R[belowVaR].mean()
    else:
        raise TypeError('Expected a pandas data series')
    
Results = pd.Series(Portfolio_sims[-1,:])

VaR = InitialInvestment - VaR_mc(Results, alpha)
CVaR = InitialInvestment - CVaR_mc(Results, alpha)
print("VaR:", VaR)
print("CVaR:", CVaR)
