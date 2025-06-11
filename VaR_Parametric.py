import pandas as pd 
import numpy as np 
import plotly.express as plt
import yfinance as yf 
import datetime as dt
import scipy.stats as sci

Time = 1
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(800)

Portfolio = [('TLT',0.2),('SPY',0.2),('QQQ',0.2),('GLD',0.2),('CL=F',0.2)]
Ticker = [Portfolio[i][0] for i in range(len(Portfolio))]
weight = np.array([Portfolio[i][1] for i in range(len(Portfolio))])

def Getdata(Ticker, start_date, end_date): 
    stock_data = yf.download(tickers=Ticker,start=start_date, end=end_date)
    if stock_data is None:
        raise('ERROR: Stock data has not been retrieved')
    stock_data = stock_data['Close'].dropna()
    r = stock_data.pct_change().dropna()
    r_mean = r.mean()
    cov = r.cov()
    return r, r_mean, cov 

def Portfolio_Performance(r_mean, cov, weight, Time):
    r_p = r_mean.dot(weight) * Time
    std_p = np.sqrt((weight.T.dot(cov).dot(weight))) * np.sqrt(Time)
    return r_p, std_p




print(Ticker)
print(weight)
r, r_mean, cov = Getdata(Ticker,start_date ,end_date)
r_p, std_p = Portfolio_Performance(r_mean, cov, weight, Time)
print(r_p, std_p)    
