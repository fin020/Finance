import numpy as np 
import datetime as dt
import pandas as pd
import yfinance as yf 
# all packages here 

#import data 
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    return stockData

List = ['AAPL', 'MSFT', 'ABC']
Stock = [stock+'.AX' for stock in List]

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

print(getData(Stock, start=startDate, end=endDate))