"Value at risk and Conditional Value at Risk using the Historical Method"

import pandas as pd 
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.express as px
import pandas as pd
import scipy.stats as stats




#Parameters to be entered:
tickers = ['TLT', 'SPY', 'QQQ', 'GLD', 'CL=F']
w = np.array([0.2,0.2,0.2,0.2,0.2,])
Time = 1#only accurate for low value of time
InitialInvestment = 1000000
alpha = 5

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=1250)

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
    std_p = np.sqrt(np.dot(w.T,np.dot(cov_matrix, w)) ) * np.sqrt(Time)
    return r_p, std_p 
# r_p is the mean rate of portfolio daily returns

# Fetch data
r, r_mean, cov_matrix = get_data(tickers, start_date, end_date)
r_p, std_p = portfolio_performance(w, r_mean, cov_matrix, Time)

#creating a daily portfolio returns column 
r['Portfolio'] = r.dot(w)

#Historical value at risk - This computes the level at which alpha is the significance level of returns
def HistoricalVaR(r, alpha=alpha):
    if isinstance(r, pd.Series):
        return np.percentile(r, alpha)
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(HistoricalVaR, alpha=alpha)
    
    else:
        raise TypeError("expected returns to be dataframe or series")

#This takes the VaR and computes the average of the returns below the significance level alpha
def HistoricalCVaR(r, alpha=alpha):
    if isinstance(r, pd.Series):
        belowVaR = r < HistoricalVaR(r, alpha = alpha)
        return r[belowVaR].mean()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(HistoricalCVaR, alpha=alpha)
    
    else:
        raise TypeError("expected returns to be dataframe or series")
    

VaR = HistoricalVaR(r['Portfolio'], alpha)*np.sqrt(Time) 
CVaR = HistoricalCVaR(r['Portfolio'], alpha)*np.sqrt(Time)


print('Expected Portfolio Return:                      ', round(InitialInvestment*r_p,2))
print('Value at Risk at the 95% Confidence:            ',round(-InitialInvestment*VaR,2))
print('Conditional Value at Risk at the 95% Confidence:',round(-InitialInvestment*CVaR,2))


#now time to create the profit-loss distribtion

print(r['Portfolio']*InitialInvestment)


    
# Convert portfolio returns to a DataFrame
df = r[['Portfolio']].copy()
df['Portfolio'] *=InitialInvestment  # Scale by initial investment

# Compute the KDE (Kernel Density Estimate)
x_vals = np.linspace(df['Portfolio'].min(), df['Portfolio'].max(), 10000)  # Smooth x-axis
kde = stats.gaussian_kde(df['Portfolio'])  # Fit KDE
kde_vals = kde(x_vals)  # Get KDE values

# Create an interactive histogram (normalized)
fig = px.histogram(df, x='Portfolio', nbins=500, opacity=0.6, histnorm='probability density', 
                   title="Probability Density Function of Portfolio Returns",
                   color_discrete_sequence=["skyblue"])

# Add the KDE curve as a line plot
fig.add_scatter(x=x_vals, y=kde_vals, mode='lines', name='KDE Estimate', line=dict(color='red'))

expected_return = InitialInvestment * r_p
VaR_point = InitialInvestment * VaR
CVaR_point = InitialInvestment * CVaR

fig.add_scatter(x=[expected_return], y=[kde(expected_return)[0]], mode='markers', 
                name='Expected Return', marker=dict(color='green', size=10))
fig.add_scatter(x=[VaR_point], y=[kde(VaR_point)[0]], mode='markers', 
                name='VaR (95%)', marker=dict(color='blue', size=10))
fig.add_scatter(x=[CVaR_point], y=[kde(CVaR_point)[0]], mode='markers', 
                name='CVaR (95%)', marker=dict(color='purple', size=10))
# Show the plot
fig.show()


