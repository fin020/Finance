"Value at risk and Conditional Value at Risk using the Historical Method"

import pandas as pd 
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as plt
import scipy.stats as sci


#Parameters to be entered:
tickers = ['TLT', 'SPY', 'QQQ', 'GLD', 'CL=F']
w = np.array([0.2,0.2,0.2,0.2,0.2,])
Time = 1#only accurate for low value of time
InitialInvestment = 1000000
alpha = 5

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=1250)

if np.sum(w)>1:
    raise TypeError('Weights exceed value of 100%')
#borrowing from Efficient Fronteir we import data:
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)
    if stock_data is None:
        raise ValueError("Failed to fetch stock data. Please check the stock symbols or network connection.")
    stock_data = stock_data['Close']
    r = stock_data.pct_change().dropna()    
    r_mean = r.mean()
      # Computes the covariance matrix for the DataFrame (no argument needed)
    return r, r_mean

def portfolio_performance(w, r_mean, Time):
    r_p = np.sum(r_mean * w) * Time 
    return r_p
# r_p is the mean rate of portfolio daily returns

# Fetch data
r, r_mean, = get_data(tickers, start_date, end_date)
r_p = portfolio_performance(w, r_mean,  Time)

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
print(f'Value at Risk at the {100-alpha}% Confidence:            ',round(-InitialInvestment*VaR,2))
print(f'Conditional Value at Risk at the {100-alpha}% Confidence:',round(-InitialInvestment*CVaR,2))


#now time to create the profit-loss distribtion

print(r['Portfolio']*InitialInvestment)


    
# Convert portfolio returns to a DataFrame
df = r[['Portfolio']].copy()
df['Portfolio'] *=InitialInvestment  # Scale by initial investment
print(df['Portfolio'])
# Compute the KDE (Kernel Density Estimate)
x_vals = np.linspace(df['Portfolio'].min(), df['Portfolio'].max(), 10000)  # Smooth x-axis
kde = sci.gaussian_kde(df['Portfolio'])  # Fit KDE
kde_vals = kde(x_vals)  # Get KDE values

expected_return = InitialInvestment * r_p
VaR_point = InitialInvestment * VaR
CVaR_point = InitialInvestment * CVaR
x_tail = x_vals[x_vals <= VaR_point]
kde_tail = kde(x_tail) 

fig = plt.Figure()

# Add the KDE curve as a line plot
fig.add_scatter(x=x_vals, y=kde_vals, mode='lines', name='KDE Estimate', line=dict(color='blue'))

# Create an interactive histogram (normalized)
fig.add_histogram(x=df['Portfolio'], nbinsx=500, opacity=0.6, histnorm='probability density', name='Histogram of Historical Returns', 
                   marker_color="skyblue")


fig.add_scatter(x=[expected_return], y=[kde(expected_return)[0]], mode='markers', 
                name='Expected Return', marker=dict(color='green', size=10))

fig.add_scatter(x=[VaR_point], y=[kde(VaR_point)[0]], mode='markers', 
                name=f'VaR ({100-alpha}%)', marker=dict(color='red', size=10))

fig.add_trace(plt.Scatter(
    x=x_tail,
    y=kde_tail,
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.4)',
    line=dict(color='rgba(255,0,0,0)'),
    name=f'Tail Area (Worst {alpha}%)'
))

fig.add_trace(plt.Scatter(
    x=[VaR_point, VaR_point],
    y=[0, kde(VaR_point)[0]],
    mode='lines',
    name=f'VaR ({100-alpha}%)  =  {VaR_point:.2f}',
    line=dict(color='red', dash='dash')    
))

fig.add_scatter(x=[CVaR_point], y=[kde(CVaR_point)[0]], mode='markers', 
                name=f'CVaR ({100-alpha}%)', marker=dict(color='purple', size=10))

fig.add_trace(plt.Scatter(
    x=[CVaR_point, CVaR_point],
    y=[0, kde(CVaR_point)[0]],
    mode='lines',
    name=f'CVaR ({100-alpha}%)  =  {CVaR_point:.2f}',
    line=dict(color='purple', dash='dash')    
))

fig.update_layout(
    title = 'Portfolio VaR and CVaR based on Historical Returns',
    xaxis_title = 'P/L',
    yaxis_title = 'Probability Density',
    template='simple_white',
    width=800,
    height=500,
    legend=dict(x=0.7, y=0.95)
)
# Show the plot
fig.show()


