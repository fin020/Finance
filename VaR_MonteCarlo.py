import yfinance as yf 
import pandas as pd 
import numpy as np
import datetime as dt
import plotly.graph_objects as plt 
import scipy.stats as sci 

# Parameters to be entered
Time = 250
alpha = 5 
InitialInvestment = 100000
Portfolio = [('TLT', 0.2),('GLD', 0.2), ('SPY', 0.2), ('QQQ',0.2), ('CL=F',0.2)]
Tickers = [Portfolio[i][0] for i in range(len(Portfolio))]
weights = np.array([Portfolio[i][1] for i in range(len(Portfolio))])

if np.sum(weights)>1:
    raise TypeError('Weights exceed value of 100%')

end = dt.datetime.today()
start = end - dt.timedelta(1250)
# Retrieve data from yahoo finance
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

# Create Portfolio Simulations
mc_sims = 10000
T = Time
meanM = np.full(shape=(T, len(weights)), fill_value=R_mean)
meanM= meanM
Portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

for m in range(mc_sims):
    z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(cov)
    R_daily = meanM + z.dot(L)
    Portfolio_sims[:,m] = np.cumprod(weights.dot(R_daily.T)+1)* InitialInvestment
    

# Calculate Value at Risk and Conditional Value at Risk
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
VaR = VaR_mc(Results, alpha) - InitialInvestment
CVaR = CVaR_mc(Results, alpha) - InitialInvestment
print(f"VaR ({100 - alpha}%):", VaR)
print(f"CVaR ({100 - alpha}%):", CVaR)

Profit = Results - InitialInvestment
x_vals = np.linspace(Profit.min(), Profit.max(), 10000)
kde = sci.gaussian_kde(Profit)
kde_vals = kde(x_vals)
x_tail = x_vals[x_vals <= VaR]
kde_tail = kde(x_tail)

fig = plt.Figure()

fig.add_scatter(x=x_vals,
                y=kde_vals,
                mode = 'lines',
                name = 'KDE Estimate',
                line=dict(color='blue')
)

fig.add_histogram(x=Profit, nbinsx=500,
                  opacity=0.6,
                  histnorm='probability density',
                  name = 'Histogram of Simulated Portfolios',
                  marker_color='skyblue')

fig.add_trace(plt.Scatter(
    x=x_tail,
    y=kde_tail,
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.4)',
    line=dict(color='rgba(255,0,0,0)'),
    name=f'Tail Area (Worst {alpha}%)'
))

fig.add_scatter(x=[VaR], y=[kde(VaR)[0]], mode='markers', 
                name=f'VaR ({100-alpha}%)', marker=dict(color='red', size=10))

fig.add_trace(plt.Scatter(
    x=[VaR, VaR],
    y=[0, kde(VaR)[0]],
    mode='lines',
    name=f'VaR ({100-alpha}%)  =  {VaR:.2f}',
    line=dict(color='red', dash='dash')    
))

fig.add_scatter(x=[CVaR], y=[kde(CVaR)[0]], mode='markers', 
                name=f'CVaR ({100-alpha}%)', marker=dict(color='purple', size=10))

fig.add_trace(plt.Scatter(
    x=[CVaR, CVaR],
    y=[0, kde(CVaR)[0]],
    mode='lines',
    name=f'CVaR ({100-alpha}%)  =  {CVaR:.2f}',
    line=dict(color='purple', dash='dash')    
))

fig.update_layout(
    title = 'Portfolio VaR and CVaR based on Monte Carlo Simulation',
    xaxis_title = 'P/L',
    yaxis_title = 'Probability Density',
    template='simple_white',
    width=800,
    height=500,
    legend=dict(x=0.7, y=0.95)
)
# Show the plot
fig.show()