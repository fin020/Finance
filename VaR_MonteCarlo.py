import yfinance as yf  #type: ignore 
import pandas as pd 
from pandas import DataFrame, Series
import numpy as np
import datetime as dt
from datetime import datetime
import plotly.graph_objects as plt  #type: ignore 
import scipy.stats as sci  #type: ignore 


CONFIG = {
    'Time': 250,
    'alpha': 5,
    'InitialInvestment': 100000,
    'mc_sims': 1000
}

 
def get_portfolio():
    Portfolio = [('TLT', 0.2),('GLD', 0.2), ('SPY', 0.2), ('QQQ',0.2),
                ('CL=F',0.2)]
    tickers = list([x[0] for x in Portfolio])
    weights = np.array([x[1] for x in Portfolio])
    return tickers, weights

end = dt.datetime.today()
start = end - dt.timedelta(1250)

# Retrieve data from yahoo finance
def get_data(
        stocks: list[str]|str, start: str|datetime =start, 
        end: str|datetime= end)-> DataFrame: 
    stock_data:DataFrame|Series = yf.download(stocks,start=start,end=end) #type: ignore
    stock_data = stock_data['Close']
    return stock_data #type: ignore

def compute_returns(
        stock_data: DataFrame):
    R = stock_data.pct_change(fill_method=None).dropna() #type:ignore
    return R

def calculate_stats(
        R: DataFrame):
    R_mean = R.mean()
    cov = R.cov() #type: ignore
    return R_mean, cov  #type: ignore

def portfolio_performance(
        weights: Series, 
        R_mean: Series, cov)-> tuple[float]: #type: ignore
    R_p: float = R_mean.dot(weights) #type: ignore
    std_p: float = weights.T.dot(cov.dot(weights)) #type: ignore
    return R_p, std_p #type: ignore

def portfolio_simulation(
        R_mean: DataFrame, weights: Series,
        cov: DataFrame, mc_sims: int=CONFIG['mc_sims'],
        initialInvestment: float=CONFIG['InitialInvestment'],
        T: int=CONFIG['Time']):
    meanM = np.full(shape=(T, len(weights)), fill_value=R_mean)
    meanM= meanM
    Portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(cov) #type: ignore
        R_daily = meanM + z.dot(L)
        Portfolio_sims[:,m] = np.cumprod(weights.dot(R_daily.T)+1)* initialInvestment #type: ignore
        
    Results = pd.Series(Portfolio_sims[-1,:])
    return Results
# Calculate Value at Risk and Conditional Value at Risk
def var_mc(
        R: Series, alpha: int=CONFIG['alpha']):
    return np.percentile(R, alpha)

    
def cvar_mc(
        R: Series, alpha: int=CONFIG['alpha']):
    belowVaR = R <= var_mc(R, alpha=alpha)
    return R[belowVaR].mean()


def print_var_results(
        initialInvestment: float, Results: Series, alpha: int):
    VaR = var_mc(Results, alpha) - initialInvestment
    CVaR = cvar_mc(Results, alpha) - initialInvestment
    print(f"VaR ({100 - alpha}%):", VaR)
    print(f"CVaR ({100 - alpha}%):", CVaR)


def _plot_histogram( #type: ignore
        Results: Series, fig, #type: ignore
        initialInvestment: float=CONFIG['InitialInvestment']):
    Profit = Results - initialInvestment
      
    fig.add_histogram( #type: ignore
                x=Profit, nbinsx=500, #type: ignore
                  opacity=0.6,
                  histnorm='probability density',
                  name = 'Histogram of Simulated Portfolios',
                  marker_color='skyblue')
    return fig #type: ignore

def _plot_kde( #type: ignore
        Results: Series, VaR: float,fig, kde, #type: ignore
        initialInvestment: float=CONFIG['InitialInvestment'],
        alpha: int=CONFIG['alpha']):
    Profit = Results - initialInvestment
    x_vals = np.linspace(Profit.min(), Profit.max(), 10000) #type: ignore
    kde_vals = kde(x_vals) #type: ignore
    
    fig.add_scatter( #type: ignore
            x=x_vals, #type: ignore
                y=kde_vals, #type: ignore
                mode = 'lines',
                name = 'KDE Estimate',
                line=dict(color='blue')
    )
    
    x_tail = x_vals[x_vals <= VaR] #type: ignore
    kde_tail = kde(x_tail) #type: ignore
    
    fig.add_trace( #type: ignore
        plt.Scatter( #type: ignore
        x=x_tail, #type: ignore
        y=kde_tail, #type: ignore
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.4)',
        line=dict(color='rgba(255,0,0,0)'),
        name=f'Tail Area (Worst {alpha}%)'
    ))
    return fig #type: ignore
    
def _var_markers( #type: ignore
        fig, VaR: float, CVaR: float, #type: ignore
        kde, alpha: int=CONFIG['alpha']):    #type: ignore 

    fig.add_scatter( #type: ignore
        x=[VaR], y=[kde(VaR)[0]], mode='markers',  #type: ignore
        name=f'VaR ({100-alpha}%)', marker=dict(color='red', size=10))
    
    fig.add_trace(plt.Scatter( #type: ignore
        x=[CVaR, CVaR],
        y=[0, kde(CVaR)[0]], #type: ignore
        mode='lines',
        name=f'CVaR ({100-alpha}%)  =  {CVaR:.2f}',
        line=dict(color='purple', dash='dash')    
    ))
    fig.add_scatter( #type: ignore
        x=[CVaR], y=[kde(CVaR)[0]], #type: ignore
        mode='markers',  #type: ignore
        name=f'CVaR ({100-alpha}%)', 
        marker=dict(color='purple', size=10))
    
    return fig #type: ignore

def plot_risk( #type: ignore
        Results: Series, alpha: int,
        initialInvestment: float):
    Profit = Results - initialInvestment
    kde = sci.gaussian_kde(Profit)
    VaR = var_mc(Results, alpha) - initialInvestment
    CVaR = cvar_mc(Results, alpha) - initialInvestment
    print(VaR, CVaR)
    
    fig = plt.Figure()
    
    fig = _plot_histogram(Results,fig, initialInvestment)
    fig = _plot_kde(Results, VaR, fig, kde, initialInvestment, alpha) #type: ignore
    fig = _var_markers( fig, VaR, CVaR, kde, alpha) #type: ignore
    


    fig.add_trace(plt.Scatter( #type: ignore
        x=[VaR, VaR],
        y=[0, kde(VaR)[0]],
        mode='lines',
        name=f'VaR ({100-alpha}%)  =  {VaR:.2f}',
        line=dict(color='red', dash='dash')    
    ))
    fig.update_layout( #type: ignore
        title = 'Monte Carlo Simulation VaR and CVaR',
        xaxis_title = 'P/L',
        yaxis_title = 'Probability Density',
        template='simple_white',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.95)
    )
    return fig #type: ignore

def main():
    tickers, weights = get_portfolio()
    data = get_data(tickers, start, end)
    returns = compute_returns(data) #type: ignore
    R_mean, cov = calculate_stats(returns)
    Results = portfolio_simulation(R_mean, weights, cov) #type: ignore
    print_var_results(CONFIG['InitialInvestment'], Results, CONFIG["alpha"])
    plot_risk(Results, CONFIG["alpha"], CONFIG['InitialInvestment'])
    
if __name__ == '__main__':
    main()