from pandas import DataFrame, Series
from datetime import datetime
from typing import Any
import numpy as np 
import yfinance as yf #type: ignore
import datetime as dt
import scipy.stats as sci #type: ignore
import scipy.integrate as integrate #type: ignore
import plotly.graph_objects as plt #type: ignore

# Retrieve data from stocks
def get_data(
        Ticker: list[str], start_date: datetime|str,
        end_date: datetime|str)->tuple[DataFrame, Series]:
    stock_data = yf.download(tickers=Ticker,start=start_date, end=end_date, threads=False) #type: ignore
    if stock_data is None:
        raise ValueError('ERROR: Stock data has not been retrieved')
    stock_data = stock_data['Close'].dropna()
    r = stock_data.pct_change().dropna()
    r_mean = r.mean()
    cov = r.cov() #type: ignore
    return r, r_mean, cov  #type: ignore

def portfolio_performance(
        r_mean: Series, cov: DataFrame, weights: Any,
        Time: int) -> tuple[float,float]:
    r_p = r_mean.dot(weights) * Time #type: ignore
    std_p = np.sqrt((weights.T.dot(cov).dot(weights))) * np.sqrt(Time)
    return r_p, std_p #type: ignore


def var_parametric(
        mu: float, sigma: float, alpha: int):
    VaR = sci.norm.ppf(alpha / 100, mu ,sigma) #type: ignore
    return VaR

def cvar_parametric(
        VaR: float, mu: float, sigma: float, alpha:int):
    z = (VaR -mu) / sigma
    CVaR = mu - sigma * sci.norm.pdf(z) / (alpha / 100) #type: ignore
    return CVaR

def create_var_plot(
        mu: float, sigma: float, VaR: float, 
        CVaR: float, alpha: int):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = sci.norm.pdf(x,mu, sigma) #type: ignore
    x_tail = x[x <= VaR]
    pdf_tail = pdf[x <= VaR] 

    fig = plt.Figure()

    fig.add_trace(plt.Scatter( #type: ignore
        x=x, y=pdf, mode='lines', name='Portfolio PDF', line=dict(color='blue')
    ))

    fig.add_trace(plt.Scatter( #type: ignore
        x=x_tail,
        y=pdf_tail,
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.4)',
        line=dict(color='rgba(255,0,0,0)'),
        name=f'Tail Area (Worst {alpha}%)'
    ))

    fig.add_trace(plt.Scatter( #type: ignore
        x=[VaR, VaR],
        y=[0, sci.norm.pdf(VaR, mu, sigma)], #type: ignore
        mode='lines',
        name=f'VaR ({100-alpha}%)  =  {VaR:.2f}',
        line=dict(color='red', dash='dash')    
    ))
    fig.add_trace(plt.Scatter( #type: ignore
        x=[CVaR, CVaR],
        y=[0, sci.norm.pdf(CVaR, mu, sigma)], #type: ignore
        mode='lines',
        name=f'CVaR ({100-alpha}%)  =  {CVaR:.2f}',
        line=dict(color='purple', dash='dash')    
    ))

    fig.update_layout( #type: ignore
        title = 'Portfolio VaR and CVaR based on Parametric assumptions',
        xaxis_title = 'P/L',
        yaxis_title = 'Probability Density',
        template='simple_white',
        width=800,
    height=500,
        legend=dict(x=0.7, y=0.95)
    )
    return fig.show() #type: ignore

if __name__ == '__main__':
    Time = 250
    alpha = 5
    InitialInvestment = 100000
    Portfolio = [('TLT',0.2),('SPY',0.2),('QQQ',0.2),('GLD',0.2),('CL=F',0.2)]
    Ticker = [Portfolio[i][0] for i in range(len(Portfolio))]
    weights = np.array([Portfolio[i][1] for i in range(len(Portfolio))])

    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(1250)
    if np.sum(weights)>1:
        raise TypeError('Weights exceed value of 100%')
    # Fetch data
    r, r_mean, cov = get_data(Ticker,start_date ,end_date) #type: ignore
    r_p, std_p = portfolio_performance(r_mean, cov, weights, Time) #type: ignore

    # Convert to normal distribution parameters
    mu = r_p * InitialInvestment
    sigma = std_p * InitialInvestment

    VaR = var_parametric(mu, sigma, alpha)
    CVaR = cvar_parametric(VaR, mu, sigma, alpha) #type: ignore

    print(f"VaR ({100-alpha}%):", VaR)
    print(f"CVaR ({100-alpha}%):", CVaR)
    create_var_plot(mu, sigma, VaR, CVaR, alpha) #type: ignore