"""
Value at risk and Conditional Value at Risk using the Historical Method
"""

from pandas import Series, DataFrame
import numpy as np
from numpy.typing import NDArray
from typing import Union, Any
from datetime import datetime
import yfinance as yf  #type: ignore
import plotly.graph_objects as plt  #type: ignore
import scipy.stats as sci  #type: ignore

def get_data(
    stocks: list[str]|str,
    start: Union[str, datetime],
    end: Union[str, datetime]
) -> tuple[DataFrame, Series]:
    stock_data = yf.download(stocks, start=start, end=end)  #type: ignore
    if stock_data is None:
        raise ValueError("Failed to fetch stock data. Please check the stock symbols or network connection.")
    stock_data = stock_data['Close']
    r = stock_data.pct_change().dropna()
    r_mean = r.mean()
    return r, r_mean #type: ignore
 
def portfolio_performance(
    w: NDArray[Any],
    r_mean: Series,
    Time: int
) -> float:
    r_p = r_mean.dot(w) * Time #type: ignore
    return r_p #type: ignore

def historical_var(
    r: Union[Series, NDArray[Any]],
    alpha: int
) -> float:
    
    return np.percentile(r, alpha) #type: ignore

def historical_cvar(
    r: Union[Series, NDArray[Any]],
    alpha: int
) -> float:
    below_var = r < historical_var(r, alpha=alpha)
    return r[below_var].mean()

def create_portfolio_returns(
    r: DataFrame,
    w: NDArray[Any]
) -> DataFrame:
    r = r.copy()
    r['Portfolio'] = r.dot(w) #type: ignore
    return r #type: ignore

def create_var_cvar_plot(
    r: DataFrame,
    alpha: int,
    Time: int,
    InitialInvestment: float,
    r_p: float
):
    fig = plt.Figure()
    T = Time  # e.g., 250 for one year
    portfolio_returns = r['Portfolio']

    # Using log returns is more accurate for compounding
    log_returns = np.log1p(portfolio_returns)  
    aggregated_log_returns = log_returns.rolling(window=T).sum().dropna() #type: ignore

    # Compute VaR and CVaR directly from the aggregated returns
    VaR = historical_var(aggregated_log_returns, alpha) #type: ignore
    CVaR = historical_cvar(aggregated_log_returns, alpha) #type: ignore
    PL = InitialInvestment * np.expm1(aggregated_log_returns) #type: ignore
    x_vals = np.linspace(PL.min(), PL.max(), 10000) #type: ignore
    kde = sci.gaussian_kde(PL)
    kde_vals = kde(x_vals) #type: ignore
    expected_return = InitialInvestment * r_p
    VaR_point = InitialInvestment * (np.expm1(VaR))
    CVaR_point = InitialInvestment * (np.expm1(CVaR))
    x_tail = x_vals[x_vals <= VaR_point] #type: ignore
    kde_tail = kde(x_tail) #type: ignore
    fig.add_scatter(x=x_vals, y=kde_vals, mode='lines', name='KDE Estimate', line=dict(color='blue')) #type: ignore
    fig.add_histogram(x=PL, nbinsx=500, opacity=0.6, histnorm='probability density', #type: ignore
                      name='Histogram of Historical Returns', marker_color="skyblue") 
    fig.add_scatter(x=[expected_return], y=[kde(expected_return)[0]], mode='markers', #type: ignore
                    name='Expected Return', marker=dict(color='green', size=10))
    fig.add_scatter(x=[VaR_point], y=[kde(VaR_point)[0]], mode='markers',  #type: ignore
                    name=f'VaR ({100-alpha}%)', marker=dict(color='red', size=10))
    fig.add_trace(plt.Scatter( #type: ignore
        x=x_tail, #type: ignore
        y=kde_tail, #type: ignore
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.4)',
        line=dict(color='rgba(255,0,0,0)'),
        name=f'Tail Area (Worst {alpha}%)'
    ))
    fig.add_trace(plt.Scatter( #type: ignore
        x=[VaR_point, VaR_point],
        y=[0, kde(VaR_point)[0]],
        mode='lines',
        name=f'VaR ({100-alpha}%)  =  {VaR_point:.2f}',
        line=dict(color='red', dash='dash')
    ))
    fig.add_scatter(x=[CVaR_point], y=[kde(CVaR_point)[0]], mode='markers', #type: ignore
                    name=f'CVaR ({100-alpha}%)', marker=dict(color='purple', size=10))
    fig.add_trace(plt.Scatter( #type: ignore
        x=[CVaR_point, CVaR_point],
        y=[0, kde(CVaR_point)[0]],
        mode='lines',
        name=f'CVaR ({100-alpha}%)  =  {CVaR_point:.2f}',
        line=dict(color='purple', dash='dash')
    ))
    fig.update_layout( #type: ignore
        title='Portfolio VaR and CVaR based on Historical Returns',
        xaxis_title='P/L',
        yaxis_title='Probability Density',
        template='simple_white',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.95)
    )
    fig.show() #type: ignore

def print_var_cvar_report(
    InitialInvestment: float,
    r_p: float,
    VaR: float,
    CVaR: float,
    r: DataFrame,
    alpha: int
):
    print('Expected Portfolio Return:                      ', round(InitialInvestment * r_p, 2))
    print(f'Value at Risk at the {100 - alpha}% Confidence:            ', round(-InitialInvestment * VaR, 2))
    print(f'Conditional Value at Risk at the {100 - alpha}% Confidence:', round(-InitialInvestment * CVaR, 2))
    print(r['Portfolio'] * InitialInvestment)
