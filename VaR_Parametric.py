import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt
import scipy.stats as sci
import scipy.integrate as integrate
import plotly.graph_objects as plt

# Retrieve data from stocks
def Getdata(Ticker, start_date, end_date): 
    stock_data = yf.download(tickers=Ticker,start=start_date, end=end_date, threads=False)
    if stock_data is None:
        raise ValueError('ERROR: Stock data has not been retrieved')
    stock_data = stock_data['Close'].dropna()
    r = stock_data.pct_change().dropna()
    r_mean = r.mean()
    cov = r.cov()
    return r, r_mean, cov 

def Portfolio_Performance(r_mean, cov, weights, Time):
    r_p = r_mean.dot(weights) * Time
    std_p = np.sqrt((weights.T.dot(cov).dot(weights))) * np.sqrt(Time)
    return r_p, std_p


def VaR_Parametric(mu, sigma, alpha):
    VaR = sci.norm.ppf(alpha / 100, mu ,sigma)
    return VaR

def CVaR_Parametric(VaR, mu, sigma, alpha,):
    z = (VaR -mu) / sigma
    CVaR = mu - sigma * sci.norm.pdf(z) / (alpha / 100)
    return CVaR

def Create_VaR_Plot(mu, sigma, VaR, CVaR, alpha):
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
        pdf = sci.norm.pdf(x,mu, sigma)
        x_tail = x[x <= VaR]
        pdf_tail = pdf[x <= VaR]

        fig = plt.Figure()

        fig.add_trace(plt.Scatter(
            x=x, y=pdf, mode='lines', name='Portfolio PDF', line=dict(color='blue')
        ))

        fig.add_trace(plt.Scatter(
            x=x_tail,
            y=pdf_tail,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.4)',
            line=dict(color='rgba(255,0,0,0)'),
            name=f'Tail Area (Worst {alpha}%)'
        ))

        fig.add_trace(plt.Scatter(
            x=[VaR, VaR],
            y=[0, sci.norm.pdf(VaR, mu, sigma)],
            mode='lines',
            name=f'VaR ({100-alpha}%)  =  {VaR:.2f}',
            line=dict(color='red', dash='dash')    
        ))
        fig.add_trace(plt.Scatter(
            x=[CVaR, CVaR],
            y=[0, sci.norm.pdf(CVaR, mu, sigma)],
            mode='lines',
            name=f'CVaR ({100-alpha}%)  =  {CVaR:.2f}',
            line=dict(color='purple', dash='dash')    
        ))

        fig.update_layout(
            title = 'Portfolio VaR and CVaR based on Parametric assumptions',
            xaxis_title = 'P/L',
            yaxis_title = 'Probability Density',
            template='simple_white',
            width=800,
        height=500,
            legend=dict(x=0.7, y=0.95)
        )
        return fig.show()

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
    r, r_mean, cov = Getdata(Ticker,start_date ,end_date)
    r_p, std_p = Portfolio_Performance(r_mean, cov, weights, Time)

    # Convert to normal distribution parameters
    mu = r_p * InitialInvestment
    sigma = std_p * InitialInvestment

    VaR = VaR_Parametric(mu, sigma, alpha)
    CVaR = CVaR_Parametric(VaR, mu, sigma, alpha)

    print(f"VaR ({100-alpha}%):", VaR)
    print(f"CVaR ({100-alpha}%):", CVaR)
    Create_VaR_Plot(mu, sigma, VaR, CVaR, alpha)