# Finance
This is my repository for all my work in modelling financial concepts using python

First project:
Using yfinance, I have retrieved a years worth of daily stock information for a portfolio of multiple assets and then calculated the mean reurns and then the covariance matrix of returns for the assets. Using this and minimising with scipy.optimize, I calcualted the weightings of the portfolio that has the highest sharpe ratio. I repeated this to also found the minimum variance portfolio. Then combining both I plotted the efficient frontier for all optimal portfolios for different levels of returns and also implemented a capital market line for the purpose of including a risk-free asset. 

Second Project:
Using libraries to implement three methods for calculating Value at Risk and Conditional Value at risk: Historical simulation, Parametric method and Monte Carlo Simulation. Eachh method estimates potential losses for a portfolio of assets under different risk modeling assumptions. The historical method uses actual past returns to simulate future risk, the parametric method assumes normally distributed returns and uses the mean and standard deviation of return to compute the risk metrics, and Monte Carlo Simulation generates a large number of random price momvements to model future outcomes and estimate risk. I am now aiming to assess the accuracy of each model using backtesting. 
