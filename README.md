# Finance
This is my repository for all my work in modelling financial concepts using python

First project:
Using yfinance, I have retrieved a years worth of daily stock information for a portfolio of multiple assets and then calculated the mean reurns and then the covariance matrix of returns for the assets. Using this and minimising with scipy.optimize, I calcualted the weightings of the portfolio that has the highest sharpe ratio. I repeated this to also found the minimum variance portfolio. Then combining both I plotted the efficient frontier for all optimal portfolios for different levels of returns and also implemented a capital market line for the purpose of including a risk-free asset. 
