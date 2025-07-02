#investing dashboard
import pandas as pd 
import numpy as np 
import scipy.stats as sci
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
from Efficient_frontier import get_data, portfolio_performance
from VaR_Parametric import VaR_Parametric, CVaR_Parametric, Create_VaR_Plot

st.set_page_config(page_title='Personal investment dashboard',layout='wide')
st.title('Personal investment dashboard')
st.markdown('---')
InitialInvestment = st.number_input('Enter the ($) initial investment in your portfolio:')
Tickers_input = st.text_input('Enter tickers for your portfolio:', 'QQQ, SPY, TLT')
Weights_input = st.text_input('Please enter the respective weightings:','0.4, 0.4, 0.2')    
Start_date = st.date_input('Start date',value =dt.datetime.now() - dt.timedelta(1), max_value=dt.datetime.now() - dt.timedelta(1))
end_date = dt.datetime.now()
if(st.button('Submit')):
    try:
        Tickers = [ticker.strip().upper() for ticker in Tickers_input.split(",")]
        Weights = [float(weight) for weight in Weights_input.split(',')]
        Weights = np.array(Weights)
        if not np.isclose(np.sum(Weights),1):
            st.error('Weights must sum to 1')
        elif InitialInvestment<=0:
            st.error('Incorrect Initial Investment')
        elif not len(Tickers) == len(Weights):
            st.error('Numer of Tickers and Weights must match.')       
        else:
            st.success('Inputs accepted.')
            st.write('Tickers:',Tickers)
            st.write('Weights:', Weights)
            with st.spinner('Fetching data...'):
                r_mean, cov_matrix = get_data(Tickers,Start_date,end_date)
                st.subheader("Mean Daily Returns")
                st.dataframe(r_mean)

                st.subheader("Covariance Matrix")
                st.dataframe(cov_matrix)

    except Exception as e:
        st.exception(e)


         
    
    
    
    