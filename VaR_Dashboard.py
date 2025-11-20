import streamlit as st
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import datetime as dt
import plotly.graph_objects as go #type: ignore
import scipy.stats as sci  #type: ignore

# Import from your existing modules
from VaR_Hist import get_data, historical_var, historical_cvar
from VaR_MonteCarlo import var_mc, cvar_mc, portfolio_simulation, plot_risk as create_monte_carlo_plot
from VaR_Parametric import ( portfolio_performance as PortfolioPerformanceParam, 
    var_parametric, 
    cvar_parametric
)

# Page configuration
st.set_page_config(
    page_title="Portfolio VaR Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  VISUALIZATION FUNCTIONS 
def create_historical_plot(
        portfolio_returns: DataFrame, initial_investment: float, 
        var_val: float, cvar_val: float, alpha: float, time_horizon:int):
    """Create Historical VaR plot"""
    log_returns = np.log1p(portfolio_returns)  
    aggregated_log_returns = log_returns.rolling(window=time_horizon).sum().dropna() #type: ignore
    df = (np.expm1(aggregated_log_returns)) * initial_investment  #type: ignore
    x_vals = np.linspace(df.min(), df.max(), 10000) #type: ignore
    kde = sci.gaussian_kde(df)
    kde_vals = kde(x_vals) #type: ignore
    
    var_point = initial_investment * np.expm1(var_val)
    cvar_point = initial_investment * np.expm1(cvar_val) 
    
    x_tail = x_vals[x_vals <= var_point] #type: ignore
    kde_tail: Series = kde(x_tail) #type: ignore
    
    fig: go.Figure = go.Figure()
    
    fig.add_trace(go.Scatter(x=x_vals, y=kde_vals, mode='lines', #type: ignore
                             name='KDE Estimate', line=dict(color='blue')))
    
    fig.add_trace(go.Histogram(x=df, nbinsx=500, opacity=0.6,  #type: ignore
                               histnorm='probability density',
                               name='Historical Returns', marker_color="skyblue"))
    
    fig.add_trace(go.Scatter(x=x_tail, y=kde_tail, fill='tozeroy', #type: ignore
                             fillcolor='rgba(255, 0, 0, 0.4)',
                             line=dict(color='rgba(255,0,0,0)'),
                             name=f'Tail Area (Worst {alpha}%)'))
    
    fig.add_trace(go.Scatter(x=[var_point, var_point], y=[0, kde(var_point)[0]], #type: ignore
                             mode='lines', name=f'VaR ({100-alpha}%) = ${var_point:,.0f}',
                             line=dict(color='red', dash='dash')))
    
    fig.add_trace(go.Scatter(x=[cvar_point, cvar_point], y=[0, kde(cvar_point)[0]], #type: ignore
                             mode='lines', name=f'CVaR ({100-alpha}%) = ${cvar_point:,.0f}',
                             line=dict(color='purple', dash='dash')))
    
    fig.update_layout( #type: ignore
        title='Historical VaR & CVaR',
        xaxis_title='Profit/Loss ($)',
        yaxis_title='Probability Density',
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    return fig

def create_parametric_plot(
        mu: float, sigma: float, var_val: float, 
        cvar_val: float, alpha: float):
    """Create Parametric VaR plot"""
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = sci.norm.pdf(x, mu, sigma) #type: ignore
    x_tail = x[x <= var_val]
    pdf_tail = pdf[x <= var_val]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', #type: ignore
                             name='Normal Distribution', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=x_tail, y=pdf_tail, fill='tozeroy', #type: ignore
                             fillcolor='rgba(255, 0, 0, 0.4)',
                             line=dict(color='rgba(255,0,0,0)'),
                             name=f'Tail Area (Worst {alpha}%)'))
    
    fig.add_trace(go.Scatter(x=[var_val, var_val], y=[0, sci.norm.pdf(var_val, mu, sigma)], #type: ignore
                             mode='lines', name=f'VaR ({100-alpha}%) = ${var_val:,.0f}',
                             line=dict(color='red', dash='dash')))
    
    fig.add_trace(go.Scatter(x=[cvar_val, cvar_val], y=[0, sci.norm.pdf(cvar_val, mu, sigma)], #type: ignore
                             mode='lines', name=f'CVaR ({100-alpha}%) = ${cvar_val:,.0f}',
                             line=dict(color='purple', dash='dash')))
    
    fig.update_layout( #type: ignore
        title='Parametric VaR & CVaR (Normal Distribution)',
        xaxis_title='Profit/Loss ($)',
        yaxis_title='Probability Density',
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    return fig

#  MAIN APP 
st.title(" Portfolio Value at Risk (VaR) Dashboard")
st.markdown("### Comprehensive risk analysis using three methodologies")
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header(" Portfolio Configuration")
    
    st.subheader("Assets")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="TLT,SPY,QQQ,GLD,CL=F",
        help="Enter ticker symbols separated by commas"
    )
    
    weights_input = st.text_input(
        "Weights (comma-separated)",
        value="0.2,0.2,0.2,0.2,0.2",
        help="Weights must sum to 1.0"
    )
    
    st.subheader("Investment Parameters")
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        value=1000000,
        step=10000,
        format="%d"
    )
    
    time_horizon = st.number_input(
        "Time Horizon (days)",
        min_value=1,
        max_value=30,
        value=1,
        step=1
    )
    
    st.subheader("Risk Parameters")
    confidence = st.selectbox(
        "Confidence Level",
        options=[90, 95, 99],
        index=1,
        format_func=lambda x: f"{x}%"
    )
    
    lookback = st.number_input(
        "Historical Data Period (days)",
        min_value=100,
        max_value=5000,
        value=1250,
        step=50
    )
    
    st.markdown("---")
    calculate_button = st.button(" Calculate VaR", type="primary", width='stretch')

# Main content area
if calculate_button:
    try:
        # Parse inputs
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        weights = np.array([float(w.strip()) for w in weights_input.split(',')])
        alpha = 100 - confidence
        
        # Validate inputs
        if not np.isclose(np.sum(weights), 1.0, atol=0.001):
            st.error(" Weights must sum to 1.0")
            st.stop()
        
        if len(tickers) != len(weights):
            st.error(" Number of tickers must match number of weights")
            st.stop()
        
        # Display portfolio summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Investment", f"${initial_investment:,.0f}")
        with col2:
            st.metric("Time Horizon", f"{time_horizon} days")
        with col3:
            st.metric("Confidence Level", f"{confidence}%")
        with col4:
            st.metric("Assets", len(tickers))
        
        # Show portfolio composition
        with st.expander(" Portfolio Composition", expanded=True):
            portfolio_df = pd.DataFrame({
                'Ticker': tickers,
                'Weight': [f"{w:.1%}" for w in weights],
                'Allocation ($)': [f"${initial_investment * w:,.0f}" for w in weights]
            })
            st.dataframe(portfolio_df, width='stretch', hide_index=True) #type: ignore
        
        # Fetch data with progress indicator
        with st.spinner(' Fetching market data...'):
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=lookback)
            
            # Use get_data from VaR_Hist.py
            r, r_mean = get_data(tickers, start_date, end_date)
            
            # Get covariance matrix
            cov = r.cov() #type: ignore
        
        st.success(" Data fetched successfully!")
        
        # Calculate all three VaR methods
        with st.spinner(' Calculating VaR metrics...'):
            
            # ========== HISTORICAL VAR ==========
            # Using functions from VaR_Hist.py
            r['Portfolio'] = r.dot(weights) #type: ignore
            portfolio_returns = r['Portfolio']
            log_returns = np.log1p(portfolio_returns)  
            aggregated_log_returns = log_returns.rolling(window=time_horizon).sum().dropna() #type: ignore
            
            hist_var_log = historical_var(aggregated_log_returns, alpha)
            hist_cvar_log = historical_cvar(aggregated_log_returns, alpha)

            hist_var_dollar = initial_investment * np.expm1(hist_var_log)
            hist_cvar_dollar = initial_investment * np.expm1(hist_cvar_log)
            
            # ========== MONTE CARLO VAR ==========
            # Using Monte Carlo simulation
            mc_results = portfolio_simulation(r_mean, weights,cov, 1000, #type: ignore
                                                   initial_investment, time_horizon)
            
            
            # Using functions from VaR_MonteCarlo.py
            mc_var = var_mc(mc_results, alpha) - initial_investment
            mc_cvar = cvar_mc(mc_results, alpha) - initial_investment
            
            # ========== PARAMETRIC VAR ==========
            # Using functions from VaR_Parametric.py
            r_p, std_p = PortfolioPerformanceParam(r_mean, cov, weights, time_horizon) #type: ignore
            mu = r_p * initial_investment
            sigma = std_p * initial_investment
            
            param_var = var_parametric(mu, sigma, alpha)
            param_cvar = cvar_parametric(param_var, mu, sigma, alpha) #type: ignore
        
        # Display summary metrics
        st.markdown("---")
        st.subheader("VaR & CVaR Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Historical Method")
            st.metric("VaR", f"${-hist_var_dollar:,.0f}", 
                     delta=f"{hist_var_dollar/initial_investment*100:.2f}%", delta_color="inverse")
            st.metric("CVaR", f"${-hist_cvar_dollar:,.0f}",
                     delta=f"{hist_cvar_dollar/initial_investment*100:.2f}%", delta_color="inverse")
        
        with col2:
            st.markdown("#### Monte Carlo Method")
            st.metric("VaR", f"${-mc_var:,.0f}",
                     delta=f"{(mc_var/initial_investment)*100:.2f}%", delta_color="inverse")
            st.metric("CVaR", f"${-mc_cvar:,.0f}",
                     delta=f"{(mc_cvar/initial_investment)*100:.2f}%", delta_color="inverse")
        
        with col3:
            st.markdown("#### Parametric Method")
            st.metric("VaR", f"${-param_var:,.0f}",
                     delta=f"{(param_var/initial_investment)*100:.2f}%", delta_color="inverse")
            st.metric("CVaR", f"${-param_cvar:,.0f}",
                     delta=f"{(param_cvar/initial_investment)*100:.2f}%", delta_color="inverse")
        
        # Comparison table
        st.markdown("---")
        st.subheader("Detailed Comparison")
        
        comparison_df = pd.DataFrame({
            'Method': ['Historical', 'Monte Carlo', 'Parametric'],
            f'VaR ({confidence}%)': [
                f"${-hist_var_dollar:,.0f}",
                f"${-mc_var:,.0f}",
                f"${-param_var:,.0f}"
            ],
            f'CVaR ({confidence}%)': [
                f"${-hist_cvar_dollar:,.0f}",
                f"${-mc_cvar:,.0f}",
                f"${-param_cvar:,.0f}"
            ],
            'VaR (%)': [
                f"{hist_var_dollar/initial_investment*100:.2f}%",
                f"{(mc_var/initial_investment)*100:.2f}%",
                f"{(param_var/initial_investment)*100:.2f}%"
            ],
            'CVaR (%)': [
                f"{hist_cvar_dollar/initial_investment*100:.2f}%",
                f"{(mc_cvar/initial_investment)*100:.2f}%",
                f"{(param_cvar/initial_investment)*100:.2f}%"
            ]
        })
        
        st.dataframe(comparison_df, width='stretch', hide_index=True) #type: ignore
        
        # Generate plots
        st.markdown("---")
        st.subheader("Distribution Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Historical", "Monte Carlo", "Parametric"])
        
        with tab1:
            hist_fig = create_historical_plot(r['Portfolio'], initial_investment,  #type: ignore
                                             hist_var_log, hist_cvar_log, alpha, time_horizon)
            st.plotly_chart(hist_fig, width='stretch') #type: ignore
            st.info("Historical method uses actual past returns to estimate risk")
        
        with tab2:
            mc_fig = create_monte_carlo_plot(mc_results, alpha ,initial_investment, 
                                            ) #type: ignore
            st.plotly_chart(mc_fig, width='stretch') #type: ignore
            st.info("Monte Carlo simulates 10,000 possible future scenarios")
        
        with tab3:
            param_fig = create_parametric_plot(mu, sigma, param_var, param_cvar, alpha) #type: ignore
            st.plotly_chart(param_fig, width='stretch') #type: ignore
            st.info("Parametric method assumes returns follow a normal distribution")
        
        # Additional statistics
        with st.expander("Portfolio Statistics"):
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.markdown("**Expected Daily Returns**")
                st.dataframe(r_mean.to_frame('Daily Return %').style.format("{:.4%}"),  #type: ignore
                           width='stretch')
            
            with stats_col2:
                st.markdown("**Correlation Matrix**")
                st.dataframe(r.corr().style.format("{:.3f}").background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), #type: ignore
                           width='stretch')
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    st.info("Configure your portfolio in the sidebar and click 'Calculate VaR' to begin analysis")
    
    # Show example usage
    with st.expander("How to Use This Dashboard"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Enter Tickers**: Add your asset ticker symbols (comma-separated)
        2. **Set Weights**: Assign portfolio weights (must sum to 1.0)
        3. **Configure Parameters**: Set investment amount, time horizon (daily, weekly, monthly), and confidence level
        4. **Calculate**: Click the Calculate VaR button to run analysis
        
        ### Understanding the Results
        
        - **VaR (Value at Risk)**: Maximum expected loss at given confidence level over a certain time period
        - **CVaR (Conditional VaR)**: Average loss beyond VaR threshold
        - **Three Methods**: Compare different risk calculation approaches
        
        ### Methodology Comparison
        
        - **Historical**: Uses actual past returns (non-parametric)
        - **Monte Carlo**: Simulates future scenarios (10,000 paths)
        - **Parametric**: Assumes normal distribution (fastest)
        """)

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance. Past performance does not guarantee future results.*")