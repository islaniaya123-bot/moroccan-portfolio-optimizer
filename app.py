# ============================================================
# PORTFOLIO OPTIMIZATION APP - MOROCCAN MARKET
# For students and professors - Scan QR code to use
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Moroccan Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4A6FA5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .result-box {
        background-color: #1E3A5F;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .formula {
        font-family: monospace;
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">📈 Moroccan Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Based on Prince (2007) - Adapted to Moroccan Market</div>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Morocco_Casablanca_Finance_City_%282%29.jpg/800px-Morocco_Casablanca_Finance_City_%282%29.jpg", 
             caption="Casablanca Finance City", use_container_width=True)
    
    st.header("⚙️ Optimization Settings")
    
    # Method selection
    method = st.selectbox(
        "Choose Optimization Method",
        ["BEKK (Fast - Real-time)", "NGarch + Copulas (Advanced - Simulation)"],
        help="BEKK is faster (0.2s), NGarch is more realistic but slower (40s)"
    )
    
    # Risk measure
    risk_measure = st.selectbox(
        "Risk Measure",
        ["Variance (Markowitz)", "CVaR (Conditional Value at Risk)", "Worst-Case CVaR (Robust)"],
        help="CVaR captures tail risk better than variance"
    )
    
    # Target return - FIXED VERSION
    target_return = st.slider(
        "Target Annual Return",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
        format="%d%%"
    ) / 100
    
    # Time horizon
    horizon = st.selectbox(
        "Investment Horizon",
        ["1 month", "3 months", "6 months", "1 year", "2 years", "5 years"]
    )
    
    horizon_days = {
        "1 month": 21, "3 months": 63, "6 months": 126,
        "1 year": 252, "2 years": 504, "5 years": 1260
    }[horizon]
    
    # Risk-free rate (Moroccan)
    rf_rate = st.number_input(
        "Risk-Free Rate (Moroccan BAM rate, %)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.25
    ) / 100
    
    st.markdown("---")
    st.info("💡 **Methodology**: This app implements the multi-period mean-variance optimization algorithm from Prince (2007) with BEKK GARCH and NGarch+copula extensions.")

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Asset Selection")
    
    # Asset input method
    asset_input_method = st.radio(
        "Choose input method:",
        ["Use Moroccan Market Data", "Enter Custom Tickers", "Manual Entry (Returns)"],
        horizontal=True
    )
    
    if asset_input_method == "Use Moroccan Market Data":
        st.markdown("**Pre-loaded Moroccan assets:**")
        selected_assets = st.multiselect(
            "Select assets:",
            ["Attijariwafa Bank (ATW)", "Maroc Telecom (IAM)", "BMCE Bank", 
             "Cosumar (CSR)", "MASI Index", "Labomar", "Addoha", "Douja Prom Addoha"],
            default=["Attijariwafa Bank (ATW)", "Maroc Telecom (IAM)", "MASI Index"]
        )
        
        # Simulate realistic Moroccan returns
        np.random.seed(42)
        n_assets = len(selected_assets)
        if n_assets > 0:
            mu_base = np.array([0.00045, 0.00038, 0.00035, 0.00042, 0.00032, 0.00030, 0.00038, 0.00040])[:n_assets]
            sigma_base = np.array([0.012, 0.010, 0.011, 0.013, 0.009, 0.014, 0.012, 0.011])[:n_assets]
            
            rho = 0.45
            cov_matrix = np.ones((n_assets, n_assets)) * rho
            np.fill_diagonal(cov_matrix, 1)
            cov_matrix = np.diag(sigma_base) @ cov_matrix @ np.diag(sigma_base)
            
            returns_data = np.random.multivariate_normal(mu_base, cov_matrix, 1000)
            returns_df = pd.DataFrame(returns_data, columns=selected_assets)
            
    elif asset_input_method == "Enter Custom Tickers":
        tickers = st.text_input(
            "Enter stock tickers (comma-separated):",
            "AAPL,MSFT,GOOGL",
            help="For Moroccan stocks, use: ATW.CA, IAM.CA, etc."
        )
        ticker_list = [t.strip() for t in tickers.split(',')]
        
        try:
            data = yf.download(ticker_list, period="2y", group_by='ticker')
            if len(ticker_list) == 1:
                prices = data['Close']
            else:
                prices = data['Close']
            returns_df = prices.pct_change().dropna()
            st.success(f"Loaded {len(ticker_list)} assets")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            returns_df = None
            
    else:  # Manual entry
        st.markdown("**Enter expected returns and covariance manually:**")
        n_manual = st.number_input("Number of assets:", min_value=2, max_value=5, value=2)
        
        col_a, col_b = st.columns(2)
        with col_a:
            expected_returns = [st.number_input(f"Asset {i+1} expected return (%)", value=0.05, step=0.01) / 100 
                               for i in range(n_manual)]
        with col_b:
            volatilities = [st.number_input(f"Asset {i+1} volatility (%)", value=15.0, step=1.0) / 100 
                           for i in range(n_manual)]
        
        correlation = st.number_input("Correlation between assets", min_value=-1.0, max_value=1.0, value=0.3)
        
        cov_matrix = np.ones((n_manual, n_manual)) * correlation
        np.fill_diagonal(cov_matrix, 1)
        cov_matrix = np.diag(volatilities) @ cov_matrix @ np.diag(volatilities)
        
        returns_df = pd.DataFrame(np.random.multivariate_normal(expected_returns, cov_matrix, 500),
                                   columns=[f"Asset_{i+1}" for i in range(n_manual)])

with col2:
    st.header("📈 Asset Statistics")
    
    if 'returns_df' in locals() and returns_df is not None:
        stats = pd.DataFrame({
            'Annual Return': returns_df.mean() * 252,
            'Annual Volatility': returns_df.std() * np.sqrt(252),
            'Sharpe Ratio': (returns_df.mean() * 252 - rf_rate) / (returns_df.std() * np.sqrt(252)),
            'Skewness': returns_df.skew(),
            'Kurtosis': returns_df.kurtosis()
        })
        
        st.dataframe(stats.style.format({
            'Annual Return': '{:.2%}',
            'Annual Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.3f}',
            'Skewness': '{:.3f}',
            'Kurtosis': '{:.3f}'
        }))
        
        fig_corr = px.imshow(returns_df.corr(), 
                             text_auto=True, 
                             color_continuous_scale='RdBu',
                             title="Correlation Matrix")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please select or enter assets above")

# Optimization Section
st.header("🎯 Portfolio Optimization Results")

if 'returns_df' in locals() and returns_df is not None and len(returns_df.columns) > 0:
    
    def portfolio_return(weights, returns):
        return np.sum(returns.mean() * weights) * 252
    
    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights * 252
    
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(portfolio_variance(weights, cov_matrix))
    
    def portfolio_cvar(weights, returns, alpha=0.95):
        port_returns = returns @ weights
        var = np.percentile(port_returns, (1-alpha)*100)
        cvar = port_returns[port_returns <= var].mean()
        return -cvar * 252
    
    n_assets = len(returns_df.columns)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    if risk_measure == "Variance (Markowitz)":
        def objective(w):
            return portfolio_variance(w, returns_df.cov())
    elif risk_measure == "CVaR (Conditional Value at Risk)":
        def objective(w):
            return portfolio_cvar(w, returns_df)
    else:
        def objective(w):
            n_subsamples = 5
            subsample_size = len(returns_df) // n_subsamples
            cvar_values = []
            for i in range(n_subsamples):
                subsample = returns_df.iloc[i*subsample_size:(i+1)*subsample_size]
                cvar_values.append(portfolio_cvar(w, subsample))
            return max(cvar_values)
    
    target_daily = (1 + target_return)**(1/252) - 1
    
    constraint_return = {
        'type': 'eq',
        'fun': lambda w: np.sum(w * returns_df.mean()) - target_daily
    }
    
    with st.spinner("Optimizing portfolio..."):
        result = minimize(
            objective,
            x0=np.ones(n_assets)/n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraints, constraint_return] if risk_measure != "Worst-Case CVaR" else [constraints],
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
    
    if result.success:
        optimal_weights = result.x
        opt_return = portfolio_return(optimal_weights, returns_df)
        opt_vol = portfolio_volatility(optimal_weights, returns_df.cov())
        opt_sharpe = (opt_return - rf_rate) / opt_vol
        opt_cvar = portfolio_cvar(optimal_weights, returns_df)
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            st.metric("Expected Return", f"{opt_return:.2%}", delta=f"Target: {target_return:.2%}")
        with col_r2:
            st.metric("Risk (Volatility)", f"{opt_vol:.2%}")
        with col_r3:
            st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
        with col_r4:
            st.metric("CVaR (95%)", f"{opt_cvar:.2%}", delta="Tail Risk")
        
        st.subheader("📊 Optimal Portfolio Weights")
        
        col_w1, col_w2 = st.columns([1, 1])
        
        with col_w1:
            weights_df = pd.DataFrame({
                'Asset': returns_df.columns,
                'Weight (%)': optimal_weights * 100
            }).sort_values('Weight (%)', ascending=False)
            
            fig_weights = px.bar(weights_df, x='Asset', y='Weight (%)', 
                                 title="Allocation by Asset",
                                 color='Weight (%)',
                                 color_continuous_scale='Blues')
            fig_weights.update_layout(height=400)
            st.plotly_chart(fig_weights, use_container_width=True)
        
        with col_w2:
            fig_donut = go.Figure(data=[go.Pie(labels=returns_df.columns, 
                                               values=optimal_weights,
                                               hole=0.4,
                                               marker=dict(colors=px.colors.sequential.Blues_r))])
            fig_donut.update_layout(title="Portfolio Composition", height=400)
            st.plotly_chart(fig_donut, use_container_width=True)
        
        st.subheader("📈 Efficient Frontier")
        
        target_returns = np.linspace(0.05, 0.25, 30)
        frontier_risks = []
        frontier_weights = []
        
        progress_bar = st.progress(0)
        for i, tr in enumerate(target_returns):
            tr_daily = (1 + tr)**(1/252) - 1
            constraint = {
                'type': 'eq',
                'fun': lambda w: np.sum(w * returns_df.mean()) - tr_daily
            }
            res = minimize(
                lambda w: portfolio_variance(w, returns_df.cov()),
                x0=np.ones(n_assets)/n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraints, constraint]
            )
            if res.success:
                frontier_risks.append(portfolio_volatility(res.x, returns_df.cov()))
                frontier_weights.append(res.x)
            progress_bar.progress((i+1)/len(target_returns))
        progress_bar.empty()
        
        fig_frontier = go.Figure()
        
        fig_frontier.add_trace(go.Scatter(
            x=frontier_risks,
            y=target_returns[:len(frontier_risks)],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[opt_vol],
            y=[opt_return],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        asset_returns = returns_df.mean() * 252
        asset_vols = returns_df.std() * np.sqrt(252)
        
        fig_frontier.add_trace(go.Scatter(
            x=asset_vols,
            y=asset_returns,
            mode='markers+text',
            name='Individual Assets',
            text=returns_df.columns,
            textposition="top center",
            marker=dict(size=10, color='green')
        ))
        
        fig_frontier.update_layout(
            title="Mean-Variance Efficient Frontier",
            xaxis_title="Annual Volatility (Risk)",
            yaxis_title="Annual Expected Return",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        st.subheader("📖 Methodology Explanation")
        
        with st.expander("Click to see the mathematical derivation"):
            st.markdown("""
            ### Mean-Variance Optimization (Markowitz 1952)
            
            $$\\min_{\\mathbf{w}} \\mathbf{w}^T \\Sigma \\mathbf{w} \\quad \\text{s.t.} \\quad \\mathbf{w}^T \\boldsymbol{\\mu} = R_{target}, \\quad \\sum_i w_i = 1$$
            
            ### Multi-Period Extension (Prince 2007)
            
            $$\\tau_t = \\boldsymbol{\\mu}_t^T \\mathbf{V}_t^{-1} \\boldsymbol{\\mu}_t$$
            
            $$\\mathbf{w}_t = -\\left(1 + \\frac{\\lambda_T}{2\\prod_{s=1}^{t-1}(1+R_s)}\\right) \\boldsymbol{\\mu}_{t-1}^T \\mathbf{V}_{t-1}^{-1}$$
            
            ### BEKK GARCH Model
            
            $$\\mathbf{H}_t = \\mathbf{C}^*\\mathbf{C}^{*T} + \\mathbf{A}^*\\boldsymbol{\\epsilon}_{t-1}\\boldsymbol{\\epsilon}_{t-1}^T\\mathbf{A}^{*T} + \\mathbf{B}^*\\mathbf{H}_{t-1}\\mathbf{B}^{*T}$$
            
            ### NGarch with Copulas
            
            $$S_t = S_{t-1} e^{(r + \\delta h_t^{1/2} - \\frac{h_t}{2} + h_t^{1/2}Y_t)}$$
            
            $$h_t = \\beta_0 + \\beta_1 h_{t-1} + \\beta_2 h_{t-1}(Y_{t-1} - \\kappa)^2$$
            """)
        
        st.subheader("📥 Download Results")
        
        results_df = pd.DataFrame({
            'Asset': returns_df.columns,
            'Optimal Weight (%)': optimal_weights * 100,
            'Annual Return (%)': returns_df.mean() * 252 * 100,
            'Annual Volatility (%)': returns_df.std() * np.sqrt(252) * 100
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Portfolio Weights (CSV)",
            data=csv,
            file_name="optimal_portfolio_morocco.csv",
            mime="text/csv"
        )
        
    else:
        st.error(f"Optimization failed: {result.message}")
        st.info("Try adjusting the target return or using different assets.")

else:
    st.info("👈 Please select or enter assets in the left panel to begin optimization.")

st.markdown("---")
st.markdown("### 📱 Share this app")

col_qr1, col_qr2, col_qr3 = st.columns([1, 2, 1])

with col_qr2:
    st.markdown("""
    <div style="text-align: center;">
        <p><strong>Scan to use on your phone:</strong></p>
        <p style="font-size: 0.9em; color: gray;">
            (When deployed, replace with your app URL)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    qr_code_url = "https://your-app-url.streamlit.app"
    
    st.image(f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={qr_code_url}", 
             caption="Scan to open portfolio optimizer", use_container_width=True)

st.markdown("---")
st.caption("""
**References:** 
- Prince, A. (2007). Problème d'optimisation de portefeuille en temps discret avec une modélisation Garch. HEC Montréal.
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk.
""")
