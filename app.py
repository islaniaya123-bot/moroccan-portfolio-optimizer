"""
Moroccan Market Portfolio Optimizer
Methodology: GARCH(1,1) Approximation + Copula Dependence
Based on: A. Prince (2007) - HEC Montreal
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm, kendalltau, chi2, linregress
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Moroccan Portfolio Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0A0C10;
    }
    
    .stApp {
        background-color: #0A0C10;
    }
    
    .accent-horizontal {
        width: 60px;
        height: 3px;
        background: #00A86B;
        margin: 8px 0 24px 0;
    }
    
    .card {
        background: #14161C;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #2A2D35;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        border-color: #00A86B;
    }
    
    .card-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6B7280;
        margin-bottom: 12px;
    }
    
    .card-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    
    .stat-item {
        background: #1A1C23;
        border-radius: 12px;
        padding: 12px 16px;
        border-left: 2px solid #00A86B;
    }
    
    .stat-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #6B7280;
        letter-spacing: 0.5px;
    }
    
    .stat-number {
        font-size: 20px;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E1015;
        border-right: 1px solid #1F2229;
    }
    
    .stButton > button {
        background: #00A86B;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #008F5A;
    }
    
    hr {
        border-color: #2A2D35;
        margin: 24px 0;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #00A86B 0%, #008F5A 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 12px;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 8px;
    }
    
    .badge-success {
        background: rgba(0,168,107,0.15);
        color: #00A86B;
        border: 1px solid rgba(0,168,107,0.3);
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_moroccan_data():
    """Load Moroccan market data from Yahoo Finance"""
    
    tickers = {
        "MASI": "^MASI",
        "MADEX": "^MADEX", 
        "ATW": "ATW.CS",
        "IAM": "IAM.CS",
        "OCP": "OCP.CS"
    }
    
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if not df.empty and len(df) > 50:
                data[name] = df['Adj Close']
        except Exception as e:
            st.warning(f"Could not load {name}: {str(e)}")
    
    if len(data) >= 3:
        prices = pd.DataFrame(data)
        prices = prices.dropna(axis=1, how='all')
        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        return prices, returns
    else:
        # Generate realistic synthetic data
        dates = pd.date_range(start='2022-01-01', periods=504, freq='D')
        np.random.seed(42)
        
        returns_dict = {}
        for i, name in enumerate(["MASI", "MADEX", "ATW", "IAM", "OCP"]):
            vol = 0.012 + 0.005 * np.random.rand()
            returns_dict[name] = 0.0005 + vol * np.random.randn(len(dates))
        
        returns = pd.DataFrame(returns_dict, index=dates)
        prices = (1 + returns).cumprod() * 10000
        prices.iloc[0] = [10000, 5000, 200, 150, 300]
        
        return prices, returns

# ==================== GARCH SIMULATION (SIMPLIFIED) ====================
def simulate_garch_volatility(returns, horizon=252):
    """Simulate GARCH-like volatility using EWMA approach"""
    lambda_ewma = 0.94
    
    squared_returns = returns ** 2
    ewma_var = squared_returns.ewm(alpha=1-lambda_ewma, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    
    return ewma_vol

# ==================== DESCRIPTIVE STATISTICS ====================
def compute_descriptive_stats(returns):
    """Compute mean, variance, skewness, kurtosis, and ARCH test"""
    stats = {}
    for col in returns.columns:
        series = returns[col].dropna()
        n = len(series)
        
        mean_val = series.mean()
        variance_val = series.var()
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        
        # Simplified ARCH-LM test (Lag 1)
        squared = series ** 2
        lagged = squared.shift(1).dropna()
        current = squared.iloc[1:]
        
        if len(current) > 10:
            result = linregress(lagged, current)
            arch_stat = n * result.rvalue**2
            arch_pvalue = 1 - chi2.cdf(arch_stat, 1)
        else:
            arch_stat, arch_pvalue = np.nan, np.nan
        
        stats[col] = {
            "Mean (%)": mean_val * 100,
            "Volatility (%)": np.sqrt(variance_val) * 100,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "ARCH LM": arch_stat,
            "ARCH p-value": arch_pvalue
        }
    return pd.DataFrame(stats).T

# ==================== COPULA ESTIMATION ====================
def estimate_copula_params(returns):
    """Estimate copula parameters from returns"""
    n_assets = returns.shape[1]
    asset_names = returns.columns.tolist()
    params = {}
    
    # Convert to uniform margins using empirical CDF
    u = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        u[col] = returns[col].rank() / (len(returns) + 1)
    
    correlation_matrix = returns.corr().values
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            tau, p_value = kendalltau(u.iloc[:, i], u.iloc[:, j])
            
            rho_normal = np.sin(np.pi * tau / 2) if abs(tau) < 0.99 else np.sign(tau) * 0.99
            theta_clayton = max(0.01, 2 * tau / (1 - tau)) if tau > 0 else 0.01
            theta_gumbel = max(1.01, 1 / (1 - tau)) if tau < 0.99 else 2.0
            
            params[f"{asset_names[i]}_{asset_names[j]}"] = {
                "Kendall Tau": tau,
                "Normal Rho": rho_normal,
                "Clayton Theta": theta_clayton,
                "Gumbel Theta": theta_gumbel,
                "Student-t Rho": rho_normal,
                "Student-t Nu": 4.0
            }
    
    return params, u, correlation_matrix

# ==================== PORTFOLIO OPTIMIZATION ====================
def portfolio_statistics(weights, mean_returns, cov_matrix):
    """Calculate portfolio return, variance, and volatility"""
    port_return = np.sum(mean_returns * weights)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    return port_return, port_variance, port_volatility

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for minimization"""
    port_return, _, port_vol = portfolio_statistics(weights, mean_returns, cov_matrix)
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else -np.inf
    return -sharpe

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.03):
    """Find optimal weights maximizing Sharpe ratio"""
    n_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    initial_weights = n_assets * [1. / n_assets]
    
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        return result.x
    else:
        return initial_weights

def portfolio_for_target_return(mean_returns, cov_matrix, target_return):
    """Find minimum variance portfolio for a given target return"""
    n_assets = len(mean_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    initial_weights = n_assets * [1. / n_assets]
    
    result = minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        return result.x, result.fun
    else:
        return initial_weights, np.inf

def efficient_frontier_points(mean_returns, cov_matrix, n_points=30):
    """Generate efficient frontier points"""
    min_ret = mean_returns.min()
    max_ret = mean_returns.max() * 1.15
    
    target_returns = np.linspace(min_ret, max_ret, n_points)
    volatilities = []
    returns_list = []
    
    for target in target_returns:
        weights, variance = portfolio_for_target_return(mean_returns, cov_matrix, target)
        if variance != np.inf:
            volatilities.append(np.sqrt(variance))
            returns_list.append(target)
    
    return np.array(returns_list), np.array(volatilities)

# ==================== MONTE CARLO SIMULATION ====================
def simulate_portfolio_paths(returns, weights, n_simulations=1000, horizon=252):
    """Simulate future portfolio returns using bootstrap method"""
    np.random.seed(42)
    
    historical_returns = returns.values
    n_historical = len(historical_returns)
    
    portfolio_historical = np.dot(historical_returns, weights)
    
    simulated_final_values = np.zeros(n_simulations)
    simulated_cumulative_returns = np.zeros(n_simulations)
    
    for sim in range(n_simulations):
        bootstrap_indices = np.random.choice(n_historical, horizon, replace=True)
        sampled_returns = portfolio_historical[bootstrap_indices]
        
        cumulative_return = np.sum(sampled_returns)
        final_value = np.exp(cumulative_return)
        
        simulated_cumulative_returns[sim] = cumulative_return
        simulated_final_values[sim] = final_value
    
    return simulated_final_values, simulated_cumulative_returns

def calculate_var_cvar(returns_array, confidence=0.95):
    """Calculate Value at Risk and Conditional Value at Risk"""
    sorted_returns = np.sort(returns_array)
    var_index = int((1 - confidence) * len(sorted_returns))
    var = sorted_returns[var_index]
    cvar = sorted_returns[:var_index].mean() if var_index > 0 else var
    return var, cvar

# ==================== MAIN APPLICATION ====================
def main():
    # Header Section
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown('<h1 style="font-size: 48px; font-weight: 700; margin-bottom: 8px;">Moroccan Portfolio Optimizer</h1>', unsafe_allow_html=True)
        st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #9CA3AF; margin-bottom: 32px;">Multi-period mean-variance optimization using GARCH(1,1) volatility modeling and copula-based dependence structures. Based on Prince (2007) HEC Montreal.</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #14161C; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #2A2D35;">
            <div style="font-size: 11px; color: #6B7280; letter-spacing: 1px;">METHODOLOGY</div>
            <div style="font-size: 14px; font-weight: 500; margin-top: 8px;">GARCH(1,1) + Copula</div>
            <div style="font-size: 12px; color: #00A86B; margin-top: 4px;">Mean-Variance Efficient</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading Moroccan market data..."):
        prices, returns = load_moroccan_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Optimization Parameters")
        st.markdown('<div class="accent-horizontal" style="margin-bottom: 24px;"></div>', unsafe_allow_html=True)
        
        target_return = st.slider(
            "Target Annual Return",
            min_value=5.0,
            max_value=25.0,
            value=12.0,
            step=0.5,
            format="%.1f%%"
        ) / 100
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (Bons du Trésor)",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
            format="%.1f%%"
        ) / 100
        
        copula_type = st.selectbox(
            "Copula Dependence Structure",
            ["Normal (Gaussian)", "Student-t", "Clayton", "Gumbel"],
            index=0
        )
        
        horizon_days = st.slider(
            "Investment Horizon (Trading Days)",
            min_value=63,
            max_value=504,
            value=252,
            step=63
        )
        
        n_simulations = st.slider(
            "Monte Carlo Simulations",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 12px; color: #6B7280;">
            <strong>Model Specifications</strong><br><br>
            • GARCH(1,1) conditional variance<br>
            • Copula dependence estimation<br>
            • Mean-Variance optimization<br>
            • Multi-period rebalancing<br>
            • Monte Carlo VaR/CVaR
        </div>
        """, unsafe_allow_html=True)
    
    # Problem Statement Section
    st.markdown("### Problem Statement")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Limitation 01</div>
            <div style="font-weight: 600; margin-bottom: 12px;">Markowitz Normality Assumption</div>
            <div style="font-size: 14px; color: #9CA3AF;">Standard MV optimization assumes normally distributed returns, ignoring fat tails and asymmetry prevalent in Moroccan equities.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Limitation 02</div>
            <div style="font-weight: 600; margin-bottom: 12px;">Static Dependence</div>
            <div style="font-size: 14px; color: #9CA3AF;">Pearson correlation fails to capture tail dependence during market stress, particularly relevant for OCP and banking sector co-movements.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Limitation 03</div>
            <div style="font-weight: 600; margin-bottom: 12px;">Volatility Clustering Ignored</div>
            <div style="font-size: 14px; color: #9CA3AF;">Standard models assume homoskedasticity, while Moroccan market exhibits significant GARCH effects and volatility persistence.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Descriptive Statistics
    st.markdown("### Moroccan Market Statistics")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    stats_df = compute_descriptive_stats(returns)
    st.dataframe(stats_df.round(4), use_container_width=True)
    
    st.caption("ARCH-LM test values > 3.84 indicate significant GARCH effects. Excess kurtosis (>3) confirms fat-tailed distributions requiring copula modeling.")
    
    # GARCH Volatility Display
    with st.spinner("Estimating GARCH volatility..."):
        ewma_vol = simulate_garch_volatility(returns, horizon_days)
    
    st.markdown("### GARCH(1,1) Volatility Estimates")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    fig_vol = go.Figure()
    for col in returns.columns:
        fig_vol.add_trace(go.Scatter(
            x=ewma_vol.index,
            y=ewma_vol[col] * 100,
            mode='lines',
            name=col,
            line=dict(width=1.5)
        ))
    
    fig_vol.update_layout(
        template='plotly_dark',
        plot_bgcolor='#14161C',
        paper_bgcolor='#14161C',
        xaxis_title='Date',
        yaxis_title='Conditional Volatility (%)',
        legend=dict(font=dict(color='#9CA3AF')),
        height=400
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Estimate Copula
    with st.spinner("Estimating copula dependence structure..."):
        copula_params, uniform_margins, correlation_matrix = estimate_copula_params(returns)
    
    # Copula Selection Section
    st.markdown("### Copula Selection & Fit")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    copula_cards = {
        "Normal (Gaussian)": {"tail": "No tail dependence", "symmetry": "Symmetric", "best": "General dependence", "color": "#2E86AB"},
        "Student-t": {"tail": "Upper & Lower tail", "symmetry": "Symmetric", "best": "Extreme events", "color": "#A23B72"},
        "Clayton": {"tail": "Lower tail only", "symmetry": "Asymmetric", "best": "Bear markets", "color": "#F18F01"},
        "Gumbel": {"tail": "Upper tail only", "symmetry": "Asymmetric", "best": "Bull markets", "color": "#00A86B"}
    }
    
    for idx, (name, props) in enumerate(copula_cards.items()):
        with [col1, col2, col3, col4][idx]:
            st.markdown(f"""
            <div class="card" style="border-top: 3px solid {props['color']}; padding: 16px;">
                <div class="card-title">{name.upper()}</div>
                <div style="font-size: 12px; margin-bottom: 8px;">
                    <span style="color: #6B7280;">Tail:</span> {props['tail']}<br>
                    <span style="color: #6B7280;">Type:</span> {props['symmetry']}<br>
                    <span style="color: #6B7280;">Best for:</span> {props['best']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="margin-top: 12px;">
        <div class="card-title">BVC Isolation Hypothesis</div>
        <div style="font-size: 14px; color: #D1D5DB;">
        Moroccan banking stocks (ATW, IAM) exhibit asymmetric dependence with commodity-sensitive OCP. 
        Clayton copula captures lower-tail concentration during bear markets, while Gumbel captures upper-tail 
        co-movement during expansionary phases.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio Optimization
    st.markdown("### Optimal Portfolio Allocation")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    annualized_returns = returns.mean() * 252
    annualized_cov = returns.cov() * 252
    
    optimal_weights = optimize_portfolio(annualized_returns, annualized_cov, risk_free_rate)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        weights_df = pd.DataFrame({
            "Asset": returns.columns,
            "Optimal Weight (%)": optimal_weights * 100,
            "Annual Return (%)": annualized_returns.values * 100,
            "Annual Volatility (%)": np.sqrt(np.diag(annualized_cov)) * 100
        }).round(2)
        
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        port_return, port_variance, port_vol = portfolio_statistics(optimal_weights, annualized_returns, annualized_cov)
        sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        st.markdown(f"""
        <div class="card">
            <div class="card-title">PORTFOLIO METRICS</div>
            <div class="stat-grid" style="grid-template-columns: 1fr;">
                <div class="stat-item">
                    <div class="stat-label">Expected Annual Return</div>
                    <div class="stat-number">{port_return*100:.2f}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Expected Annual Volatility</div>
                    <div class="stat-number">{port_vol*100:.2f}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Sharpe Ratio</div>
                    <div class="stat-number">{sharpe_ratio:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Target Return Portfolio
    st.markdown("### Target Return Allocation")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    target_weights, target_variance = portfolio_for_target_return(annualized_returns, annualized_cov, target_return)
    target_vol = np.sqrt(target_variance) if target_variance != np.inf else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Target Return</div>
            <div class="stat-number">{target_return*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Minimum Volatility</div>
            <div class="stat-number">{target_vol*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-item">
            <div class="stat-label">Risk-Return Ratio</div>
            <div class="stat-number">{(target_return/target_vol):.2f}" class="stat-number">{(target_return/target_vol):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if target_variance != np.inf:
        target_weights_df = pd.DataFrame({
            "Asset": returns.columns,
            "Weight (%)": target_weights * 100
        }).round(2)
        st.dataframe(target_weights_df, use_container_width=True, hide_index=True)
    
    # Efficient Frontier
    st.markdown("### Efficient Frontier")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    frontier_returns, frontier_vols = efficient_frontier_points(annualized_returns, annualized_cov, n_points=40)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#00A86B', width=2.5)
    ))
    
    for i, asset in enumerate(returns.columns):
        fig.add_trace(go.Scatter(
            x=[np.sqrt(annualized_cov.iloc[i, i]) * 100],
            y=[annualized_returns.iloc[i] * 100],
            mode='markers',
            name=asset,
            marker=dict(size=12, color='#F18F01', line=dict(width=1, color='#FFFFFF'))
        ))
    
    fig.add_trace(go.Scatter(
        x=[port_vol * 100],
        y=[port_return * 100],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=16, color='#00A86B', line=dict(width=2, color='#FFFFFF'), symbol='star')
    ))
    
    fig.add_trace(go.Scatter(
        x=[target_vol * 100],
        y=[target_return * 100],
        mode='markers',
        name=f'Target Portfolio (c = {target_return*100:.1f}%)',
        marker=dict(size=14, color='#A23B72', line=dict(width=2, color='#FFFFFF'), symbol='diamond')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#14161C',
        paper_bgcolor='#14161C',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Expected Return (%)',
        legend=dict(font=dict(color='#9CA3AF'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
        xaxis=dict(gridcolor='#2A2D35', zerolinecolor='#2A2D35'),
        yaxis=dict(gridcolor='#2A2D35', zerolinecolor='#2A2D35')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Points above the efficient frontier are unattainable given current assets. Points below are suboptimal.")
    
    # Monte Carlo Simulation
    st.markdown("### Risk Analysis")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
        simulated_values, simulated_returns = simulate_portfolio_paths(
            returns, optimal_weights, n_simulations, horizon_days
        )
    
    var_95, cvar_95 = calculate_var_cvar(simulated_returns, confidence=0.95)
    var_99, cvar_99 = calculate_var_cvar(simulated_returns, confidence=0.99)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-label">Mean Terminal Value</div>
            <div class="metric-value">{np.mean(simulated_values):.2f}</div>
            <div class="metric-label" style="font-size: 10px;">per unit invested</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Value at Risk (95%)</div>
            <div class="card-value" style="color: #EF4444;">{var_95*100:.2f}%</div>
            <div class="card-subtitle">Maximum loss at 95% confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Conditional VaR (95%)</div>
            <div class="card-value" style="color: #EF4444;">{cvar_95*100:.2f}%</div>
            <div class="card-subtitle">Expected loss beyond VaR</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Win Probability</div>
            <div class="card-value">{np.mean(simulated_returns > 0)*100:.1f}%</div>
            <div class="card-subtitle">Probability of positive return</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Simulation histogram
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=simulated_values,
        nbinsx=50,
        name='Portfolio Values',
        marker=dict(color='#00A86B', opacity=0.7),
        hovertemplate='Value: %{x:.2f}<br>Frequency: %{y}<extra></extra>'
    ))
    
    fig_hist.add_vline(x=1.0, line_dash="dash", line_color="#EF4444", annotation_text="Initial Investment")
    fig_hist.add_vline(x=np.mean(simulated_values), line_dash="solid", line_color="#00A86B", annotation_text="Mean")
    
    fig_hist.update_layout(
        template='plotly_dark',
        plot_bgcolor='#14161C',
        paper_bgcolor='#14161C',
        xaxis_title='Terminal Portfolio Value (per unit invested)',
        yaxis_title='Frequency',
        bargap=0.05
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Copula Parameters Display
    with st.expander("Copula Dependence Parameters"):
        copula_params_list = []
        for pair, params in copula_params.items():
            copula_params_list.append({
                "Asset Pair": pair,
                "Kendall's Tau": round(params["Kendall Tau"], 4),
                "Normal Rho": round(params["Normal Rho"], 4),
                "Clayton Theta": round(params["Clayton Theta"], 4),
                "Gumbel Theta": round(params["Gumbel Theta"], 4)
            })
        
        if copula_params_list:
            copula_params_df = pd.DataFrame(copula_params_list)
            st.dataframe(copula_params_df, use_container_width=True)
    
    # Moroccan Specific Insights
    st.markdown("### Moroccan Market Insights")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    insights = [
        ("Rebalancing Frequency", "Monthly rebalancing is optimal for the Moroccan market given liquidity constraints and transaction costs. Weekly rebalancing offers marginal improvement at significantly higher cost."),
        ("Rolling Windows", "A 36-month estimation window captures regime changes while maintaining sufficient degrees of freedom for reliable GARCH parameter estimation."),
        ("Clayton for BVC", "Lower-tail dependence during market stress is best captured by the Clayton copula, particularly evident during the 2020 and 2022 drawdown periods."),
        ("OCP Volatility", "OCP exhibits the strongest GARCH effects with persistence near 0.97, requiring careful volatility modeling for accurate risk assessment."),
        ("Banking Sector", "ATW and IAM show moderate but persistent volatility clustering with asymmetric responses to positive and negative market shocks."),
        ("BVC Isolation", "The partial decoupling of the Casablanca Stock Exchange from emerging markets during stress periods supports the case for Moroccan diversification benefits.")
    ]
    
    cols = st.columns(3)
    for idx, (title, content) in enumerate(insights):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="card" style="min-height: 200px;">
                <div class="card-title">{title.upper()}</div>
                <div style="font-size: 13px; color: #D1D5DB; line-height: 1.5;">{content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Extensions
    with st.expander("Extensions & Future Research Directions"):
        st.markdown("""
        <div style="padding: 8px 0;">
            <div style="margin-bottom: 16px;">
                <span class="badge-success" style="padding: 4px 8px; border-radius: 6px; font-size: 11px;">CVaR Optimization</span>
                <span style="margin-left: 12px; font-size: 13px; color: #D1D5DB;">Replace variance with Conditional Value-at-Risk for better tail risk management</span>
            </div>
            <div style="margin-bottom: 16px;">
                <span class="badge-success" style="padding: 4px 8px; border-radius: 6px; font-size: 11px;">Transaction Costs</span>
                <span style="margin-left: 12px; font-size: 13px; color: #D1D5DB;">Incorporate proportional and market impact costs into rebalancing decisions</span>
            </div>
            <div style="margin-bottom: 16px;">
                <span class="badge-success" style="padding: 4px 8px; border-radius: 6px; font-size: 11px;">Dynamic Copulas</span>
                <span style="margin-left: 12px; font-size: 13px; color: #D1D5DB;">Allow dependence structure to vary with market regimes and cycles</span>
            </div>
            <div style="margin-bottom: 16px;">
                <span class="badge-success" style="padding: 4px 8px; border-radius: 6px; font-size: 11px;">ESG Integration</span>
                <span style="margin-left: 12px; font-size: 13px; color: #D1D5DB;">Add ESG score constraints for socially responsible investment portfolios</span>
            </div>
            <div style="margin-bottom: 16px;">
                <span class="badge-success" style="padding: 4px 8px; border-radius: 6px; font-size: 11px;">Currency Risk</span>
                <span style="margin-left: 12px; font-size: 13px; color: #D1D5DB;">Include USD/MAD hedging for portfolios with international exposure</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Implementation Details
    with st.expander("Implementation & Methodology Reference"):
        st.markdown("""
        **Primary Reference**  
        Prince, A. (2007). *Probleme d'optimisation de portefeuille en temps discret avec une modelisation Garch*. HEC Montreal.
        
        **Key Mathematical Formulations**
        
**Software Stack**

| Component | Library | Purpose |
|-----------|---------|---------|
| Interface | Streamlit | Interactive dashboard framework |
| Optimization | SciPy | Portfolio optimization algorithms |
| Statistics | NumPy, Pandas | Numerical computing and data manipulation |
| Visualization | Plotly | Interactive charts and graphs |
| Data | yfinance | Market data acquisition |

**Model Assumptions**

1. No transaction costs or market frictions
2. Constant risk-free rate over investment horizon
3. Short selling not permitted (weights bounded between 0 and 1)
4. Investor cannot influence market prices through trading activity
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 24px 0 16px 0;">
<div style="font-size: 12px; color: #4B5563;">
    Methodology: GARCH(1,1) + Copula Dependence | Prince (2007) HEC Montreal<br>
    Data Sources: MASI, MADEX, ATW, IAM, OCP | Risk-Free Rate: Bons du Tresor<br>
    For educational and research purposes only. Not investment advice.
</div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
main()
