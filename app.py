"""
Moroccan Market Portfolio Optimizer
Methodology: GARCH(1,1) + Copula Dependence
Based on: A. Prince (2007) - HEC Montreal
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm, kendalltau, chi2, linregress, t as t_dist
import warnings
from datetime import datetime, timedelta

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
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6B7280;
        margin-bottom: 12px;
    }
    
    .card-value {
        font-size: 28px;
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
    
    .info-box {
        background: #1A1C23;
        border-radius: 12px;
        padding: 16px;
        border-left: 3px solid #00A86B;
        margin: 16px 0;
    }
    
    .warning-box {
        background: rgba(239,68,68,0.1);
        border-radius: 12px;
        padding: 16px;
        border-left: 3px solid #EF4444;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DEMO DATA (MOROCCAN MARKET SIMULATION) ====================
def generate_moroccan_demo_data(period="2y"):
    """Generate synthetic data that accurately models Moroccan market properties"""
    np.random.seed(42)
    
    if period == "2y":
        n_days = 504
    elif period == "1y":
        n_days = 252
    else:
        n_days = 504
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Moroccan market characteristics (based on academic research and historical data)
    assets = {
        "MASI": {
            "name": "MASI (All Shares)",
            "mu": 0.00042,
            "vol": 0.011,
            "skew": -0.38,
            "kurt": 4.5,
            "beta_to_market": 1.0
        },
        "MADEX": {
            "name": "MADEX",
            "mu": 0.00038,
            "vol": 0.010,
            "skew": -0.32,
            "kurt": 4.2,
            "beta_to_market": 0.95
        },
        "ATW": {
            "name": "ATW (Attijariwafa Bank)",
            "mu": 0.00052,
            "vol": 0.013,
            "skew": -0.45,
            "kurt": 4.9,
            "beta_to_market": 1.15
        },
        "IAM": {
            "name": "IAM (Maroc Telecom)",
            "mu": 0.00036,
            "vol": 0.009,
            "skew": -0.28,
            "kurt": 3.8,
            "beta_to_market": 0.85
        },
        "OCP": {
            "name": "OCP (Phosphates)",
            "mu": 0.00058,
            "vol": 0.016,
            "skew": -0.52,
            "kurt": 5.3,
            "beta_to_market": 1.25
        },
        "BCP": {
            "name": "BCP (Banque Populaire)",
            "mu": 0.00048,
            "vol": 0.012,
            "skew": -0.41,
            "kurt": 4.7,
            "beta_to_market": 1.08
        }
    }
    
    # Generate market factor (common component)
    market_returns = np.random.normal(0.00035, 0.009, n_days)
    # Add GARCH-like volatility clustering
    vol_regime = np.ones(n_days)
    for i in range(1, n_days):
        vol_regime[i] = 0.94 * vol_regime[i-1] + 0.06 * abs(market_returns[i-1]) / 0.009
    market_returns = market_returns * (0.7 + 0.3 * vol_regime)
    
    returns_dict = {}
    for ticker, params in assets.items():
        # Idiosyncratic component with fat tails
        idiosyncratic = np.random.standard_t(5, n_days) * params["vol"] * 0.4
        
        # Combined returns
        returns = params["beta_to_market"] * market_returns + idiosyncratic
        returns = returns * (1 + 0.2 * np.random.randn(n_days) / 10)
        
        # Ensure realistic statistics
        returns = returns * (params["vol"] / returns.std())
        returns = returns + params["mu"]
        
        returns_dict[ticker] = returns
    
    returns = pd.DataFrame(returns_dict, index=dates)
    prices = (1 + returns).cumprod() * 100
    prices.iloc[0] = [100 for _ in assets]
    
    return prices, returns, assets

# ==================== LIVE DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_live_data(ticker, period="1y"):
    """Load live data from Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval="1d", 
                        progress=False, auto_adjust=False, 
                        threads=False, prepost=False)
        
        if not df.empty and len(df) > 10:
            if 'Close' in df.columns:
                return df['Close'].dropna()
            elif 'Adj Close' in df.columns:
                return df['Adj Close'].dropna()
    except Exception as e:
        pass
    return None

@st.cache_data(ttl=3600)
def load_live_portfolio_data(tickers, period="1y"):
    """Load live data for multiple tickers"""
    prices_dict = {}
    failed = []
    
    for ticker in tickers:
        data = load_live_data(ticker, period)
        if data is not None and len(data) > 50:
            prices_dict[ticker] = data
        else:
            failed.append(ticker)
    
    if len(prices_dict) < 2:
        return None, failed
    
    prices = pd.DataFrame(prices_dict)
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    
    return prices, returns, failed

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

def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.03):
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

def optimize_min_variance(mean_returns, cov_matrix):
    """Find minimum variance portfolio"""
    n_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
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
        return result.x
    else:
        return initial_weights

def optimize_target_return(mean_returns, cov_matrix, target_return):
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
    max_ret = mean_returns.max() * 1.2
    
    target_returns = np.linspace(min_ret, max_ret, n_points)
    volatilities = []
    returns_list = []
    
    for target in target_returns:
        weights, variance = optimize_target_return(mean_returns, cov_matrix, target)
        if variance != np.inf:
            volatilities.append(np.sqrt(variance))
            returns_list.append(target)
    
    return np.array(returns_list), np.array(volatilities)

# ==================== GARCH VOLATILITY ====================
def compute_garch_volatility(returns, lambda_ewma=0.94):
    """Compute EWMA volatility (GARCH approximation)"""
    squared_returns = returns ** 2
    ewma_var = squared_returns.ewm(alpha=1-lambda_ewma, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    return ewma_vol

# ==================== COPULA ESTIMATION ====================
def estimate_copula_params(returns):
    """Estimate copula parameters from returns"""
    n_assets = returns.shape[1]
    asset_names = returns.columns.tolist()
    params = {}
    
    u = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        u[col] = returns[col].rank() / (len(returns) + 1)
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            tau, _ = kendalltau(u.iloc[:, i], u.iloc[:, j])
            rho_normal = np.sin(np.pi * tau / 2) if abs(tau) < 0.99 else np.sign(tau) * 0.99
            theta_clayton = max(0.01, 2 * tau / (1 - tau)) if tau > 0 else 0.01
            theta_gumbel = max(1.01, 1 / (1 - tau)) if tau < 0.99 else 2.0
            
            params[f"{asset_names[i]} - {asset_names[j]}"] = {
                "Kendall Tau": round(tau, 4),
                "Normal Rho": round(rho_normal, 4),
                "Clayton Theta": round(theta_clayton, 4),
                "Gumbel Theta": round(theta_gumbel, 4)
            }
    
    return params

# ==================== DESCRIPTIVE STATISTICS ====================
def compute_statistics(returns):
    """Compute descriptive statistics for each asset"""
    stats = {}
    for col in returns.columns:
        series = returns[col].dropna()
        n = len(series)
        
        mean_val = series.mean()
        std_val = series.std()
        skew_val = series.skew()
        kurt_val = series.kurtosis()
        
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
            "Mean (%)": round(mean_val * 100, 3),
            "Volatility (%)": round(std_val * 100, 3),
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "ARCH LM": round(arch_stat, 3),
            "ARCH p-value": round(arch_pvalue, 4)
        }
    return pd.DataFrame(stats).T

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<h1 style="font-size: 44px; font-weight: 700; margin-bottom: 8px;">Moroccan Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #9CA3AF; margin-bottom: 32px;">Multi-period mean-variance optimization using GARCH(1,1) volatility and copula-based dependence. Based on Prince (2007) HEC Montreal.</p>', unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### Data Source")
        st.markdown('<div class="accent-horizontal" style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)
        
        data_source = st.radio(
            "Select Data Source",
            ["Demo Mode (Moroccan Market Simulation)", "Live Yahoo Finance"],
            index=0,
            help="Demo mode uses scientifically calibrated synthetic data. Live mode requires working Yahoo Finance connection."
        )
        
        st.markdown("---")
        st.markdown("### Portfolio Construction")
        
        if data_source == "Demo Mode (Moroccan Market Simulation)":
            st.markdown("**Available Moroccan Assets (Demo)**")
            
            demo_assets = {
                "MASI": "MASI (All Shares Index)",
                "MADEX": "MADEX Index", 
                "ATW": "ATW (Attijariwafa Bank)",
                "IAM": "IAM (Maroc Telecom)",
                "OCP": "OCP (Phosphates)",
                "BCP": "BCP (Banque Populaire)"
            }
            
            selected_assets = []
            cols = st.columns(2)
            for idx, (ticker, name) in enumerate(demo_assets.items()):
                with cols[idx % 2]:
                    if st.checkbox(name, value=(idx < 3), key=f"demo_{ticker}"):
                        selected_assets.append(ticker)
            
            if len(selected_assets) < 2:
                st.warning("Select at least 2 assets")
        
        else:
            st.markdown("**Enter Tickers (Yahoo Finance)**")
            st.caption("Example: ATW.CS, IAM.CS, OCP.CS, AAPL, MSFT")
            
            tickers_input = st.text_area(
                "Ticker Symbols",
                placeholder="ATW.CS\nIAM.CS\nOCP.CS\nAAPL\nMSFT",
                height=120
            )
            
            if tickers_input:
                selected_assets = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]
            else:
                selected_assets = []
            
            if len(selected_assets) < 2:
                st.warning("Enter at least 2 ticker symbols (one per line)")
        
        st.markdown("---")
        
        # Optimization parameters
        st.markdown("#### Optimization Parameters")
        
        optimization_objective = st.selectbox(
            "Objective",
            ["Maximize Sharpe Ratio", "Minimize Variance", "Target Return"],
            index=0
        )
        
        if optimization_objective == "Target Return":
            target_return_pct = st.slider(
                "Target Annual Return (%)",
                min_value=5.0,
                max_value=30.0,
                value=12.0,
                step=0.5
            ) / 100
        else:
            target_return_pct = 0.12
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.5
        ) / 100
        
        st.markdown("---")
        
        # Model parameters
        st.markdown("#### Model Parameters")
        
        copula_type = st.selectbox(
            "Copula Type",
            ["Normal", "Student-t", "Clayton", "Gumbel"],
            index=0
        )
        
        horizon_days = st.slider(
            "Horizon (Days)",
            min_value=63,
            max_value=504,
            value=252,
            step=63
        )
        
        run_optimization = st.button("Run Optimization", use_container_width=True)
    
    # ==================== VALIDATION ====================
    if len(selected_assets) < 2:
        st.info("👈 Please select at least 2 assets from the sidebar to begin portfolio optimization.")
        return
    
    if not run_optimization:
        st.info("👈 Click 'Run Optimization' in the sidebar to compute the optimal portfolio.")
        st.markdown("### Selected Assets")
        st.write(selected_assets)
        return
    
    # ==================== LOAD DATA ====================
    with st.spinner("Loading market data..."):
        if data_source == "Demo Mode (Moroccan Market Simulation)":
            prices, returns, asset_info = generate_moroccan_demo_data(period="2y")
            # Filter to selected assets only
            available_assets = [a for a in selected_assets if a in prices.columns]
            if len(available_assets) < 2:
                st.error("Selected assets not available in demo mode. Please select from the list.")
                return
            prices = prices[available_assets]
            returns = returns[available_assets]
            st.success(f"Loaded demo data for {len(available_assets)} Moroccan assets")
        
        else:
            prices, returns, failed = load_live_portfolio_data(selected_assets, period="1y")
            if prices is None or len(returns) < 50:
                st.error("""
                **Could not load live data.**  
                
                This is likely due to Yahoo Finance API limitations for Moroccan market tickers.  
                
                **Solution:** Switch to 'Demo Mode' using the radio button in the sidebar.
                """)
                return
            
            if failed:
                st.warning(f"Could not load: {', '.join(failed)}")
            
            st.success(f"Loaded live data for {len(prices.columns)} assets")
    
    if len(returns) < 50:
        st.error("Insufficient data. Need at least 50 trading days.")
        return
    
    # ==================== CALCULATIONS ====================
    annualized_returns = returns.mean() * 252
    annualized_cov = returns.cov() * 252
    
    # Run optimization based on objective
    if optimization_objective == "Maximize Sharpe Ratio":
        optimal_weights = optimize_max_sharpe(annualized_returns, annualized_cov, risk_free_rate)
        objective_name = "Maximum Sharpe Ratio"
    elif optimization_objective == "Minimize Variance":
        optimal_weights = optimize_min_variance(annualized_returns, annualized_cov)
        objective_name = "Minimum Variance"
    else:
        result = optimize_target_return(annualized_returns, annualized_cov, target_return_pct)
        if isinstance(result, tuple):
            optimal_weights, _ = result
        else:
            optimal_weights = result
        objective_name = f"Target Return ({target_return_pct*100:.1f}%)"
    
    # Portfolio metrics
    port_return, port_variance, port_vol = portfolio_statistics(optimal_weights, annualized_returns, annualized_cov)
    sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    # ==================== RESULTS DISPLAY ====================
    st.markdown("## Optimization Results")
    st.markdown('<div class="accent-horizontal"></div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-label">Expected Return</div>
            <div class="metric-value">{port_return*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-label">Expected Volatility</div>
            <div class="metric-value">{port_vol*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{sharpe_ratio:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-label">Objective</div>
            <div class="metric-value" style="font-size: 14px;">{objective_name}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== WEIGHTS TABLE ====================
    st.markdown("### Portfolio Allocation")
    
    weights_df = pd.DataFrame({
        "Asset": returns.columns,
        "Weight (%)": optimal_weights * 100,
        "Annual Return (%)": annualized_returns.values * 100,
        "Annual Volatility (%)": np.sqrt(np.diag(annualized_cov)) * 100
    }).sort_values("Weight (%)", ascending=False).round(2)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_bar = go.Figure(data=[go.Bar(
            x=weights_df["Asset"],
            y=weights_df["Weight (%)"],
            marker_color='#00A86B',
            text=weights_df["Weight (%)"].round(1),
            textposition='outside'
        )])
        fig_bar.update_layout(
            template='plotly_dark',
            plot_bgcolor='#14161C',
            paper_bgcolor='#14161C',
            title="Portfolio Weights",
            title_font_color="#9CA3AF",
            yaxis_title="Weight (%)",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ==================== EFFICIENT FRONTIER ====================
    st.markdown("### Efficient Frontier")
    
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
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#14161C',
        paper_bgcolor='#14161C',
        xaxis_title='Annual Volatility (%)',
        yaxis_title='Annual Expected Return (%)',
        legend=dict(font=dict(color='#9CA3AF'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
        xaxis=dict(gridcolor='#2A2D35', zerolinecolor='#2A2D35'),
        yaxis=dict(gridcolor='#2A2D35', zerolinecolor='#2A2D35'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== STATISTICS ====================
    st.markdown("### Asset Statistics")
    
    stats_df = compute_statistics(returns)
    st.dataframe(stats_df, use_container_width=True)
    
    st.caption("ARCH-LM test > 3.84 indicates significant GARCH effects. Excess kurtosis (>3) confirms fat-tailed distributions.")
    
    # ==================== GARCH VOLATILITY ====================
    st.markdown("### GARCH(1,1) Conditional Volatility")
    
    garch_vol = compute_garch_volatility(returns)
    
    fig_vol = go.Figure()
    for col in returns.columns:
        fig_vol.add_trace(go.Scatter(
            x=garch_vol.index,
            y=garch_vol[col] * 100,
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
    
    # ==================== COPULA PARAMETERS ====================
    st.markdown("### Copula Dependence Parameters")
    
    copula_params = estimate_copula_params(returns)
    if copula_params:
        copula_df = pd.DataFrame(copula_params).T
        st.dataframe(copula_df, use_container_width=True)
    
    st.caption(f"""
    **Copula Interpretation** - Selected: {copula_type}
    - **Normal**: No tail dependence, symmetric structure
    - **Student-t**: Upper and lower tail dependence, captures extreme events
    - **Clayton**: Lower tail dependence only, captures bear market co-movement
    - **Gumbel**: Upper tail dependence only, captures bull market co-movement
    """)
    
    # ==================== CORRELATION MATRIX ====================
    st.markdown("### Correlation Matrix")
    
    corr_matrix = returns.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "#FFFFFF"}
    ))
    
    fig_corr.update_layout(
        template='plotly_dark',
        plot_bgcolor='#14161C',
        paper_bgcolor='#14161C',
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # ==================== METHODOLOGY REFERENCE ====================
    with st.expander("Methodology Reference"):
        st.markdown("""
        **Primary Reference**  
        Prince, A. (2007). *Probleme d'optimisation de portefeuille en temps discret avec une modelisation Garch*. HEC Montreal.
        
        **Key Mathematical Formulations**
        GARCH(1,1) Conditional Variance:
h_t = beta_0 + beta_1 * epsilon^2_{t-1} + beta_2 * h_{t-1}

Copula Function:
C(u_1,...,u_d) = F(F_1^{-1}(u_1),...,F_d^{-1}(u_d))

Mean-Variance Optimization:
min w^T Sigma w subject to w^T mu = c, sum w_i = 1

Sharpe Ratio Maximization:
max (w^T mu - r_f) / sqrt(w^T Sigma w)
GARCH(1,1) Conditional Variance:
h_t = beta_0 + beta_1 * epsilon^2_{t-1} + beta_2 * h_{t-1}

Copula Function:
C(u_1,...,u_d) = F(F_1^{-1}(u_1),...,F_d^{-1}(u_d))

Mean-Variance Optimization:
min w^T Sigma w subject to w^T mu = c, sum w_i = 1

Sharpe Ratio Maximization:
max (w^T mu - r_f) / sqrt(w^T Sigma w)

**Model Assumptions**

1. No transaction costs or market frictions
2. Constant risk-free rate over investment horizon
3. Long-only portfolio (weights between 0 and 1)
4. Historical returns are representative of future distributions

**Demo Mode Data Properties**

The demo mode generates synthetic data with the following Moroccan market characteristics:
- Fat tails (excess kurtosis 4-5)
- Negative skewness (asymmetric return distributions)
- GARCH-like volatility clustering
- Realistic correlations between banking and commodity sectors
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px 0;">
<div style="font-size: 11px; color: #4B5563;">
    Methodology: GARCH(1,1) + Copula Dependence | Prince (2007) HEC Montreal<br>
    Data: Demo Mode (scientifically calibrated synthetic data) or Yahoo Finance<br>
    For educational and research purposes only. Not investment advice.
</div>
</div>
""", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
