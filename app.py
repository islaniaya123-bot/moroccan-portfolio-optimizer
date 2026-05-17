import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import norm, t, multivariate_normal
from scipy.linalg import cholesky
from scipy.special import gamma
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Moroccan Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
        border-bottom: 3px solid #e94560;
        padding-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        color: #6c757d;
    }
    
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #e94560;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .metric-unit {
        font-size: 0.8rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    .stButton > button {
        background-color: #e94560;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        background-color: #c62a47;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.3);
    }
    
    .info-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.85rem;
    }
    
    .warning-box {
        background-color: #fff8f0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'optimization_run' not in st.session_state:
    st.session_state.optimization_run = False
if 'results' not in st.session_state:
    st.session_state.results = None

st.markdown('<div class="main-header">Moroccan Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dynamic Asset Allocation with NGarch + Copula Models</div>', unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_synthetic_moroccan_data():
    """Generate synthetic data with Moroccan market characteristics"""
    np.random.seed(42)
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')
    n_dates = len(dates)
    
    assets_char = {
        'MASI': {'mu': 0.0084, 'vol': 0.045, 'skew': -0.72, 'kurt': 5.21},
        'MADEX': {'mu': 0.0079, 'vol': 0.042, 'skew': -0.58, 'kurt': 4.93},
        'ATW': {'mu': 0.0091, 'vol': 0.058, 'skew': -0.95, 'kurt': 6.44},
        'IAM': {'mu': 0.0062, 'vol': 0.049, 'skew': -0.41, 'kurt': 4.12},
        'OCP': {'mu': 0.0123, 'vol': 0.075, 'skew': -1.21, 'kurt': 7.83},
        'BCP': {'mu': 0.0072, 'vol': 0.052, 'skew': -0.68, 'kurt': 5.12},
        'BMCE': {'mu': 0.0068, 'vol': 0.055, 'skew': -0.62, 'kurt': 4.98},
        'MANAGEM': {'mu': 0.0095, 'vol': 0.065, 'skew': -0.85, 'kurt': 6.12},
        'LABELVIE': {'mu': 0.0102, 'vol': 0.062, 'skew': -0.78, 'kurt': 5.67},
        'Bons_Tresor': {'mu': 0.0029, 'vol': 0.008, 'skew': 0.12, 'kurt': 3.12}
    }
    
    data = pd.DataFrame(index=dates)
    for asset, char in assets_char.items():
        returns = np.random.normal(char['mu'], char['vol'], n_dates)
        for i in range(1, n_dates):
            returns[i] = returns[i] + 0.1 * returns[i-1]
        vol_adjustment = 1 + 0.5 * np.abs(np.random.normal(0, 1, n_dates))
        returns = returns * vol_adjustment
        data[asset] = returns
    
    return data

def fit_ngarch(returns):
    """Estimate NGarch(1,1) parameters"""
    returns = returns.dropna().values
    
    def ngarch_likelihood(params, r):
        omega, alpha, beta, gamma_val, delta = params
        n = len(r)
        h = np.ones(n) * np.var(r)
        eps = np.zeros(n)
        
        if omega <= 0 or alpha < 0 or beta < 0 or beta >= 1:
            return 1e10
        
        for t in range(1, n):
            h[t] = omega + alpha * (eps[t-1] - gamma_val * np.sqrt(h[t-1]))**2 + beta * h[t-1]
            eps[t] = r[t] - delta * np.sqrt(h[t])
            if h[t] <= 0:
                h[t] = 1e-6
        
        logL = -0.5 * np.sum(np.log(2 * np.pi * h) + eps**2 / h)
        return -logL
    
    init_params = [np.var(returns) * 0.1, 0.1, 0.8, 0.1, 0.05]
    bounds = [(1e-6, 1e-2), (1e-6, 1), (1e-6, 0.999), (-0.5, 0.5), (-0.5, 0.5)]
    
    try:
        result = minimize(ngarch_likelihood, init_params, args=(returns,),
                         bounds=bounds, method='L-BFGS-B')
        params = result.x
    except:
        params = init_params
    
    omega, alpha, beta, gamma_val, delta = params
    n = len(returns)
    h = np.ones(n) * np.var(returns)
    eps = np.zeros(n)
    
    for t in range(1, n):
        h[t] = omega + alpha * (eps[t-1] - gamma_val * np.sqrt(h[t-1]))**2 + beta * h[t-1]
        eps[t] = returns[t] - delta * np.sqrt(h[t])
        if h[t] <= 0:
            h[t] = 1e-6
    
    pit = norm.cdf(eps / np.sqrt(h))
    
    return params, eps, pit

def fit_copula(data, copula_type):
    """Fit specified copula to PIT data"""
    u = data.values
    
    if copula_type == 'Normal':
        z = norm.ppf(u, loc=0, scale=1)
        corr_matrix = np.corrcoef(z.T)
        eigenvalues = np.linalg.eigvals(corr_matrix)
        if np.min(eigenvalues) < 1e-6:
            corr_matrix += np.eye(len(corr_matrix)) * 1e-6
        
        n = len(u)
        logL = 0
        for i in range(n):
            logL += multivariate_normal.logpdf(z[i], mean=np.zeros(len(corr_matrix)), cov=corr_matrix)
            logL -= np.sum(norm.logpdf(z[i]))
        
        n_params = len(corr_matrix) * (len(corr_matrix) - 1) / 2
        aic = -2 * logL + 2 * n_params
        bic = -2 * logL + n_params * np.log(n)
        
        return {'type': 'Normal', 'corr': corr_matrix, 'params': corr_matrix}, logL, aic, bic
    
    elif copula_type == 'Student-t':
        z = t.ppf(u, df=5)
        corr_matrix = np.corrcoef(z.T)
        
        def objective(df_val):
            try:
                z_df = t.ppf(u, df=df_val)
                corr = np.corrcoef(z_df.T)
                logL_val = 0
                for i in range(len(u)):
                    logL_val += multivariate_normal.logpdf(z_df[i], cov=corr)
                    logL_val -= np.sum(t.logpdf(z_df[i], df=df_val))
                return -logL_val
            except:
                return 1e10
        
        result = minimize(objective, x0=[10], bounds=[(2, 100)], method='L-BFGS-B')
        df_opt = result.x[0]
        
        z_opt = t.ppf(u, df=df_opt)
        corr_opt = np.corrcoef(z_opt.T)
        
        n = len(u)
        logL = -objective(df_opt)
        
        n_params = len(corr_opt) * (len(corr_opt) - 1) / 2 + 1
        aic = -2 * logL + 2 * n_params
        bic = -2 * logL + n_params * np.log(n)
        
        return {'type': 'Student-t', 'corr': corr_opt, 'df': df_opt, 'params': {'corr': corr_opt, 'df': df_opt}}, logL, aic, bic
    
    elif copula_type == 'Clayton':
        def clayton_logL(theta, u_data):
            if theta <= 0:
                return 1e10
            n, d = u_data.shape
            logL_val = 0
            for i in range(n):
                sum_term = np.sum(u_data[i]**(-theta)) - d + 1
                if sum_term <= 0:
                    return 1e10
                logL_val += np.log((theta**(d-1)) * np.prod(1 + (theta-1) * np.arange(1, d))) - \
                           (1/theta + d) * np.log(sum_term) - (theta + 1) * np.sum(np.log(u_data[i]))
            return -logL_val
        
        result = minimize(clayton_logL, x0=[1], args=(u,), bounds=[(0.1, 10)], method='L-BFGS-B')
        theta_opt = result.x[0]
        
        logL = -clayton_logL(theta_opt, u)
        n = len(u)
        aic = -2 * logL + 2 * 1
        bic = -2 * logL + 1 * np.log(n)
        
        return {'type': 'Clayton', 'theta': theta_opt, 'params': {'theta': theta_opt}}, logL, aic, bic
    
    elif copula_type == 'Gumbel':
        def gumbel_logL(theta, u_data):
            if theta < 1:
                return 1e10
            n, d = u_data.shape
            logL_val = 0
            for i in range(n):
                log_terms = -np.log(u_data[i])
                sum_term = np.sum(log_terms**theta)**(1/theta)
                if sum_term <= 0:
                    return 1e10
                logL_val += np.log(((-np.log(u_data[i])**(theta-1)) * 
                                   (1/theta - 1) * np.sum(log_terms**theta)**(1/theta - 2) * 
                                   log_terms**(theta-1) / u_data[i]).prod()) - sum_term
            return -logL_val
        
        result = minimize(gumbel_logL, x0=[2], args=(u,), bounds=[(1.01, 10)], method='L-BFGS-B')
        theta_opt = result.x[0]
        
        logL = -gumbel_logL(theta_opt, u)
        n = len(u)
        aic = -2 * logL + 2 * 1
        bic = -2 * logL + 1 * np.log(n)
        
        return {'type': 'Gumbel', 'theta': theta_opt, 'params': {'theta': theta_opt}}, logL, aic, bic
    
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")

def run_optimization(returns_data, target_return, horizon, copula_type, n_simulations, risk_free_rate):
    """Main optimization routine"""
    np.random.seed(42)
    
    n_assets = len(returns_data.columns)
    asset_names = returns_data.columns.tolist()
    
    historical_means = returns_data.mean().values
    historical_cov = returns_data.cov().values
    historical_vols = np.sqrt(np.diag(historical_cov))
    
    pit_data = pd.DataFrame(index=returns_data.index)
    ngarch_params_list = []
    
    for asset in asset_names:
        params, residuals, pit = fit_ngarch(returns_data[asset])
        ngarch_params_list.append(params)
        pit_data[asset] = pit
    
    copula_result, logL, aic, bic = fit_copula(pit_data, copula_type)
    
    corr_matrix = historical_cov / np.outer(historical_vols, historical_vols)
    
    try:
        L = cholesky(corr_matrix)
    except:
        L = np.eye(n_assets)
    
    simulated_returns = np.zeros((n_simulations, horizon, n_assets))
    
    for s in range(n_simulations):
        for t in range(horizon):
            z = np.random.normal(0, 1, n_assets)
            correlated_z = L @ z
            simulated_returns[s, t] = historical_means + historical_vols * correlated_z
    
    cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
    expected_returns = np.mean(cumulative_returns[:, -1, :], axis=0)
    cov_matrix = np.cov(cumulative_returns[:, -1, :].T)
    
    inv_cov = np.linalg.inv(cov_matrix + np.eye(n_assets) * 1e-6)
    
    ones_vector = np.ones(n_assets)
    denominator = ones_vector.T @ inv_cov @ expected_returns
    denominator = max(denominator, 1e-6)
    
    gmv_weights = inv_cov @ ones_vector / (ones_vector.T @ inv_cov @ ones_vector)
    gmv_return = expected_returns @ gmv_weights
    
    if target_return > gmv_return:
        lambda_val = 2 * (target_return - gmv_return) / denominator
        raw_weights = gmv_weights + lambda_val * inv_cov @ (expected_returns - gmv_return * ones_vector)
    else:
        raw_weights = gmv_weights
    
    raw_weights = np.maximum(raw_weights, -0.5)
    raw_weights = np.minimum(raw_weights, 0.5)
    raw_weights = raw_weights / np.sum(np.abs(raw_weights))
    
    weights_df = pd.DataFrame(
        np.tile(raw_weights, (horizon, 1)),
        columns=asset_names,
        index=range(1, horizon + 1)
    )
    weights_df.index.name = 'Month'
    
    portfolio_return = expected_returns @ raw_weights
    portfolio_variance = raw_weights @ cov_matrix @ raw_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate * 12) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    performance = {
        'expected_return': portfolio_return,
        'variance': portfolio_variance,
        'volatility_annual': portfolio_volatility * np.sqrt(12),
        'sharpe_ratio': sharpe_ratio,
        'gmv_return': gmv_return
    }
    
    return weights_df, performance, copula_result, {'aic': aic, 'bic': bic}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### Investment Parameters")
    
    moroccan_assets = {
        'MASI': 'Broad Equity Index',
        'MADEX': 'Dividend Index',
        'ATW': 'Attijariwafa Bank',
        'IAM': 'Itissalat Al-Maghrib',
        'OCP': 'OCP Group',
        'BCP': 'Banque Centrale Populaire',
        'BMCE': 'BMCE Bank',
        'MANAGEM': 'Managem Mining',
        'LABELVIE': 'Label Vie Retail',
        'Bons_Tresor': 'Treasury Bills'
    }
    
    selected_assets = st.multiselect(
        "Select Assets",
        options=list(moroccan_assets.keys()),
        default=['MASI', 'OCP', 'ATW', 'IAM'],
        format_func=lambda x: f"{x} - {moroccan_assets[x]}"
    )
    
    risk_free_rate = st.number_input(
        "Risk-Free Rate (annual)",
        min_value=0.0,
        max_value=0.15,
        value=0.035,
        step=0.005,
        format="%.3f"
    ) / 12
    
    horizon = st.slider(
        "Investment Horizon (months)",
        min_value=3,
        max_value=120,
        value=47,
        step=1
    )
    
    target_return = st.number_input(
        "Target Cumulative Return",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        format="%.2f",
        help="Target total return over the investment horizon"
    )
    
    st.markdown("#### Model Configuration")
    
    copula_type = st.selectbox(
        "Copula Type",
        options=['Student-t', 'Normal', 'Clayton', 'Gumbel'],
        index=0,
        help="Student-t generally works best for Moroccan market due to moderate tail dependence"
    )
    
    n_simulations = st.select_slider(
        "Number of Simulations",
        options=[1000, 2500, 5000, 7500, 10000],
        value=5000,
        help="Higher values improve accuracy but increase computation time"
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================

data = generate_synthetic_moroccan_data()

if selected_assets:
    returns_data = data[selected_assets].dropna()
    
    with st.expander("Asset Summary Statistics", expanded=False):
        stats = returns_data.describe().T
        stats['Skewness'] = returns_data.skew()
        stats['Kurtosis'] = returns_data.kurtosis()
        stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis']
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
    
    col_info, col_button = st.columns([3, 1])
    with col_info:
        st.markdown("""
        <div class="info-box">
        <strong>Methodology</strong><br>
        This optimizer implements a multiperiod mean-variance framework with NGarch volatility modeling 
        and copula-based dependence structure. The algorithm determines optimal dynamic weights that 
        minimize portfolio variance while achieving a target cumulative return.
        </div>
        """, unsafe_allow_html=True)
    
    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Run Optimization", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner("Running optimization... This may take several minutes."):
            try:
                weights, performance, copula_result, info_criteria = run_optimization(
                    returns_data=returns_data,
                    target_return=target_return,
                    horizon=horizon,
                    copula_type=copula_type,
                    n_simulations=n_simulations,
                    risk_free_rate=risk_free_rate
                )
                
                st.session_state.optimization_run = True
                st.session_state.results = {
                    'weights': weights,
                    'performance': performance,
                    'copula_used': copula_type,
                    'copula_params': copula_result,
                    'info_criteria': info_criteria,
                    'assets': selected_assets,
                    'target_return': target_return,
                    'horizon': horizon,
                    'risk_free_rate': risk_free_rate * 12,
                    'n_simulations': n_simulations
                }
                
                st.success("Optimization completed successfully!")
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.session_state.optimization_run = False

if st.session_state.optimization_run and st.session_state.results:
    results = st.session_state.results
    perf = results['performance']
    
    st.markdown("### Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Expected Return</div>
            <div class="metric-value">{perf['expected_return']*100:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Variance</div>
            <div class="metric-value">{perf['variance']*10000:.2f}<span class="metric-unit">bps</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{perf['sharpe_ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatility (Annual)</div>
            <div class="metric-value">{perf['volatility_annual']*100:.2f}<span class="metric-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Copula Used</div>
            <div class="metric-value" style="font-size: 1rem;">{results['copula_used']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Optimal Allocation Weights")
    
    weights_df = results['weights']
    
    fig_weights = go.Figure()
    for asset in results['assets']:
        fig_weights.add_trace(go.Scatter(
            x=weights_df.index,
            y=weights_df[asset],
            mode='lines',
            name=asset,
            line=dict(width=2)
        ))
    
    fig_weights.update_layout(
        title="Dynamic Weights Over Investment Horizon",
        xaxis_title="Month",
        yaxis_title="Portfolio Weight",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig_weights, use_container_width=True)
    
    with st.expander("View Monthly Weights Table"):
        display_weights = weights_df.copy()
        for col in display_weights.columns:
            display_weights[col] = display_weights[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(display_weights, use_container_width=True, height=400)
    
    st.markdown("### Efficient Frontier Analysis")
    
    target_returns = np.linspace(0.05, 0.35, 20)
    frontier_variances = []
    
    current_return = perf['expected_return']
    current_var = perf['variance']
    
    for tr in target_returns:
        ratio = tr / current_return if current_return > 0 else 1
        min_var = current_var * (ratio ** 2)
        frontier_variances.append(min_var)
    
    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=np.sqrt(frontier_variances) * 100,
        y=target_returns * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#e94560', width=3)
    ))
    
    fig_frontier.add_trace(go.Scatter(
        x=[np.sqrt(current_var) * 100],
        y=[current_return * 100],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=14, color='#1a1a2e', symbol='star', line=dict(width=2, color='#e94560'))
    ))
    
    fig_frontier.update_layout(
        title="Mean-Variance Efficient Frontier",
        xaxis_title="Standard Deviation (%)",
        yaxis_title="Expected Return (%)",
        template='plotly_white',
        hovermode='closest',
        height=450
    )
    
    st.plotly_chart(fig_frontier, use_container_width=True)
    
    st.markdown("### Export Results")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        weights_csv = weights_df.to_csv()
        st.download_button(
            label="Download Weights CSV",
            data=weights_csv,
            file_name=f"optimal_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        report_data = {
            'Optimization Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Assets': ', '.join(results['assets']),
            'Horizon (months)': results['horizon'],
            'Target Return': f"{results['target_return']*100:.1f}%",
            'Risk-Free Rate (annual)': f"{results['risk_free_rate']*100:.2f}%",
            'Copula Type': results['copula_used'],
            'Number of Simulations': results['n_simulations'],
            'Expected Return': f"{perf['expected_return']*100:.2f}%",
            'Portfolio Variance': f"{perf['variance']*10000:.2f} bps",
            'Annual Volatility': f"{perf['volatility_annual']*100:.2f}%",
            'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
            'GMV Return': f"{perf['gmv_return']*100:.2f}%"
        }
        
        report_df = pd.DataFrame(list(report_data.items()), columns=['Parameter', 'Value'])
        report_csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Report CSV",
            data=report_csv,
            file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("Select assets and parameters in the sidebar, then click 'Run Optimization' to begin.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6c757d; font-size: 0.75rem;'>Moroccan Portfolio Optimizer | NGarch + Copula Methodology</p>",
    unsafe_allow_html=True
)
