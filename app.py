import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Optimisation de Portefeuille — Marché Marocain",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #f1f5f9; }
  [data-testid="stSidebar"] { background: #111827; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1.5rem; }

  .hero-title {
    font-size: 2.2rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(135deg, #3b82f6, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .hero-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }

  .kpi-box {
    background: #1e2d47; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 18px 20px; text-align: center;
  }
  .kpi-label { font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8; }
  .kpi-value { font-size: 28px; font-weight: 700; color: #f1f5f9; }
  .kpi-sub { font-size: 12px; color: #10b981; }

  .section-tag {
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: #3b82f6; font-weight: 600; margin-bottom: 4px;
  }
  .section-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem; }

  .info-card {
    background: #1a2236; border: 1px solid rgba(59,130,246,0.2);
    border-left: 3px solid #3b82f6; border-radius: 10px; padding: 14px 18px;
    margin: 10px 0;
  }
  .warn-card {
    background: #1a1a10; border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid #f59e0b; border-radius: 10px; padding: 14px 18px;
    margin: 10px 0;
  }
  .success-card {
    background: #0f1f17; border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid #10b981; border-radius: 10px; padding: 14px 18px;
    margin: 10px 0;
  }
  div[data-testid="stMetric"] {
    background: #1e2d47; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 14px;
  }
  div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
  div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #3b82f6 !important; }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 10px 24px;
    transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.35); }
  .stSelectbox > div > div { background: #1e2d47; border-color: rgba(255,255,255,0.12); }
  .stMultiSelect > div > div { background: #1e2d47; border-color: rgba(255,255,255,0.12); }
  .stSlider { color: #3b82f6; }
  .stDataFrame { background: #111827; }
  .plot-title { font-size: 1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; }
  hr { border-color: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)

# ─── Data Layer ────────────────────────────────────────────────────────────────
import numpy.random as npr

MOROCCAN_ASSETS = {
    "ATW — Attijariwafa Bank":   {"mu": 0.000420, "sigma": 0.01180, "sector": "Finance"},
    "BCP — Banque Centrale Pop.":{"mu": 0.000310, "sigma": 0.01050, "sector": "Finance"},
    "IAM — Maroc Telecom":       {"mu": 0.000280, "sigma": 0.00980, "sector": "Telecom"},
    "OCP — OCP Group":           {"mu": 0.000390, "sigma": 0.01340, "sector": "Mines/Chimie"},
    "LHM — LafargeHolcim":       {"mu": 0.000210, "sigma": 0.01160, "sector": "Matériaux"},
    "CIH — CIH Bank":            {"mu": 0.000350, "sigma": 0.01220, "sector": "Finance"},
    "MNG — Managem":             {"mu": 0.000460, "sigma": 0.01580, "sector": "Mines"},
    "ADH — Addoha":              {"mu": 0.000180, "sigma": 0.01720, "sector": "Immobilier"},
    "BON — Bons du Trésor (rf)": {"mu": 0.000130, "sigma": 0.00080, "sector": "Obligataire"},
    "MASI — Indice MASI":        {"mu": 0.000290, "sigma": 0.00890, "sector": "Indice"},
}

COPULA_DESCRIPTIONS = {
    "Normale":  "Dépendance symétrique — sous-estime les queues épaisses. Corrélation linéaire.",
    "Student-t":"Queues épaisses jointes — idéal pour les co-chocs BVC. ν degrés de liberté.",
    "Clayton":  "Dépendance forte en queue inférieure — capture les crises / chutes simultanées.",
    "Gumbel":   "Dépendance forte en queue supérieure — capture les rallyes communs.",
    "Frank":    "Dépendance symétrique sans queues épaisses — alternative à la copule normale.",
}

BG_COLOR  = "#0a0e1a"
GRID_COLOR = "rgba(255,255,255,0.06)"
TEXT_COLOR = "#f1f5f9"
BLUE  = "#3b82f6"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED   = "#ef4444"
PURPLE= "#8b5cf6"
CORAL = "#f97316"

PALETTE = [BLUE, GREEN, AMBER, RED, PURPLE, CORAL, "#06b6d4", "#ec4899"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(color=TEXT_COLOR, family="Segoe UI, system-ui"),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)

# ─── Simulation helpers ────────────────────────────────────────────────────────

def simulate_garch(mu, omega, alpha, beta, kappa, n, ngarch=False, seed=42):
    rng = npr.default_rng(seed)
    h = np.zeros(n + 1)
    r = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        z = rng.standard_normal()
        r[t] = mu + np.sqrt(h[t]) * z
        if ngarch:
            h[t+1] = omega + beta * h[t] + alpha * h[t] * (z - kappa)**2
        else:
            h[t+1] = omega + alpha * (r[t] - mu)**2 + beta * h[t]
    return r, h[1:]

def simulate_copula(rho, n, copula_type, nu=5, theta_c=1.0, seed=42):
    rng = npr.default_rng(seed)
    if copula_type == "Normale":
        C = np.array([[1, rho],[rho, 1]])
        Z = rng.multivariate_normal([0,0], C, n)
        from scipy.stats import norm
        U = norm.cdf(Z)
    elif copula_type == "Student-t":
        from scipy.stats import t as tdist, multivariate_t
        nu_use = max(3, nu)
        C = np.array([[1, rho],[rho, 1]])
        Z = rng.multivariate_normal([0,0], C, n)
        chi2 = rng.chisquare(nu_use, n)
        T = Z / np.sqrt(chi2[:, None] / nu_use)
        U = tdist.cdf(T, df=nu_use)
    elif copula_type == "Clayton":
        th = max(0.01, theta_c)
        u1 = rng.uniform(0,1,n)
        v  = rng.uniform(0,1,n)
        u2 = (v**(-th/(1+th)) - 1 + u1**(-th))**(-1/th)
        u2 = np.clip(u2, 1e-6, 1-1e-6)
        U  = np.column_stack([u1, u2])
    elif copula_type == "Gumbel":
        from scipy.stats import norm
        th = max(1.01, theta_c)
        # Marshall-Olkin approach approximation
        u1 = rng.uniform(0,1,n)
        u2 = rng.uniform(0,1,n)
        dep = np.exp(-(((-np.log(u1))**th + (-np.log(u2))**th)**(1/th)))
        mix = dep * (1 - rho) + rho * u1
        mix = np.clip(mix, 1e-6, 1-1e-6)
        U   = np.column_stack([u1, mix])
    elif copula_type == "Frank":
        th = max(0.01, abs(rho) * 8 + 0.1) * np.sign(rho) if rho != 0 else 0.1
        u1 = rng.uniform(0,1,n)
        v  = rng.uniform(0,1,n)
        u2 = -np.log(1 + v*(np.exp(-th)-1)/(v*(np.exp(-th*u1)-1) - np.exp(-th*u1))) / th
        u2 = np.clip(u2, 1e-6, 1-1e-6)
        U  = np.column_stack([u1, u2])
    else:
        C = np.array([[1, rho],[rho, 1]])
        Z = rng.multivariate_normal([0,0], C, n)
        from scipy.stats import norm
        U = norm.cdf(Z)
    return U

def uniforms_to_returns(U, params_list):
    from scipy.stats import norm
    rets = []
    for i, p in enumerate(params_list):
        z = norm.ppf(np.clip(U[:, i], 1e-6, 1-1e-6))
        r_sim = p["mu"] + p["sigma"] * z
        rets.append(r_sim)
    return np.column_stack(rets)

def compute_optimal_weights_meanvar(mu_vec, cov_mat, target_return, allow_short=True):
    n = len(mu_vec)
    mu_vec = np.array(mu_vec)
    cov_mat = np.array(cov_mat)
    from scipy.optimize import minimize

    def portfolio_variance(w):
        return w @ cov_mat @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "eq", "fun": lambda w: w @ mu_vec - target_return}]
    bounds = None if allow_short else [(0, None)] * n
    w0 = np.ones(n) / n
    res = minimize(portfolio_variance, w0, method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"ftol": 1e-12, "maxiter": 1000})
    if res.success:
        return res.x
    return np.ones(n) / n

def compute_efficient_frontier(mu_vec, cov_mat, allow_short=True, n_points=50):
    mu_vec = np.array(mu_vec)
    mu_min = mu_vec.min() * 0.8
    mu_max = mu_vec.max() * 1.2
    targets = np.linspace(mu_min, mu_max, n_points)
    vols, rets, weights_all = [], [], []
    for t in targets:
        try:
            w = compute_optimal_weights_meanvar(mu_vec, cov_mat, t, allow_short)
            var = w @ cov_mat @ w
            vols.append(np.sqrt(var) * np.sqrt(252))
            rets.append((w @ mu_vec) * 252)
            weights_all.append(w)
        except:
            pass
    return np.array(vols), np.array(rets), weights_all

def estimate_garch_params(returns):
    mu_hat  = returns.mean()
    sig2    = returns.var()
    omega   = sig2 * 0.05
    alpha   = 0.10
    beta    = 0.85
    kappa   = 0.0
    return {"mu": mu_hat, "omega": omega, "alpha": alpha, "beta": beta, "kappa": kappa}

def compute_cvar(weights, simulated_returns, alpha=0.05):
    port_ret = simulated_returns @ weights
    var_level = np.percentile(port_ret, alpha * 100)
    cvar = port_ret[port_ret <= var_level].mean()
    return float(var_level), float(cvar)

def rolling_sharpe(port_returns, rf=0.0003, window=20):
    series = pd.Series(port_returns)
    roll_mean = series.rolling(window).mean() - rf
    roll_std  = series.rolling(window).std()
    return (roll_mean / roll_std).values

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px;'>
      <div style='font-size:1.5rem; font-weight:800; background:linear-gradient(135deg,#3b82f6,#10b981);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        BVC Portfolio Lab
      </div>
      <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>
        GARCH · Copulas · Optimisation
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📊 Univers d'actifs")
    selected_assets = st.multiselect(
        "Choisir les actifs (≥ 2)",
        list(MOROCCAN_ASSETS.keys()),
        default=["ATW — Attijariwafa Bank", "IAM — Maroc Telecom",
                 "OCP — OCP Group", "BON — Bons du Trésor (rf)"],
        help="Sélectionner au moins 2 actifs du marché boursier marocain (BVC/MASI)"
    )

    st.divider()
    st.markdown("### ⚙️ Modèle GARCH")
    garch_type = st.selectbox("Type de modèle", ["GARCH(1,1)", "NGarch (asymétrique)"])
    use_ngarch = garch_type == "NGarch (asymétrique)"

    col1, col2 = st.columns(2)
    with col1:
        alpha_garch = st.slider("α (choc)", 0.01, 0.30, 0.10, 0.01,
                                help="Poids du choc au carré (ARCH effect)")
    with col2:
        beta_garch  = st.slider("β (persistance)", 0.50, 0.98, 0.85, 0.01,
                                help="Persistance de la variance conditionnelle")

    if use_ngarch:
        kappa_ngarch = st.slider("κ (asymétrie NGarch)", -1.0, 3.0, 0.5, 0.1,
                                 help="Asymétrie : chocs négatifs > chocs positifs")
    else:
        kappa_ngarch = 0.0

    st.divider()
    st.markdown("### 🔗 Copule")
    copula_type = st.selectbox("Type de copule",
                               ["Normale","Student-t","Clayton","Gumbel","Frank"])

    rho_copula = st.slider("ρ — corrélation de base", -0.8, 0.8, 0.25, 0.05,
                           help="Paramètre de corrélation de la copule")

    if copula_type == "Student-t":
        nu_student = st.slider("ν — degrés de liberté", 3, 30, 5, 1,
                               help="Faible ν = queues plus épaisses (risque extrême)")
    else:
        nu_student = 5

    if copula_type in ["Clayton","Gumbel","Frank"]:
        theta_copula = st.slider("θ — paramètre de dépendance", 0.1, 5.0, 1.0, 0.1)
    else:
        theta_copula = 1.0

    st.divider()
    st.markdown("### 🎯 Optimisation")
    allow_short = st.toggle("Ventes à découvert", value=True,
                            help="Autoriser les poids négatifs (short selling)")
    target_mode = st.radio("Mode d'optimisation",
                           ["Minimiser variance (rendement cible)",
                            "Maximiser rendement (variance cible)"])
    risk_free_rate = st.slider("Taux sans risque annuel (%)", 0.0, 6.0, 3.0, 0.25) / 100

    if "rendement cible" in target_mode:
        target_return_pct = st.slider("Rendement cible annuel (%)", 1.0, 30.0, 12.0, 0.5)
        target_return = target_return_pct / 100
        target_var = None
    else:
        target_var_pct = st.slider("Variance cible annuelle (%)", 1.0, 25.0, 8.0, 0.5)
        target_var = (target_var_pct / 100) ** 2
        target_return = None

    st.divider()
    st.markdown("### 🔄 Simulation")
    n_simulations = st.select_slider("Simulations Monte Carlo",
                                     options=[500,1000,2000,5000,10000], value=2000)
    n_periods     = st.slider("Horizon (jours ouvrés)", 21, 252, 47,
                              help="Nombre de périodes de la simulation hors échantillon")
    rolling_window = st.slider("Fenêtre glissante (jours)", 10, 60, 20)

    run_btn = st.button("▶  Lancer l'optimisation", use_container_width=True)

# ─── Main content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class='hero-title'>Optimisation de Portefeuille — Marché Marocain (BVC)</div>
<div class='hero-sub'>Modélisation GARCH · Copules · Approche Moyenne-Variance Multipériodique</div>
""", unsafe_allow_html=True)

if len(selected_assets) < 2:
    st.warning("⚠️  Veuillez sélectionner au moins 2 actifs dans la barre latérale.")
    st.stop()

# ─── Simulate data ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_simulated_data(selected, alpha_g, beta_g, kappa_g, ngarch,
                       cop_type, rho, nu, theta_c, n_sim, n_per, seed=42):
    asset_keys = selected
    params_list = []
    all_returns = {}
    all_variances = {}

    rng = npr.default_rng(seed)
    n_hist = 600

    for key in asset_keys:
        p = MOROCCAN_ASSETS[key]
        omega = p["sigma"]**2 * (1 - alpha_g - beta_g)
        omega = max(omega, 1e-8)
        r, h = simulate_garch(p["mu"], omega, alpha_g, beta_g, kappa_g,
                               n_hist, ngarch=ngarch, seed=rng.integers(1e6))
        all_returns[key] = r
        all_variances[key] = h
        params_list.append({"mu": r.mean(), "sigma": r.std()})

    hist_df = pd.DataFrame(all_returns)
    hist_df.index = pd.bdate_range(end="2024-12-31", periods=n_hist)

    corr_mat = hist_df.corr().values
    n_assets = len(asset_keys)
    rho_effective = rho

    U = simulate_copula(rho_effective, n_sim, cop_type, nu=nu,
                        theta_c=theta_c, seed=seed)

    if n_assets == 2:
        sim_ret = uniforms_to_returns(U, params_list)
    else:
        sim_all = []
        for i in range(n_assets):
            row = []
            for j in range(n_assets):
                if i == j:
                    row.append(1.0)
                else:
                    row.append(rho_effective * corr_mat[i,j] / max(abs(corr_mat[i,j]),0.01))
            sim_all.append(row)

        sim_ret_list = []
        for i in range(n_assets):
            rho_pair = np.clip(rho_effective * abs(corr_mat[0, i]), -0.95, 0.95)
            Ui = simulate_copula(rho_pair, n_sim, cop_type, nu=nu,
                                 theta_c=theta_c, seed=seed + i)
            from scipy.stats import norm
            zi = norm.ppf(np.clip(Ui[:, 0], 1e-6, 1-1e-6))
            r_i = params_list[i]["mu"] + params_list[i]["sigma"] * zi
            sim_ret_list.append(r_i)
        sim_ret = np.column_stack(sim_ret_list)

    mu_vec  = sim_ret.mean(axis=0)
    cov_mat = np.cov(sim_ret.T)

    return hist_df, sim_ret, mu_vec, cov_mat, all_variances, params_list

with st.spinner("⚙️  Simulation GARCH + Copule en cours…"):
    hist_df, sim_ret, mu_vec, cov_mat, all_variances, params_list = get_simulated_data(
        tuple(selected_assets), alpha_garch, beta_garch, kappa_ngarch, use_ngarch,
        copula_type, rho_copula, nu_student, theta_copula,
        n_simulations, n_periods
    )

n_assets = len(selected_assets)
asset_labels = [k.split("—")[0].strip() for k in selected_assets]

# ─── Compute optimal weights ───────────────────────────────────────────────────

if "rendement cible" in target_mode:
    daily_target = target_return / 252
else:
    mu_best = mu_vec.max()
    daily_target = mu_best * 0.9

try:
    optimal_w = compute_optimal_weights_meanvar(
        mu_vec, cov_mat, daily_target, allow_short=allow_short)
except:
    optimal_w = np.ones(n_assets) / n_assets

port_return_daily = float(optimal_w @ mu_vec)
port_var_daily    = float(optimal_w @ cov_mat @ optimal_w)
port_vol_annual   = float(np.sqrt(port_var_daily * 252))
port_ret_annual   = float(port_return_daily * 252)
sharpe            = (port_ret_annual - risk_free_rate) / max(port_vol_annual, 1e-6)
var_level, cvar   = compute_cvar(optimal_w, sim_ret, alpha=0.05)
var_annual  = var_level * np.sqrt(252)
cvar_annual = cvar   * np.sqrt(252)

# ─── TABS ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    " Vue d'ensemble",
    " GARCH & Volatilité",
    " Analyse Copule",
    " Portefeuille Optimal",
    " Frontière Efficiente",
    "  Risque & CVaR",
    " Statistiques",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendement annuel", f"{port_ret_annual*100:.2f}%",
              f"Cible: {(daily_target*252*100):.1f}%")
    c2.metric("Volatilité annuelle", f"{port_vol_annual*100:.2f}%")
    c3.metric("Ratio de Sharpe", f"{sharpe:.3f}")
    c4.metric("VaR 95% (annuel)", f"{abs(var_annual)*100:.2f}%")
    c5.metric("CVaR 95% (annuel)", f"{abs(cvar_annual)*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-tag">ALLOCATION OPTIMALE</div>'
                    '<div class="section-title">Poids du portefeuille</div>',
                    unsafe_allow_html=True)

        colors_pie = PALETTE[:n_assets]
        fig_pie = go.Figure(go.Pie(
            labels=asset_labels,
            values=[abs(w) for w in optimal_w],
            hole=0.52,
            marker=dict(colors=colors_pie,
                        line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color=TEXT_COLOR, size=12),
            hovertemplate="<b>%{label}</b><br>Poids: %{value:.4f}<br>%{percent}<extra></extra>",
            textposition="outside",
            pull=[0.04 if w == optimal_w.max() else 0 for w in optimal_w],
        ))
        fig_pie.add_annotation(
            text=f"<b>{n_assets}</b><br>actifs",
            x=0.5, y=0.5, font=dict(size=16, color=TEXT_COLOR),
            showarrow=False, align="center"
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=340,
                              title=dict(text="", x=0),
                              showlegend=True,
                              legend=dict(orientation="v", x=1.02))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-tag">DÉCOMPOSITION</div>'
                    '<div class="section-title">Poids détaillés</div>',
                    unsafe_allow_html=True)

        weight_df = pd.DataFrame({
            "Actif": asset_labels,
            "Poids (%)": [f"{w*100:+.2f}" for w in optimal_w],
            "Contribution σ (%)": [
                f"{abs(optimal_w[i] * cov_mat[i,:] @ optimal_w) / port_var_daily * 100:.1f}"
                for i in range(n_assets)
            ],
            "Rend. annuel (%)": [f"{mu_vec[i]*252*100:.2f}" for i in range(n_assets)],
        })

        fig_bar = go.Figure(go.Bar(
            x=asset_labels,
            y=optimal_w * 100,
            marker_color=[GREEN if w >= 0 else RED for w in optimal_w],
            marker_line=dict(color="#0a0e1a", width=1),
            hovertemplate="<b>%{x}</b><br>Poids: %{y:.2f}%<extra></extra>",
        ))
        fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=dict(text="Poids optimaux (%)", font=dict(size=13, color="#94a3b8"), x=0),
                              yaxis_title="%",
                              showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        weight_df.style.applymap(
            lambda v: f"color: {'#10b981' if '+' in str(v) else '#ef4444' if '-' in str(v) else '#f1f5f9'}"
            if "%" in str(v) else "", subset=["Poids (%)"]
        ),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.markdown(f"""
    <div class="info-card">
    <b>🧠 Modèle actif</b> — {garch_type} + Copule {copula_type} (ρ = {rho_copula:.2f})
    &nbsp;|&nbsp; {n_simulations} simulations &nbsp;|&nbsp; Horizon: {n_periods} jours ouvrés
    &nbsp;|&nbsp; Ventes à découvert: {"✅ autorisées" if allow_short else "❌ interdites"}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: GARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-tag">MODÉLISATION GARCH</div>'
                '<div class="section-title">Variance conditionnelle & rendements simulés</div>',
                unsafe_allow_html=True)

    n_show = min(4, n_assets)
    fig_garch = make_subplots(
        rows=n_show, cols=2,
        subplot_titles=[
            title
            for key in selected_assets[:n_show]
            for title in [
                f"{key.split('—')[0].strip()} — Rendements journaliers",
                f"{key.split('—')[0].strip()} — Volatilité conditionnelle √h_t"
            ]
        ],
        vertical_spacing=0.06,
    )

    for i, key in enumerate(selected_assets[:n_show]):
        p = MOROCCAN_ASSETS[key]
        omega_i = p["sigma"]**2 * (1 - alpha_garch - beta_garch)
        omega_i = max(omega_i, 1e-8)
        r_i, h_i = simulate_garch(p["mu"], omega_i, alpha_garch, beta_garch,
                                   kappa_ngarch, 300, ngarch=use_ngarch, seed=i+42)
        color_i = PALETTE[i % len(PALETTE)]
        x_idx = list(range(len(r_i)))

        fig_garch.add_trace(
            go.Scatter(x=x_idx, y=r_i*100,
                       mode="lines", line=dict(color=color_i, width=0.9),
                       name=key.split("—")[0].strip(), showlegend=(i==0),
                       hovertemplate="%{y:.3f}%<extra></extra>"),
            row=i+1, col=1
        )
        fig_garch.add_trace(
            go.Scatter(x=x_idx, y=np.sqrt(h_i)*100,
                       mode="lines", line=dict(color=AMBER, width=1.2),
                       name="√h_t", showlegend=(i==0),
                       hovertemplate="%{y:.3f}%<extra></extra>"),
            row=i+1, col=2
        )
        fig_garch.add_hrect(
            y0=-np.std(r_i)*2*100, y1=np.std(r_i)*2*100,
            fillcolor="rgba(59,130,246,0.04)", line_width=0,
            row=i+1, col=1
        )

    fig_garch.update_layout(
        **PLOTLY_LAYOUT,
        height=220 * n_show,
        title=dict(text=f"Simulation GARCH — {garch_type}", font=dict(size=14, color="#94a3b8")),
        showlegend=False,
    )
    for i in range(1, n_show*2+1):
        fig_garch.update_xaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, col=(i%2+1) if i%2 else 1)
        fig_garch.update_yaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)

    st.plotly_chart(fig_garch, use_container_width=True)

    st.divider()
    st.markdown("### 📐 Paramètres GARCH estimés")

    garch_table = []
    for key in selected_assets:
        p = MOROCCAN_ASSETS[key]
        omega_i = p["sigma"]**2 * (1 - alpha_garch - beta_garch)
        h0 = omega_i / max(1 - alpha_garch - beta_garch, 0.01)
        garch_table.append({
            "Actif": key.split("—")[0].strip(),
            "μ journalier": f"{p['mu']*100:.4f}%",
            "σ annualisée": f"{p['sigma']*np.sqrt(252)*100:.2f}%",
            "ω": f"{omega_i:.6f}",
            "α": f"{alpha_garch:.3f}",
            "β": f"{beta_garch:.3f}",
            "α+β": f"{alpha_garch+beta_garch:.3f}",
            "h₀ (var. init.)": f"{h0:.6f}",
            "κ (NGarch)": f"{kappa_ngarch:.2f}" if use_ngarch else "—",
            "Stationnarité": "✅" if alpha_garch+beta_garch < 1 else "⚠️",
        })
    st.dataframe(pd.DataFrame(garch_table), use_container_width=True, hide_index=True)

    if alpha_garch + beta_garch >= 0.99:
        st.markdown('<div class="warn-card">⚠️ α+β ≥ 0.99 — Le processus GARCH est proche d\'une racine unitaire (IGARCH). Réduire α ou β.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-card">✅ Stationnarité confirmée : α+β = {alpha_garch+beta_garch:.3f} &lt; 1 — La variance conditionnelle converge vers sa valeur à long terme.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: COPULA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-tag">STRUCTURE DE DÉPENDANCE</div>'
                '<div class="section-title">Analyse des copules</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
    <b>Copule {copula_type}</b> — {COPULA_DESCRIPTIONS[copula_type]}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Nuage de points U(0,1) — espace copule")
        U_plot = simulate_copula(rho_copula, min(1000, n_simulations), copula_type,
                                 nu=nu_student, theta_c=theta_copula, seed=42)
        pair_0 = 0
        pair_1 = min(1, U_plot.shape[1]-1)
        density_colors = np.sqrt((U_plot[:,pair_0]-0.5)**2 + (U_plot[:,pair_1]-0.5)**2)

        fig_cop = go.Figure(go.Scatter(
            x=U_plot[:,pair_0], y=U_plot[:,pair_1],
            mode="markers",
            marker=dict(
                size=3.5,
                color=density_colors,
                colorscale=[[0,"#3b82f6"],[0.5,"#10b981"],[1,"#f59e0b"]],
                opacity=0.55,
            ),
            hovertemplate="U₁: %{x:.3f}<br>U₂: %{y:.3f}<extra></extra>",
        ))
        fig_cop.update_layout(**PLOTLY_LAYOUT, height=320,
                              xaxis_title="U₁", yaxis_title="U₂",
                              title=dict(text=f"Copule {copula_type} (ρ={rho_copula:.2f})",
                                         font=dict(size=13, color="#94a3b8")))
        st.plotly_chart(fig_cop, use_container_width=True)

    with col_b:
        st.markdown("#### Heatmap de corrélation des rendements simulés")
        if n_assets >= 2:
            corr_sim = np.corrcoef(sim_ret.T)
            fig_hm = go.Figure(go.Heatmap(
                z=corr_sim,
                x=asset_labels, y=asset_labels,
                colorscale=[[0, "#ef4444"],[0.5, "#111827"],[1, "#3b82f6"]],
                zmin=-1, zmax=1,
                text=[[f"{corr_sim[i,j]:.2f}" for j in range(n_assets)]
                       for i in range(n_assets)],
                texttemplate="%{text}",
                textfont=dict(size=11, color=TEXT_COLOR),
                hovertemplate="(%{x}, %{y})<br>ρ = %{z:.3f}<extra></extra>",
            ))
            fig_hm.update_layout(**PLOTLY_LAYOUT, height=320,
                                  title=dict(text="Matrice de corrélation simulée",
                                             font=dict(size=13, color="#94a3b8")))
            st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()
    st.markdown("### 🔍 Comparaison des copules")

    copule_compare = []
    for cop_name in ["Normale","Student-t","Clayton","Gumbel","Frank"]:
        U_c = simulate_copula(rho_copula, 2000, cop_name, nu=nu_student,
                              theta_c=theta_copula, seed=42)
        tail_dep_lower = np.mean((U_c[:,0] < 0.05) & (U_c[:,1] < 0.05)) / 0.05
        tail_dep_upper = np.mean((U_c[:,0] > 0.95) & (U_c[:,1] > 0.95)) / 0.05
        kendall_tau = np.corrcoef(U_c[:,0], U_c[:,1])[0,1]
        copule_compare.append({
            "Copule": cop_name,
            "Queue inf. λL": f"{tail_dep_lower:.3f}",
            "Queue sup. λU": f"{tail_dep_upper:.3f}",
            "Tau de Kendall": f"{kendall_tau:.3f}",
            "Sélectionnée": "✅" if cop_name == copula_type else "",
            "Usage BVC": {
                "Normale": "Marché calme",
                "Student-t": "Chocs communs ⭐",
                "Clayton": "Crises / chutes",
                "Gumbel": "Rallyes haussiers",
                "Frank": "Dépendance modérée",
            }[cop_name]
        })
    st.dataframe(pd.DataFrame(copule_compare), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-tag">STRATÉGIE OPTIMALE</div>'
                '<div class="section-title">Évolution des poids et performance</div>',
                unsafe_allow_html=True)

    # Simulate one-step-ahead strategy
    rng_strat = npr.default_rng(99)
    n_path = n_periods
    weights_path = np.zeros((n_path, n_assets))
    port_returns_path = np.zeros(n_path)

    current_w = optimal_w.copy()
    cumulative_wealth = 1.0
    wealth_path = [1.0]
    bvc_path = [1.0]

    for t in range(n_path):
        # One-step-ahead weight update (simplified SIMMI-style decay)
        decay = 1 - (t / n_path) * 0.3
        noise = rng_strat.normal(0, 0.02, n_assets)
        current_w = optimal_w * decay + noise
        if allow_short is False:
            current_w = np.maximum(current_w, 0)
        current_w /= current_w.sum() if current_w.sum() != 0 else 1
        weights_path[t] = current_w

        r_t = sim_ret[t % len(sim_ret)]
        port_r = float(current_w @ r_t)
        port_returns_path[t] = port_r
        cumulative_wealth *= (1 + port_r)
        wealth_path.append(cumulative_wealth)

        bvc_r = float(np.mean([MOROCCAN_ASSETS[k]["mu"] for k in selected_assets])
                      + rng_strat.normal(0, 0.008))
        bvc_path.append(bvc_path[-1] * (1 + bvc_r))

    # Plot weights over time
    fig_w = go.Figure()
    for i, label in enumerate(asset_labels):
        fig_w.add_trace(go.Scatter(
            x=list(range(n_path)), y=weights_path[:,i],
            mode="lines", name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"<b>{label}</b><br>Période: %{{x}}<br>Poids: %{{y:.3f}}<extra></extra>",
            stackgroup=None,
        ))
    fig_w.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1, line_dash="dot")
    fig_w.update_layout(**PLOTLY_LAYOUT, height=300,
                         title=dict(text="Évolution des poids (one-step-ahead)",
                                    font=dict(size=13, color="#94a3b8")),
                         yaxis_title="Poids w_t", xaxis_title="Période (jours)")
    st.plotly_chart(fig_w, use_container_width=True)

    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        fig_wealth = go.Figure()
        fig_wealth.add_trace(go.Scatter(
            x=list(range(n_path+1)), y=wealth_path,
            mode="lines", name="Portefeuille optimal",
            line=dict(color=BLUE, width=2.2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        ))
        fig_wealth.add_trace(go.Scatter(
            x=list(range(n_path+1)), y=bvc_path,
            mode="lines", name="Benchmark MASI",
            line=dict(color=AMBER, width=1.5, dash="dot"),
        ))
        fig_wealth.update_layout(**PLOTLY_LAYOUT, height=280,
                                  title=dict(text="Richesse cumulée (base 1)",
                                             font=dict(size=13, color="#94a3b8")),
                                  yaxis_title="Valeur du portefeuille")
        st.plotly_chart(fig_wealth, use_container_width=True)

    with col_perf2:
        sharpe_roll = rolling_sharpe(port_returns_path, rf=risk_free_rate/252, window=rolling_window)
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=list(range(len(sharpe_roll))), y=sharpe_roll,
            mode="lines", name="Sharpe glissant",
            line=dict(color=GREEN, width=1.8),
        ))
        fig_sharpe.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_sharpe.add_hline(y=1, line_color=GREEN, line_width=1, line_dash="dot",
                              annotation_text="Seuil 1.0", annotation_font_color=GREEN)
        fig_sharpe.update_layout(**PLOTLY_LAYOUT, height=280,
                                  title=dict(text=f"Sharpe glissant ({rolling_window}j)",
                                             font=dict(size=13, color="#94a3b8")),
                                  yaxis_title="Sharpe")
        st.plotly_chart(fig_sharpe, use_container_width=True)

    st.divider()
    perf_final = wealth_path[-1] - 1
    benchmark_final = bvc_path[-1] - 1
    alpha_excess = perf_final - benchmark_final
    max_dd = min(0, min([w / max(wealth_path[:i+1]) - 1 for i, w in enumerate(wealth_path)]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance totale", f"{perf_final*100:.2f}%",
              f"vs benchmark: {alpha_excess*100:+.2f}%")
    c2.metric("Rendement annualisé", f"{port_ret_annual*100:.2f}%",
              f"Cible: {(daily_target*252*100):.1f}%")
    c3.metric("Sharpe (annualisé)", f"{sharpe:.3f}")
    c4.metric("Drawdown maximum", f"{max_dd*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-tag">FRONTIÈRE EFFICIENTE</div>'
                '<div class="section-title">Espace risque-rendement multipériodique</div>',
                unsafe_allow_html=True)

    with st.spinner("Calcul de la frontière efficiente…"):
        vols_ef, rets_ef, weights_ef = compute_efficient_frontier(
            mu_vec, cov_mat, allow_short=allow_short, n_points=60)

    sharpes_ef = (rets_ef - risk_free_rate) / np.where(vols_ef > 0, vols_ef, 1e-6)
    max_sharpe_idx = np.argmax(sharpes_ef)
    min_vol_idx    = np.argmin(vols_ef)

    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=vols_ef*100, y=rets_ef*100,
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
        marker=dict(
            size=7,
            color=sharpes_ef,
            colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#10b981"]],
            showscale=True,
            colorbar=dict(title="Sharpe", tickfont=dict(color=TEXT_COLOR),
                          titlefont=dict(color=TEXT_COLOR)),
        ),
        name="Frontière efficiente",
        hovertemplate="Vol: %{x:.2f}%<br>Rend: %{y:.2f}%<extra></extra>",
    ))

    # Max Sharpe
    fig_ef.add_trace(go.Scatter(
        x=[vols_ef[max_sharpe_idx]*100], y=[rets_ef[max_sharpe_idx]*100],
        mode="markers+text",
        marker=dict(size=14, color=GREEN, symbol="star"),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(color=GREEN, size=12),
        name=f"Max Sharpe ({sharpes_ef[max_sharpe_idx]:.2f})",
    ))

    # Min Vol
    fig_ef.add_trace(go.Scatter(
        x=[vols_ef[min_vol_idx]*100], y=[rets_ef[min_vol_idx]*100],
        mode="markers+text",
        marker=dict(size=12, color=AMBER, symbol="diamond"),
        text=["Min Variance"], textposition="top left",
        textfont=dict(color=AMBER, size=12),
        name="Min Variance",
    ))

    # Current portfolio
    fig_ef.add_trace(go.Scatter(
        x=[port_vol_annual*100], y=[port_ret_annual*100],
        mode="markers+text",
        marker=dict(size=14, color=RED, symbol="x"),
        text=["Portefeuille actuel"], textposition="top right",
        textfont=dict(color=RED, size=12),
        name="Portefeuille sélectionné",
    ))

    # Individual assets
    for i, key in enumerate(selected_assets):
        p = MOROCCAN_ASSETS[key]
        fig_ef.add_trace(go.Scatter(
            x=[p["sigma"]*np.sqrt(252)*100], y=[p["mu"]*252*100],
            mode="markers+text",
            marker=dict(size=9, color=PALETTE[i % len(PALETTE)], symbol="circle"),
            text=[asset_labels[i]], textposition="top right",
            textfont=dict(size=10, color=PALETTE[i % len(PALETTE)]),
            name=asset_labels[i], showlegend=False,
        ))

    fig_ef.update_layout(
        **PLOTLY_LAYOUT, height=500,
        title=dict(text="Frontière efficiente — Marché Marocain (BVC)",
                   font=dict(size=14, color="#94a3b8")),
        xaxis_title="Volatilité annualisée (%)",
        yaxis_title="Rendement annualisé (%)",
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    col_ef1, col_ef2, col_ef3 = st.columns(3)
    if len(rets_ef) > max_sharpe_idx:
        col_ef1.metric("Max Sharpe — Rendement", f"{rets_ef[max_sharpe_idx]*100:.2f}%")
        col_ef1.metric("Max Sharpe — Volatilité", f"{vols_ef[max_sharpe_idx]*100:.2f}%")
    if len(rets_ef) > min_vol_idx:
        col_ef2.metric("Min Variance — Rendement", f"{rets_ef[min_vol_idx]*100:.2f}%")
        col_ef2.metric("Min Variance — Volatilité", f"{vols_ef[min_vol_idx]*100:.2f}%")
    col_ef3.metric("Portefeuille actuel — Rendement", f"{port_ret_annual*100:.2f}%")
    col_ef3.metric("Portefeuille actuel — Volatilité", f"{port_vol_annual*100:.2f}%")

    st.divider()
    st.markdown("### 🎯 Portefeuilles cibles (rendements c = 10.5% → 17%)")
    target_pts = np.arange(0.105, 0.175, 0.015)
    target_rows = []
    for c in target_pts:
        try:
            w_c = compute_optimal_weights_meanvar(mu_vec, cov_mat, c/252, allow_short)
            var_c = float(w_c @ cov_mat @ w_c)
            sh_c  = (c - risk_free_rate) / max(np.sqrt(var_c*252), 1e-6)
            target_rows.append({
                "Rendement cible (%)": f"{c*100:.1f}",
                "Variance min. (%)": f"{np.sqrt(var_c*252)*100:.2f}",
                "Sharpe": f"{sh_c:.3f}",
                **{asset_labels[i]: f"{w_c[i]*100:+.1f}%" for i in range(n_assets)}
            })
        except:
            pass
    if target_rows:
        st.dataframe(pd.DataFrame(target_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-tag">MESURES DE RISQUE</div>'
                '<div class="section-title">VaR, CVaR & distribution des pertes</div>',
                unsafe_allow_html=True)

    port_sim_returns = sim_ret @ optimal_w
    var_5, cvar_5   = compute_cvar(optimal_w, sim_ret, alpha=0.05)
    var_1, cvar_1   = compute_cvar(optimal_w, sim_ret, alpha=0.01)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95% (journalier)", f"{abs(var_5)*100:.3f}%",
              f"Annualisé: {abs(var_5)*np.sqrt(252)*100:.2f}%")
    c2.metric("CVaR 95% (journalier)", f"{abs(cvar_5)*100:.3f}%",
              f"Annualisé: {abs(cvar_5)*np.sqrt(252)*100:.2f}%")
    c3.metric("VaR 99% (journalier)", f"{abs(var_1)*100:.3f}%",
              f"Annualisé: {abs(var_1)*np.sqrt(252)*100:.2f}%")
    c4.metric("CVaR 99% (journalier)", f"{abs(cvar_1)*100:.3f}%",
              f"Annualisé: {abs(cvar_1)*np.sqrt(252)*100:.2f}%")

    fig_dist = go.Figure()

    ret_pct = port_sim_returns * 100
    fig_dist.add_trace(go.Histogram(
        x=ret_pct, nbinsx=80,
        name="Distribution simulée",
        marker_color="rgba(59,130,246,0.55)",
        marker_line=dict(color="#0a0e1a", width=0.3),
        hovertemplate="Rendement: %{x:.3f}%<br>Fréquence: %{y}<extra></extra>",
    ))
    fig_dist.add_vline(x=var_5*100, line_color=AMBER, line_width=2,
                        annotation_text=f"VaR 95%: {var_5*100:.3f}%",
                        annotation_font_color=AMBER, annotation_position="top right")
    fig_dist.add_vline(x=cvar_5*100, line_color=RED, line_width=2, line_dash="dash",
                        annotation_text=f"CVaR 95%: {cvar_5*100:.3f}%",
                        annotation_font_color=RED, annotation_position="top left")
    fig_dist.add_vline(x=var_1*100, line_color=CORAL, line_width=1.5, line_dash="dot",
                        annotation_text=f"VaR 99%", annotation_font_color=CORAL)

    fig_dist.add_vrect(x0=ret_pct.min(), x1=var_5*100,
                        fillcolor="rgba(239,68,68,0.08)", line_width=0,
                        annotation_text="Zone de perte extrême",
                        annotation_font_color="#ef4444", annotation_position="top left")

    fig_dist.update_layout(
        **PLOTLY_LAYOUT, height=380,
        title=dict(text=f"Distribution des rendements simulés — Copule {copula_type}",
                   font=dict(size=13, color="#94a3b8")),
        xaxis_title="Rendement journalier (%)",
        yaxis_title="Fréquence",
        bargap=0.05,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Risk decomposition
    st.markdown("### 📊 Contribution au risque par actif")
    risk_contrib = np.array([
        abs(optimal_w[i] * (cov_mat[i,:] @ optimal_w)) for i in range(n_assets)
    ])
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

    fig_rc = go.Figure(go.Bar(
        x=asset_labels, y=risk_contrib_pct,
        marker_color=PALETTE[:n_assets],
        marker_line=dict(color="#0a0e1a", width=1),
        text=[f"{v:.1f}%" for v in risk_contrib_pct],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
        hovertemplate="<b>%{x}</b><br>Contribution au risque: %{y:.1f}%<extra></extra>",
    ))
    fig_rc.update_layout(**PLOTLY_LAYOUT, height=280,
                          title=dict(text="Contribution marginale au risque de variance (%)",
                                     font=dict(size=13, color="#94a3b8")),
                          yaxis_title="%", showlegend=False)
    st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown(f"""
    <div class="warn-card">
    <b>⚠️ Note méthodologique BVC</b><br>
    La CVaR est préférée à la VaR pour la BVC car elle satisfait les 4 propriétés de cohérence
    (Artzner et al., 1999) : monotonie, sous-additivité, homogénéité positive et invariance par translation.
    La VaR n'est pas sous-additive — elle peut sous-estimer le risque agrégé dans les marchés frontières
    comme le Maroc où la liquidité est limitée.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-tag">STATISTIQUES DESCRIPTIVES</div>'
                '<div class="section-title">Propriétés des séries simulées</div>',
                unsafe_allow_html=True)

    from scipy.stats import kurtosis, skew, jarque_bera

    stat_rows = []
    for i, key in enumerate(selected_assets):
        p = MOROCCAN_ASSETS[key]
        r_i = sim_ret[:, i]
        jb_stat, jb_p = jarque_bera(r_i)
        arch_lm = np.corrcoef(r_i[:-1]**2, r_i[1:]**2)[0,1]
        stat_rows.append({
            "Actif": key.split("—")[0].strip(),
            "Secteur": p["sector"],
            "Moyenne journ. (%)": f"{r_i.mean()*100:.4f}",
            "Vol. annualisée (%)": f"{r_i.std()*np.sqrt(252)*100:.2f}",
            "Asymétrie (skew)": f"{skew(r_i):.4f}",
            "Aplatissement (kurt)": f"{kurtosis(r_i, fisher=True):.4f}",
            "Jarque-Bera (stat)": f"{jb_stat:.2f}",
            "JB p-valeur": f"{jb_p:.4f}",
            "Corrél. ARCH(1)": f"{arch_lm:.4f}",
            "Non-normalité": "⚠️ Oui" if jb_p < 0.05 else "✅ Non",
            "Effet ARCH": "⚠️ Oui" if abs(arch_lm) > 0.1 else "✅ Non",
        })

    stat_df = pd.DataFrame(stat_rows)
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    st.divider()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### 📐 Matrice de covariance (annualisée)")
        cov_annual = cov_mat * 252
        fig_cov = go.Figure(go.Heatmap(
            z=cov_annual,
            x=asset_labels, y=asset_labels,
            colorscale=[[0,"#111827"],[0.5,"#1e3a5f"],[1,"#3b82f6"]],
            text=[[f"{cov_annual[i,j]:.4f}" for j in range(n_assets)] for i in range(n_assets)],
            texttemplate="%{text}", textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="Cov(%{x},%{y}) = %{z:.6f}<extra></extra>",
        ))
        fig_cov.update_layout(**PLOTLY_LAYOUT, height=300,
                               title=dict(text="Σ (annualisée)",
                                          font=dict(size=13, color="#94a3b8")))
        st.plotly_chart(fig_cov, use_container_width=True)

    with col_s2:
        st.markdown("### 📦 Q-Q Plot des résidus normalisés")
        from scipy.stats import probplot
        r_port = sim_ret @ optimal_w
        r_std  = (r_port - r_port.mean()) / r_port.std()
        (theor, sample), _ = probplot(r_std, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theor, y=sample, mode="markers",
            marker=dict(size=3, color=BLUE, opacity=0.5),
            name="Quantiles observés",
        ))
        max_val = max(abs(theor.max()), abs(sample.max()))
        fig_qq.add_trace(go.Scatter(
            x=[-max_val, max_val], y=[-max_val, max_val],
            mode="lines", line=dict(color=RED, width=1.5, dash="dot"),
            name="Loi normale",
        ))
        fig_qq.update_layout(**PLOTLY_LAYOUT, height=300,
                              title=dict(text="Q-Q Plot rendements du portefeuille",
                                         font=dict(size=13, color="#94a3b8")),
                              xaxis_title="Quantiles théoriques (N(0,1))",
                              yaxis_title="Quantiles observés")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.divider()
    st.markdown("""
    <div class="info-card">
    <b>📚 Références méthodologiques</b><br>
    Prince (2007) — <i>Problème d'optimisation de portefeuille en temps discret avec une modélisation GARCH</i>, HEC Montréal<br>
    Vaillancourt & Watier (2005) — Moyenne-variance multipériodique multivariée<br>
    Bollerslev (1986) — GARCH généralisé &nbsp;|&nbsp; Engle & Kroner (1995) — BEKK multivarié<br>
    Longstaff & Schwartz (2001) — Méthode LSM pour les espérances conditionnelles<br>
    Artzner et al. (1999) — Mesures de risque cohérentes (CVaR)
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#374151; font-size:12px; padding: 8px 0 20px;'>
  BVC Portfolio Lab &nbsp;·&nbsp; GARCH + Copules + Optimisation Moyenne-Variance &nbsp;·&nbsp;
  Marché Boursier de Casablanca (BVC/MASI) &nbsp;·&nbsp;
  Basé sur Prince (2007) — HEC Montréal
</div>
""", unsafe_allow_html=True)
