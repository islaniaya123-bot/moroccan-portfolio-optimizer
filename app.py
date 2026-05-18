"""
Moroccan Portfolio Optimizer
Dynamic Portfolio Optimization — NGarch + Copula Framework
Bourse des Valeurs de Casablanca
Vaillancourt & Watier (2005)

ENHANCED VERSION:
- Beautiful charts with Plotly
- Professional Moroccan design theme
- Interactive visualizations
- Comprehensive results dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BVC Portfolio Optimizer | Moroccan Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "ink":     "#0d1117",
    "ink2":    "#1e2530",
    "ink3":    "#5a6070",
    "bg":      "#f7f5f0",
    "bg2":     "#edeae3",
    "bg3":     "#e2ddd4",
    "gold":    "#c9a84c",
    "gold2":   "#e2c068",
    "teal":    "#1c4f4a",
    "teal2":   "#2a7a72",
    "teal3":   "#3a9a8a",
    "rust":    "#8b3a2a",
    "rust2":   "#a84a3a",
    "green":   "#2a7a4a",
    "green2":  "#3a9a5a",
    "red":     "#8b2a2a",
    "red2":    "#a83a3a",
    "border":  "#ccc8be",
    "white":   "#ffffff",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Barlow:wght@300;400;500;600&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

html, body, [class*="css"] {{
    font-family: 'Barlow', sans-serif;
    font-weight: 400;
    background: {PALETTE['bg']};
    color: {PALETTE['ink']};
}}

/* Main header */
.app-header {{
    background: linear-gradient(135deg, {PALETTE['ink']} 0%, {PALETTE['ink2']} 100%);
    padding: 2rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 2px solid {PALETTE['gold']};
    position: relative;
    overflow: hidden;
}}
.app-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {PALETTE['gold']}, {PALETTE['teal2']}, {PALETTE['gold']});
}}
.header-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.5rem;
}}
.header-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 300;
    color: #fff;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}}
.header-title em {{
    font-style: italic;
    color: {PALETTE['gold2']};
}}
.header-sub {{
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.5;
    max-width: 650px;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PALETTE['ink']} 0%, {PALETTE['ink2']} 100%);
    border-right: 1px solid rgba(201,168,76,0.2);
}}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label {{
    color: rgba(255,255,255,0.7) !important;
}}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {{
    font-family: 'Cormorant Garamond', serif !important;
    color: {PALETTE['gold']} !important;
    font-weight: 300 !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid rgba(201,168,76,0.3);
    padding-bottom: 0.5rem;
}}

/* Section headings */
.sec-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}
.sec-eyebrow::before {{
    content: '';
    display: block;
    width: 30px;
    height: 1px;
    background: {PALETTE['gold']};
}}
.sec-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: {PALETTE['ink']};
    margin-bottom: 0.3rem;
    line-height: 1.2;
}}
.sec-title em {{
    font-style: italic;
    color: {PALETTE['gold']};
}}
.sec-desc {{
    font-size: 0.85rem;
    color: {PALETTE['ink3']};
    line-height: 1.6;
    margin-bottom: 1.5rem;
    max-width: 600px;
}}

/* Metric cards */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}}
.metric-card {{
    background: {PALETTE['white']};
    border: 1px solid {PALETTE['border']};
    border-radius: 0;
    border-top: 3px solid {PALETTE['gold']};
    padding: 1rem 1.2rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
.metric-card.dark {{
    background: {PALETTE['ink']};
    border-color: {PALETTE['ink2']};
}}
.metric-card.dark .mc-label,
.metric-card.dark .mc-sub {{
    color: rgba(255,255,255,0.4);
}}
.metric-card.dark .mc-value {{
    color: {PALETTE['gold2']};
}}
.mc-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
    margin-bottom: 0.4rem;
}}
.mc-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: {PALETTE['ink']};
    line-height: 1.1;
    margin-bottom: 0.2rem;
}}
.mc-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: {PALETTE['ink3']};
}}

/* Table styling */
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: {PALETTE['white']};
    border: 1px solid {PALETTE['border']};
}}
.styled-table th {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
    padding: 0.8rem 1rem;
    background: {PALETTE['bg2']};
    border-bottom: 1px solid {PALETTE['border']};
    text-align: left;
    font-weight: 400;
}}
.styled-table td {{
    padding: 0.8rem 1rem;
    border-bottom: 1px solid {PALETTE['border']};
    color: {PALETTE['ink']};
    vertical-align: middle;
}}
.styled-table tr:hover td {{
    background: {PALETTE['bg2']};
}}

/* Buttons */
.stButton > button {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
    border-radius: 0 !important;
    background: {PALETTE['teal']} !important;
    color: #fff !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: {PALETTE['gold']} !important;
    color: {PALETTE['ink']} !important;
}}

/* Alert boxes */
.alert-success {{
    background: rgba(42,122,74,0.1);
    border-left: 3px solid {PALETTE['green']};
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}}
.alert-warning {{
    background: rgba(201,168,76,0.1);
    border-left: 3px solid {PALETTE['gold']};
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}}
.alert-info {{
    background: rgba(28,79,74,0.08);
    border-left: 3px solid {PALETTE['teal']};
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}}

/* Divider */
.divider {{
    border: none;
    border-top: 1px solid {PALETTE['border']};
    margin: 1.5rem 0;
}}
.divider-gold {{
    border: none;
    border-top: 1px solid {PALETTE['gold']};
    margin: 1.5rem 0;
}}

/* Utility */
.mono {{ font-family: 'DM Mono', monospace; font-size: 0.8rem; }}
.teal-text {{ color: {PALETTE['teal']}; }}
.gold-text {{ color: {PALETTE['gold']}; }}
.green-text {{ color: {PALETTE['green']}; }}
.red-text {{ color: {PALETTE['red']}; }}
.muted {{ color: {PALETTE['ink3']}; font-size: 0.8rem; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ASSET DATA (WITH MASI)
# ─────────────────────────────────────────────────────────────────────────────

BVC_ASSETS = {
    "MASI": {"name": "MASI Index", "sector": "Index", "beta1": 0.720, "kappa": -0.25, "mcap": 650, "beta0": 3.5e-6, "delta": 0.10, "color": PALETTE["gold"]},
    "ATW": {"name": "Attijariwafa Bank", "sector": "Banking", "beta1": 0.871, "kappa": -0.38, "mcap": 92, "beta0": 5.2e-6, "delta": 0.11, "color": PALETTE["teal"]},
    "IAM": {"name": "Maroc Telecom", "sector": "Telecoms", "beta1": 0.842, "kappa": -0.22, "mcap": 71, "beta0": 4.1e-6, "delta": 0.09, "color": PALETTE["teal2"]},
    "BCP": {"name": "Banque Centrale Populaire", "sector": "Banking", "beta1": 0.858, "kappa": -0.41, "mcap": 58, "beta0": 4.8e-6, "delta": 0.10, "color": PALETTE["teal"]},
    "OCP": {"name": "OCP Group", "sector": "Phosphates", "beta1": 0.804, "kappa": -0.45, "mcap": 48, "beta0": 6.1e-6, "delta": 0.13, "color": PALETTE["rust"]},
    "MNG": {"name": "Managem", "sector": "Mining", "beta1": 0.793, "kappa": -0.51, "mcap": 34, "beta0": 5.9e-6, "delta": 0.12, "color": "#4a6a5a"},
    "TQM": {"name": "TotalEnergies MM", "sector": "Energy", "beta1": 0.811, "kappa": -0.29, "mcap": 28, "beta0": 4.5e-6, "delta": 0.09, "color": "#5a6a4a"},
    "CIH": {"name": "CIH Bank", "sector": "Banking", "beta1": 0.834, "kappa": -0.35, "mcap": 22, "beta0": 4.7e-6, "delta": 0.10, "color": PALETTE["teal"]},
    "LBV": {"name": "Label Vie", "sector": "Retail", "beta1": 0.769, "kappa": -0.18, "mcap": 18, "beta0": 3.9e-6, "delta": 0.08, "color": "#6a5a7a"},
    "WAA": {"name": "Wafa Assurance", "sector": "Insurance", "beta1": 0.822, "kappa": -0.27, "mcap": 16, "beta0": 4.4e-6, "delta": 0.09, "color": "#5a6a7a"},
    "MUT": {"name": "Mutandis", "sector": "Consumer", "beta1": 0.748, "kappa": -0.14, "mcap": 12, "beta0": 3.7e-6, "delta": 0.08, "color": "#7a6a5a"},
    "ADDH": {"name": "Addoha", "sector": "Real Estate", "beta1": 0.701, "kappa": -0.62, "mcap": 9, "beta0": 7.2e-6, "delta": 0.14, "color": PALETTE["ink3"]},
}

SECTOR_COLORS = {
    "Index": PALETTE["gold"],
    "Banking": PALETTE["teal"],
    "Telecoms": PALETTE["teal2"],
    "Phosphates": PALETTE["rust"],
    "Mining": "#4a6a5a",
    "Energy": "#5a6a4a",
    "Retail": "#6a5a7a",
    "Insurance": "#5a6a7a",
    "Consumer": "#7a6a5a",
    "Real Estate": PALETTE["ink3"],
}


# ─────────────────────────────────────────────────────────────────────────────
# NGARCH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class NGarchModel:
    def __init__(self, info: dict, rf_monthly: float):
        self.beta0 = info["beta0"]
        self.beta1 = info["beta1"]
        self.beta2 = 0.09
        self.kappa = info["kappa"]
        self.delta = info["delta"]
        self.rf = rf_monthly

    def h0(self) -> float:
        denom = 1 - self.beta1 - self.beta2 * (1 + self.kappa ** 2)
        return self.beta0 / max(denom, 1e-10)

    def simulate(self, T: int, M: int, innovations: np.ndarray) -> tuple:
        h = np.zeros((M, T))
        ret = np.zeros((M, T))
        h[:, 0] = self.h0()
        for t in range(T):
            Z = innovations[:, t]
            log_ret = self.rf + self.delta * np.sqrt(h[:, t]) - h[:, t] / 2 + np.sqrt(h[:, t]) * Z
            ret[:, t] = np.exp(log_ret) - 1
            if t < T - 1:
                h[:, t + 1] = self.beta0 + self.beta1 * h[:, t] + self.beta2 * h[:, t] * (Z - self.kappa) ** 2
        return ret, h

    def forecast_variance(self, steps: int) -> np.ndarray:
        h = np.zeros(steps)
        h[0] = self.h0()
        persist = self.beta1 + self.beta2 * (1 + self.kappa ** 2)
        for t in range(1, steps):
            h[t] = self.beta0 + persist * h[t - 1]
        return h


# ─────────────────────────────────────────────────────────────────────────────
# COPULA ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CopulaEngine:
    def __init__(self, copula: str, N: int, theta: float = 1.5, nu: int = 5):
        self.copula = copula
        self.N = N
        self.theta = theta
        self.nu = nu
        rho_val = 0.35
        self.rho = rho_val * np.ones((N, N))
        np.fill_diagonal(self.rho, 1.0)
        A = (self.rho + self.rho.T) / 2
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, 1e-8)
        self.rho = vecs @ np.diag(vals) @ vecs.T

    def sample_innovations(self, M: int, T: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.rho)
        innovations = np.zeros((self.N, M, T))

        if self.copula == "Gaussian":
            Z = rng.standard_normal((M, T, self.N))
            Z_corr = Z @ L.T
            for n in range(self.N):
                innovations[n] = Z_corr[:, :, n]

        elif self.copula == "Student-t":
            Z = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            chi2 = rng.chisquare(self.nu, (M, T, 1))
            T_rv = Z_c / np.sqrt(chi2 / self.nu)
            U = stats.t.cdf(T_rv, df=self.nu)
            U = np.clip(U, 1e-6, 1 - 1e-6)
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        elif self.copula == "Clayton":
            th = max(self.theta, 0.01)
            Z = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U = stats.norm.cdf(Z_c)
            for n in range(1, self.N):
                u0 = np.clip(U[:, :, 0], 1e-6, 1 - 1e-6)
                raw = (rng.uniform(0, 1, (M, T)) ** (-th / (1 + th)) - 1 + u0 ** (-th)) ** (-1 / th)
                u_n = np.clip(raw, 1e-6, 1 - 1e-6)
                alpha_blend = 0.65 if n == 1 else max(0.4 - (n - 2) * 0.1, 0.15)
                U[:, :, n] = alpha_blend * u_n + (1 - alpha_blend) * np.clip(U[:, :, n], 1e-6, 1 - 1e-6)
                mask = U[:, :, n] < 0.2
                U[:, :, n][mask] *= 0.8
            U = np.clip(U, 1e-6, 1 - 1e-6)
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        elif self.copula == "Gumbel":
            Z = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U = stats.norm.cdf(Z_c)
            th = max(self.theta, 1.001)
            alpha = 1 / th
            for n in range(self.N):
                u = np.clip(U[:, :, n], 1e-6, 1 - 1e-6)
                mask = u > 0.8
                u[mask] = np.clip(u[mask] ** alpha, 1e-6, 1 - 1e-6)
                U[:, :, n] = u
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        return innovations


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINED OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedLSMOptimizer:
    def __init__(self, tickers: list, ngarch_models: dict, copula_engine: CopulaEngine,
                 T: int, M: int, target_return: float, rf_monthly: float,
                 tx_cost: float = 0.005, max_single_weight: float = 0.30,
                 min_single_weight: float = 0.05, max_sector: float = 0.40,
                 div_penalty: float = 0.1, turnover_penalty: float = 0.05):
        self.tickers = tickers
        self.models = ngarch_models
        self.copula = copula_engine
        self.T = T
        self.M = M
        self.c = target_return
        self.rf = rf_monthly
        self.tx_cost = tx_cost
        self.max_single = max_single_weight
        self.min_single = min_single_weight
        self.max_sector = max_sector
        self.div_penalty = div_penalty
        self.turnover_penalty = turnover_penalty
        self.N = len(tickers)

    def _excess_return(self, R: np.ndarray) -> np.ndarray:
        return (R - self.rf) / (1 + self.rf)

    def _basis(self, R_bar: np.ndarray) -> np.ndarray:
        M = R_bar.shape[0]
        cols = [np.ones(M)]
        for n in range(self.N):
            cols.append(R_bar[:, n])
            cols.append(R_bar[:, n] ** 2)
        for n in range(self.N):
            for j in range(n + 1, self.N):
                cols.append(R_bar[:, n] * R_bar[:, j])
        return np.column_stack(cols)

    def _diversification_score(self, w: np.ndarray) -> float:
        return 1 - np.sum(w ** 2)

    def _sector_weights(self, w: np.ndarray) -> dict:
        sector_w = {}
        for i, tk in enumerate(self.tickers):
            s = BVC_ASSETS[tk]["sector"]
            sector_w[s] = sector_w.get(s, 0) + w[i]
        return sector_w

    def _objective_with_penalties(self, w: np.ndarray, mu0: np.ndarray, V0_inv: np.ndarray,
                                   prev_w: np.ndarray = None) -> float:
        var_obj = w @ V0_inv @ w
        div_score = self._diversification_score(w)
        div_pen = -self.div_penalty * div_score
        turnover_pen = 0
        if prev_w is not None:
            turnover = np.sum(np.abs(w - prev_w))
            turnover_pen = self.turnover_penalty * turnover
        return var_obj + div_pen + turnover_pen

    def _get_constraints(self) -> list:
        N = self.N
        constraints = []
        constraints.append(LinearConstraint(np.ones(N), 1.0, 1.0))
        
        sectors = list(set(BVC_ASSETS[t]["sector"] for t in self.tickers))
        for sec in sectors:
            sec_indices = [i for i, tk in enumerate(self.tickers) if BVC_ASSETS[tk]["sector"] == sec]
            if sec_indices:
                A_sec = np.zeros(N)
                for idx in sec_indices:
                    A_sec[idx] = 1.0
                constraints.append(LinearConstraint(A_sec, 0, self.max_sector))
        
        return constraints

    def _solve_weights(self, mu0: np.ndarray, V0_inv: np.ndarray, prev_w: np.ndarray = None) -> np.ndarray:
        N = self.N
        bounds = Bounds([self.min_single] * N, [self.max_single] * N)
        constraints = self._get_constraints()
        w0 = np.ones(N) / N
        
        result = minimize(
            lambda w: self._objective_with_penalties(w, mu0, V0_inv, prev_w),
            w0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if result.success:
            w = result.x
            return w / w.sum()
        else:
            return np.ones(N) / N

    def run(self, seed: int = 42) -> dict:
        N, T, M = self.N, self.T, self.M

        innovations = self.copula.sample_innovations(M, T, seed=seed)
        
        all_ret = np.zeros((N, M, T))
        for i, tk in enumerate(self.tickers):
            ret, _ = self.models[tk].simulate(T, M, innovations[i])
            all_ret[i] = ret

        ex_ret = np.stack([self._excess_return(all_ret[n]) for n in range(N)], axis=2)
        
        tau_list = []
        mu_list = []
        V_list = []

        for t in range(T - 1, -1, -1):
            R_t = ex_ret[:, t, :]
            basis = self._basis(R_t)
            
            mu_fit = np.zeros((M, N))
            for n in range(N):
                coef, _, _, _ = np.linalg.lstsq(basis, R_t[:, n], rcond=None)
                mu_fit[:, n] = basis @ coef
            
            V_fit = np.zeros((M, N, N))
            for n in range(N):
                for j in range(n, N):
                    y = R_t[:, n] * R_t[:, j]
                    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
                    V_fit[:, n, j] = basis @ coef
                    V_fit[:, j, n] = V_fit[:, n, j]
            
            tau_path = np.zeros(M)
            for m in range(M):
                Vm = V_fit[m] + 1e-9 * np.eye(N)
                mum = mu_fit[m]
                try:
                    Vi = np.linalg.pinv(Vm)
                    tau_path[m] = max(float(mum @ Vi @ mum), 0.0)
                except Exception:
                    pass
            
            tau_list.append(float(np.mean(tau_path)))
            mu_list.append(np.mean(mu_fit, axis=0))
            V_list.append(np.mean(V_fit, axis=0))
        
        tau_list.reverse()
        mu_list.reverse()
        V_list.reverse()
        
        E_tau = max(sum(tau_list), 1e-10)
        lam_T = -2.0 * (self.c / E_tau + 1.0)
        
        mu0 = mu_list[0]
        V0 = V_list[0] + 1e-8 * np.eye(N)
        V0_inv = np.linalg.pinv(V0)
        w_opt = self._solve_weights(mu0, V0_inv)
        
        dynamic_w = self._compute_dynamic_weights(mu_list, V_list, all_ret, lam_T, w_opt)
        
        min_var = abs(self.c ** 2 / max(E_tau - 1.0, 1e-6))
        opt_vol = float(np.sqrt(min_var))
        sharpe = (self.c - self.rf * 12) / max(opt_vol, 1e-8)
        achieved = float(np.prod(1 + np.mean(all_ret, axis=(0, 1))) - 1)
        div_score = self._diversification_score(w_opt)
        effective_n = 1 / np.sum(w_opt ** 2)
        sector_weights = self._sector_weights(w_opt)
        sector_warning = any(v > self.max_sector for v in sector_weights.values())
        concentration_warning = div_score < 0.7
        
        return {
            "weights": dict(zip(self.tickers, w_opt)),
            "weights_arr": w_opt,
            "tau_list": tau_list,
            "E_tau": E_tau,
            "lambda_T": lam_T,
            "min_var": min_var,
            "opt_vol": opt_vol,
            "sharpe": sharpe,
            "achieved": achieved,
            "dynamic_w": dynamic_w,
            "all_ret": all_ret,
            "diversification_score": div_score,
            "effective_number_assets": effective_n,
            "sector_weights": sector_weights,
            "sector_warning": sector_warning,
            "concentration_warning": concentration_warning,
        }

    def _compute_dynamic_weights(self, mu_list, V_list, all_ret, lam_T, w0):
        T, N = self.T, self.N
        dyn = np.zeros((T, N))
        dyn[0] = w0
        cum_wealth = 1.0
        
        for t in range(1, T):
            port_ret = float(np.mean(
                sum(dyn[t-1, n] * all_ret[n, :, t-1] for n in range(N))
            ))
            cum_wealth *= (1 + port_ret)
            mu_t = mu_list[min(t, len(mu_list)-1)]
            V_t = V_list[min(t, len(V_list)-1)] + 1e-8 * np.eye(N)
            Vi = np.linalg.pinv(V_t)
            w_raw = self._solve_weights(mu_t, Vi, dyn[t-1])
            dyn[t] = w_raw
        
        return dyn


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_dashboard_plot(weights: dict, sector_weights: dict, result: dict, tickers: list):
    """Create a comprehensive dashboard with multiple charts"""
    
    # Sort weights for bar chart
    sorted_items = sorted(weights.items(), key=lambda x: -x[1])
    ticker_list = [f"{t} — {BVC_ASSETS[t]['name'][:15]}" for t, _ in sorted_items]
    weight_vals = [w * 100 for _, w in sorted_items]
    colors = [BVC_ASSETS[t]["color"] for t, _ in sorted_items]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Optimal Portfolio Weights", "Sector Allocation", "Diversification Metrics",
            "Dynamic Weight Evolution", "Information Ratio τ_t", "Efficient Frontier"
        ),
        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. Bar chart - weights
    fig.add_trace(
        go.Bar(
            x=weight_vals, y=ticker_list, orientation='h',
            marker_color=colors, text=[f"{v:.1f}%" for v in weight_vals],
            textposition='outside', name="Weights",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Weight (%)", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    
    # 2. Pie chart - sector allocation
    sector_labels = list(sector_weights.keys())
    sector_values = [v for v in sector_weights.values()]
    sector_colors = [SECTOR_COLORS.get(s, PALETTE["ink3"]) for s in sector_labels]
    fig.add_trace(
        go.Pie(
            labels=sector_labels, values=sector_values,
            marker_colors=sector_colors, hole=0.45,
            textinfo="percent", textposition="auto",
            hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Indicator - diversification score
    div_score = result["diversification_score"]
    effective_n = result["effective_number_assets"]
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=div_score * 100,
            title={"text": f"Diversification Score<br>(higher = better)", "font": {"size": 12}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["ink3"]},
                "bar": {"color": PALETTE["teal"]},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": PALETTE["bg2"]},
                    {"range": [50, 70], "color": PALETTE["bg3"]},
                    {"range": [70, 100], "color": PALETTE["bg2"]}
                ],
                "threshold": {
                    "line": {"color": PALETTE["gold"], "width": 2},
                    "thickness": 0.75, "value": 70
                }
            },
            number={"suffix": "%", "font": {"size": 24, "color": PALETTE["teal"]}},
            delta={"reference": 70, "valueformat": ".0f"}
        ),
        row=1, col=3
    )
    
    # Add effective assets as annotation
    fig.add_annotation(
        text=f"Effective N = {effective_n:.1f} assets",
        x=0.5, y=0.3, xref="x3 domain", yref="y3 domain",
        showarrow=False, font=dict(size=11, color=PALETTE["ink3"])
    )
    
    # 4. Dynamic weight evolution
    dyn_w = result["dynamic_w"]
    months = list(range(1, result["dynamic_w"].shape[0] + 1))
    asset_colors = [BVC_ASSETS[t]["color"] for t in tickers]
    for i, tk in enumerate(tickers):
        fig.add_trace(
            go.Scatter(
                x=months, y=dyn_w[:, i] * 100,
                mode='lines', name=tk,
                line=dict(color=asset_colors[i], width=2),
                stackgroup='one',
                groupnorm='percent',
                hovertemplate=f"{tk}: %{{y:.1f}}%<extra></extra>"
            ),
            row=2, col=1
        )
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Weight (%)", row=2, col=1, range=[0, 100])
    
    # 5. Tau series
    tau_list = result["tau_list"]
    fig.add_trace(
        go.Scatter(
            x=months[:len(tau_list)], y=tau_list,
            mode='lines+markers', name="τ_t",
            line=dict(color=PALETTE["gold"], width=2.5),
            marker=dict(size=5, color=PALETTE["teal"]),
            fill='tozeroy', fillcolor=f"rgba(201,168,76,0.1)",
            hovertemplate="τ_t = %{y:.4f}<extra></extra>"
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Time Step t", row=2, col=2)
    fig.update_yaxes(title_text="τ_t = μ'V⁻¹μ", row=2, col=2)
    
    # 6. Efficient frontier
    E_tau = result["E_tau"]
    ret_rng = np.linspace(0.05, 0.30, 50)
    vols = [np.sqrt(max(r ** 2 / max(E_tau - 1, 0.001), 1e-6)) * 100 for r in ret_rng]
    
    fig.add_trace(
        go.Scatter(
            x=vols, y=ret_rng * 100,
            mode='lines', name="Efficient Frontier",
            line=dict(color=PALETTE["teal"], width=3),
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
        ),
        row=2, col=3
    )
    
    # Current portfolio point
    current_vol = result["opt_vol"] * 100
    current_ret = result["c"] * 100
    fig.add_trace(
        go.Scatter(
            x=[current_vol], y=[current_ret],
            mode='markers', name="Your Portfolio",
            marker=dict(size=15, symbol="star-diamond", color=PALETTE["gold"],
                        line=dict(width=2, color=PALETTE["ink"])),
            hovertemplate=f"Target: {current_ret:.1f}%<br>Vol: {current_vol:.1f}%<extra></extra>"
        ),
        row=2, col=3
    )
    
    # Achieved return point
    achieved = result["achieved"] * 100
    if abs(achieved - current_ret) > 0.5:
        fig.add_trace(
            go.Scatter(
                x=[current_vol], y=[achieved],
                mode='markers', name="Achieved",
                marker=dict(size=12, symbol="circle", color=PALETTE["rust"],
                            line=dict(width=1, color=PALETTE["ink"])),
                hovertemplate=f"Achieved: {achieved:.1f}%<extra></extra>"
            ),
            row=2, col=3
        )
    
    fig.update_xaxes(title_text="Volatility σ (%)", row=2, col=3)
    fig.update_yaxes(title_text="Expected Return (%)", row=2, col=3)
    
    # Overall layout
    fig.update_layout(
        height=900,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Barlow", size=11, color=PALETTE["ink"]),
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        margin=dict(l=40, r=40, t=80, b=40),
    )
    
    # Update subplot titles styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family="Cormorant Garamond", size=13, color=PALETTE["ink"])
    
    return fig


def create_copula_comparison_chart(comparison_results: dict) -> go.Figure:
    """Create a bar chart comparing copula performance"""
    
    copulas = list(comparison_results.keys())
    sharpes = [comparison_results[c]["sharpe"] for c in copulas]
    vols = [comparison_results[c]["opt_vol"] * 100 for c in copulas]
    returns = [comparison_results[c]["achieved"] * 100 for c in copulas]
    
    colors = [PALETTE["teal"] if c == "Clayton" else PALETTE["ink3"] for c in copulas]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sharpe Ratio", "Volatility (%)", "Achieved Return (%)"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )
    
    fig.add_trace(
        go.Bar(x=copulas, y=sharpes, marker_color=colors, text=[f"{s:.3f}" for s in sharpes],
               textposition="outside", name="Sharpe"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=copulas, y=vols, marker_color=colors, text=[f"{v:.1f}%" for v in vols],
               textposition="outside", name="Volatility"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=copulas, y=returns, marker_color=colors, text=[f"{r:.1f}%" for r in returns],
               textposition="outside", name="Return"),
        row=1, col=3
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Barlow", size=11, color=PALETTE["ink"]),
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        margin=dict(l=40, r=40, t=60, b=30),
    )
    
    return fig


def create_radar_chart(weights: dict, sector_weights: dict) -> go.Figure:
    """Create a radar chart for sector exposure"""
    
    all_sectors = list(SECTOR_COLORS.keys())
    sector_exposures = [sector_weights.get(s, 0) * 100 for s in all_sectors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=sector_exposures,
        theta=all_sectors,
        fill='toself',
        name='Portfolio Exposure',
        line=dict(color=PALETTE["teal"], width=2),
        fillcolor=f"rgba(28,79,74,0.3)",
        marker=dict(color=PALETTE["gold"], size=4)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(50, max(sector_exposures) + 5)],
                          tickfont=dict(family="DM Mono", size=9),
                          gridcolor=PALETTE["border"]),
            angularaxis=dict(tickfont=dict(family="Barlow", size=10, color=PALETTE["ink"]),
                            gridcolor=PALETTE["border"])
        ),
        title=dict(text="Sector Exposure Radar", font=dict(family="Cormorant Garamond", size=16)),
        height=400,
        template="plotly_white",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
    )
    
    return fig


def create_risk_return_scatter(assets_data: dict, weights: dict) -> go.Figure:
    """Create a risk-return scatter plot for individual assets"""
    
    # Calculate asset returns and volatilities (simplified for visualization)
    assets = list(assets_data.keys())
    returns = [assets_data[a]["delta"] * 100 for a in assets]
    volatilities = [np.sqrt(assets_data[a]["beta0"] * 252) * 100 for a in assets]
    colors = [assets_data[a]["color"] for a in assets]
    sizes = [assets_data[a]["mcap"] / 5 for a in assets]
    
    fig = go.Figure()
    
    # Add asset points
    fig.add_trace(go.Scatter(
        x=volatilities, y=returns,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=1, color=PALETTE["white"]), opacity=0.8),
        text=assets,
        textposition="top center",
        textfont=dict(family="DM Mono", size=9, color=PALETTE["ink"]),
        hovertemplate="<b>%{text}</b><br>Return: %{y:.1f}%<br>Volatility: %{x:.1f}%<extra></extra>"
    ))
    
    # Add efficient frontier marker for portfolio
    port_return = sum(weights[a] * assets_data[a]["delta"] * 100 for a in assets if a in weights)
    port_vol = np.sqrt(sum(weights[a]**2 * (np.sqrt(assets_data[a]["beta0"]*252)*100)**2 for a in assets if a in weights))
    
    fig.add_trace(go.Scatter(
        x=[port_vol], y=[port_return],
        mode='markers',
        marker=dict(size=20, symbol="star", color=PALETTE["gold"], line=dict(width=2, color=PALETTE["ink"])),
        name="Your Portfolio",
        hovertemplate=f"Portfolio: {port_return:.1f}% / {port_vol:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="Asset Risk-Return Profile", font=dict(family="Cormorant Garamond", size=16)),
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=400,
        template="plotly_white",
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        hovermode="closest",
    )
    
    fig.update_xaxes(gridcolor=PALETTE["border"], gridwidth=0.5)
    fig.update_yaxes(gridcolor=PALETTE["border"], gridwidth=0.5)
    
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEADER SECTION
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="header-eyebrow">Bourse des Valeurs de Casablanca</div>
    <div class="header-title">Dynamic Portfolio Optimization<br>for the <em>Moroccan Market</em></div>
    <div class="header-sub">
        NGarch volatility marginals · Copula dependence structures · Longstaff-Schwartz LSM
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Asset Universe")
    
    default_assets = ["MASI", "ATW", "IAM", "OCP", "TQM"]
    selected_tickers = st.multiselect(
        "Select BVC Constituents",
        options=list(BVC_ASSETS.keys()),
        default=default_assets,
        format_func=lambda t: f"{t} — {BVC_ASSETS[t]['name']}",
        help="Choose 3-8 assets for optimal diversification"
    )
    
    st.markdown("### 🎯 Optimization Parameters")
    target_pct = st.slider("Target Annual Return (%)", 5, 25, 12, 1)
    horizon = st.slider("Investment Horizon (months)", 6, 36, 12, 3)
    
    st.markdown("### 💰 Risk Parameters")
    rf_pct = st.slider("Risk-Free Rate (% p.a.)", 1.0, 6.0, 2.75, 0.25)
    tx_cost_pct = st.slider("Transaction Costs (% per trade)", 0.0, 0.75, 0.50, 0.05)
    
    st.markdown("### ⚖️ Constraint Settings")
    col1, col2 = st.columns(2)
    with col1:
        max_single = st.slider("Max Single Asset", 20, 40, 30, 5) / 100
    with col2:
        min_single = st.slider("Min Single Asset", 3, 15, 5, 1) / 100
    
    max_sector = st.slider("Max Sector Exposure (%)", 30, 60, 40, 5) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 0.3, 0.1, 0.05)
    
    st.markdown("### 🔬 Simulation")
    n_paths = st.select_slider("Monte Carlo Paths", options=[1000, 2000, 5000, 10000], value=5000)
    
    copula_choice = st.selectbox("Copula Model", ["Clayton", "Student-t", "Gaussian", "Gumbel"], index=0)
    
    if copula_choice in ["Clayton", "Gumbel"]:
        theta = st.slider("Dependence Parameter θ", 0.5, 4.0, 1.5, 0.1)
    else:
        theta = 1.5
    
    if copula_choice == "Student-t":
        nu = st.slider("Degrees of Freedom ν", 3, 20, 5, 1)
    else:
        nu = 5
    
    st.markdown("---")
    
    seed = st.number_input("Random Seed", value=42, step=1)
    run_button = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

if run_button and len(selected_tickers) >= 3:
    
    rf_monthly = rf_pct / 100 / 12
    target_return = target_pct / 100
    
    # Build models
    ngarch_models = {tk: NGarchModel(BVC_ASSETS[tk], rf_monthly) for tk in selected_tickers}
    copula_engine = CopulaEngine(copula_choice, len(selected_tickers), theta=theta, nu=nu)
    
    optimizer = ConstrainedLSMOptimizer(
        selected_tickers, ngarch_models, copula_engine,
        horizon, n_paths, target_return, rf_monthly,
        tx_cost=tx_cost_pct / 100,
        max_single_weight=max_single,
        min_single_weight=min_single,
        max_sector=max_sector,
        div_penalty=div_penalty,
    )
    
    with st.spinner("Running LSM optimization... This may take a few seconds."):
        result = optimizer.run(seed=int(seed))
    
    # ─────────────────────────────────────────────────────────────────────────
    # METRICS ROW
    # ─────────────────────────────────────────────────────────────────────────
    
    st.markdown('<div class="sec-eyebrow">Optimization Complete</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Portfolio <em>Results</em></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Optimal Volatility</div>
            <div class="mc-value">{result['opt_vol']*100:.2f}%</div>
            <div class="mc-sub">Annualised σ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Sharpe Ratio</div>
            <div class="mc-value">{result['sharpe']:.3f}</div>
            <div class="mc-sub">Risk-adjusted return</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "green-text" if result['achieved']*100 >= target_pct else "red-text"
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Achieved Return</div>
            <div class="mc-value {color}">{result['achieved']*100:.2f}%</div>
            <div class="mc-sub">Target: {target_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Diversification Score</div>
            <div class="mc-value">{result['diversification_score']:.3f}</div>
            <div class="mc-sub">1 = perfectly diversified</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Effective Assets</div>
            <div class="mc-value">{result['effective_number_assets']:.1f}</div>
            <div class="mc-sub">out of {len(selected_tickers)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning messages
    if result['concentration_warning']:
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ Portfolio concentration is moderate. Consider increasing diversification penalty or reducing max single asset weight.
        </div>
        """, unsafe_allow_html=True)
    
    if result['sector_warning']:
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ Sector exposure limit reached. Some sectors are at maximum allowed allocation.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider-gold">', unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DASHBOARD CHART
    # ─────────────────────────────────────────────────────────────────────────
    
    st.plotly_chart(
        create_dashboard_plot(result['weights'], result['sector_weights'], result, selected_tickers),
        use_container_width=True
    )
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DETAILED WEIGHTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    
    st.markdown('<div class="sec-eyebrow">Allocation Detail</div>', unsafe_allow_html=True)
    
    table_rows = ""
    for sym, w in sorted(result['weights'].items(), key=lambda x: -x[1]):
        a = BVC_ASSETS[sym]
        persist = a["beta1"] + 0.09 * (1 + a["kappa"] ** 2)
        lever_str = "Strong" if abs(a["kappa"]) > 0.4 else "Moderate" if abs(a["kappa"]) > 0.2 else "Weak"
        table_rows += f"""
        <tr>
            <td class="mono teal-text">{sym}</td>
            <td>{a['name']}</td>
            <td class="muted">{a['sector']}</td>
            <td class="gold-text mono">{w*100:.2f}%</td>
            <td class="mono">{a['beta1']:.3f}</td>
            <td class="mono">{a['kappa']:.3f}</td>
            <td class="muted">{lever_str}</td>
            <td class="mono">{persist:.4f}</td>
        </tr>
        """
    
    st.markdown(f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Ticker</th><th>Name</th><th>Sector</th><th>Weight</th>
                <th>β₁</th><th>κ</th><th>Leverage</th><th>Persistence</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ADDITIONAL CHARTS ROW
    # ─────────────────────────────────────────────────────────────────────────
    
    col_radar, col_scatter = st.columns(2)
    
    with col_radar:
        st.plotly_chart(create_radar_chart(result['weights'], result['sector_weights']), use_container_width=True)
    
    with col_scatter:
        st.plotly_chart(create_risk_return_scatter(BVC_ASSETS, result['weights']), use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # COPULA COMPARISON (optional button)
    # ─────────────────────────────────────────────────────────────────────────
    
    if st.button("📊 Compare All Copulas", use_container_width=True):
        with st.spinner("Running comparison across all copulas..."):
            comparison_results = {}
            for cop in ["Clayton", "Student-t", "Gaussian", "Gumbel"]:
                engine = CopulaEngine(cop, len(selected_tickers), theta=1.5, nu=5)
                opt = ConstrainedLSMOptimizer(
                    selected_tickers, ngarch_models, engine,
                    horizon, min(n_paths, 2000), target_return, rf_monthly,
                    tx_cost=tx_cost_pct/100, max_single_weight=max_single,
                    min_single_weight=min_single, max_sector=max_sector,
                    div_penalty=div_penalty
                )
                comparison_results[cop] = opt.run(seed=int(seed))
        
        st.plotly_chart(create_copula_comparison_chart(comparison_results), use_container_width=True)
        
        # Display comparison table
        comp_data = []
        for cop, res in comparison_results.items():
            comp_data.append({
                "Copula": cop,
                "Return": f"{res['achieved']*100:.2f}%",
                "Volatility": f"{res['opt_vol']*100:.2f}%",
                "Sharpe": f"{res['sharpe']:.3f}",
                "Div Score": f"{res['diversification_score']:.3f}",
                "Eff N": f"{res['effective_number_assets']:.1f}"
            })
        
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

elif run_button and len(selected_tickers) < 3:
    st.markdown(f"""
    <div class="alert-warning">
        ⚠️ Please select at least 3 assets. Current selection: {len(selected_tickers)} asset(s).<br>
        <strong>Recommended:</strong> MASI + ATW + IAM + OCP for cross-sector diversification.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="sec-eyebrow">Ready</div>
    <div class="sec-title">Configure your portfolio in the <em>sidebar</em></div>
    <div class="sec-desc">
        Select BVC assets, set your target return, and click <strong>Run Optimizer</strong>.
        The optimizer will compute the optimal dynamic weights using NGarch volatility,
        copula dependence, and LSM backward induction.
    </div>
    """, unsafe_allow_html=True)
    
    # Sample portfolio suggestion
    st.markdown(f"""
    <div class="alert-info">
        <strong>📌 Recommended starting portfolio:</strong><br>
        • <strong>MASI</strong> — Index benchmark<br>
        • <strong>ATW</strong> — Largest Moroccan bank<br>
        • <strong>IAM</strong> — Defensive telecom<br>
        • <strong>OCP</strong> — Phosphate commodity exposure<br>
        • <strong>TQM</strong> — Energy diversifier<br><br>
        <span class="muted">This provides cross-sector diversification across Morocco's key economic drivers.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="muted" style="text-align: center; font-size: 0.7rem;">
        Based on Vaillancourt &amp; Watier (2005) · NGarch + Copula + LSM<br>
        Data: Bourse des Valeurs de Casablanca · Risk-free: Bank Al-Maghrib Bons du Trésor
    </div>
    """, unsafe_allow_html=True)
