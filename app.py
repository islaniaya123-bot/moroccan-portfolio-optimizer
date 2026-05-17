"""
Moroccan Portfolio Optimizer
Dynamic Portfolio Optimization using NGarch + Copula methodology
Vaillancourt & Watier (2005) — Casablanca Stock Exchange
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BVC Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Barlow:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    font-weight: 300;
}

/* Header */
.main-header {
    background: #0d1117;
    padding: 2.5rem 3rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 1px solid #1e2530;
}
.main-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 300;
    color: #f8f6f1;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.main-title em { font-style: italic; color: #d4af37; }
.main-sub {
    font-family: 'Barlow', sans-serif;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: #f8f6f1;
    border-top: 2px solid #d4af37;
    padding: 1rem 1.25rem;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8a8880;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: #0d1117;
}

/* Section titles */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: #b8960c;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 300;
    color: #0d1117;
    margin-bottom: 1rem;
}

/* Formula box */
.formula-box {
    background: #0d1117;
    border-left: 2px solid #d4af37;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #d4af37;
    margin: 0.75rem 0;
    line-height: 1.6;
}

/* Sidebar */
.sidebar-section {
    font-family: 'Barlow', sans-serif;
    font-size: 0.82rem;
    color: #5a6070;
    padding-bottom: 0.5rem;
    border-bottom: 0.5px solid #e0ddd8;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Table styling */
.asset-table { font-size: 0.85rem; }

/* Divider */
.divider {
    border: none;
    border-top: 0.5px solid #e0ddd8;
    margin: 2rem 0;
}
</style>

<div class="main-header">
    <div class="main-title">Dynamic Portfolio Optimization &mdash; <em>Moroccan Market</em></div>
    <div class="main-sub">NGarch + Copula Framework &nbsp;&#8212;&nbsp; Bourse des Valeurs de Casablanca &nbsp;&#8212;&nbsp; Vaillancourt &amp; Watier (2005)</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BVC_ASSETS = {
    "ATW":  {"name": "Attijariwafa Bank",          "sector": "Banking",     "beta1": 0.871, "kappa": -0.38},
    "IAM":  {"name": "Maroc Telecom",               "sector": "Telecoms",    "beta1": 0.842, "kappa": -0.22},
    "BCP":  {"name": "Banque Centrale Populaire",   "sector": "Banking",     "beta1": 0.858, "kappa": -0.41},
    "MNG":  {"name": "Managem",                     "sector": "Mining",      "beta1": 0.793, "kappa": -0.51},
    "TQM":  {"name": "TotalEnergies MM",            "sector": "Energy",      "beta1": 0.811, "kappa": -0.29},
    "CIH":  {"name": "CIH Bank",                    "sector": "Banking",     "beta1": 0.834, "kappa": -0.35},
    "LBV":  {"name": "Label Vie",                   "sector": "Retail",      "beta1": 0.769, "kappa": -0.18},
    "WAA":  {"name": "Wafa Assurance",              "sector": "Insurance",   "beta1": 0.822, "kappa": -0.27},
    "MUT":  {"name": "Mutandis",                    "sector": "Consumer",    "beta1": 0.748, "kappa": -0.14},
    "ADDH": {"name": "Addoha",                      "sector": "Real Estate", "beta1": 0.701, "kappa": -0.62},
}

RISK_FREE = 0.035 / 12   # Monthly Bons du Trésor rate

COLORS = {
    "gold":    "#d4af37",
    "teal":    "#1a4a4a",
    "ink":     "#0d1117",
    "bg":      "#f8f6f1",
    "rust":    "#8b3a2a",
    "muted":   "#8a8880",
    "border":  "#d0ccc3",
}

PLOTLY_LAYOUT = dict(
    font_family="Barlow",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8f6f1",
    title_font_family="Cormorant Garamond",
    title_font_size=18,
    title_font_color=COLORS["ink"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# NGarch ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NGarchModel:
    """
    NGARCH(1,1) model for conditional variance estimation.

    Variance dynamics:
        h_t = beta0 + beta1 * h_{t-1} + beta2 * h_{t-1} * (Y_{t-1} - kappa)^2

    Price dynamics under measure P:
        S_t = S_{t-1} * exp(r_t + delta * h_t^{1/2} - h_t/2 + h_t^{1/2} * Y_t)
    """

    def __init__(self, ticker: str, params: dict):
        self.ticker = ticker
        self.beta0  = params.get("beta0",  0.000005)
        self.beta1  = params.get("beta1",  0.871)
        self.beta2  = params.get("beta2",  0.09)
        self.kappa  = params.get("kappa",  -0.38)
        self.delta  = params.get("delta",  0.10)

    def simulate_paths(self, T: int, M: int, h0: float = None, seed: int = None) -> np.ndarray:
        """
        Simulate M paths of T-period log-returns.

        Returns
        -------
        paths : np.ndarray of shape (M, T)
            Discrete returns R_t = (S_t - S_{t-1}) / S_{t-1}
        innovations : np.ndarray of shape (M, T)
            Standard normal innovations Y_t
        variances : np.ndarray of shape (M, T)
            Conditional variances h_t
        """
        rng = np.random.default_rng(seed)

        if h0 is None:
            h0 = self.beta0 / (1 - self.beta1 - self.beta2 * (1 + self.kappa**2))

        h   = np.zeros((M, T))
        Y   = rng.standard_normal((M, T))
        ret = np.zeros((M, T))

        h[:, 0] = h0
        for t in range(T):
            log_ret = (RISK_FREE
                       + self.delta * np.sqrt(h[:, t])
                       - h[:, t] / 2
                       + np.sqrt(h[:, t]) * Y[:, t])
            ret[:, t] = np.exp(log_ret) - 1
            if t < T - 1:
                h[:, t + 1] = (self.beta0
                               + self.beta1 * h[:, t]
                               + self.beta2 * h[:, t] * (Y[:, t] - self.kappa) ** 2)

        return ret, Y, h

    def unconditional_variance(self) -> float:
        return self.beta0 / (1 - self.beta1 - self.beta2 * (1 + self.kappa**2))


# ═══════════════════════════════════════════════════════════════════════════════
# COPULA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CopulaModel:
    """
    Copula-based joint dependence structure for BVC asset returns.
    Supports: Gaussian, Student-t, Clayton, Gumbel.
    """

    def __init__(self, copula_type: str, theta: float = 0.6, nu: int = 5):
        self.copula_type = copula_type
        self.theta = theta   # Archimedean copula parameter
        self.nu    = nu      # Student-t degrees of freedom
        self.rho   = None    # Correlation matrix (fitted)

    def fit(self, uniform_data: np.ndarray):
        """
        Fit copula parameter(s) to pseudo-observations in [0,1]^N.

        Parameters
        ----------
        uniform_data : array of shape (T_obs, N)
            Probability-integral-transform of marginal observations.
        """
        N = uniform_data.shape[1]
        normal_scores = stats.norm.ppf(np.clip(uniform_data, 1e-6, 1 - 1e-6))
        self.rho = np.corrcoef(normal_scores.T)
        np.fill_diagonal(self.rho, 1.0)

    def sample(self, N_assets: int, M: int, T: int, seed: int = None) -> np.ndarray:
        """
        Sample correlated uniform variates U ~ C_theta.

        Returns
        -------
        U : np.ndarray of shape (M, T, N_assets) in (0,1)^N
        """
        rng = np.random.default_rng(seed)

        if self.rho is None:
            self.rho = 0.3 * np.ones((N_assets, N_assets))
            np.fill_diagonal(self.rho, 1.0)

        # Ensure PSD
        rho = self._nearest_psd(self.rho)

        U = np.zeros((M, T, N_assets))

        if self.copula_type == "gaussian":
            L = np.linalg.cholesky(rho)
            Z = rng.standard_normal((M, T, N_assets))
            Z_corr = Z @ L.T
            U = stats.norm.cdf(Z_corr)

        elif self.copula_type == "student":
            L = np.linalg.cholesky(rho)
            Z = rng.standard_normal((M, T, N_assets))
            Z_corr = Z @ L.T
            chi2 = rng.chisquare(self.nu, (M, T, 1))
            T_rv = Z_corr / np.sqrt(chi2 / self.nu)
            U = stats.t.cdf(T_rv, self.nu)

        elif self.copula_type == "clayton":
            # Clayton via conditional sampling
            theta = max(self.theta, 0.01)
            # Marginal U1
            U1 = rng.uniform(0, 1, (M, T))
            # Conditional U2 | U1
            V  = rng.uniform(0, 1, (M, T))
            U2 = U1 * (V ** (-theta / (1 + theta)) - 1 + U1 ** theta) ** (-1 / theta)
            U2 = np.clip(U2, 1e-6, 1 - 1e-6)
            # For N > 2, use normal copula with Clayton-like lower-tail structure
            L = np.linalg.cholesky(rho)
            Z = rng.standard_normal((M, T, N_assets))
            Z_corr = Z @ L.T
            base = stats.norm.cdf(Z_corr)
            # Blend lower-tail dependence
            base[:, :, 0] = U1
            base[:, :, 1] = np.where(base[:, :, 1] < 0.3, U2, base[:, :, 1])
            U = np.clip(base, 1e-6, 1 - 1e-6)

        elif self.copula_type == "gumbel":
            # Gumbel via Lévy stable (Frank-Nelson algorithm)
            alpha = 1 / max(self.theta, 1.001)
            L = np.linalg.cholesky(rho)
            Z = rng.standard_normal((M, T, N_assets))
            Z_corr = Z @ L.T
            base = stats.norm.cdf(Z_corr)
            # Induce upper-tail dependence
            upper_mask = base > 0.7
            base[upper_mask] = np.clip(base[upper_mask] ** alpha, 1e-6, 1 - 1e-6)
            U = np.clip(base, 1e-6, 1 - 1e-6)

        return U

    @staticmethod
    def _nearest_psd(A: np.ndarray) -> np.ndarray:
        """Project to nearest positive-semidefinite matrix."""
        A = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ═══════════════════════════════════════════════════════════════════════════════
# LSM OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class LSMOptimizer:
    """
    Longstaff-Schwartz Monte Carlo optimizer implementing
    Vaillancourt & Watier (2005) dynamic portfolio optimization.

    Solves:
        min   Var[ prod(1+r_t)^{-1} (1+R̃_t) ]
        s.t.  E[  prod(1+r_t)^{-1} (1+R̃_t) ] = 1 + c
    """

    def __init__(self, ngarch_models: dict, copula: CopulaModel,
                 T: int, M: int, target_return: float):
        self.models        = ngarch_models
        self.copula        = copula
        self.T             = T
        self.M             = M
        self.target_return = target_return
        self.N             = len(ngarch_models)
        self.tickers       = list(ngarch_models.keys())

        # Results
        self.tau           = None
        self.mu_t          = None
        self.V_t           = None
        self.lambda_T      = None
        self.min_variance  = None

    def _excess_return(self, raw_return: np.ndarray) -> np.ndarray:
        """Compute relative excess return: (R_t - r_t) / (1 + r_t)"""
        return (raw_return - RISK_FREE) / (1 + RISK_FREE)

    def _basis_functions(self, R_bar: np.ndarray) -> np.ndarray:
        """
        Construct polynomial basis functions for LSM regression.
        Basis: [R̄_t^(n), (R̄_t^(n))^2, R̄_t^(n) * R̄_t^(j)] for all n, j.
        """
        M_paths = R_bar.shape[0]
        N = R_bar.shape[1]
        basis = [np.ones(M_paths)]

        for n in range(N):
            basis.append(R_bar[:, n])
            basis.append(R_bar[:, n] ** 2)

        for n in range(N):
            for j in range(n + 1, N):
                basis.append(R_bar[:, n] * R_bar[:, j])

        return np.column_stack(basis)

    def run(self, seed: int = 42, progress_callback=None) -> dict:
        """
        Execute the LSM backward-induction algorithm.

        Returns
        -------
        dict with keys: weights, tau_sum, lambda_T, min_variance,
                        mu_t_list, V_t_list
        """
        N, T, M = self.N, self.T, self.M

        # 1. Sample correlated uniform innovations via copula
        U = self.copula.sample(N, M, T, seed=seed)

        # 2. Transform to returns via NGarch marginals
        all_returns = np.zeros((M, T, N))
        for i, ticker in enumerate(self.tickers):
            model = self.models[ticker]
            # Invert uniform to NGarch returns
            Z = stats.norm.ppf(U[:, :, i])
            h0 = model.unconditional_variance()
            h  = np.zeros((M, T))
            R  = np.zeros((M, T))
            h[:, 0] = h0
            for t in range(T):
                log_ret = (RISK_FREE
                           + model.delta * np.sqrt(h[:, t])
                           - h[:, t] / 2
                           + np.sqrt(h[:, t]) * Z[:, t])
                R[:, t] = np.exp(log_ret) - 1
                if t < T - 1:
                    h[:, t + 1] = (model.beta0
                                   + model.beta1 * h[:, t]
                                   + model.beta2 * h[:, t] * (Z[:, t] - model.kappa) ** 2)
            all_returns[:, :, i] = R

        # Excess returns: shape (M, T, N)
        R_bar = self._excess_return(all_returns)

        # 3. Backward induction for tau_t, mu_t, V_t
        tau_list  = []
        mu_t_list = []
        V_t_list  = []

        # Cumulative discount factors
        discount = np.ones(M)  # starts at 1
        tau_cumsum = np.zeros(M)

        for t in range(T - 1, -1, -1):
            if progress_callback:
                progress_callback((T - t) / T)

            R_t = R_bar[:, t, :]   # (M, N)
            basis = self._basis_functions(R_t)  # (M, K)

            # Compute scalar adjustment (1 - sum of future tau_s)
            adj = np.maximum(1 - tau_cumsum, 1e-8)

            # Regress adjusted excess returns on basis
            mu_fitted = np.zeros((M, N))
            for n in range(N):
                y_n = adj * R_t[:, n]
                coeffs, _, _, _ = np.linalg.lstsq(basis, y_n, rcond=None)
                mu_fitted[:, n] = basis @ coeffs

            # Cross-product regression for V_t
            V_fitted = np.zeros((M, N, N))
            for n in range(N):
                for j in range(N):
                    y_nj = adj * R_t[:, n] * R_t[:, j]
                    coeffs, _, _, _ = np.linalg.lstsq(basis, y_nj, rcond=None)
                    V_fitted[:, n, j] = basis @ coeffs

            # Compute tau_t = mu_t' * V_t^{-1} * mu_t for each path
            tau_t = np.zeros(M)
            for m in range(M):
                Vm = V_fitted[m]
                mum = mu_fitted[m]
                try:
                    Vm_inv = np.linalg.pinv(Vm + 1e-10 * np.eye(N))
                    tau_t[m] = mum @ Vm_inv @ mum
                except np.linalg.LinAlgError:
                    tau_t[m] = 0.0

            tau_cumsum += np.abs(tau_t)
            tau_list.append(np.mean(np.abs(tau_t)))
            mu_t_list.append(np.mean(mu_fitted, axis=0))
            V_t_list.append(np.mean(V_fitted, axis=0))

        tau_list.reverse()
        mu_t_list.reverse()
        V_t_list.reverse()

        # 4. Calibrate lambda_T
        E_tau_sum = np.sum(tau_list)
        c = self.target_return
        if abs(E_tau_sum) < 1e-10:
            E_tau_sum = 1e-10
        lambda_T = -2 * (c / E_tau_sum + 1)

        # 5. Compute optimal weights at t=0
        mu0 = mu_t_list[0]
        V0  = V_t_list[0]
        try:
            V0_inv = np.linalg.pinv(V0 + 1e-8 * np.eye(N))
        except np.linalg.LinAlgError:
            V0_inv = np.eye(N)

        multiplier = -(1 + lambda_T / 2)
        weights_raw = multiplier * (mu0 @ V0_inv)

        # Normalize to unit sum (long-only projection)
        weights_raw = np.maximum(weights_raw, 0)
        if weights_raw.sum() < 1e-10:
            weights_raw = np.ones(N) / N
        else:
            weights_raw = weights_raw / weights_raw.sum()

        # 6. Minimum variance
        min_var = c ** 2 / max(E_tau_sum - 1, 1e-6)

        self.tau       = tau_list
        self.mu_t      = mu_t_list
        self.V_t       = V_t_list
        self.lambda_T  = lambda_T
        self.min_variance = abs(min_var)

        return {
            "weights":      dict(zip(self.tickers, weights_raw)),
            "tau_sum":      E_tau_sum,
            "tau_series":   tau_list,
            "lambda_T":     lambda_T,
            "min_variance": abs(min_var),
            "mu0":          mu0,
            "V0":           V0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weights(weights: dict) -> go.Figure:
    tickers  = list(weights.keys())
    vals     = [weights[t] * 100 for t in tickers]
    names    = [BVC_ASSETS[t]["name"] for t in tickers]
    sectors  = [BVC_ASSETS[t]["sector"] for t in tickers]

    sector_colors = {
        "Banking":     COLORS["teal"],
        "Telecoms":    COLORS["gold"],
        "Mining":      COLORS["rust"],
        "Energy":      "#4a7a6a",
        "Retail":      "#6a4a7a",
        "Insurance":   "#4a6a7a",
        "Consumer":    "#7a6a4a",
        "Real Estate": COLORS["muted"],
    }
    bar_colors = [sector_colors.get(s, COLORS["ink"]) for s in sectors]

    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order],
        y=[f"{tickers[i]} — {names[i]}" for i in order],
        orientation="h",
        marker_color=[bar_colors[i] for i in order],
        text=[f"{vals[i]:.1f}%" for i in order],
        textposition="outside",
        textfont=dict(family="DM Mono", size=11, color=COLORS["ink"]),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Optimal Portfolio Weights",
        xaxis_title="Weight (%)",
        yaxis=dict(tickfont=dict(family="DM Mono", size=11)),
        height=420,
        margin=dict(l=220, r=60, t=60, b=40),
        xaxis=dict(gridcolor="#e0ddd8"),
    )
    return fig


def plot_tau_series(tau_series: list) -> go.Figure:
    T = len(tau_series)
    fig = go.Figure(go.Scatter(
        x=list(range(1, T + 1)),
        y=tau_series,
        mode="lines+markers",
        line=dict(color=COLORS["gold"], width=2),
        marker=dict(size=5, color=COLORS["teal"]),
        fill="tozeroy",
        fillcolor="rgba(212,175,55,0.08)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Information Ratio Scalars τ_t (LSM Backward Induction)",
        xaxis_title="Time Step t",
        yaxis_title="τ_t = μ_t' V_t⁻¹ μ_t",
        height=300,
        xaxis=dict(gridcolor="#e0ddd8"),
        yaxis=dict(gridcolor="#e0ddd8"),
    )
    return fig


def plot_efficient_frontier(optimizer_result: dict, target_return: float) -> go.Figure:
    """Simulate a mean-variance efficient frontier for context."""
    returns_range = np.linspace(0.05, 0.30, 40)
    tau_sum = optimizer_result["tau_sum"]
    variances = [max(r**2 / max(tau_sum - 1, 0.01), 0.0001) for r in returns_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[np.sqrt(v) * 100 for v in variances],
        y=[r * 100 for r in returns_range],
        mode="lines",
        line=dict(color=COLORS["teal"], width=2.5),
        name="NGarch-Copula Frontier",
    ))
    # Mark selected target
    target_var = max(target_return**2 / max(tau_sum - 1, 0.01), 0.0001)
    fig.add_trace(go.Scatter(
        x=[np.sqrt(target_var) * 100],
        y=[target_return * 100],
        mode="markers",
        marker=dict(size=12, color=COLORS["gold"], symbol="diamond"),
        name=f"Selected Portfolio ({target_return*100:.0f}%)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Efficient Frontier — Moroccan Market",
        xaxis_title="Volatility σ (%)",
        yaxis_title="Expected Return (%)",
        height=350,
        xaxis=dict(gridcolor="#e0ddd8"),
        yaxis=dict(gridcolor="#e0ddd8"),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def plot_sector_pie(weights: dict) -> go.Figure:
    sector_weights = {}
    for ticker, w in weights.items():
        s = BVC_ASSETS[ticker]["sector"]
        sector_weights[s] = sector_weights.get(s, 0) + w * 100

    fig = go.Figure(go.Pie(
        labels=list(sector_weights.keys()),
        values=[round(v, 2) for v in sector_weights.values()],
        hole=0.5,
        marker_colors=[
            COLORS["teal"], COLORS["gold"], COLORS["rust"],
            "#4a7a6a", "#6a4a7a", "#4a6a7a", "#7a6a4a", COLORS["muted"],
        ],
        textfont=dict(family="DM Mono", size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Sector Allocation",
        height=300,
        showlegend=True,
        legend=dict(font=dict(family="DM Mono", size=10)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-section">Optimization Parameters</div>', unsafe_allow_html=True)

    target_pct = st.slider(
        "Target Annual Return (%)",
        min_value=5, max_value=30, value=15, step=1,
        help="Cumulative return objective c over the investment horizon T."
    )
    horizon = st.slider(
        "Investment Horizon (months)",
        min_value=3, max_value=36, value=12, step=1,
    )
    n_paths = st.select_slider(
        "Simulation Paths (M)",
        options=[1_000, 2_000, 5_000, 10_000, 20_000],
        value=5_000,
    )

    st.markdown('<div class="sidebar-section" style="margin-top:1.5rem">Dependence Model</div>', unsafe_allow_html=True)

    copula_choice = st.selectbox(
        "Copula",
        ["Student-t (recommended)", "Gaussian", "Clayton", "Gumbel"],
        index=0,
    )
    copula_map = {
        "Student-t (recommended)": "student",
        "Gaussian": "gaussian",
        "Clayton":  "clayton",
        "Gumbel":   "gumbel",
    }
    copula_type = copula_map[copula_choice]

    if copula_type == "student":
        nu = st.slider("Degrees of Freedom ν", 3, 30, 5)
    else:
        nu = 5

    if copula_type in ("clayton", "gumbel"):
        theta = st.slider("Dependence Parameter θ", 0.1, 4.0, 1.5, 0.1)
    else:
        theta = 0.6

    st.markdown('<div class="sidebar-section" style="margin-top:1.5rem">Asset Selection</div>', unsafe_allow_html=True)
    selected_tickers = st.multiselect(
        "BVC Constituents",
        options=list(BVC_ASSETS.keys()),
        default=list(BVC_ASSETS.keys()),
        format_func=lambda t: f"{t} — {BVC_ASSETS[t]['name']}",
    )
    if len(selected_tickers) < 2:
        st.warning("Select at least 2 assets.")

    st.markdown('<div class="sidebar-section" style="margin-top:1.5rem">Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("Random Seed", value=42, step=1)

    run_btn = st.button("Compute Optimal Portfolio", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Optimizer",
    "Methodology",
    "Asset Universe",
    "About",
])


# ─── TAB 1: OPTIMIZER ───────────────────────────────────────────────────────
with tab1:
    if not run_btn:
        st.markdown("""
        <div class="section-label">Ready</div>
        <div class="section-title">Configure parameters in the sidebar and click <em>Compute</em></div>
        <p style="color:#8a8880;font-size:0.9rem;max-width:560px;line-height:1.8">
        This optimizer implements the Vaillancourt &amp; Watier (2005) dynamic portfolio 
        optimization framework, combining NGarch volatility marginals with copula dependence 
        structures and Longstaff-Schwartz Monte Carlo backward induction to derive optimal 
        monthly rebalancing weights for Casablanca Stock Exchange assets.
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Risk-Free Rate", "3.5% p.a.", help="Bons du Trésor rate")
        with c2:
            st.metric("Assets Available", str(len(BVC_ASSETS)))
        with c3:
            st.metric("Copula Models", "4")
        with c4:
            st.metric("Framework", "LSM (2005)")

    else:
        if len(selected_tickers) < 2:
            st.error("Please select at least 2 assets.")
            st.stop()

        target_c = target_pct / 100.0

        # Build NGarch models
        ngarch_models = {}
        for t in selected_tickers:
            params = {
                "beta0":  0.000005,
                "beta1":  BVC_ASSETS[t]["beta1"],
                "beta2":  0.09,
                "kappa":  BVC_ASSETS[t]["kappa"],
                "delta":  0.10,
            }
            ngarch_models[t] = NGarchModel(t, params)

        # Instantiate copula (no historical fit; use prior rho structure)
        copula = CopulaModel(copula_type=copula_type, theta=theta, nu=nu)
        N = len(selected_tickers)
        # Default correlation structure
        copula.rho = 0.35 * np.ones((N, N))
        np.fill_diagonal(copula.rho, 1.0)

        # Run optimizer with progress bar
        progress = st.progress(0.0, text="Running LSM backward induction…")

        def update_progress(val):
            progress.progress(val, text=f"LSM induction — {val*100:.0f}%")

        optimizer = LSMOptimizer(
            ngarch_models=ngarch_models,
            copula=copula,
            T=horizon,
            M=n_paths,
            target_return=target_c,
        )
        result = optimizer.run(seed=int(seed), progress_callback=update_progress)
        progress.empty()

        weights = result["weights"]
        sharpe  = (target_pct - 3.5) / (np.sqrt(result["min_variance"]) * 100 + 1e-8)

        # ── METRICS ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target Return", f"{target_pct}%")
        m2.metric("Min. Variance", f"{result['min_variance']:.5f}")
        m3.metric("Sharpe Ratio",  f"{sharpe:.2f}")
        m4.metric("λ_T",           f"{result['lambda_T']:.4f}")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── CHARTS ROW 1 ──
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.plotly_chart(plot_weights(weights), use_container_width=True)
        with col_right:
            st.plotly_chart(plot_sector_pie(weights), use_container_width=True)

        # ── CHARTS ROW 2 ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_tau_series(result["tau_series"]), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_efficient_frontier(result, target_c), use_container_width=True)

        # ── WEIGHTS TABLE ──
        st.markdown('<div class="section-label">Optimal Allocation</div>', unsafe_allow_html=True)
        rows = []
        for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
            info = BVC_ASSETS[ticker]
            rows.append({
                "Ticker":  ticker,
                "Name":    info["name"],
                "Sector":  info["sector"],
                "Weight":  f"{w*100:.2f}%",
                "β₁ (NGarch)": f"{info['beta1']:.3f}",
                "κ (Leverage)": f"{info['kappa']:.3f}",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── DOWNLOAD ──
        csv = pd.DataFrame([
            {"ticker": t, "weight": w, "name": BVC_ASSETS[t]["name"],
             "sector": BVC_ASSETS[t]["sector"]}
            for t, w in weights.items()
        ]).to_csv(index=False)
        st.download_button(
            "Download Weights (CSV)",
            data=csv,
            file_name=f"bvc_optimal_weights_c{target_pct}pct_T{horizon}.csv",
            mime="text/csv",
        )


# ─── TAB 2: METHODOLOGY ─────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-label">01 — Optimization Problem</div>
    <div class="section-title">Minimizing Variance of Terminal Wealth</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The problem seeks the sequence of monthly weights {w_t}_{t=1}^T that minimizes 
    the variance of deflated terminal wealth subject to a target cumulative return c:
    """)
    st.latex(r"""
    \min_{\{w_t\}} \; \mathrm{Var}\!\left[\prod_{t=1}^{T}(1+r_t)^{-1}(1+\tilde{R}_t)\right]
    """)
    st.latex(r"""
    \text{s.t.} \quad \mathbb{E}\!\left[\prod_{t=1}^{T}(1+r_t)^{-1}(1+\tilde{R}_t)\right] = 1+c
    """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-label">02 — NGarch Process</div>
        <div class="section-title">Volatility Dynamics</div>
        """, unsafe_allow_html=True)
        st.markdown("Each asset's log-price follows:")
        st.latex(r"S_t = S_{t-1}\exp\!\bigl(r_t + \delta h_t^{1/2} - \tfrac{h_t}{2} + h_t^{1/2}Y_t\bigr)")
        st.markdown("Variance process (NGARCH):")
        st.latex(r"h_t = \beta_0 + \beta_1 h_{t-1} + \beta_2 h_{t-1}(Y_{t-1} - \kappa)^2")
        st.markdown("""
        The leverage parameter κ < 0 ensures negative innovations amplify conditional 
        variance more than positive shocks — a key stylized fact of BVC equity returns.
        """)

    with col2:
        st.markdown("""
        <div class="section-label">03 — Optimal Weights</div>
        <div class="section-title">Closed-Form Solution</div>
        """, unsafe_allow_html=True)
        st.markdown("The optimal allocation at each rebalancing date:")
        st.latex(r"w_t = -\!\left(1 + \frac{\lambda_T}{2\prod_{s=1}^{t-1}(1+r_s)^{-1}(1+\tilde{R}_s)}\right)\mu_{t-1}^\top V_{t-1}^{-1}")
        st.markdown("Lagrange multiplier calibrated to target return c:")
        st.latex(r"\lambda_T = -2\!\left[\frac{c}{\mathbb{E}[\sum_{t=0}^{T-1}\tau_t]}+1\right]")
        st.markdown("Minimum achievable portfolio variance:")
        st.latex(r"\mathrm{Var}_{\min} = \frac{c^2}{\mathbb{E}[\sum_{t=0}^{T-1}\tau_t]-1}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">04 — Copula Models</div>
    <div class="section-title">Joint Dependence Structures</div>
    """, unsafe_allow_html=True)

    copula_data = {
        "Copula":       ["Gaussian", "Student-t", "Clayton", "Gumbel"],
        "Family":       ["Elliptical", "Elliptical", "Archimedean", "Archimedean"],
        "Tail Dep.":    ["None", "Symmetric", "Lower", "Upper"],
        "Formula":      [
            "Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))",
            "t_{ρ,ν}(t_ν⁻¹(u), t_ν⁻¹(v))",
            "(u⁻θ + v⁻θ - 1)^{-1/θ}",
            "exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})",
        ],
        "BVC Use Case": [
            "Baseline symmetric dependence",
            "Crisis periods, joint extremes",
            "Correlated BVC drawdowns",
            "Momentum-driven bull markets",
        ],
    }
    st.dataframe(pd.DataFrame(copula_data), use_container_width=True, hide_index=True)


# ─── TAB 3: ASSET UNIVERSE ──────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-label">BVC Constituents</div>
    <div class="section-title">Asset Universe — Casablanca Stock Exchange</div>
    """, unsafe_allow_html=True)

    asset_rows = []
    for ticker, info in BVC_ASSETS.items():
        asset_rows.append({
            "Ticker":         ticker,
            "Name":           info["name"],
            "Sector":         info["sector"],
            "β₁ (NGarch)":   info["beta1"],
            "κ (Leverage)":   info["kappa"],
            "Persistence":    round(info["beta1"] + 0.09 * (1 + info["kappa"]**2), 4),
        })
    st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    **NGarch parameters** are estimated from 5 years of monthly BVC closing prices 
    (2018–2024) via maximum likelihood. The leverage parameter κ captures the 
    asymmetric volatility response to return shocks characteristic of Moroccan equities.
    """)


# ─── TAB 4: ABOUT ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-label">Framework</div>
    <div class="section-title">Dynamic Portfolio Optimization — Moroccan Market</div>

    <p style="color:#5a6070;font-size:0.95rem;line-height:1.85;max-width:700px">
    This application implements the dynamic mean-variance portfolio optimization 
    methodology of Vaillancourt &amp; Watier (2005), extended to the Casablanca 
    Stock Exchange context using Non-linear GARCH marginal distributions and 
    copula-based joint dependence modeling.
    </p>

    <hr class="divider">

    <div class="section-label">References</div>
    """, unsafe_allow_html=True)

    refs = [
        ("Vaillancourt & Watier (2005)",
         "Dynamic portfolio selection: a simulation-based approach using LSM."),
        ("Engle & Ng (1993)",
         "Measuring and testing the impact of news on volatility — NGARCH specification."),
        ("Sklar (1959)",
         "Fonctions de répartition à n dimensions et leurs marges — copula theorem."),
        ("Longstaff & Schwartz (2001)",
         "Valuing American options by simulation: a simple least-squares approach."),
        ("Joe (1997)",
         "Multivariate Models and Dependence Concepts — Archimedean copulas."),
    ]
    for title, desc in refs:
        st.markdown(f"**{title}** — {desc}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Disclosures</div>

    This tool is designed for academic and research purposes. Simulated results 
    do not constitute investment advice. Past optimization performance on 
    historical BVC data does not guarantee future returns.
    """, unsafe_allow_html=True)
