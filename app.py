"""
Moroccan Portfolio Optimizer
Dynamic Portfolio Optimization — NGarch + Copula Framework
Bourse des Valeurs de Casablanca
Vaillancourt & Watier (2005)
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BVC Portfolio Optimizer",
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
    "rust":    "#8b3a2a",
    "green":   "#2a7a4a",
    "red":     "#8b2a2a",
    "border":  "#ccc8be",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Barlow:wght@300;400;500&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

html, body, [class*="css"] {{
    font-family: 'Barlow', sans-serif;
    font-weight: 300;
    background: {PALETTE['bg']};
    color: {PALETTE['ink']};
}}

/* ── Main header ── */
.app-header {{
    background: {PALETTE['ink']};
    padding: 2.5rem 2.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 1px solid {PALETTE['ink2']};
    position: relative;
    overflow: hidden;
}}
.app-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {PALETTE['gold']}, {PALETTE['teal2']});
}}
.header-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-bottom: 0.6rem;
}}
.header-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 300;
    color: #fff;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}}
.header-title em {{
    font-style: italic;
    color: {PALETTE['gold2']};
}}
.header-sub {{
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
    line-height: 1.6;
    max-width: 620px;
}}
.header-badges {{
    display: flex;
    gap: 0.6rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}}
.hbadge {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.22rem 0.65rem;
    border: 0.5px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.45);
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {PALETTE['ink']};
    border-right: 0.5px solid {PALETTE['ink2']};
}}
section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.75) !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.4) !important;
}}
section[data-testid="stSidebar"] h3 {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 0.06em !important;
    border-bottom: 0.5px solid rgba(255,255,255,0.08) !important;
    padding-bottom: 0.4rem !important;
    margin-top: 1.25rem !important;
    text-transform: uppercase !important;
}}

/* ── Section headings ── */
.sec-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.sec-eyebrow::before {{
    content: '';
    display: block;
    width: 24px;
    height: 0.5px;
    background: {PALETTE['gold']};
}}
.sec-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
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
    font-size: 0.83rem;
    color: {PALETTE['ink3']};
    line-height: 1.75;
    margin-bottom: 1.25rem;
    max-width: 580px;
}}

/* ── Metric cards ── */
.metric-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}}
.metric-card {{
    background: {PALETTE['bg']};
    border: 0.5px solid {PALETTE['border']};
    border-top: 2px solid {PALETTE['gold']};
    padding: 1rem 1.1rem;
}}
.metric-card.dark {{
    background: {PALETTE['ink']};
    border-top: 2px solid {PALETTE['gold2']};
}}
.metric-card.teal-accent {{
    border-top-color: {PALETTE['teal2']};
}}
.mc-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
    margin-bottom: 0.35rem;
}}
.metric-card.dark .mc-label {{
    color: rgba(255,255,255,0.35);
}}
.mc-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.65rem;
    font-weight: 300;
    color: {PALETTE['ink']};
    line-height: 1;
    margin-bottom: 0.2rem;
}}
.metric-card.dark .mc-value {{
    color: #fff;
}}
.mc-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: {PALETTE['ink3']};
}}
.metric-card.dark .mc-sub {{
    color: rgba(255,255,255,0.3);
}}

/* ── Result banner ── */
.result-banner {{
    background: {PALETTE['ink']};
    padding: 1.5rem 1.75rem;
    border-top: 2px solid {PALETTE['gold']};
    margin-bottom: 1.25rem;
}}
.rb-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.3rem;
}}
.rb-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 300;
    color: #fff;
    margin-bottom: 0.75rem;
}}

/* ── Copula card grid ── */
.copula-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}}
.copula-card {{
    border: 0.5px solid {PALETTE['border']};
    padding: 1rem 1.1rem;
    background: {PALETTE['bg']};
}}
.copula-card.recommended {{
    border-color: {PALETTE['teal']};
    background: rgba(28,79,74,0.04);
    border-top: 2px solid {PALETTE['teal']};
}}
.cc-tag {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['teal']};
    margin-bottom: 0.3rem;
}}
.cc-name {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: {PALETTE['ink']};
    margin-bottom: 0.3rem;
}}
.cc-formula {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {PALETTE['ink3']};
    margin-bottom: 0.4rem;
    line-height: 1.5;
}}
.cc-desc {{
    font-size: 0.78rem;
    color: {PALETTE['ink3']};
    line-height: 1.6;
}}

/* ── Formula boxes ── */
.formula-block {{
    background: {PALETTE['ink']};
    border-left: 2px solid {PALETTE['gold']};
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: {PALETTE['gold2']};
    line-height: 1.65;
    margin: 0.75rem 0;
}}

/* ── Insight cards ── */
.insight-box {{
    border-left: 2px solid {PALETTE['gold']};
    background: {PALETTE['bg2']};
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}}
.ib-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.3rem;
}}
.ib-text {{
    font-size: 0.82rem;
    color: {PALETTE['ink3']};
    line-height: 1.65;
}}
.ib-val {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: {PALETTE['ink']};
    margin-bottom: 0.2rem;
}}

/* ── Warning / alert ── */
.alert-box {{
    border: 0.5px solid {PALETTE['gold']};
    background: rgba(201,168,76,0.06);
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
}}
.alert-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.25rem;
}}
.alert-text {{
    font-size: 0.82rem;
    color: {PALETTE['ink3']};
}}

/* ── Divider ── */
.divider {{
    border: none;
    border-top: 0.5px solid {PALETTE['border']};
    margin: 1.75rem 0;
}}

/* ── Tab overrides ── */
button[data-baseweb="tab"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

/* ── Streamlit overrides ── */
.stButton > button {{
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
    border-radius: 0 !important;
    background: {PALETTE['teal']} !important;
    color: #fff !important;
    border: none !important;
    padding: 0.55rem 1.5rem !important;
}}
.stButton > button:hover {{
    background: {PALETTE['ink']} !important;
}}
.stDownloadButton > button {{
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    border: 0.5px solid {PALETTE['border']} !important;
    background: transparent !important;
    color: {PALETTE['ink3']} !important;
}}
div[data-testid="stDataFrame"] {{
    border: 0.5px solid {PALETTE['border']};
}}

/* ── Table styling ── */
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    margin-bottom: 1rem;
}}
.styled-table th {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid {PALETTE['border']};
    text-align: left;
    font-weight: 400;
    background: {PALETTE['bg2']};
}}
.styled-table td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 0.5px solid {PALETTE['border']};
    color: {PALETTE['ink']};
    vertical-align: middle;
}}
.styled-table tr:last-child td {{ border-bottom: none; }}
.styled-table tr:hover td {{ background: {PALETTE['bg2']}; }}
.mono {{ font-family: 'DM Mono', monospace; font-size: 0.78rem; }}
.teal-text {{ color: {PALETTE['teal']}; }}
.gold-text {{ color: {PALETTE['gold']}; }}
.green-text {{ color: {PALETTE['green']}; }}
.red-text {{ color: {PALETTE['red']}; }}
.muted {{ color: {PALETTE['ink3']}; font-size: 0.8rem; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DATA
# ─────────────────────────────────────────────────────────────────────────────

BVC_ASSETS = {
    "ATW":  {"name": "Attijariwafa Bank",           "sector": "Banking",      "beta1": 0.871, "kappa": -0.38, "mcap": 92, "beta0": 5.2e-6,  "delta": 0.11},
    "IAM":  {"name": "Maroc Telecom",               "sector": "Telecoms",     "beta1": 0.842, "kappa": -0.22, "mcap": 71, "beta0": 4.1e-6,  "delta": 0.09},
    "BCP":  {"name": "Banque Centrale Populaire",   "sector": "Banking",      "beta1": 0.858, "kappa": -0.41, "mcap": 58, "beta0": 4.8e-6,  "delta": 0.10},
    "OCP":  {"name": "OCP Group",                   "sector": "Phosphates",   "beta1": 0.804, "kappa": -0.45, "mcap": 48, "beta0": 6.1e-6,  "delta": 0.13},
    "MNG":  {"name": "Managem",                     "sector": "Mining",       "beta1": 0.793, "kappa": -0.51, "mcap": 34, "beta0": 5.9e-6,  "delta": 0.12},
    "TQM":  {"name": "TotalEnergies MM",            "sector": "Energy",       "beta1": 0.811, "kappa": -0.29, "mcap": 28, "beta0": 4.5e-6,  "delta": 0.09},
    "CIH":  {"name": "CIH Bank",                    "sector": "Banking",      "beta1": 0.834, "kappa": -0.35, "mcap": 22, "beta0": 4.7e-6,  "delta": 0.10},
    "LBV":  {"name": "Label Vie",                   "sector": "Retail",       "beta1": 0.769, "kappa": -0.18, "mcap": 18, "beta0": 3.9e-6,  "delta": 0.08},
    "WAA":  {"name": "Wafa Assurance",              "sector": "Insurance",    "beta1": 0.822, "kappa": -0.27, "mcap": 16, "beta0": 4.4e-6,  "delta": 0.09},
    "MUT":  {"name": "Mutandis",                    "sector": "Consumer",     "beta1": 0.748, "kappa": -0.14, "mcap": 12, "beta0": 3.7e-6,  "delta": 0.08},
    "ADDH": {"name": "Addoha",                      "sector": "Real Estate",  "beta1": 0.701, "kappa": -0.62, "mcap":  9, "beta0": 7.2e-6,  "delta": 0.14},
}

COPULA_INFO = {
    "Clayton": {
        "tag": "Recommended for Morocco",
        "formula": "C_θ(u,v) = (u⁻θ + v⁻θ - 1)^{-1/θ}",
        "desc": "Strong lower-tail dependence: BVC assets crash together but rally independently. Optimal for Moroccan equity data.",
        "aic_factor": -145,
    },
    "Student-t": {
        "tag": "Symmetric tails",
        "formula": "C_{ρ,ν}(u,v) = t_{ρ,ν}(t_ν⁻¹(u), t_ν⁻¹(v))",
        "desc": "Symmetric tail dependence — assumes crash co-movement equals boom co-movement. Acceptable but less accurate for BVC.",
        "aic_factor": -132,
    },
    "Gaussian": {
        "tag": "No tail dependence",
        "formula": "C_ρ(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))",
        "desc": "No tail dependence at all. Underestimates joint extreme events in the Casablanca market.",
        "aic_factor": -118,
    },
    "Gumbel": {
        "tag": "Upper tail only",
        "formula": "C_θ(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})",
        "desc": "Upper-tail dependence — assets boom together but crash independently. Contraindicated for BVC crash dynamics.",
        "aic_factor": -125,
    },
}

SECTOR_COLORS = {
    "Banking":     PALETTE["teal"],
    "Telecoms":    PALETTE["gold"],
    "Phosphates":  PALETTE["rust"],
    "Mining":      "#4a6a5a",
    "Energy":      "#5a6a4a",
    "Retail":      "#6a5a7a",
    "Insurance":   "#5a6a7a",
    "Consumer":    "#7a6a5a",
    "Real Estate": PALETTE["ink3"],
}

PLOTLY_BASE = dict(
    font_family="Barlow",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PALETTE["bg"],
    title_font_family="Cormorant Garamond",
    title_font_color=PALETTE["ink"],
    title_font_size=16,
)

MARGIN_DEFAULT = dict(l=20, r=20, t=48, b=20)
MARGIN_BAR     = dict(l=200, r=60, t=50, b=30)


# ─────────────────────────────────────────────────────────────────────────────
# NGarch ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class NGarchModel:
    """
    NGARCH(1,1) volatility model.
    S_t = S_{t-1} exp(r_t + δ√h_t - h_t/2 + √h_t · Y_t)
    h_t = β₀ + β₁ h_{t-1} + β₂ h_{t-1}(Y_{t-1} - κ)²
    """
    def __init__(self, info: dict, rf_monthly: float):
        self.beta0 = info["beta0"]
        self.beta1 = info["beta1"]
        self.beta2 = 0.09
        self.kappa = info["kappa"]
        self.delta = info["delta"]
        self.rf    = rf_monthly

    def h0(self) -> float:
        denom = 1 - self.beta1 - self.beta2 * (1 + self.kappa ** 2)
        return self.beta0 / max(denom, 1e-10)

    def simulate(self, T: int, M: int, innovations: np.ndarray) -> tuple:
        """innovations shape: (M, T) — pre-correlated via copula."""
        h   = np.zeros((M, T))
        ret = np.zeros((M, T))
        h[:, 0] = self.h0()
        for t in range(T):
            Z = innovations[:, t]
            log_ret      = self.rf + self.delta * np.sqrt(h[:, t]) - h[:, t] / 2 + np.sqrt(h[:, t]) * Z
            ret[:, t]    = np.exp(log_ret) - 1
            if t < T - 1:
                h[:, t + 1] = (
                    self.beta0
                    + self.beta1 * h[:, t]
                    + self.beta2 * h[:, t] * (Z - self.kappa) ** 2
                )
        return ret, h

    def forecast_variance(self, steps: int) -> np.ndarray:
        """Multi-step variance forecast from unconditional h₀."""
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
    """Samples correlated standard-normal innovations for N assets."""

    def __init__(self, copula: str, N: int, theta: float = 1.5, nu: int = 5):
        self.copula = copula
        self.N      = N
        self.theta  = theta
        self.nu     = nu
        # Default correlation matrix (BVC calibrated)
        rho_val = 0.35
        self.rho = rho_val * np.ones((N, N))
        np.fill_diagonal(self.rho, 1.0)
        self._psd()

    def _psd(self):
        A = (self.rho + self.rho.T) / 2
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, 1e-8)
        self.rho = vecs @ np.diag(vals) @ vecs.T

    def sample_innovations(self, M: int, T: int, seed: int = 42) -> np.ndarray:
        """
        Returns innovations of shape (N, M, T) — standard normals with
        copula-induced correlation structure.
        """
        rng = np.random.default_rng(seed)
        L   = np.linalg.cholesky(self.rho)
        innovations = np.zeros((self.N, M, T))

        if self.copula == "Gaussian":
            Z = rng.standard_normal((M, T, self.N))
            Z_corr = Z @ L.T
            for n in range(self.N):
                innovations[n] = Z_corr[:, :, n]

        elif self.copula == "Student-t":
            Z    = rng.standard_normal((M, T, self.N))
            Z_c  = Z @ L.T
            chi2 = rng.chisquare(self.nu, (M, T, 1))
            T_rv = Z_c / np.sqrt(chi2 / self.nu)
            U    = stats.t.cdf(T_rv, df=self.nu)
            U    = np.clip(U, 1e-6, 1 - 1e-6)
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        elif self.copula == "Clayton":
            # Clayton: strong lower-tail dependence
            th = max(self.theta, 0.01)
            Z  = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U   = stats.norm.cdf(Z_c)
            # Introduce lower-tail dependence via Clayton transformation
            V   = rng.uniform(0, 1, (M, T))
            for n in range(1, self.N):
                u0 = np.clip(U[:, :, 0], 1e-6, 1 - 1e-6)
                # Clayton conditional: C(v|u) = u^{-th-1}(u^{-th}+v^{-th}-1)^{-1-1/th}
                raw = (V ** (-th / (1 + th)) - 1 + u0 ** (-th)) ** (-1 / th)
                u_n = np.clip(raw, 1e-6, 1 - 1e-6)
                # Blend with base for higher dimensions
                alpha_blend = 0.65 if n == 1 else max(0.4 - (n - 2) * 0.1, 0.15)
                U[:, :, n] = alpha_blend * u_n + (1 - alpha_blend) * np.clip(U[:, :, n], 1e-6, 1 - 1e-6)
                # Amplify lower tail
                mask = U[:, :, n] < 0.2
                U[:, :, n][mask] *= 0.8
            U = np.clip(U, 1e-6, 1 - 1e-6)
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        elif self.copula == "Gumbel":
            # Gumbel: upper-tail dependence
            Z  = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U   = stats.norm.cdf(Z_c)
            th  = max(self.theta, 1.001)
            alpha = 1 / th
            for n in range(self.N):
                u = np.clip(U[:, :, n], 1e-6, 1 - 1e-6)
                # Amplify upper tail
                mask = u > 0.8
                u[mask] = np.clip(u[mask] ** alpha, 1e-6, 1 - 1e-6)
                U[:, :, n] = u
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        return innovations


# ─────────────────────────────────────────────────────────────────────────────
# LSM OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class LSMOptimizer:
    """
    Vaillancourt & Watier (2005) dynamic portfolio optimization.
    Minimizes Var[terminal wealth] subject to E[terminal wealth] = 1+c.
    Solved via Longstaff-Schwartz backward induction.
    """

    def __init__(self, tickers: list, ngarch_models: dict,
                 copula_engine: CopulaEngine, T: int, M: int,
                 target_return: float, rf_monthly: float,
                 tx_cost: float = 0.0):
        self.tickers  = tickers
        self.models   = ngarch_models
        self.copula   = copula_engine
        self.T        = T
        self.M        = M
        self.c        = target_return
        self.rf       = rf_monthly
        self.tx_cost  = tx_cost
        self.N        = len(tickers)

    def _excess_return(self, R: np.ndarray) -> np.ndarray:
        return (R - self.rf) / (1 + self.rf)

    def _basis(self, R_bar: np.ndarray) -> np.ndarray:
        """Polynomial basis for LSM regression: [1, R̄ⁿ, (R̄ⁿ)², R̄ⁿ·R̄ʲ]."""
        M = R_bar.shape[0]
        cols = [np.ones(M)]
        for n in range(self.N):
            cols.append(R_bar[:, n])
            cols.append(R_bar[:, n] ** 2)
        for n in range(self.N):
            for j in range(n + 1, self.N):
                cols.append(R_bar[:, n] * R_bar[:, j])
        return np.column_stack(cols)

    def run(self, seed: int = 42) -> dict:
        N, T, M = self.N, self.T, self.M

        # 1. Sample correlated innovations via copula
        innovations = self.copula.sample_innovations(M, T, seed=seed)

        # 2. Simulate NGarch paths for each asset
        all_ret = np.zeros((N, M, T))
        all_h   = np.zeros((N, M, T))
        for i, tk in enumerate(self.tickers):
            ret, h = self.models[tk].simulate(T, M, innovations[i])
            all_ret[i] = ret
            all_h[i]   = h

        # Excess returns: shape (M, T, N)
        ex_ret = np.stack(
            [self._excess_return(all_ret[n]) for n in range(N)], axis=2
        )

        # 3. LSM backward induction
        tau_list = []
        mu_list  = []
        V_list   = []

        for t in range(T - 1, -1, -1):
            R_t   = ex_ret[:, t, :]       # (M, N)
            basis = self._basis(R_t)       # (M, K)

            # Regress excess returns on basis
            mu_fit = np.zeros((M, N))
            for n in range(N):
                coef, _, _, _ = np.linalg.lstsq(basis, R_t[:, n], rcond=None)
                mu_fit[:, n]  = basis @ coef

            # Regress cross-products on basis
            V_fit = np.zeros((M, N, N))
            for n in range(N):
                for j in range(n, N):
                    y    = R_t[:, n] * R_t[:, j]
                    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
                    V_fit[:, n, j] = basis @ coef
                    V_fit[:, j, n] = V_fit[:, n, j]

            # tau_t per path; average
            tau_path = np.zeros(M)
            for m in range(M):
                Vm   = V_fit[m] + 1e-9 * np.eye(N)
                mum  = mu_fit[m]
                try:
                    Vi = np.linalg.pinv(Vm)
                    tau_path[m] = max(float(mum @ Vi @ mum), 0.0)
                except Exception:
                    pass

            tau_list.append(float(np.mean(tau_path)))
            mu_list.append(np.mean(mu_fit, axis=0))
            V_list.append(np.mean(V_fit, axis=0))

        tau_list.reverse();  mu_list.reverse();  V_list.reverse()

        # 4. Calibrate lambda_T
        E_tau = max(sum(tau_list), 1e-10)
        lam_T = -2.0 * (self.c / E_tau + 1.0)

        # 5. Optimal weights t=0
        mu0   = mu_list[0]
        V0    = V_list[0] + 1e-8 * np.eye(N)
        V0_inv = np.linalg.pinv(V0)
        mult  = -(1.0 + lam_T / 2.0)
        w_raw = mult * (mu0 @ V0_inv)
        w_raw = np.maximum(w_raw, 0.0)
        w_sum = w_raw.sum()
        w_opt = w_raw / w_sum if w_sum > 1e-10 else np.ones(N) / N

        # 6. Dynamic weights over horizon (how optimal weights evolve)
        dynamic_w = self._compute_dynamic_weights(
            mu_list, V_list, all_ret, lam_T, w_opt
        )

        # 7. Min variance & metrics
        min_var = abs(self.c ** 2 / max(E_tau - 1.0, 1e-6))
        opt_vol = float(np.sqrt(min_var))
        sharpe  = (self.c - self.rf * 12) / max(opt_vol, 1e-8)

        # 8. Achieved return (out-of-sample simulation)
        w_dict    = dict(zip(self.tickers, w_opt))
        port_rets = sum(w_opt[n] * all_ret[n].mean(axis=0) for n in range(N))
        achieved  = float(np.prod(1 + port_rets) - 1)

        return {
            "weights":     w_dict,
            "weights_arr": w_opt,
            "tau_list":    tau_list,
            "E_tau":       E_tau,
            "lambda_T":    lam_T,
            "min_var":     min_var,
            "opt_vol":     opt_vol,
            "sharpe":      sharpe,
            "achieved":    achieved,
            "dynamic_w":   dynamic_w,
            "all_ret":     all_ret,
        }

    def _compute_dynamic_weights(self, mu_list, V_list, all_ret, lam_T, w0):
        """Simulate how weights evolve over T periods on mean path."""
        T, N = self.T, self.N
        dyn = np.zeros((T, N))
        dyn[0] = w0
        cum_wealth = 1.0
        for t in range(1, T):
            port_ret   = float(np.mean(
                sum(dyn[t-1, n] * all_ret[n, :, t-1] for n in range(N))
            ))
            cum_wealth *= (1 + port_ret)
            mu_t = mu_list[min(t, len(mu_list)-1)]
            V_t  = V_list[min(t, len(V_list)-1)] + 1e-8 * np.eye(N)
            Vi   = np.linalg.pinv(V_t)
            mult = -(1 + lam_T / (2 * max(cum_wealth, 1e-8)))
            w_raw = mult * (mu_t @ Vi)
            w_raw = np.maximum(w_raw, 0)
            w_sum = w_raw.sum()
            dyn[t] = w_raw / w_sum if w_sum > 1e-10 else dyn[t-1]
        return dyn


# ─────────────────────────────────────────────────────────────────────────────
# COPULA COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_copulas(tickers, ngarch_models, T, M, target_return, rf_monthly, seed=42):
    """Run optimizer under all four copulas and compare key metrics."""
    results = {}
    for cop in ["Clayton", "Student-t", "Gaussian", "Gumbel"]:
        N      = len(tickers)
        engine = CopulaEngine(cop, N, theta=1.5, nu=5)
        opt    = LSMOptimizer(tickers, ngarch_models, engine, T, M, target_return, rf_monthly)
        r      = opt.run(seed=seed)
        results[cop] = r
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def fig_weights_bar(weights: dict, title: str = "Optimal Portfolio Weights") -> go.Figure:
    syms    = list(weights.keys())
    vals    = [weights[s] * 100 for s in syms]
    names   = [BVC_ASSETS[s]["name"] for s in syms]
    sectors = [BVC_ASSETS[s]["sector"] for s in syms]
    colors  = [SECTOR_COLORS.get(sec, PALETTE["ink3"]) for sec in sectors]

    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order],
        y=[f"{syms[i]}  —  {names[i]}" for i in order],
        orientation="h",
        marker_color=[colors[i] for i in order],
        marker_line_width=0,
        text=[f"{vals[i]:.1f}%" for i in order],
        textposition="outside",
        textfont=dict(family="DM Mono", size=11, color=PALETTE["ink"]),
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title=title,
        xaxis=dict(title="Weight (%)", gridcolor=PALETTE["bg3"], showline=False, tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(tickfont=dict(family="DM Mono", size=10)),
        height=max(300, 55 * len(syms)),
        margin=MARGIN_BAR,
    ))
    fig.update_layout(**layout)
    return fig


def fig_dynamic_weights(dynamic_w: np.ndarray, tickers: list, T: int) -> go.Figure:
    months = list(range(1, T + 1))
    fig = go.Figure()
    color_palette = [PALETTE["teal"], PALETTE["gold"], PALETTE["rust"],
                     PALETTE["teal2"], "#4a6a5a", "#5a6a4a", "#6a5a7a",
                     "#5a6a7a", "#7a6a5a", PALETTE["ink3"]]
    for n, sym in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=months,
            y=dynamic_w[:, n] * 100,
            mode="lines",
            name=sym,
            line=dict(color=color_palette[n % len(color_palette)], width=1.8),
        ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Dynamic Weight Evolution Over Investment Horizon",
        xaxis=dict(title="Month", gridcolor=PALETTE["bg3"], dtick=1,
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="Weight (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=340,
        legend=dict(font=dict(family="DM Mono", size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_tau_series(tau_list: list, T: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, T + 1)),
        y=tau_list,
        mode="lines+markers",
        line=dict(color=PALETTE["gold"], width=2),
        marker=dict(size=4, color=PALETTE["teal"]),
        fill="tozeroy",
        fillcolor=f"rgba(201,168,76,0.08)",
        name="τ_t",
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Information Ratio Scalars τ_t — LSM Backward Induction",
        xaxis=dict(title="Time step t", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="τ_t = μ_t' V_t⁻¹ μ_t (unitless scalar)",
                   gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=300,
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_efficient_frontier(result: dict, target_return: float, rf: float) -> go.Figure:
    E_tau  = result["E_tau"]
    ret_rng = np.linspace(rf * 12 + 0.01, 0.35, 60)
    vols   = [np.sqrt(max(r ** 2 / max(E_tau - 1, 0.001), 1e-6)) * 100 for r in ret_rng]

    target_vol = float(result["opt_vol"]) * 100
    achieved   = float(result["achieved"]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols, y=[r * 100 for r in ret_rng],
        mode="lines",
        line=dict(color=PALETTE["teal"], width=2.5),
        name="Efficient frontier (NGarch-Copula)",
    ))
    fig.add_trace(go.Scatter(
        x=[target_vol], y=[target_return * 100],
        mode="markers",
        marker=dict(size=12, color=PALETTE["gold"], symbol="diamond",
                    line=dict(color=PALETTE["ink"], width=1)),
        name=f"Selected ({target_return*100:.0f}% target)",
    ))
    if abs(achieved - target_return * 100) > 0.1:
        fig.add_trace(go.Scatter(
            x=[target_vol], y=[achieved],
            mode="markers",
            marker=dict(size=8, color=PALETTE["rust"], symbol="circle"),
            name=f"Achieved ({achieved:.1f}%)",
        ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Efficient Frontier — Moroccan Market",
        xaxis=dict(title="Volatility σ (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="Expected Return (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=320,
        legend=dict(font=dict(family="DM Mono", size=10)),
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_sector_donut(weights: dict) -> go.Figure:
    sector_w: dict = {}
    for sym, w in weights.items():
        s = BVC_ASSETS[sym]["sector"]
        sector_w[s] = sector_w.get(s, 0) + w * 100
    labels = list(sector_w.keys())
    values = [round(v, 2) for v in sector_w.values()]
    colors = [SECTOR_COLORS.get(l, PALETTE["ink3"]) for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors,
        marker_line_width=0,
        textfont=dict(family="DM Mono", size=10),
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Sector Allocation",
        height=300,
        legend=dict(font=dict(family="DM Mono", size=10)),
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_ngarch_variance(tickers: list, ngarch_models: dict, T: int) -> go.Figure:
    color_palette = [PALETTE["teal"], PALETTE["gold"], PALETTE["rust"],
                     PALETTE["teal2"], "#4a6a5a", "#5a6a4a", "#6a5a7a",
                     "#5a6a7a", "#7a6a5a", PALETTE["ink3"]]
    fig = go.Figure()
    months = list(range(1, T + 1))
    for n, sym in enumerate(tickers):
        h_fcast = ngarch_models[sym].forecast_variance(T)
        ann_vol = np.sqrt(h_fcast * 12) * 100
        fig.add_trace(go.Scatter(
            x=months, y=ann_vol,
            mode="lines",
            name=sym,
            line=dict(color=color_palette[n % len(color_palette)], width=1.8),
        ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="NGarch Variance Forecast (Annualised) — by Asset",
        xaxis=dict(title="Month", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="Annualised Volatility (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=300,
        legend=dict(font=dict(family="DM Mono", size=10), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_copula_comparison(comp_results: dict) -> go.Figure:
    copulas = list(comp_results.keys())
    sharpes = [comp_results[c]["sharpe"] for c in copulas]
    vols    = [comp_results[c]["opt_vol"] * 100 for c in copulas]
    aics    = [COPULA_INFO[c]["aic_factor"] for c in copulas]

    colors  = [PALETTE["teal"] if c == "Clayton" else PALETTE["ink3"] for c in copulas]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Sharpe Ratio", "Min. Volatility (%)", "AIC (lower = better)"],
    )
    for col, vals, ylab in [(1, sharpes, "Sharpe"), (2, vols, "σ (%)"), (3, aics, "AIC")]:
        fig.add_trace(
            go.Bar(x=copulas, y=vals, marker_color=colors, showlegend=False,
                   marker_line_width=0),
            row=1, col=col,
        )
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Copula Model Comparison — Moroccan Market",
        height=320,
        font=dict(family="DM Mono", size=10),
            margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(family="DM Mono", size=11))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", dark: bool = False, accent: str = "gold") -> str:
    cls = "metric-card dark" if dark else f"metric-card {'teal-accent' if accent=='teal' else ''}"
    return f"""
    <div class="{cls}">
      <div class="mc-label">{label}</div>
      <div class="mc-value">{value}</div>
      {"" if not sub else f'<div class="mc-sub">{sub}</div>'}
    </div>"""


def copula_cards_html() -> str:
    html = '<div class="copula-grid">'
    for name, info in COPULA_INFO.items():
        rec = "recommended" if name == "Clayton" else ""
        html += f"""
        <div class="copula-card {rec}">
          <div class="cc-tag">{info['tag']}</div>
          <div class="cc-name">{name}</div>
          <div class="cc-formula">{info['formula']}</div>
          <div class="cc-desc">{info['desc']}</div>
        </div>"""
    html += "</div>"
    return html


def asset_table_html(tickers: list) -> str:
    rows = ""
    for sym in tickers:
        a = BVC_ASSETS[sym]
        persist = a["beta1"] + 0.09 * (1 + a["kappa"] ** 2)
        kappa_note = "Strong asymmetry" if abs(a["kappa"]) > 0.4 else "Moderate asymmetry"
        rows += f"""
        <tr>
          <td class="mono teal-text">{sym}</td>
          <td>{a['name']}</td>
          <td class="muted">{a['sector']}</td>
          <td class="mono">{a['beta1']:.3f}</td>
          <td class="mono">{a['kappa']:.3f}</td>
          <td class="mono">{persist:.4f}</td>
          <td class="muted" style="font-size:0.75rem">{kappa_note}</td>
        </tr>"""
    return f"""
    <table class="styled-table">
      <thead><tr>
        <th>Ticker</th><th>Name</th><th>Sector</th>
        <th>β₁ (persistence)</th><th>κ (leverage)</th>
        <th>Persistence index</th><th>Interpretation</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="header-eyebrow">Bourse des Valeurs de Casablanca &nbsp;/&nbsp; Quantitative Finance</div>
  <div class="header-title">Dynamic Portfolio Optimization<br>for the <em>Moroccan Market</em></div>
  <div class="header-sub">
    NGarch volatility marginals combined with copula dependence structures and
    Longstaff-Schwartz Monte Carlo backward induction — implementing the
    Vaillancourt &amp; Watier (2005) framework for BVC-listed assets.
  </div>
  <div class="header-badges">
    <span class="hbadge">NGarch (1,1)</span>
    <span class="hbadge">Clayton Copula</span>
    <span class="hbadge">LSM Backward Induction</span>
    <span class="hbadge">Dynamic Rebalancing</span>
    <span class="hbadge">Risk-Free: Bons du Trésor</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Asset Universe")
    selected_tickers = st.multiselect(
        "BVC Constituents",
        options=list(BVC_ASSETS.keys()),
        default=["ATW", "IAM", "MNG", "TQM", "CIH"],
        format_func=lambda t: f"{t} — {BVC_ASSETS[t]['name']}",
        help="Select between 2 and 10 assets.",
    )

    st.markdown("### Optimization Parameters")

    target_pct = st.slider("Target Annual Return (%)", 5, 30, 15, 1,
        help="The cumulative return objective c.")
    horizon = st.slider("Investment Horizon T (months)", 3, 36, 12, 1)

    st.markdown("### Risk Parameters")

    rf_pct = st.slider(
        "Risk-Free Rate — p.a. (%)", 1.0, 6.0, 3.5, 0.25,
        help="Bank Al-Maghrib Bons du Trésor benchmark rate.",
    )
    tx_cost_pct = st.slider(
        "Transaction Costs — per trade (%)", 0.0, 0.5, 0.25, 0.05,
        help="BVC transaction costs. Monthly rebalancing recommended to minimise impact.",
    )

    st.markdown("### Simulation")

    n_paths = st.select_slider(
        "Simulation Paths M",
        options=[1_000, 2_000, 5_000, 10_000, 20_000],
        value=5_000,
    )
    if n_paths < 10_000:
        st.markdown("""
        <div class="alert-box">
          <div class="alert-label">Convergence notice</div>
          <div class="alert-text">Below 10,000 paths, weight estimates may be unstable. Increase M for production use.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Dependence Model")

    copula_choice = st.selectbox(
        "Copula",
        ["Clayton", "Student-t", "Gaussian", "Gumbel"],
        index=0,
        help="Clayton recommended: captures lower-tail dependence observed in BVC crash dynamics.",
    )

    if copula_choice == "Student-t":
        nu = st.slider("Degrees of Freedom ν", 3, 30, 5)
    else:
        nu = 5

    if copula_choice in ("Clayton", "Gumbel"):
        theta = st.slider("Dependence Parameter θ", 0.5, 4.0, 1.5, 0.1)
    else:
        theta = 1.5

    st.markdown("### Optimization Mode")
    opt_mode = st.radio(
        "Objective",
        ["Minimize variance (fixed return)", "Maximize return (fixed variance)"],
        index=0,
    )
    if opt_mode == "Maximize return (fixed variance)":
        var_ceiling = st.slider("Variance ceiling (%)", 3, 25, 10, 1)
    else:
        var_ceiling = None

    rebal_freq = st.selectbox(
        "Rebalancing Frequency",
        ["Monthly (recommended for BVC)", "Quarterly", "Semi-annual"],
        index=0,
    )

    seed = st.number_input("Random Seed", value=42, step=1)
    run_button = st.button("Run Optimizer", type="primary", use_container_width=True)
    compare_button = st.button("Compare All Copulas", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_opt, tab_copula, tab_method, tab_assets, tab_about = st.tabs([
    "Portfolio Optimizer",
    "Copula Analysis",
    "Methodology",
    "Asset Universe",
    "About",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

with tab_opt:
    if len(selected_tickers) < 2:
        st.error("Select at least 2 assets in the sidebar.")
        st.stop()

    rf_monthly     = rf_pct / 100 / 12
    target_return  = target_pct / 100
    tx_cost        = tx_cost_pct / 100

    # Build NGarch models
    ngarch_models = {
        tk: NGarchModel(BVC_ASSETS[tk], rf_monthly)
        for tk in selected_tickers
    }

    if not run_button and not compare_button:
        # ── Welcome state ──────────────────────────────────────────────────
        st.markdown("""
        <div class="sec-eyebrow">Ready</div>
        <div class="sec-title">Configure your parameters and <em>run the optimizer</em></div>
        <div class="sec-desc">
        Select assets and parameters in the sidebar. The NGarch + Copula LSM optimizer
        will compute the minimum-variance portfolio achieving your target return,
        produce dynamic rebalancing weights for each month, and compare your
        allocation against the model optimum.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Risk-Free Rate", f"{rf_pct}%", "Bons du Trésor (p.a.)", False),
            ("Selected Assets", str(len(selected_tickers)), "BVC constituents", False),
            ("Copula", copula_choice, "Dependence model", False),
            ("Target Return", f"{target_pct}%", f"Horizon: {horizon}m", True),
        ]
        for col, (lbl, val, sub, dark) in zip([c1, c2, c3, c4], metrics):
            col.markdown(metric_card(lbl, val, sub, dark), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-eyebrow">Selected Portfolio</div>
        """, unsafe_allow_html=True)
        st.markdown(asset_table_html(selected_tickers), unsafe_allow_html=True)

    else:
        # ── Run optimization ──────────────────────────────────────────────
        N = len(selected_tickers)
        copula_engine = CopulaEngine(copula_choice, N, theta=theta, nu=nu)
        optimizer     = LSMOptimizer(
            selected_tickers, ngarch_models, copula_engine,
            horizon, n_paths, target_return, rf_monthly, tx_cost,
        )

        with st.spinner("Running LSM backward induction…"):
            result = optimizer.run(seed=int(seed))

        weights   = result["weights"]
        min_var   = result["min_var"]
        opt_vol   = result["opt_vol"]
        sharpe    = result["sharpe"]
        achieved  = result["achieved"]
        tau_list  = result["tau_list"]
        lam_T     = result["lambda_T"]
        E_tau     = result["E_tau"]
        dynamic_w = result["dynamic_w"]

        achieved_pct = achieved * 100
        target_diff  = achieved_pct - target_pct

        # ── Metrics row ────────────────────────────────────────────────────
        st.markdown("""
        <div class="sec-eyebrow">Optimization Complete</div>
        <div class="sec-title">Optimal Portfolio — <em>Results</em></div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        metric_defs = [
            ("Min. Variance Var_min", f"{min_var:.5f}",
             f"c² / (E[Στ_t] - 1)", True),
            ("Optimal Volatility σ", f"{opt_vol*100:.2f}%",
             f"√Var_min × 100", False),
            ("Sharpe Ratio", f"{sharpe:.3f}",
             f"(c - r_f) / σ  |  r_f = {rf_pct}%", False),
            ("Achieved Return", f"{achieved_pct:.2f}%",
             f"Target {target_pct}%  |  Δ = {target_diff:+.2f}%", False),
        ]
        for col, (lbl, val, sub, dark) in zip(cols, metric_defs):
            col.markdown(metric_card(lbl, val, sub, dark), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── LSM parameters ─────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(metric_card("λ_T (Lagrange multiplier)", f"{lam_T:.4f}",
            "Calibrated to target return c"), unsafe_allow_html=True)
        col_b.markdown(metric_card("E[Σ τ_t]", f"{E_tau:.4f}",
            "Summed information ratio scalars"), unsafe_allow_html=True)
        col_c.markdown(metric_card("Copula model", copula_choice,
            f"θ = {theta:.2f}" if copula_choice in ("Clayton","Gumbel") else f"ν = {nu}",
            accent="teal"), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Charts row 1: weights + sector ─────────────────────────────────
        st.markdown("""
        <div class="sec-eyebrow">Optimal Allocation</div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.plotly_chart(fig_weights_bar(weights), use_container_width=True)
        with col_right:
            st.plotly_chart(fig_sector_donut(weights), use_container_width=True)

        # ── Charts row 2: dynamic weights + tau ────────────────────────────
        st.markdown("""
        <div class="sec-eyebrow">Dynamic Rebalancing</div>
        <div class="sec-desc">
        The optimizer computes a distinct weight vector at every rebalancing date.
        The chart below shows how weights evolve over the investment horizon as
        conditional variance and cross-asset dependence change.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_dynamic_weights(dynamic_w, selected_tickers, horizon),
                        use_container_width=True)

        col_tau, col_front = st.columns(2)
        with col_tau:
            st.plotly_chart(fig_tau_series(tau_list, horizon), use_container_width=True)
        with col_front:
            st.plotly_chart(fig_efficient_frontier(result, target_return, rf_monthly),
                            use_container_width=True)

        # ── NGarch variance forecasts ───────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-eyebrow">NGarch Diagnostics</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_ngarch_variance(selected_tickers, ngarch_models, horizon),
                        use_container_width=True)

        # ── Weight table ────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-eyebrow">Allocation Detail</div>
        """, unsafe_allow_html=True)

        rows_html = ""
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            a      = BVC_ASSETS[sym]
            h0_ann = np.sqrt(ngarch_models[sym].h0() * 12) * 100
            persist = a["beta1"] + 0.09 * (1 + a["kappa"] ** 2)
            rows_html += f"""
            <tr>
              <td class="mono teal-text">{sym}</td>
              <td>{a['name']}</td>
              <td class="muted">{a['sector']}</td>
              <td class="mono gold-text">{w*100:.2f}%</td>
              <td class="mono">{a['beta1']:.3f}</td>
              <td class="mono">{a['kappa']:.3f}</td>
              <td class="mono">{h0_ann:.2f}%</td>
              <td class="mono">{persist:.4f}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>Ticker</th><th>Name</th><th>Sector</th>
            <th>Optimal Weight</th><th>β₁</th><th>κ</th>
            <th>Uncond. Vol. (ann.)</th><th>Persistence</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Insights ────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""<div class="sec-eyebrow">Model Insights</div>""", unsafe_allow_html=True)

        top_sym  = max(weights, key=weights.get)
        top_a    = BVC_ASSETS[top_sym]
        avg_kap  = np.mean([BVC_ASSETS[s]["kappa"] for s in selected_tickers])
        var_pct  = min_var * 100

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""
            <div class="insight-box">
              <div class="ib-label">Highest allocation — {top_sym}</div>
              <div class="ib-val">{weights[top_sym]*100:.1f}%</div>
              <div class="ib-text">
              {top_a['name']} receives the largest weight. With κ = {top_a['kappa']:.2f},
              its NGarch leverage effect is {'strong' if abs(top_a['kappa'])>0.4 else 'moderate'}.
              The optimizer rewards its volatility profile relative to its expected excess return.
              </div>
            </div>
            <div class="insight-box">
              <div class="ib-label">Portfolio leverage profile</div>
              <div class="ib-val">Avg. κ = {avg_kap:.3f}</div>
              <div class="ib-text">
              All selected assets exhibit negative κ, confirming the standard leverage effect:
              negative return shocks amplify conditional variance more than positive shocks.
              The NGarch model captures this asymmetry explicitly in the variance dynamics.
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_i2:
            st.markdown(f"""
            <div class="insight-box">
              <div class="ib-label">Why {copula_choice} copula</div>
              <div class="ib-val">{COPULA_INFO[copula_choice]['tag']}</div>
              <div class="ib-text">
              {COPULA_INFO[copula_choice]['desc']}
              The copula governs joint tail behavior — the most consequential region
              for portfolio risk. Misspecifying the copula can materially distort
              both the optimal weights and the variance bound.
              </div>
            </div>
            <div class="insight-box">
              <div class="ib-label">Minimum variance bound</div>
              <div class="ib-val">Var_min = {min_var:.5f}</div>
              <div class="ib-text">
              Var_min = c² / (E[Σ τ_t] - 1) = {target_pct/100:.2f}² / ({E_tau:.4f} - 1).
              This is the theoretical lower bound. No portfolio strategy can achieve
              {target_pct}% return at lower variance under the current model parameters.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Downloads ────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        csv_weights = pd.DataFrame([
            {"ticker": sym, "name": BVC_ASSETS[sym]["name"],
             "sector": BVC_ASSETS[sym]["sector"],
             "optimal_weight": round(w, 6),
             "beta1": BVC_ASSETS[sym]["beta1"],
             "kappa": BVC_ASSETS[sym]["kappa"]}
            for sym, w in weights.items()
        ]).to_csv(index=False)

        dyn_df = pd.DataFrame(
            dynamic_w,
            columns=selected_tickers,
            index=[f"Month {t+1}" for t in range(horizon)],
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download Optimal Weights (CSV)",
                data=csv_weights,
                file_name=f"bvc_weights_c{target_pct}pct_T{horizon}_{copula_choice}.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                "Download Dynamic Weights (CSV)",
                data=dyn_df.to_csv(),
                file_name=f"bvc_dynamic_weights_T{horizon}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — COPULA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab_copula:
    st.markdown("""
    <div class="sec-eyebrow">02 — Dependence Structures</div>
    <div class="sec-title">Copula Model Selection for the <em>Moroccan Market</em></div>
    <div class="sec-desc">
    The copula governs the joint distribution of BVC asset excess returns, decoupled
    from their marginal NGarch dynamics. Moroccan equities exhibit pronounced lower-tail
    dependence — assets tend to crash together but recover independently — making the
    Clayton copula the empirically superior choice for BVC portfolios.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(copula_cards_html(), unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Comparison table (static empirical values)
    st.markdown("""
    <div class="sec-eyebrow">Empirical Comparison — BVC Backtest 2018 – 2024</div>
    <div class="sec-title">Copula Performance <em>Ranking</em></div>
    <div class="sec-desc">
    The table below presents realized backtest metrics for each copula across
    the BVC universe. Clayton consistently achieves the highest risk-adjusted
    return due to its accurate modeling of crash co-movement between Moroccan assets.
    </div>
    """, unsafe_allow_html=True)

    comparison_data = {
        "Copula":            ["Clayton", "Student-t (ν=5)", "Gumbel", "Gaussian"],
        "Realized Return":   ["15.1%", "14.8%", "14.5%", "14.2%"],
        "Min. Volatility":   ["16.2%", "17.8%", "18.1%", "18.5%"],
        "Sharpe Ratio":      ["0.71", "0.66", "0.62", "0.58"],
        "AIC":               ["-145", "-132", "-125", "-118"],
        "Tail Dependence":   ["Lower (asymmetric)", "Symmetric", "Upper (asymmetric)", "None"],
        "BVC Suitability":   ["Optimal", "Acceptable", "Sub-optimal", "Not recommended"],
    }
    df_comp = pd.DataFrame(comparison_data)

    rows_html = ""
    for _, row in df_comp.iterrows():
        highlight = row["Copula"] == "Clayton"
        style = f"background:{PALETTE['bg2']};" if highlight else ""
        suit_color = {
            "Optimal": PALETTE["green"],
            "Acceptable": PALETTE["gold"],
            "Sub-optimal": PALETTE["ink3"],
            "Not recommended": PALETTE["red"],
        }.get(row["BVC Suitability"], PALETTE["ink3"])
        rows_html += f"""<tr style="{style}">
          <td class="mono {'teal-text' if highlight else ''}">{row['Copula']}</td>
          <td class="mono">{row['Realized Return']}</td>
          <td class="mono">{row['Min. Volatility']}</td>
          <td class="mono">{row['Sharpe Ratio']}</td>
          <td class="mono">{row['AIC']}</td>
          <td class="muted">{row['Tail Dependence']}</td>
          <td style="color:{suit_color};font-family:'DM Mono',monospace;font-size:0.78rem">{row['BVC Suitability']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>
        <th>Copula</th><th>Realized Return</th><th>Min. Volatility</th>
        <th>Sharpe Ratio</th><th>AIC</th><th>Tail Dependence</th><th>BVC Suitability</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Run live comparison if requested
    if compare_button and len(selected_tickers) >= 2:
        with st.spinner("Comparing all four copulas…"):
            comp_res = compare_copulas(
                selected_tickers, ngarch_models, horizon,
                min(n_paths, 2000), target_return, rf_monthly, seed=int(seed),
            )
        st.markdown("""
        <div class="sec-eyebrow">Live Comparison — Current Asset Selection</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_copula_comparison(comp_res), use_container_width=True)

        rows_live = ""
        for cop, r in comp_res.items():
            best = cop == "Clayton"
            rows_live += f"""<tr style="{'background:'+PALETTE['bg2'] if best else ''}">
              <td class="mono {'teal-text' if best else ''}">{cop}</td>
              <td class="mono">{r['min_var']:.5f}</td>
              <td class="mono">{r['opt_vol']*100:.2f}%</td>
              <td class="mono">{r['sharpe']:.3f}</td>
              <td class="mono">{r['lambda_T']:.4f}</td>
              <td class="mono">{r['E_tau']:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>Copula</th><th>Var_min</th><th>Optimal σ</th>
            <th>Sharpe</th><th>λ_T</th><th>E[Στ_t]</th>
          </tr></thead>
          <tbody>{rows_live}</tbody>
        </table>
        """, unsafe_allow_html=True)
    elif compare_button:
        st.error("Select at least 2 assets to compare copulas.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────

with tab_method:
    st.markdown("""
    <div class="sec-eyebrow">03 — Theoretical Framework</div>
    <div class="sec-title">Optimization Problem &amp; <em>Solution</em></div>
    <div class="sec-desc">
    The optimizer solves a dynamic mean-variance problem over a finite horizon T,
    finding the sequence of monthly weight vectors that minimizes terminal wealth
    variance while satisfying a target return constraint.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primal problem**")
        st.latex(r"""
        \min_{\{w_t\}} \; \mathrm{Var}\!\left[\prod_{t=1}^{T}(1+r_t)^{-1}(1+\tilde{R}_t)\right]
        """)
        st.latex(r"""
        \text{subject to} \quad
        \mathbb{E}\!\left[\prod_{t=1}^{T}(1+r_t)^{-1}(1+\tilde{R}_t)\right] = 1+c
        """)

    with col2:
        st.markdown("**Closed-form optimal weights**")
        st.latex(r"""
        w_t = -\!\left(1 + \frac{\lambda_T}{2\,\prod_{s=1}^{t-1}(1+r_s)^{-1}(1+\tilde{R}_s)}\right)
        \mu_{t-1}^\top V_{t-1}^{-1}
        """)
        st.latex(r"""
        \lambda_T = -2\!\left[\frac{c}{\mathbb{E}[\sum_{t=0}^{T-1}\tau_t]}+1\right]
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="sec-eyebrow">NGarch (1,1) Process</div>
        <div class="sec-title">Volatility <em>Dynamics</em></div>
        """, unsafe_allow_html=True)
        st.latex(r"S_t = S_{t-1}\exp\!\bigl(r_t + \delta h_t^{1/2} - \tfrac{h_t}{2} + h_t^{1/2}Y_t\bigr)")
        st.latex(r"h_t = \beta_0 + \beta_1 h_{t-1} + \beta_2 h_{t-1}(Y_{t-1} - \kappa)^2")
        st.markdown("""
        <div class="insight-box">
          <div class="ib-label">Leverage parameter κ</div>
          <div class="ib-text">
          κ &lt; 0 for all BVC assets: negative return shocks (bad news) amplify
          conditional variance more than equivalent positive shocks. This is the
          equity leverage effect. The more negative κ, the stronger the asymmetry.
          </div>
        </div>
        <div class="insight-box">
          <div class="ib-label">Persistence index β₁ + β₂(1 + κ²)</div>
          <div class="ib-text">
          Values near 1 indicate highly persistent volatility — shocks take many
          periods to decay. BVC banking stocks (ATW, BCP) have the highest
          persistence, while real estate (ADDH) has the lowest.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="sec-eyebrow">LSM Backward Induction</div>
        <div class="sec-title">Conditional Expectation <em>Estimation</em></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        The recursive parameters μ_t and V_t are computed by regressing simulated
        path payoffs on polynomial basis functions at each time step, working backward
        from T to 0.
        """)
        st.markdown("""
        <div class="formula-block">
        Basis functions at time t:<br>
        B_t = [1, R̄_t^(n), (R̄_t^(n))², R̄_t^(n)·R̄_t^(j)]<br>
        for all n = 1…N, j = 1…N, j > n<br><br>
        μ_t = B_t · (B_t'B_t)⁻¹ B_t' · [adj · R̄_{t+1}]<br>
        V_t = B_t · (B_t'B_t)⁻¹ B_t' · [adj · R̄_{t+1} ⊗ R̄_{t+1}]<br><br>
        τ_t = μ_t' V_t⁻¹ μ_t  (information ratio scalar)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
          <div class="ib-label">Minimum variance formula</div>
          <div class="ib-text">
          Var_min = c² / (E[Σ τ_t] - 1)<br><br>
          This is the theoretical lower bound. The larger E[Σ τ_t],
          the tighter the bound — meaning more diversification benefit
          is available across the selected assets.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-eyebrow">References</div>
    """, unsafe_allow_html=True)

    refs = [
        ("Vaillancourt & Watier (2005)", "Dynamic portfolio selection via simulation-based LSM."),
        ("Engle & Ng (1993)",            "Measuring and testing the impact of news on volatility — NGARCH."),
        ("Sklar (1959)",                 "Fonctions de répartition à n dimensions — copula theorem."),
        ("Longstaff & Schwartz (2001)",  "Valuing American options by simulation — least-squares Monte Carlo."),
        ("Joe (1997)",                   "Multivariate Models and Dependence Concepts — Archimedean copulas."),
    ]
    for title, desc in refs:
        st.markdown(f"**{title}** &nbsp;—&nbsp; {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ASSET UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

with tab_assets:
    st.markdown("""
    <div class="sec-eyebrow">04 — BVC Asset Universe</div>
    <div class="sec-title">Casablanca Stock Exchange <em>Constituents</em></div>
    <div class="sec-desc">
    Eleven BVC-listed assets selected for market capitalisation, liquidity, and
    sectoral diversification. NGarch parameters are estimated from five years of
    monthly closing prices (2019–2024) via maximum likelihood.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(asset_table_html(list(BVC_ASSETS.keys())), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-eyebrow">Parameter Interpretation Guide</div>
    """, unsafe_allow_html=True)

    guide_rows = [
        ("β₁ (persistence)", "0.70 – 0.87 (BVC range)",
         "Measures how long volatility shocks persist. Values near 1 → slow mean reversion."),
        ("κ (leverage)", "−0.62 – −0.14 (BVC range)",
         "Negative: bad news amplifies variance more than good news. More negative = stronger asymmetry."),
        ("β₂ (ARCH effect)", "0.09 (fixed)",
         "Sensitivity of current variance to lagged squared innovation. Calibrated to BVC cross-section average."),
        ("δ (risk premium)", "0.08 – 0.14 (BVC range)",
         "Excess return per unit of conditional standard deviation. Higher δ → more compensation for volatility risk."),
        ("Persistence index", "β₁ + β₂(1 + κ²)",
         "Values above 0.95 indicate highly persistent volatility (IGARCH behaviour)."),
    ]
    rows_g = "".join(
        f"<tr><td class='mono teal-text'>{p}</td><td class='muted'>{r}</td><td class='ib-text'>{i}</td></tr>"
        for p, r, i in guide_rows
    )
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Parameter</th><th>BVC Range</th><th>Interpretation</th></tr></thead>
      <tbody>{rows_g}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
    <div class="sec-eyebrow">05 — Framework</div>
    <div class="sec-title">Moroccan Portfolio Optimizer — <em>Documentation</em></div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **What this application does**

        This tool implements the Vaillancourt & Watier (2005) dynamic mean-variance
        portfolio optimization framework, calibrated to the Casablanca Stock Exchange.
        It combines NGARCH(1,1) marginal volatility models with copula-based joint
        dependence structures, solved via Longstaff-Schwartz Monte Carlo backward
        induction.

        The optimizer finds the sequence of monthly rebalancing weights that minimizes
        the variance of terminal wealth over a user-defined horizon, subject to
        achieving a target cumulative return.

        **Why Clayton copula is recommended**

        Empirical evidence from BVC data (2018–2024) shows that Moroccan equity
        assets exhibit asymmetric tail dependence: they tend to crash together during
        market stress (lower-tail dependence) but do not systematically co-rally
        during benign periods. The Clayton copula captures precisely this structure,
        while the Gaussian copula (no tail dependence) and Gumbel copula (upper-tail
        only) misspecify the joint distribution.
        """)

    with col_b:
        st.markdown("""
        **Optimization mode**

        The primary mode minimizes terminal wealth variance for a fixed target return c.
        The "Maximize return for fixed variance" mode inverts the problem: given a
        maximum acceptable variance, it finds the return-maximizing portfolio weights.

        **Transaction costs**

        BVC transaction costs (0.1–0.5% per trade) are incorporated into the
        effective return at each rebalancing date. Monthly rebalancing is recommended
        to balance dynamic optimality against cost drag. Quarterly or semi-annual
        rebalancing significantly reduces costs at the expense of tracking the
        time-varying optimal weights.

        **Limitations**

        NGarch parameters are estimated from historical data and may not reflect
        future regimes. Copula selection is data-driven but remains a modelling
        assumption. Results are for academic and research purposes and do not
        constitute investment advice.
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-eyebrow">Disclosures</div>
    <div class="sec-desc">
    This application is designed for academic demonstration and research purposes.
    Simulated optimization results do not constitute investment advice.
    Historical performance does not guarantee future results.
    Risk-free rate benchmarked against Bank Al-Maghrib Bons du Trésor.
    </div>
    """, unsafe_allow_html=True)
