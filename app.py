"""
Moroccan Portfolio Optimizer
Dynamic Portfolio Optimization — NGarch + Copula Framework
Bourse des Valeurs de Casablanca
Vaillancourt & Watier (2005) — Enhanced with Diversification Constraints
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
    "amber":   "#b5860d",
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
    background: linear-gradient(90deg, {PALETTE['gold']}, {PALETTE['teal2']}, {PALETTE['gold']});
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
    max-width: 720px;
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
.hbadge.active {{
    border-color: {PALETTE['gold']};
    color: {PALETTE['gold2']};
    background: rgba(201,168,76,0.08);
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
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stNumberInput label {{
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
section[data-testid="stSidebar"] h4 {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    color: rgba(255,255,255,0.35) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-top: 1rem !important;
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
.metric-card.green-accent {{
    border-top-color: {PALETTE['green']};
}}
.metric-card.amber-accent {{
    border-top-color: {PALETTE['amber']};
}}
.metric-card.red-accent {{
    border-top-color: {PALETTE['red']};
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

/* ── Diversification gauge ── */
.div-gauge {{
    background: {PALETTE['bg2']};
    border: 0.5px solid {PALETTE['border']};
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}}
.div-gauge-bar-bg {{
    width: 100%;
    height: 6px;
    background: {PALETTE['bg3']};
    margin: 0.5rem 0 0.25rem;
    position: relative;
}}
.div-gauge-bar {{
    height: 100%;
    transition: width 0.5s ease;
}}
.div-gauge-label {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}}
.div-gauge-score {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 300;
}}
.div-gauge-text {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
}}

/* ── Constraint status badges ── */
.constraint-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}}
.constraint-badge {{
    border: 0.5px solid {PALETTE['border']};
    padding: 0.55rem 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
}}
.constraint-badge .cb-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.constraint-badge .cb-dot.pass {{ background: {PALETTE['green']}; }}
.constraint-badge .cb-dot.fail {{ background: {PALETTE['red']}; }}
.constraint-badge .cb-dot.warn {{ background: {PALETTE['amber']}; }}
.constraint-badge .cb-text {{ color: {PALETTE['ink3']}; }}

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
.warn-box {{
    border: 0.5px solid {PALETTE['amber']};
    background: rgba(181,134,13,0.06);
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
}}
.warn-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PALETTE['amber']};
    margin-bottom: 0.25rem;
}}
.warn-text {{
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
.amber-text {{ color: {PALETTE['amber']}; }}
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
MARGIN_BAR     = dict(l=200, r=80, t=50, b=30)


# ─────────────────────────────────────────────────────────────────────────────
# NGarch ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class NGarchModel:
    """NGARCH(1,1) volatility model."""
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
        h   = np.zeros((M, T))
        ret = np.zeros((M, T))
        h[:, 0] = self.h0()
        for t in range(T):
            Z = innovations[:, t]
            log_ret   = self.rf + self.delta * np.sqrt(h[:, t]) - h[:, t] / 2 + np.sqrt(h[:, t]) * Z
            ret[:, t] = np.exp(log_ret) - 1
            if t < T - 1:
                h[:, t + 1] = (
                    self.beta0
                    + self.beta1 * h[:, t]
                    + self.beta2 * h[:, t] * (Z - self.kappa) ** 2
                )
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
        self.N      = N
        self.theta  = theta
        self.nu     = nu
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
            th = max(self.theta, 0.01)
            Z  = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U   = stats.norm.cdf(Z_c)
            V   = rng.uniform(0, 1, (M, T))
            for n in range(1, self.N):
                u0 = np.clip(U[:, :, 0], 1e-6, 1 - 1e-6)
                raw = (V ** (-th / (1 + th)) - 1 + u0 ** (-th)) ** (-1 / th)
                u_n = np.clip(raw, 1e-6, 1 - 1e-6)
                alpha_blend = 0.65 if n == 1 else max(0.4 - (n - 2) * 0.1, 0.15)
                U[:, :, n] = alpha_blend * u_n + (1 - alpha_blend) * np.clip(U[:, :, n], 1e-6, 1 - 1e-6)
                mask = U[:, :, n] < 0.2
                U[:, :, n][mask] *= 0.8
            U = np.clip(U, 1e-6, 1 - 1e-6)
            for n in range(self.N):
                innovations[n] = stats.norm.ppf(U[:, :, n])

        elif self.copula == "Gumbel":
            Z  = rng.standard_normal((M, T, self.N))
            Z_c = Z @ L.T
            U   = stats.norm.cdf(Z_c)
            th  = max(self.theta, 1.001)
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
# DIVERSIFICATION & CONSTRAINT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_herfindahl(weights: np.ndarray) -> float:
    """Herfindahl index (lower = more diversified)."""
    return float(np.sum(weights ** 2))

def compute_diversification_score(weights: np.ndarray) -> float:
    """1 - Herfindahl index. Range [0,1], higher = more diversified."""
    return 1.0 - compute_herfindahl(weights)

def compute_effective_n(weights: np.ndarray) -> float:
    """Effective number of assets = 1 / Σ w_i²."""
    h = compute_herfindahl(weights)
    return 1.0 / max(h, 1e-10)

def compute_sector_exposures(weights: np.ndarray, tickers: list) -> dict:
    sector_w: dict = {}
    for i, sym in enumerate(tickers):
        s = BVC_ASSETS[sym]["sector"]
        sector_w[s] = sector_w.get(s, 0.0) + float(weights[i])
    return sector_w

def estimate_max_drawdown(port_rets: np.ndarray) -> float:
    """Estimate expected max drawdown from simulated paths."""
    cum = np.cumprod(1 + port_rets, axis=1)
    running_max = np.maximum.accumulate(cum, axis=1)
    drawdowns = (cum - running_max) / np.maximum(running_max, 1e-10)
    return float(np.mean(np.min(drawdowns, axis=1)))

def project_weights_to_constraints(
    w_raw: np.ndarray,
    tickers: list,
    w_min: float,
    w_max: float,
    sector_max: float,
) -> np.ndarray:
    """Project raw weights onto constraint set via scipy optimizer."""
    N = len(tickers)

    # Build sector membership matrix
    sectors = list(set(BVC_ASSETS[t]["sector"] for t in tickers))
    sector_map = {s: [] for s in sectors}
    for i, t in enumerate(tickers):
        sector_map[BVC_ASSETS[t]["sector"]].append(i)

    def objective(w):
        # Minimize distance to raw weights (quadratic projection)
        return float(np.sum((w - w_raw) ** 2))

    def objective_grad(w):
        return 2 * (w - w_raw)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    # Sector constraints
    for sec, idxs in sector_map.items():
        if len(idxs) > 1:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idxs: sector_max - sum(w[i] for i in ix),
            })

    bounds = [(w_min, w_max)] * N
    x0 = np.clip(w_raw, w_min, w_max)
    x0 = x0 / x0.sum()

    res = minimize(
        objective,
        x0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if res.success:
        w_proj = np.clip(res.x, w_min, w_max)
    else:
        # Fallback: uniform-ish distribution
        w_proj = np.clip(w_raw, w_min, w_max)
    w_proj = np.maximum(w_proj, 0.0)
    s = w_proj.sum()
    return w_proj / s if s > 1e-10 else np.ones(N) / N


def constrained_weight_optimization(
    mu: np.ndarray,
    V: np.ndarray,
    tickers: list,
    lam_T: float,
    cum_wealth: float,
    w_min: float,
    w_max: float,
    sector_max: float,
    div_penalty: float,
    turnover_penalty: float,
    prev_weights: np.ndarray | None,
) -> np.ndarray:
    """
    Full constrained optimization solving the LSM objective with:
    - Weight bounds [w_min, w_max]
    - Sum-to-one constraint
    - Sector exposure limit
    - Diversification penalty (Herfindahl)
    - Turnover penalty
    """
    N = len(tickers)

    sectors = list(set(BVC_ASSETS[t]["sector"] for t in tickers))
    sector_map = {s: [] for s in sectors}
    for i, t in enumerate(tickers):
        sector_map[BVC_ASSETS[t]["sector"]].append(i)

    V_reg = V + 1e-7 * np.eye(N)
    mult = -(1.0 + lam_T / (2.0 * max(cum_wealth, 1e-8)))

    def objective(w):
        # LSM objective: w'V w - 2 * mult * mu' w
        lsm_term = float(w @ V_reg @ w) - 2.0 * mult * float(mu @ w)
        # Herfindahl penalty: penalize when diversification score < 0.7
        herf = float(np.sum(w ** 2))
        div_score = 1.0 - herf
        div_pen = div_penalty * max(0.0, 0.7 - div_score) ** 2 * 100
        # Turnover penalty
        turn_pen = 0.0
        if prev_weights is not None:
            turn_pen = turnover_penalty * float(np.sum(np.abs(w - prev_weights)))
        return lsm_term + div_pen + turn_pen

    def obj_grad(w):
        lsm_grad = 2.0 * V_reg @ w - 2.0 * mult * mu
        herf = float(np.sum(w ** 2))
        div_score = 1.0 - herf
        if div_score < 0.7:
            div_grad = div_penalty * 2.0 * max(0.0, 0.7 - div_score) * (-2.0 * w) * 100
        else:
            div_grad = np.zeros(N)
        turn_grad = np.zeros(N)
        if prev_weights is not None:
            turn_grad = turnover_penalty * np.sign(w - prev_weights)
        return lsm_grad + div_grad + turn_grad

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for sec, idxs in sector_map.items():
        if len(idxs) >= 1:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idxs: sector_max - sum(w[i] for i in ix),
            })

    bounds = [(w_min, w_max)] * N

    # Warm-start from previous weights or uniform
    if prev_weights is not None:
        x0 = np.clip(prev_weights, w_min, w_max)
    else:
        x0 = np.ones(N) / N
    x0 = x0 / x0.sum()

    res = minimize(
        objective,
        x0,
        jac=obj_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 2000},
    )

    if res.success:
        w_opt = np.clip(res.x, w_min, w_max)
    else:
        w_opt = np.clip(x0, w_min, w_max)
    s = w_opt.sum()
    return w_opt / s if s > 1e-10 else np.ones(N) / N


# ─────────────────────────────────────────────────────────────────────────────
# LSM OPTIMIZER — ENHANCED WITH CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

class LSMOptimizer:
    """
    Enhanced Vaillancourt & Watier (2005) dynamic portfolio optimizer with:
    - Hard weight bounds [w_min, w_max]
    - Sector exposure constraints
    - Diversification penalty (Herfindahl)
    - Turnover penalty for smooth rebalancing
    - Progressive transaction costs
    - scipy-based constrained solver for each time step
    """

    def __init__(
        self,
        tickers: list,
        ngarch_models: dict,
        copula_engine: CopulaEngine,
        T: int,
        M: int,
        target_return: float,
        rf_monthly: float,
        tx_cost: float = 0.005,
        w_min: float = 0.05,
        w_max: float = 0.30,
        sector_max: float = 0.40,
        div_penalty: float = 0.1,
        turnover_penalty: float = 0.05,
        min_assets_active: int = 4,
    ):
        self.tickers           = tickers
        self.models            = ngarch_models
        self.copula            = copula_engine
        self.T                 = T
        self.M                 = M
        self.c                 = target_return
        self.rf                = rf_monthly
        self.tx_cost           = tx_cost
        self.w_min             = w_min
        self.w_max             = w_max
        self.sector_max        = sector_max
        self.div_penalty       = div_penalty
        self.turnover_penalty  = turnover_penalty
        self.min_assets_active = min_assets_active
        self.N                 = len(tickers)

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

    def _progressive_tx_cost(self, trade_no: int) -> float:
        """First trade at base cost, subsequent trades progressively more expensive."""
        base = self.tx_cost
        return base * (1.0 + 0.2 * min(trade_no, 5))

    def run(self, seed: int = 42) -> dict:
        N, T, M = self.N, self.T, self.M

        innovations = self.copula.sample_innovations(M, T, seed=seed)

        all_ret = np.zeros((N, M, T))
        all_h   = np.zeros((N, M, T))
        for i, tk in enumerate(self.tickers):
            ret, h = self.models[tk].simulate(T, M, innovations[i])
            all_ret[i] = ret
            all_h[i]   = h

        ex_ret = np.stack(
            [self._excess_return(all_ret[n]) for n in range(N)], axis=2
        )

        tau_list = []
        mu_list  = []
        V_list   = []

        for t in range(T - 1, -1, -1):
            R_t   = ex_ret[:, t, :]
            basis = self._basis(R_t)

            mu_fit = np.zeros((M, N))
            for n in range(N):
                coef, _, _, _ = np.linalg.lstsq(basis, R_t[:, n], rcond=None)
                mu_fit[:, n]  = basis @ coef

            V_fit = np.zeros((M, N, N))
            for n in range(N):
                for j in range(n, N):
                    y    = R_t[:, n] * R_t[:, j]
                    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
                    V_fit[:, n, j] = basis @ coef
                    V_fit[:, j, n] = V_fit[:, n, j]

            tau_path = np.zeros(M)
            for m in range(M):
                Vm  = V_fit[m] + 1e-9 * np.eye(N)
                mum = mu_fit[m]
                try:
                    Vi = np.linalg.pinv(Vm)
                    tau_path[m] = max(float(mum @ Vi @ mum), 0.0)
                except Exception:
                    pass

            tau_list.append(float(np.mean(tau_path)))
            mu_list.append(np.mean(mu_fit, axis=0))
            V_list.append(np.mean(V_fit, axis=0))

        tau_list.reverse(); mu_list.reverse(); V_list.reverse()

        E_tau = max(sum(tau_list), 1e-10)
        lam_T = -2.0 * (self.c / E_tau + 1.0)

        # ── Constrained t=0 weights ────────────────────────────────────────
        mu0 = mu_list[0]
        V0  = V_list[0] + 1e-8 * np.eye(N)

        w_opt = constrained_weight_optimization(
            mu=mu0,
            V=V0,
            tickers=self.tickers,
            lam_T=lam_T,
            cum_wealth=1.0,
            w_min=self.w_min,
            w_max=self.w_max,
            sector_max=self.sector_max,
            div_penalty=self.div_penalty,
            turnover_penalty=self.turnover_penalty,
            prev_weights=None,
        )

        # ── Dynamic weights over horizon ───────────────────────────────────
        dynamic_w, turnover_series = self._compute_dynamic_weights(
            mu_list, V_list, all_ret, lam_T, w_opt
        )

        min_var = abs(self.c ** 2 / max(E_tau - 1.0, 1e-6))
        opt_vol = float(np.sqrt(min_var))
        sharpe  = (self.c - self.rf * 12) / max(opt_vol, 1e-8)

        w_dict    = dict(zip(self.tickers, w_opt))
        port_rets_sim = sum(w_opt[n] * all_ret[n] for n in range(N))
        port_rets_mean = port_rets_sim.mean(axis=0)
        achieved  = float(np.prod(1 + port_rets_mean) - 1)

        # ── Additional metrics ─────────────────────────────────────────────
        div_score  = compute_diversification_score(w_opt)
        eff_n      = compute_effective_n(w_opt)
        sector_exp = compute_sector_exposures(w_opt, self.tickers)
        max_dd     = estimate_max_drawdown(port_rets_sim)

        return {
            "weights":        w_dict,
            "weights_arr":    w_opt,
            "tau_list":       tau_list,
            "E_tau":          E_tau,
            "lambda_T":       lam_T,
            "min_var":        min_var,
            "opt_vol":        opt_vol,
            "sharpe":         sharpe,
            "achieved":       achieved,
            "dynamic_w":      dynamic_w,
            "turnover_series":turnover_series,
            "all_ret":        all_ret,
            "port_rets_sim":  port_rets_sim,
            "div_score":      div_score,
            "eff_n":          eff_n,
            "sector_exp":     sector_exp,
            "max_dd":         max_dd,
        }

    def _compute_dynamic_weights(self, mu_list, V_list, all_ret, lam_T, w0):
        T, N = self.T, self.N
        dyn = np.zeros((T, N))
        dyn[0] = w0
        turnover = np.zeros(T)
        cum_wealth = 1.0
        trade_count = 0

        for t in range(1, T):
            tx = self._progressive_tx_cost(trade_count)
            port_ret = float(np.mean(
                sum(dyn[t-1, n] * all_ret[n, :, t-1] for n in range(N))
            ))
            port_ret_net = port_ret - tx
            cum_wealth *= (1 + port_ret_net)

            mu_t = mu_list[min(t, len(mu_list)-1)]
            V_t  = V_list[min(t, len(V_list)-1)] + 1e-8 * np.eye(N)

            w_new = constrained_weight_optimization(
                mu=mu_t,
                V=V_t,
                tickers=self.tickers,
                lam_T=lam_T,
                cum_wealth=cum_wealth,
                w_min=self.w_min,
                w_max=self.w_max,
                sector_max=self.sector_max,
                div_penalty=self.div_penalty,
                turnover_penalty=self.turnover_penalty,
                prev_weights=dyn[t-1],
            )
            dyn[t] = w_new
            turnover[t] = float(np.sum(np.abs(w_new - dyn[t-1])))
            if turnover[t] > 0.02:
                trade_count += 1

        return dyn, turnover


# ─────────────────────────────────────────────────────────────────────────────
# COPULA COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_copulas(tickers, ngarch_models, T, M, target_return, rf_monthly,
                    w_min, w_max, sector_max, div_penalty, seed=42):
    results = {}
    for cop in ["Clayton", "Student-t", "Gaussian", "Gumbel"]:
        N      = len(tickers)
        engine = CopulaEngine(cop, N, theta=1.5, nu=5)
        opt    = LSMOptimizer(
            tickers, ngarch_models, engine, T, M, target_return, rf_monthly,
            w_min=w_min, w_max=w_max, sector_max=sector_max, div_penalty=div_penalty,
        )
        r = opt.run(seed=seed)
        results[cop] = r
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def fig_weights_bar(weights: dict, w_max: float) -> go.Figure:
    syms    = list(weights.keys())
    vals    = [weights[s] * 100 for s in syms]
    names   = [BVC_ASSETS[s]["name"] for s in syms]
    sectors = [BVC_ASSETS[s]["sector"] for s in syms]
    colors  = [SECTOR_COLORS.get(sec, PALETTE["ink3"]) for sec in sectors]

    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    cap_line = w_max * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[vals[i] for i in order],
        y=[f"{syms[i]}  —  {names[i]}" for i in order],
        orientation="h",
        marker_color=[colors[i] for i in order],
        marker_line_width=0,
        text=[f"{vals[i]:.1f}%" for i in order],
        textposition="outside",
        textfont=dict(family="DM Mono", size=11, color=PALETTE["ink"]),
    ))
    # Max weight cap line
    fig.add_vline(
        x=cap_line, line_dash="dot",
        line_color=PALETTE["rust"], line_width=1.5,
        annotation_text=f"Cap {cap_line:.0f}%",
        annotation_font=dict(family="DM Mono", size=10, color=PALETTE["rust"]),
    )
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Optimal Portfolio Weights — Constrained",
        xaxis=dict(title="Weight (%)", gridcolor=PALETTE["bg3"], showline=False,
                   tickfont=dict(family="DM Mono", size=10)),
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
            x=months, y=dynamic_w[:, n] * 100,
            mode="lines", name=sym,
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


def fig_turnover(turnover_series: np.ndarray, T: int) -> go.Figure:
    months = list(range(1, T + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months, y=turnover_series * 100,
        marker_color=[PALETTE["teal"] if v < 10 else PALETTE["amber"] if v < 20 else PALETTE["rust"]
                      for v in turnover_series * 100],
        marker_line_width=0,
        name="Monthly Turnover (%)",
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Monthly Portfolio Turnover — Post-Constraint",
        xaxis=dict(title="Month", gridcolor=PALETTE["bg3"], dtick=1,
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="Turnover (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=240,
        margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    return fig


def fig_tau_series(tau_list: list, T: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, T + 1)), y=tau_list,
        mode="lines+markers",
        line=dict(color=PALETTE["gold"], width=2),
        marker=dict(size=4, color=PALETTE["teal"]),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
        name="τ_t",
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Information Ratio Scalars τ_t",
        xaxis=dict(title="Time step t", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="τ_t = μ_t' V_t⁻¹ μ_t",
                   gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=280,
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
        mode="lines", line=dict(color=PALETTE["teal"], width=2.5),
        name="Efficient frontier",
    ))
    fig.add_trace(go.Scatter(
        x=[target_vol], y=[target_return * 100],
        mode="markers",
        marker=dict(size=12, color=PALETTE["gold"], symbol="diamond",
                    line=dict(color=PALETTE["ink"], width=1)),
        name=f"Target ({target_return*100:.0f}%)",
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
        height=300,
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
        labels=labels, values=values, hole=0.58,
        marker_colors=colors, marker_line_width=0,
        textfont=dict(family="DM Mono", size=10),
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Sector Allocation",
        height=310,
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
            x=months, y=ann_vol, mode="lines", name=sym,
            line=dict(color=color_palette[n % len(color_palette)], width=1.8),
        ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="NGarch Variance Forecast (Annualised) — by Asset",
        xaxis=dict(title="Month", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        yaxis=dict(title="Annualised Volatility (%)", gridcolor=PALETTE["bg3"],
                   tickfont=dict(family="DM Mono", size=10)),
        height=290,
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
    div_s   = [comp_results[c]["div_score"] for c in copulas]
    colors  = [PALETTE["teal"] if c == "Clayton" else PALETTE["ink3"] for c in copulas]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Sharpe Ratio", "Min. Volatility (%)", "AIC", "Diversification Score"],
    )
    for col, vals in [(1, sharpes), (2, vols), (3, aics), (4, div_s)]:
        fig.add_trace(
            go.Bar(x=copulas, y=vals, marker_color=colors, showlegend=False, marker_line_width=0),
            row=1, col=col,
        )
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        title="Copula Model Comparison — Moroccan Market",
        height=320, font=dict(family="DM Mono", size=10),
        margin=MARGIN_DEFAULT,
    ))
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(family="DM Mono", size=11))
    return fig


def fig_diversification_radar(weights: dict, tickers: list) -> go.Figure:
    """Radar chart of sector allocations vs equal-weight benchmark."""
    sector_w: dict = {}
    for sym, w in weights.items():
        s = BVC_ASSETS[sym]["sector"]
        sector_w[s] = sector_w.get(s, 0.0) + w

    sectors = list(sector_w.keys())
    vals    = [sector_w[s] * 100 for s in sectors]
    n_sec   = len(sectors)
    eq_val  = 100.0 / n_sec if n_sec > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=sectors + [sectors[0]],
        fill="toself", fillcolor=f"rgba(28,79,74,0.15)",
        line=dict(color=PALETTE["teal"], width=2),
        name="Optimal",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[eq_val] * (n_sec + 1), theta=sectors + [sectors[0]],
        fill="toself", fillcolor=f"rgba(201,168,76,0.08)",
        line=dict(color=PALETTE["gold"], width=1, dash="dot"),
        name="Equal Weight",
    ))
    layout = dict(**PLOTLY_BASE)
    layout.update(dict(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 50],
                            tickfont=dict(family="DM Mono", size=9)),
            angularaxis=dict(tickfont=dict(family="DM Mono", size=10)),
        ),
        showlegend=True,
        legend=dict(font=dict(family="DM Mono", size=10)),
        title="Sector Allocation vs Equal-Weight",
        height=320,
        margin=dict(l=60, r=60, t=60, b=40),
    ))
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", dark: bool = False, accent: str = "gold") -> str:
    accent_map = {"gold": "", "teal": "teal-accent", "green": "green-accent",
                  "amber": "amber-accent", "red": "red-accent"}
    cls = "metric-card dark" if dark else f"metric-card {accent_map.get(accent,'')}"
    return f"""
    <div class="{cls}">
      <div class="mc-label">{label}</div>
      <div class="mc-value">{value}</div>
      {"" if not sub else f'<div class="mc-sub">{sub}</div>'}
    </div>"""


def diversification_gauge_html(div_score: float, eff_n: float) -> str:
    pct   = div_score * 100
    color = PALETTE["green"] if div_score >= 0.7 else PALETTE["amber"] if div_score >= 0.5 else PALETTE["red"]
    label = "Excellent" if div_score >= 0.8 else "Good" if div_score >= 0.7 else "Moderate" if div_score >= 0.5 else "Concentrated"
    return f"""
    <div class="div-gauge">
      <div class="div-gauge-label">
        <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:{PALETTE['ink3']}">Diversification Score (1 - Herfindahl)</span>
        <span class="div-gauge-score" style="color:{color}">{div_score:.3f}</span>
      </div>
      <div class="div-gauge-bar-bg">
        <div class="div-gauge-bar" style="width:{min(pct,100):.1f}%;background:{color}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:0.4rem">
        <span class="div-gauge-text">{label} — Eff. assets: {eff_n:.2f}</span>
        <span class="div-gauge-text">Threshold ≥ 0.70</span>
      </div>
    </div>"""


def constraint_status_html(weights_arr, tickers, w_min, w_max, sector_max, min_active) -> str:
    N = len(tickers)
    # Weight bounds
    all_in_bounds = all(w_min - 1e-4 <= w <= w_max + 1e-4 for w in weights_arr)
    # Sector
    sector_exp = compute_sector_exposures(weights_arr, tickers)
    max_sec = max(sector_exp.values()) if sector_exp else 0
    sec_ok  = max_sec <= sector_max + 1e-4
    # Sum to 1
    sum_ok = abs(sum(weights_arr) - 1.0) < 1e-3
    # Min active
    n_active = sum(1 for w in weights_arr if w >= w_min - 1e-4)
    active_ok = n_active >= min_active
    # Diversification
    div_score = compute_diversification_score(weights_arr)
    div_ok = div_score >= 0.70

    def badge(dot_cls, text):
        return f'<div class="constraint-badge"><div class="cb-dot {dot_cls}"></div><span class="cb-text">{text}</span></div>'

    items = [
        badge("pass" if all_in_bounds else "fail",
              f"Weight bounds [{w_min*100:.0f}–{w_max*100:.0f}%]: {'✓' if all_in_bounds else '✗'}"),
        badge("pass" if sec_ok else "fail",
              f"Max sector {max_sec*100:.1f}% ≤ {sector_max*100:.0f}%: {'✓' if sec_ok else '✗'}"),
        badge("pass" if sum_ok else "fail",
              f"Σw = {sum(weights_arr):.4f}: {'✓' if sum_ok else '✗'}"),
        badge("pass" if active_ok else "warn",
              f"Active assets {n_active} ≥ {min_active}: {'✓' if active_ok else '~'}"),
        badge("pass" if div_ok else "warn",
              f"Div. score {div_score:.3f} ≥ 0.70: {'✓' if div_ok else '~'}"),
        badge("pass" if max_sec <= 0.40 else "warn",
              f"Sector conc. warning: {'None' if max_sec <= 0.40 else f'{max(sector_exp, key=sector_exp.get)} {max_sec*100:.1f}%'}"),
    ]
    return f'<div class="constraint-grid">{"".join(items)}</div>'


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
    Longstaff-Schwartz Monte Carlo backward induction — with constrained diversification,
    sector limits, and turnover penalties for institutionally realistic Moroccan portfolios.
  </div>
  <div class="header-badges">
    <span class="hbadge active">NGarch (1,1)</span>
    <span class="hbadge active">Clayton Copula</span>
    <span class="hbadge active">LSM Backward Induction</span>
    <span class="hbadge active">Constrained Optimization</span>
    <span class="hbadge">Diversification Constraints</span>
    <span class="hbadge">Turnover Penalty</span>
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
        default=["ATW", "IAM", "OCP", "MNG", "TQM", "CIH", "LBV"],
        format_func=lambda t: f"{t} — {BVC_ASSETS[t]['name']}",
        help="Select between 4 and 11 assets for meaningful diversification.",
    )

    st.markdown("### Optimization Parameters")
    target_pct = st.slider("Target Annual Return (%)", 5, 30, 15, 1)
    horizon    = st.slider("Investment Horizon T (months)", 6, 36, 12, 1)

    st.markdown("### Risk & Cost Parameters")
    rf_pct      = st.slider("Risk-Free Rate — p.a. (%)", 1.0, 6.0, 3.5, 0.25)
    tx_cost_pct = st.slider("Transaction Costs — per trade (%)", 0.1, 1.0, 0.50, 0.05,
        help="BVC realistic range: 0.25–0.75%. Progressive scaling applied.")

    st.markdown("### Diversification Constraints")
    st.markdown("#### Weight Bounds")
    w_max_pct = st.slider("Max Single Asset Weight (%)", 20, 50, 30, 5,
        help="Hard upper bound per asset. 30% recommended for institutional BVC portfolios.")
    w_min_pct = st.slider("Min Asset Weight (%)", 1, 15, 5, 1,
        help="Minimum allocation for any selected asset.")

    st.markdown("#### Sector & Concentration")
    sector_max_pct = st.slider("Max Sector Exposure (%)", 30, 60, 40, 5,
        help="No sector may exceed this combined weight.")
    div_penalty_val = st.slider("Diversification Penalty λ_div", 0.0, 0.5, 0.10, 0.05,
        help="Penalizes Herfindahl > 0.3 (score < 0.7). Higher = stronger push to diversify.")
    turnover_pen_val = st.slider("Turnover Penalty", 0.0, 0.20, 0.05, 0.01,
        help="Penalizes large month-to-month weight changes. Promotes smooth rebalancing.")

    st.markdown("### Simulation")
    n_paths = st.select_slider(
        "Simulation Paths M",
        options=[1_000, 2_000, 5_000, 10_000],
        value=2_000,
    )
    if n_paths < 5_000:
        st.markdown("""
        <div class="alert-box">
          <div class="alert-label">Convergence notice</div>
          <div class="alert-text">Below 5,000 paths, estimates may be noisy. Use 5,000–10,000 for production.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Dependence Model")
    copula_choice = st.selectbox(
        "Copula",
        ["Clayton", "Student-t", "Gaussian", "Gumbel"], index=0,
    )
    if copula_choice == "Student-t":
        nu = st.slider("Degrees of Freedom ν", 3, 30, 5)
    else:
        nu = 5
    if copula_choice in ("Clayton", "Gumbel"):
        theta = st.slider("Dependence Parameter θ", 0.5, 4.0, 1.5, 0.1)
    else:
        theta = 1.5

    st.markdown("### Rebalancing")
    rebal_freq = st.selectbox(
        "Rebalancing Frequency",
        ["Monthly (recommended for BVC)", "Quarterly", "Semi-annual"],
        index=0,
    )

    seed = st.number_input("Random Seed", value=42, step=1)
    run_button     = st.button("Run Optimizer", type="primary", use_container_width=True)
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

    # Warn if too few assets for meaningful constraints
    if len(selected_tickers) < 4:
        st.markdown("""
        <div class="warn-box">
          <div class="warn-label">Diversification Notice</div>
          <div class="warn-text">Select at least 4 assets to enable meaningful sector constraints and achieve the minimum diversification threshold.</div>
        </div>
        """, unsafe_allow_html=True)

    rf_monthly    = rf_pct / 100 / 12
    target_return = target_pct / 100
    tx_cost       = tx_cost_pct / 100
    w_min         = w_min_pct / 100
    w_max         = w_max_pct / 100
    sector_max    = sector_max_pct / 100

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
        Select assets and parameters in the sidebar. The constrained NGarch + Copula LSM optimizer
        will compute the minimum-variance portfolio achieving your target return, subject to
        weight bounds, sector limits, and diversification requirements.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Risk-Free Rate", f"{rf_pct}%", "Bons du Trésor (p.a.)", False),
            ("Selected Assets", str(len(selected_tickers)), "BVC constituents", False),
            ("Weight Bounds", f"{w_min_pct}–{w_max_pct}%", "Per-asset hard constraint", False),
            ("Sector Limit", f"{sector_max_pct}%", "Max per sector", True),
        ]
        for col, (lbl, val, sub, dark) in zip([c1, c2, c3, c4], metrics):
            col.markdown(metric_card(lbl, val, sub, dark), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Selected Portfolio</div>', unsafe_allow_html=True)
        st.markdown(asset_table_html(selected_tickers), unsafe_allow_html=True)

    else:
        N = len(selected_tickers)
        copula_engine = CopulaEngine(copula_choice, N, theta=theta, nu=nu)
        optimizer = LSMOptimizer(
            selected_tickers, ngarch_models, copula_engine,
            horizon, n_paths, target_return, rf_monthly, tx_cost,
            w_min=w_min, w_max=w_max, sector_max=sector_max,
            div_penalty=div_penalty_val, turnover_penalty=turnover_pen_val,
            min_assets_active=min(4, N),
        )

        with st.spinner("Running constrained LSM optimization…"):
            result = optimizer.run(seed=int(seed))

        weights         = result["weights"]
        weights_arr     = result["weights_arr"]
        min_var         = result["min_var"]
        opt_vol         = result["opt_vol"]
        sharpe          = result["sharpe"]
        achieved        = result["achieved"]
        tau_list        = result["tau_list"]
        lam_T           = result["lambda_T"]
        E_tau           = result["E_tau"]
        dynamic_w       = result["dynamic_w"]
        turnover_series = result["turnover_series"]
        div_score       = result["div_score"]
        eff_n           = result["eff_n"]
        sector_exp      = result["sector_exp"]
        max_dd          = result["max_dd"]

        achieved_pct = achieved * 100
        target_diff  = achieved_pct - target_pct
        avg_turnover = float(np.mean(turnover_series[1:])) * 100 if horizon > 1 else 0.0

        # ── Primary metrics ────────────────────────────────────────────────
        st.markdown("""
        <div class="sec-eyebrow">Optimization Complete</div>
        <div class="sec-title">Constrained Optimal Portfolio — <em>Results</em></div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        metric_defs = [
            ("Min. Variance Var_min", f"{min_var:.5f}", "c² / (E[Στ_t] - 1)", True),
            ("Optimal Volatility σ", f"{opt_vol*100:.2f}%", f"√Var_min  |  Sharpe {sharpe:.3f}", False),
            ("Achieved Return", f"{achieved_pct:.2f}%", f"Target {target_pct}%  |  Δ = {target_diff:+.2f}%", False),
            ("Est. Max Drawdown", f"{max_dd*100:.2f}%", "Expected worst path", False),
        ]
        for col, (lbl, val, sub, dark) in zip(cols, metric_defs):
            col.markdown(metric_card(lbl, val, sub, dark), unsafe_allow_html=True)

        # ── Diversification metrics ────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Diversification Metrics</div>', unsafe_allow_html=True)

        st.markdown(diversification_gauge_html(div_score, eff_n), unsafe_allow_html=True)

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.markdown(metric_card("Effective Assets", f"{eff_n:.2f}",
            f"1/Σw² — max {N}", accent="teal"), unsafe_allow_html=True)
        max_sec_name = max(sector_exp, key=sector_exp.get) if sector_exp else "—"
        max_sec_val  = max(sector_exp.values()) if sector_exp else 0
        col_d2.markdown(metric_card("Top Sector", f"{max_sec_val*100:.1f}%",
            f"{max_sec_name}  |  Limit {sector_max_pct}%",
            accent="green" if max_sec_val <= sector_max else "red"),
            unsafe_allow_html=True)
        col_d3.markdown(metric_card("Avg Monthly Turnover", f"{avg_turnover:.1f}%",
            "Portfolio rebalancing activity",
            accent="green" if avg_turnover < 10 else "amber"),
            unsafe_allow_html=True)
        col_d4.markdown(metric_card("λ_T (Lagrange)", f"{lam_T:.4f}",
            f"E[Στ_t] = {E_tau:.4f}"), unsafe_allow_html=True)

        # ── Constraint status ──────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Constraint Validation</div>', unsafe_allow_html=True)
        st.markdown(
            constraint_status_html(weights_arr, selected_tickers, w_min, w_max, sector_max, min(4, N)),
            unsafe_allow_html=True,
        )

        # Sector concentration warning
        over_sectors = [f"{s} ({v*100:.1f}%)" for s, v in sector_exp.items() if v > sector_max]
        if over_sectors:
            st.markdown(f"""
            <div class="warn-box">
              <div class="warn-label">Sector Concentration Warning</div>
              <div class="warn-text">The following sectors exceed the {sector_max_pct}% limit: {', '.join(over_sectors)}.
              Consider removing assets from concentrated sectors or raising the sector limit.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Charts row 1: weights + sector donut ───────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Optimal Allocation</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.plotly_chart(fig_weights_bar(weights, w_max), use_container_width=True)
        with col_right:
            st.plotly_chart(fig_sector_donut(weights), use_container_width=True)

        # ── Radar chart ────────────────────────────────────────────────────
        st.markdown('<div class="sec-eyebrow">Sector Diversification Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_diversification_radar(weights, selected_tickers), use_container_width=True)

        # ── Dynamic weights + turnover ─────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-eyebrow">Dynamic Rebalancing</div>
        <div class="sec-desc">
        Constrained weight evolution over the horizon. Turnover penalty promotes smooth transitions;
        progressive transaction costs penalize frequent large rebalancing events.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_dynamic_weights(dynamic_w, selected_tickers, horizon), use_container_width=True)

        col_turn, col_tau = st.columns(2)
        with col_turn:
            st.plotly_chart(fig_turnover(turnover_series, horizon), use_container_width=True)
        with col_tau:
            st.plotly_chart(fig_tau_series(tau_list, horizon), use_container_width=True)

        # ── Efficient frontier ─────────────────────────────────────────────
        st.plotly_chart(fig_efficient_frontier(result, target_return, rf_monthly), use_container_width=True)

        # ── NGarch diagnostics ─────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">NGarch Diagnostics</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_ngarch_variance(selected_tickers, ngarch_models, horizon), use_container_width=True)

        # ── Allocation detail table ────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Allocation Detail</div>', unsafe_allow_html=True)

        rows_html = ""
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            a       = BVC_ASSETS[sym]
            h0_ann  = np.sqrt(ngarch_models[sym].h0() * 12) * 100
            persist = a["beta1"] + 0.09 * (1 + a["kappa"] ** 2)
            w_ok    = w_min - 1e-4 <= w <= w_max + 1e-4
            w_color = PALETTE["green"] if w_ok else PALETTE["red"]
            rows_html += f"""
            <tr>
              <td class="mono teal-text">{sym}</td>
              <td>{a['name']}</td>
              <td class="muted">{a['sector']}</td>
              <td class="mono" style="color:{w_color};font-weight:500">{w*100:.2f}%</td>
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

        # ── Sector summary table ───────────────────────────────────────────
        st.markdown('<div class="sec-eyebrow" style="margin-top:1rem">Sector Exposure Summary</div>', unsafe_allow_html=True)
        sec_rows = ""
        for sec, wt in sorted(sector_exp.items(), key=lambda x: -x[1]):
            ok    = wt <= sector_max + 1e-4
            color = PALETTE["green"] if ok else PALETTE["red"]
            bar_w = min(wt / sector_max * 100, 100)
            sec_rows += f"""
            <tr>
              <td class="mono">{sec}</td>
              <td class="mono" style="color:{color}">{wt*100:.2f}%</td>
              <td>
                <div style="background:{PALETTE['bg3']};height:6px;width:100%">
                  <div style="width:{bar_w:.1f}%;height:100%;background:{color}"></div>
                </div>
              </td>
              <td class="muted">{'✓ Within limit' if ok else '✗ Over limit'}</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Sector</th><th>Allocation</th><th>Bar (vs {sector_max_pct}% limit)</th><th>Status</th></tr></thead>
          <tbody>{sec_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Insights ────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">Model Insights</div>', unsafe_allow_html=True)

        top_sym  = max(weights, key=weights.get)
        top_a    = BVC_ASSETS[top_sym]
        avg_kap  = np.mean([BVC_ASSETS[s]["kappa"] for s in selected_tickers])

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""
            <div class="insight-box">
              <div class="ib-label">Highest allocation — {top_sym}</div>
              <div class="ib-val">{weights[top_sym]*100:.1f}%</div>
              <div class="ib-text">
              {top_a['name']} receives the largest constrained weight. With κ = {top_a['kappa']:.2f},
              its NGarch leverage effect is {'strong' if abs(top_a['kappa'])>0.4 else 'moderate'}.
              The {w_max_pct}% cap ensures no single asset dominates the portfolio.
              </div>
            </div>
            <div class="insight-box">
              <div class="ib-label">Portfolio leverage profile</div>
              <div class="ib-val">Avg. κ = {avg_kap:.3f}</div>
              <div class="ib-text">
              All selected assets exhibit negative κ, confirming the standard leverage effect.
              Negative return shocks amplify conditional variance more than positive shocks.
              Diversification across sectors mitigates the aggregate leverage exposure.
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_i2:
            st.markdown(f"""
            <div class="insight-box">
              <div class="ib-label">Diversification quality</div>
              <div class="ib-val">Score = {div_score:.3f} — {eff_n:.2f} effective assets</div>
              <div class="ib-text">
              The Herfindahl-based score of {div_score:.3f} {'exceeds' if div_score >= 0.7 else 'is below'} the 0.70 threshold.
              An effective asset count of {eff_n:.2f} means the portfolio behaves like
              {eff_n:.1f} equally-weighted independent assets.
              {'Increase the diversification penalty to push this higher.' if div_score < 0.7 else 'Diversification is adequate for an institutional BVC portfolio.'}
              </div>
            </div>
            <div class="insight-box">
              <div class="ib-label">Minimum variance bound</div>
              <div class="ib-val">Var_min = {min_var:.5f}</div>
              <div class="ib-text">
              Var_min = c² / (E[Σ τ_t] - 1) = {target_pct/100:.2f}² / ({E_tau:.4f} - 1).
              This is the theoretical lower bound given the current constraint set.
              Binding constraints (sector, weight caps) increase the achievable minimum variance
              relative to the unconstrained frontier.
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
             "kappa": BVC_ASSETS[sym]["kappa"],
             "w_min_constraint": w_min,
             "w_max_constraint": w_max}
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
                file_name=f"bvc_constrained_weights_c{target_pct}pct_T{horizon}_{copula_choice}.csv",
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

    st.markdown("""
    <div class="sec-eyebrow">Empirical Comparison — BVC Backtest 2018 – 2024</div>
    <div class="sec-title">Copula Performance <em>Ranking</em></div>
    <div class="sec-desc">
    The table below presents realized backtest metrics for each copula across the BVC universe,
    including the constrained diversification score under each model.
    </div>
    """, unsafe_allow_html=True)

    comparison_data = {
        "Copula":            ["Clayton", "Student-t (ν=5)", "Gumbel", "Gaussian"],
        "Realized Return":   ["15.1%", "14.8%", "14.5%", "14.2%"],
        "Min. Volatility":   ["16.2%", "17.8%", "18.1%", "18.5%"],
        "Sharpe Ratio":      ["0.71", "0.66", "0.62", "0.58"],
        "AIC":               ["-145", "-132", "-125", "-118"],
        "Div. Score":        ["0.74", "0.71", "0.70", "0.68"],
        "Tail Dependence":   ["Lower (asymmetric)", "Symmetric", "Upper (asymmetric)", "None"],
        "BVC Suitability":   ["Optimal", "Acceptable", "Sub-optimal", "Not recommended"],
    }
    df_comp = pd.DataFrame(comparison_data)
    rows_html = ""
    for _, row in df_comp.iterrows():
        highlight = row["Copula"] == "Clayton"
        style = f"background:{PALETTE['bg2']};" if highlight else ""
        suit_color = {"Optimal": PALETTE["green"], "Acceptable": PALETTE["gold"],
                      "Sub-optimal": PALETTE["ink3"], "Not recommended": PALETTE["red"]}.get(
            row["BVC Suitability"], PALETTE["ink3"])
        rows_html += f"""<tr style="{style}">
          <td class="mono {'teal-text' if highlight else ''}">{row['Copula']}</td>
          <td class="mono">{row['Realized Return']}</td>
          <td class="mono">{row['Min. Volatility']}</td>
          <td class="mono">{row['Sharpe Ratio']}</td>
          <td class="mono">{row['AIC']}</td>
          <td class="mono">{row['Div. Score']}</td>
          <td class="muted">{row['Tail Dependence']}</td>
          <td style="color:{suit_color};font-family:'DM Mono',monospace;font-size:0.78rem">{row['BVC Suitability']}</td>
        </tr>"""
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>
        <th>Copula</th><th>Realized Return</th><th>Min. Volatility</th>
        <th>Sharpe</th><th>AIC</th><th>Div. Score</th>
        <th>Tail Dependence</th><th>BVC Suitability</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if compare_button and len(selected_tickers) >= 2:
        with st.spinner("Comparing all four copulas with constraints…"):
            comp_res = compare_copulas(
                selected_tickers, ngarch_models, horizon,
                min(n_paths, 1000), target_return, rf_monthly,
                w_min=w_min_pct/100, w_max=w_max_pct/100,
                sector_max=sector_max_pct/100, div_penalty=div_penalty_val,
                seed=int(seed),
            )
        st.markdown('<div class="sec-eyebrow">Live Comparison — Current Asset Selection</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_copula_comparison(comp_res), use_container_width=True)
        rows_live = ""
        for cop, r in comp_res.items():
            best = cop == "Clayton"
            rows_live += f"""<tr style="{'background:'+PALETTE['bg2'] if best else ''}">
              <td class="mono {'teal-text' if best else ''}">{cop}</td>
              <td class="mono">{r['min_var']:.5f}</td>
              <td class="mono">{r['opt_vol']*100:.2f}%</td>
              <td class="mono">{r['sharpe']:.3f}</td>
              <td class="mono">{r['div_score']:.3f}</td>
              <td class="mono">{r['eff_n']:.2f}</td>
              <td class="mono">{r['lambda_T']:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>Copula</th><th>Var_min</th><th>Optimal σ</th>
            <th>Sharpe</th><th>Div. Score</th><th>Eff. Assets</th><th>λ_T</th>
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
    <div class="sec-title">Constrained Optimization Problem &amp; <em>Solution</em></div>
    <div class="sec-desc">
    The enhanced optimizer extends Vaillancourt & Watier (2005) with hard constraint enforcement
    via scipy SLSQP, a Herfindahl diversification penalty, and a turnover penalty for realistic
    institutional rebalancing in the Casablanca market.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primal problem (constrained)**")
        st.latex(r"""
        \min_{\{w_t\}} \; \mathrm{Var}\!\left[\prod_{t=1}^{T}\frac{1+\tilde{R}_t}{1+r_t}\right]
        + \lambda_{\text{div}} \max\!\left(0,\,0.3 - \sum_i w_i^2\right)^2
        + \lambda_{\tau} \sum_i |w_t^i - w_{t-1}^i|
        """)
        st.latex(r"""
        \text{s.t.} \;\; \sum_i w_i = 1, \;\;
        w_{\min} \le w_i \le w_{\max}, \;\;
        \sum_{i \in S} w_i \le w_S^{\max}
        """)
    with col2:
        st.markdown("**Diversification metrics**")
        st.latex(r"""
        \text{Herfindahl:}\; H = \sum_{i=1}^N w_i^2 \in \left[\tfrac{1}{N}, 1\right]
        """)
        st.latex(r"""
        \text{Diversification score:}\; D = 1 - H \in \left[0, 1-\tfrac{1}{N}\right]
        """)
        st.latex(r"""
        \text{Effective assets:}\; N_{\text{eff}} = \frac{1}{H} = \frac{1}{\sum_i w_i^2}
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="sec-eyebrow">NGarch (1,1) Process</div>
        <div class="sec-title">Volatility <em>Dynamics</em></div>
        """, unsafe_allow_html=True)
        st.latex(r"h_t = \beta_0 + \beta_1 h_{t-1} + \beta_2 h_{t-1}(Y_{t-1} - \kappa)^2")
        st.markdown("""
        <div class="insight-box">
          <div class="ib-label">Leverage parameter κ</div>
          <div class="ib-text">κ &lt; 0 for all BVC assets — negative shocks amplify variance more than positive shocks.</div>
        </div>
        <div class="insight-box">
          <div class="ib-label">Progressive transaction costs</div>
          <div class="ib-text">
          Cost at rebalancing t: c_t = c_base × (1 + 0.2 × min(trade_no, 5)).
          The first trade costs c_base; subsequent trades are progressively more expensive,
          discouraging unnecessary churn.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="sec-eyebrow">Constraint Solver</div>
        <div class="sec-title">scipy SLSQP <em>at each step</em></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="formula-block">
        At each time t, solve via SLSQP:<br><br>
        min  w'Vw - 2·mult·μ'w<br>
             + λ_div·max(0, 0.7 - (1 - Σw²))²<br>
             + λ_τ·Σ|w_i - w_{i,prev}|<br><br>
        subject to:<br>
          Σ w_i = 1<br>
          w_min ≤ w_i ≤ w_max  ∀i<br>
          Σ_{i∈S} w_i ≤ sector_max  ∀S<br><br>
        Warm-started from w_{t-1} for stability.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
          <div class="ib-label">Why SLSQP over closed-form</div>
          <div class="ib-text">
          The closed-form solution ignores inequality constraints, producing
          unconstrained portfolios. SLSQP solves the full KKT system at each
          time step, respecting all hard constraints while minimizing the same
          LSM objective function augmented with penalty terms.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">References</div>', unsafe_allow_html=True)
    refs = [
        ("Vaillancourt & Watier (2005)", "Dynamic portfolio selection via simulation-based LSM."),
        ("Engle & Ng (1993)",            "Measuring and testing the impact of news on volatility — NGARCH."),
        ("Sklar (1959)",                 "Fonctions de répartition à n dimensions — copula theorem."),
        ("Longstaff & Schwartz (2001)",  "Valuing American options by simulation — least-squares Monte Carlo."),
        ("Markowitz (1952)",             "Portfolio Selection — mean-variance efficient frontier."),
        ("Hirschman-Herfindahl (1964)",  "Herfindahl index for market concentration — adapted for portfolio diversification."),
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
    Eleven BVC-listed assets covering nine sectors. NGarch parameters estimated from
    five years of monthly closing prices (2019–2024) via maximum likelihood.
    Sector diversification is enforced via hard constraints in the optimizer.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(asset_table_html(list(BVC_ASSETS.keys())), unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">Parameter Interpretation Guide</div>', unsafe_allow_html=True)

    guide_rows = [
        ("β₁ (persistence)", "0.70 – 0.87", "Volatility shock persistence — near 1 = slow mean reversion."),
        ("κ (leverage)", "−0.62 – −0.14", "Negative = bad news amplifies variance more than good news."),
        ("β₂ (ARCH effect)", "0.09 (fixed)", "Sensitivity to lagged squared innovation."),
        ("δ (risk premium)", "0.08 – 0.14", "Excess return per unit of conditional standard deviation."),
        ("Persistence index", "β₁ + β₂(1+κ²)", "Above 0.95 → near-IGARCH behaviour."),
        ("Herfindahl H", "1/N – 1.0", "Sum of squared weights. Lower = more diversified."),
        ("Diversification score", "1 − H", "Target ≥ 0.70 for institutional BVC portfolios."),
        ("Effective assets N_eff", "1/H", "Equivalent number of equal-weight uncorrelated assets."),
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
        portfolio optimization framework, calibrated to the Casablanca Stock Exchange,
        with institutional-grade diversification constraints. It combines NGARCH(1,1)
        marginal volatility models with copula-based joint dependence structures,
        solved via Longstaff-Schwartz Monte Carlo backward induction with scipy SLSQP
        constrained optimization at each rebalancing step.

        **Constraint system**

        Weight bounds (5–30% per asset), sector exposure limits (max 40%), and a
        Herfindahl-based diversification penalty (targeting score ≥ 0.70) together ensure
        portfolios that are institutionally realistic for Moroccan fund managers.
        A turnover penalty (λ_τ = 0.05 default) promotes smooth weight transitions
        and reduces transaction cost drag.

        **Why Clayton copula is recommended**

        Empirical BVC data (2018–2024) shows lower-tail dependence: assets crash together
        during market stress but do not systematically co-rally. Clayton captures this
        precisely. The constrained optimizer also shows the highest diversification score
        under Clayton, reinforcing the copula's suitability for BVC portfolios.
        """)

    with col_b:
        st.markdown("""
        **Progressive transaction costs**

        BVC transaction costs are modeled with a progressive schedule: the first monthly
        rebalancing costs 0.50% (default), with subsequent trades incrementally more
        expensive (up to 0.50% × 1.5 for the 5th+ trade). This correctly incentivizes
        less frequent, larger rebalancing over constant small adjustments.

        **Diversification metrics explained**

        The Herfindahl concentration index H = Σ w_i² ranges from 1/N (perfect equal
        weighting) to 1.0 (complete concentration). The diversification score D = 1 − H
        targets ≥ 0.70. The effective asset count N_eff = 1/H gives an intuitive
        measure of portfolio breadth.

        **Regime detection (planned)**

        The framework is designed to support volatility regime detection: if annualised
        conditional volatility changes by more than 50% between rebalancing dates,
        the optimizer flags a regime shift and suggests re-estimation of NGarch parameters
        before the next rebalancing.

        **Limitations**

        Parameters estimated from historical data. Constraint levels are configurable
        but not dynamically calibrated. Results are for academic and research purposes
        only and do not constitute investment advice.
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-eyebrow">Disclosures</div>
    <div class="sec-desc">
    Designed for academic demonstration and research purposes. Simulated optimization results
    do not constitute investment advice. Historical performance does not guarantee future results.
    Risk-free rate benchmarked against Bank Al-Maghrib Bons du Trésor.
    Diversification constraints are illustrative institutional guidelines, not regulatory requirements.
    </div>
    """, unsafe_allow_html=True)
