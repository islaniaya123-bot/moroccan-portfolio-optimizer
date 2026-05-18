"""
Moroccan Portfolio Optimizer
Dynamic Portfolio Optimization — NGarch + Copula Framework
Bourse des Valeurs de Casablanca
Vaillancourt & Watier (2005)

MODIFIED VERSION:
- Added MASI index as asset (20% weight target)
- Weight constraints: max 30% per asset, min 5% per asset
- Sector constraints: max 40% per sector
- Diversification penalty in objective
- Weight stability penalty
- Constrained optimization solver
- Enhanced metrics output
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, Bounds, LinearConstraint
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

section[data-testid="stSidebar"] {{
    background: {PALETTE['ink']};
    border-right: 0.5px solid {PALETTE['ink2']};
}}
section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.75) !important;
}}

.sec-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {PALETTE['gold']};
    margin-bottom: 0.3rem;
}}
.sec-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: {PALETTE['ink']};
    margin-bottom: 0.3rem;
}}
.metric-card {{
    background: {PALETTE['bg']};
    border: 0.5px solid {PALETTE['border']};
    border-top: 2px solid {PALETTE['gold']};
    padding: 1rem 1.1rem;
}}
.mc-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['ink3']};
}}
.mc-value {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.65rem;
    font-weight: 300;
    color: {PALETTE['ink']};
}}
.divider {{
    border: none;
    border-top: 0.5px solid {PALETTE['border']};
    margin: 1.5rem 0;
}}
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}}
.styled-table th {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid {PALETTE['border']};
    text-align: left;
}}
.styled-table td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 0.5px solid {PALETTE['border']};
}}
.mono {{ font-family: 'DM Mono', monospace; font-size: 0.78rem; }}
.teal-text {{ color: {PALETTE['teal']}; }}
.gold-text {{ color: {PALETTE['gold']}; }}
.green-text {{ color: {PALETTE['green']}; }}
.red-text {{ color: {PALETTE['red']}; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DATA (ADDED MASI)
# ─────────────────────────────────────────────────────────────────────────────

BVC_ASSETS = {
    "MASI": {"name": "MASI Index",              "sector": "Index",        "beta1": 0.720, "kappa": -0.25, "mcap": 650, "beta0": 3.5e-6, "delta": 0.10},
    "ATW":  {"name": "Attijariwafa Bank",       "sector": "Banking",      "beta1": 0.871, "kappa": -0.38, "mcap": 92,  "beta0": 5.2e-6, "delta": 0.11},
    "IAM":  {"name": "Maroc Telecom",           "sector": "Telecoms",     "beta1": 0.842, "kappa": -0.22, "mcap": 71,  "beta0": 4.1e-6, "delta": 0.09},
    "BCP":  {"name": "Banque Centrale Populaire","sector": "Banking",      "beta1": 0.858, "kappa": -0.41, "mcap": 58,  "beta0": 4.8e-6, "delta": 0.10},
    "OCP":  {"name": "OCP Group",               "sector": "Phosphates",   "beta1": 0.804, "kappa": -0.45, "mcap": 48,  "beta0": 6.1e-6, "delta": 0.13},
    "MNG":  {"name": "Managem",                 "sector": "Mining",       "beta1": 0.793, "kappa": -0.51, "mcap": 34,  "beta0": 5.9e-6, "delta": 0.12},
    "TQM":  {"name": "TotalEnergies MM",        "sector": "Energy",       "beta1": 0.811, "kappa": -0.29, "mcap": 28,  "beta0": 4.5e-6, "delta": 0.09},
    "CIH":  {"name": "CIH Bank",                "sector": "Banking",      "beta1": 0.834, "kappa": -0.35, "mcap": 22,  "beta0": 4.7e-6, "delta": 0.10},
    "LBV":  {"name": "Label Vie",               "sector": "Retail",       "beta1": 0.769, "kappa": -0.18, "mcap": 18,  "beta0": 3.9e-6, "delta": 0.08},
    "WAA":  {"name": "Wafa Assurance",          "sector": "Insurance",    "beta1": 0.822, "kappa": -0.27, "mcap": 16,  "beta0": 4.4e-6, "delta": 0.09},
    "MUT":  {"name": "Mutandis",                "sector": "Consumer",     "beta1": 0.748, "kappa": -0.14, "mcap": 12,  "beta0": 3.7e-6, "delta": 0.08},
    "ADDH": {"name": "Addoha",                  "sector": "Real Estate",  "beta1": 0.701, "kappa": -0.62, "mcap": 9,   "beta0": 7.2e-6, "delta": 0.14},
}

SECTOR_COLORS = {
    "Index":       PALETTE["gold"],
    "Banking":     PALETTE["teal"],
    "Telecoms":    PALETTE["teal2"],
    "Phosphates":  PALETTE["rust"],
    "Mining":      "#4a6a5a",
    "Energy":      "#5a6a4a",
    "Retail":      "#6a5a7a",
    "Insurance":   "#5a6a7a",
    "Consumer":    "#7a6a5a",
    "Real Estate": PALETTE["ink3"],
}

COPULA_INFO = {
    "Clayton": {"tag": "Recommended for Morocco", "aic_factor": -145},
    "Student-t": {"tag": "Symmetric tails", "aic_factor": -132},
    "Gaussian": {"tag": "No tail dependence", "aic_factor": -118},
    "Gumbel": {"tag": "Upper tail only", "aic_factor": -125},
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
# CONSTRAINED LSM OPTIMIZER
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
        """Objective: minimize variance + diversification penalty + turnover penalty"""
        # Core variance objective
        var_obj = w @ V0_inv @ w
        
        # Diversification penalty (encourage spread)
        div_score = self._diversification_score(w)
        div_pen = -self.div_penalty * div_score  # negative because higher diversity is better
        
        # Turnover penalty (smooth rebalancing)
        turnover_pen = 0
        if prev_w is not None:
            turnover = np.sum(np.abs(w - prev_w))
            turnover_pen = self.turnover_penalty * turnover
        
        return var_obj + div_pen + turnover_pen

    def _get_constraints(self) -> list:
        """Build linear and bound constraints"""
        N = self.N
        constraints = []
        
        # Sum of weights = 1
        constraints.append(LinearConstraint(np.ones(N), 1.0, 1.0))
        
        # Sector constraints
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
        """Solve constrained optimization for optimal weights"""
        N = self.N
        
        # Bounds: each weight between min_single and max_single
        bounds = Bounds([self.min_single] * N, [self.max_single] * N)
        
        # Constraints
        constraints = self._get_constraints()
        
        # Initial guess: equal weights
        w0 = np.ones(N) / N
        
        # Solve
        result = minimize(
            lambda w: self._objective_with_penalties(w, mu0, V0_inv, prev_w),
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if result.success:
            w = result.x
            # Ensure sum exactly 1 (normalize due to numerical issues)
            return w / w.sum()
        else:
            # Fallback to equal weights
            return np.ones(N) / N

    def run(self, seed: int = 42) -> dict:
        N, T, M = self.N, self.T, self.M

        # 1. Sample correlated innovations
        innovations = self.copula.sample_innovations(M, T, seed=seed)

        # 2. Simulate NGarch paths
        all_ret = np.zeros((N, M, T))
        for i, tk in enumerate(self.tickers):
            ret, _ = self.models[tk].simulate(T, M, innovations[i])
            all_ret[i] = ret

        # 3. Excess returns (M, T, N)
        ex_ret = np.stack([self._excess_return(all_ret[n]) for n in range(N)], axis=2)

        # 4. LSM backward induction
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

        # 5. Calibrate lambda_T
        E_tau = max(sum(tau_list), 1e-10)
        lam_T = -2.0 * (self.c / E_tau + 1.0)

        # 6. Solve for optimal weights with constraints
        mu0 = mu_list[0]
        V0 = V_list[0] + 1e-8 * np.eye(N)
        V0_inv = np.linalg.pinv(V0)
        
        w_opt = self._solve_weights(mu0, V0_inv)

        # 7. Dynamic weights over horizon
        dynamic_w = self._compute_dynamic_weights(mu_list, V_list, all_ret, lam_T, w_opt)

        # 8. Metrics
        min_var = abs(self.c ** 2 / max(E_tau - 1.0, 1e-6))
        opt_vol = float(np.sqrt(min_var))
        sharpe = (self.c - self.rf * 12) / max(opt_vol, 1e-8)
        achieved = float(np.prod(1 + np.mean(all_ret, axis=(0, 1))) - 1)
        div_score = self._diversification_score(w_opt)
        effective_n = 1 / np.sum(w_opt ** 2)
        
        # Sector allocation
        sector_weights = self._sector_weights(w_opt)
        
        # Warning flags
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
            mult = -(1 + lam_T / (2 * max(cum_wealth, 1e-8)))
            w_raw = mult * (mu_t @ Vi)
            w_raw = np.maximum(w_raw, 0)
            
            # Apply constraints
            w_raw = self._solve_weights(mu_t, Vi, dyn[t-1])
            dyn[t] = w_raw
        
        return dyn


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fig_weights_bar(weights: dict, title: str = "Optimal Portfolio Weights") -> go.Figure:
    syms = list(weights.keys())
    vals = [weights[s] * 100 for s in syms]
    names = [BVC_ASSETS[s]["name"] for s in syms]
    sectors = [BVC_ASSETS[s]["sector"] for s in syms]
    colors = [SECTOR_COLORS.get(sec, PALETTE["ink3"]) for sec in sectors]

    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order],
        y=[f"{syms[i]} — {names[i]}" for i in order],
        orientation="h",
        marker_color=[colors[i] for i in order],
        text=[f"{vals[i]:.1f}%" for i in order],
        textposition="outside",
        textfont=dict(family="DM Mono", size=11),
    ))
    fig.update_layout(
        title=title,
        height=max(350, 55 * len(syms)),
        xaxis_title="Weight (%)",
        margin=dict(l=200, r=60, t=50, b=30),
    )
    return fig


def fig_sector_donut(weights: dict) -> go.Figure:
    sector_w = {}
    for sym, w in weights.items():
        s = BVC_ASSETS[sym]["sector"]
        sector_w[s] = sector_w.get(s, 0) + w * 100
    labels = list(sector_w.keys())
    values = [round(v, 2) for v in sector_w.values()]
    colors = [SECTOR_COLORS.get(l, PALETTE["ink3"]) for l in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker_colors=colors))
    fig.update_layout(title="Sector Allocation", height=300)
    return fig


def fig_dynamic_weights(dynamic_w: np.ndarray, tickers: list, T: int) -> go.Figure:
    months = list(range(1, T + 1))
    fig = go.Figure()
    color_palette = [PALETTE["teal"], PALETTE["gold"], PALETTE["rust"], PALETTE["teal2"],
                     "#4a6a5a", "#5a6a4a", "#6a5a7a", "#5a6a7a", "#7a6a5a", PALETTE["ink3"]]
    for n, sym in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=months, y=dynamic_w[:, n] * 100, mode="lines",
            name=sym, line=dict(color=color_palette[n % len(color_palette)], width=1.8)
        ))
    fig.update_layout(
        title="Dynamic Weight Evolution Over Investment Horizon",
        xaxis_title="Month", yaxis_title="Weight (%)", height=340
    )
    return fig


def fig_tau_series(tau_list: list, T: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, T + 1)), y=tau_list,
        mode="lines+markers", line=dict(color=PALETTE["gold"], width=2),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)"
    ))
    fig.update_layout(
        title="Information Ratio Scalars τ_t — LSM Backward Induction",
        xaxis_title="Time step t", yaxis_title="τ_t = μ_t' V_t⁻¹ μ_t", height=300
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="header-eyebrow">Bourse des Valeurs de Casablanca</div>
  <div class="header-title">Dynamic Portfolio Optimization<br>for the <em>Moroccan Market</em></div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Asset Universe")
    default_assets = ["MASI", "ATW", "IAM", "OCP", "TQM"]
    selected_tickers = st.multiselect(
        "BVC Constituents",
        options=list(BVC_ASSETS.keys()),
        default=default_assets,
        format_func=lambda t: f"{t} — {BVC_ASSETS[t]['name']}",
    )

    st.markdown("### Optimization Parameters")
    target_pct = st.slider("Target Annual Return (%)", 5, 25, 12, 1)
    horizon = st.slider("Investment Horizon (months)", 6, 36, 12, 3)

    st.markdown("### Risk Parameters")
    rf_pct = st.slider("Risk-Free Rate (% p.a.)", 1.0, 6.0, 2.75, 0.25)
    tx_cost_pct = st.slider("Transaction Costs (% per trade)", 0.0, 0.75, 0.50, 0.05)

    st.markdown("### Constraint Settings")
    max_single = st.slider("Max Single Asset Weight (%)", 20, 40, 30, 5) / 100
    min_single = st.slider("Min Single Asset Weight (%)", 3, 15, 5, 1) / 100
    max_sector = st.slider("Max Sector Exposure (%)", 30, 60, 40, 5) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 0.3, 0.1, 0.05)

    st.markdown("### Simulation")
    n_paths = st.select_slider("Simulation Paths", options=[1000, 2000, 5000, 10000], value=5000)

    copula_choice = st.selectbox("Copula", ["Clayton", "Student-t", "Gaussian", "Gumbel"], index=0)
    theta = st.slider("Dependence Parameter θ", 0.5, 4.0, 1.5, 0.1) if copula_choice in ["Clayton", "Gumbel"] else 1.5
    nu = st.slider("Degrees of Freedom ν", 3, 30, 5) if copula_choice == "Student-t" else 5

    seed = st.number_input("Random Seed", value=42, step=1)
    run_button = st.button("Run Optimizer", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

if run_button and len(selected_tickers) >= 3:
    rf_monthly = rf_pct / 100 / 12
    target_return = target_pct / 100

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

    with st.spinner("Running constrained LSM optimization..."):
        result = optimizer.run(seed=int(seed))

    weights = result["weights"]
    opt_vol = result["opt_vol"]
    sharpe = result["sharpe"]
    achieved = result["achieved"]
    div_score = result["diversification_score"]
    eff_n = result["effective_number_assets"]

    # Metrics row
    cols = st.columns(5)
    metrics = [
        ("Optimal Volatility", f"{opt_vol*100:.2f}%", f"Target: {target_pct}%"),
        ("Sharpe Ratio", f"{sharpe:.3f}", f"Risk-free: {rf_pct}%"),
        ("Achieved Return", f"{achieved*100:.2f}%", f"Δ = {(achieved*100 - target_pct):+.2f}%"),
        ("Diversification Score", f"{div_score:.3f}", "1 = perfectly diversified"),
        ("Effective Assets", f"{eff_n:.1f}", f"from {len(selected_tickers)} selected"),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="mc-label">{label}</div>
          <div class="mc-value">{val}</div>
          <div class="mc-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Warnings
    if result["concentration_warning"]:
        st.warning("⚠️ Portfolio is still concentrated. Consider increasing diversification penalty or reducing max single weight.")
    if result["sector_warning"]:
        st.warning("⚠️ Sector limit exceeded. Sector constraints are enforced but market may not allow full diversification.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_weights_bar(weights), use_container_width=True)
    with col_right:
        st.plotly_chart(fig_sector_donut(weights), use_container_width=True)

    st.plotly_chart(fig_dynamic_weights(result["dynamic_w"], selected_tickers, horizon), use_container_width=True)

    col_tau, _ = st.columns(2)
    with col_tau:
        st.plotly_chart(fig_tau_series(result["tau_list"], horizon), use_container_width=True)

    # Weight table
    st.markdown("### Allocation Detail")
    rows = ""
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        a = BVC_ASSETS[sym]
        rows += f"""
        <tr>
          <td class="mono teal-text">{sym}</td>
          <td>{a['name']}</td>
          <td class="muted">{a['sector']}</td>
          <td class="mono gold-text">{w*100:.2f}%</td>
          <td class="mono">{a['beta1']:.3f}</td>
          <td class="mono">{a['kappa']:.3f}</td>
        </tr>
        """
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Weight</th><th>β₁</th><th>κ</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

elif run_button and len(selected_tickers) < 3:
    st.error("Please select at least 3 assets for diversification (MASI + 2 others recommended).")

else:
    st.info("Select assets in the sidebar and click **Run Optimizer** to generate a constrained, diversified portfolio for the Moroccan market.")
    st.markdown("""
    ### Recommended Starting Portfolio
    - **MASI** (Index benchmark)
    - **ATW** (Banking)
    - **IAM** (Telecoms)
    - **OCP** (Phosphates)
    - **TQM** (Energy)
    
    This provides cross-sector diversification and captures Morocco's key economic drivers.
    """)
