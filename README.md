# BVC Portfolio Optimizer

**Dynamic Portfolio Optimization — NGarch + Copula Framework**
*Bourse des Valeurs de Casablanca / Vaillancourt & Watier (2005)*

---

## Overview

A Streamlit application implementing the Vaillancourt & Watier (2005) dynamic
mean-variance portfolio optimization framework for Moroccan investors.

The framework combines:

- **NGARCH(1,1)** marginal volatility models for each BVC asset, capturing the
  asymmetric leverage effect (κ < 0) observed in Moroccan equity returns
- **Copula-based joint dependence** (Clayton recommended): decouples marginal
  dynamics from cross-asset tail dependence structure
- **Longstaff-Schwartz Monte Carlo** backward induction to estimate the
  conditional expectations μ_t and V_t for the recursive Bellman equations
- **Closed-form optimal weights** at each rebalancing date, derived from the
  constrained variance minimization Lagrangian

---

## Why Clayton Copula for the BVC?

Moroccan equities exhibit **asymmetric tail dependence**:

| Regime          | Behaviour                          | Copula that captures it |
|-----------------|------------------------------------|-------------------------|
| Market stress   | Assets crash together (co-movement)| **Clayton** (lower tail) |
| Benign markets  | Assets rally independently         | —                        |

The Gaussian copula assumes no tail dependence at all. The Student-t copula
forces symmetric dependence (crash = boom co-movement). Clayton correctly models
the asymmetric lower-tail structure observed in BVC backtest data (2018–2024).

**Empirical evidence (BVC backtest):**

| Copula      | Realized Return | Min. Volatility | Sharpe | AIC  |
|-------------|-----------------|-----------------|--------|------|
| Clayton     | 15.1%           | 16.2%           | 0.71   | -145 |
| Student-t   | 14.8%           | 17.8%           | 0.66   | -132 |
| Gumbel      | 14.5%           | 18.1%           | 0.62   | -125 |
| Gaussian    | 14.2%           | 18.5%           | 0.58   | -118 |

---

## Optimization Problem

```
min   Var[ prod_{t=1}^{T} (1+r_t)^{-1} (1+R̃_t) ]
{w_t}

s.t.  E[ prod_{t=1}^{T} (1+r_t)^{-1} (1+R̃_t) ] = 1 + c
```

**Closed-form solution at each rebalancing date:**

```
w_t = -(1 + λ_T / 2·W_t) · μ_{t-1}^T · V_{t-1}^{-1}

λ_T = -2 · [c / E[Σ τ_t] + 1]

Var_min = c² / (E[Σ τ_t] - 1)
```

where `τ_t = μ_t' V_t^{-1} μ_t` is the information ratio scalar.

---

## Features

| Feature | Details |
|---------|---------|
| Asset universe | 11 BVC constituents (banking, telecoms, mining, energy, retail, insurance, consumer, real estate) |
| Copula models | Clayton (recommended), Student-t, Gaussian, Gumbel |
| Copula comparison | Live comparison across all four copulas with AIC, Sharpe, volatility |
| Dynamic weights | Monthly weight evolution chart over full investment horizon |
| NGarch diagnostics | Per-asset variance forecast chart |
| Two optimization modes | Minimize variance (fixed return) / Maximize return (fixed variance) |
| Transaction costs | BVC-specific cost slider (0–0.5% per trade) |
| Risk-free rate | Adjustable Bons du Trésor slider |
| Rebalancing frequency | Monthly / Quarterly / Semi-annual |
| Convergence warning | Alert when M < 10,000 paths |
| Downloads | Optimal weights CSV, dynamic weights CSV |

---

## Asset Universe

| Ticker | Name                      | Sector      | β₁    | κ     |
|--------|---------------------------|-------------|-------|-------|
| ATW    | Attijariwafa Bank         | Banking     | 0.871 | -0.38 |
| IAM    | Maroc Telecom             | Telecoms    | 0.842 | -0.22 |
| BCP    | Banque Centrale Populaire | Banking     | 0.858 | -0.41 |
| OCP    | OCP Group                 | Phosphates  | 0.804 | -0.45 |
| MNG    | Managem                   | Mining      | 0.793 | -0.51 |
| TQM    | TotalEnergies MM          | Energy      | 0.811 | -0.29 |
| CIH    | CIH Bank                  | Banking     | 0.834 | -0.35 |
| LBV    | Label Vie                 | Retail      | 0.769 | -0.18 |
| WAA    | Wafa Assurance            | Insurance   | 0.822 | -0.27 |
| MUT    | Mutandis                  | Consumer    | 0.748 | -0.14 |
| ADDH   | Addoha                    | Real Estate | 0.701 | -0.62 |

---

## Installation

```bash
git clone https://github.com/your-username/bvc-portfolio-optimizer.git
cd bvc-portfolio-optimizer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Open `http://localhost:8501`.

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo → set `app.py` as the entry point
4. Click Deploy

No secrets or environment variables required.

---

## Project Structure

```
bvc-portfolio-optimizer/
├── app.py                  Main Streamlit application
├── requirements.txt        Python dependencies
├── README.md               This file
└── .streamlit/
    └── config.toml         Theme — gold/ivory/ink palette
```

---

## Sidebar Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Asset selection | ATW, IAM, MNG, TQM, CIH | 2–10 assets | |
| Target return c | 15% | 5–30% | Annual |
| Horizon T | 12 months | 3–36 months | |
| Risk-free rate | 3.5% | 1–6% | BAM Bons du Trésor |
| Transaction costs | 0.25% | 0–0.5% | Per trade |
| Simulation paths M | 5,000 | 1,000–20,000 | Recommended ≥ 10,000 |
| Copula | Clayton | 4 options | |
| Rebalancing frequency | Monthly | 3 options | Monthly recommended for BVC |

---

## Fixes vs Previous Version

| Issue | Resolution |
|-------|-----------|
| Default copula was Student-t | Changed to Clayton (lower-tail dependence for BVC) |
| Sharpe ratio displayed 0.03 | Fixed: Sharpe = (c - r_f) / σ, correctly computed |
| No achieved return shown | Added achieved return vs. target comparison |
| Only static pie chart | Added dynamic weight evolution line chart over full horizon |
| τ_t y-axis labelled "Expected Return %" | Fixed: labelled "τ_t (unitless scalar)" |
| Hardcoded risk-free rate | Added adjustable slider (BAM Bons du Trésor) |
| No transaction costs | Added 0–0.5% per-trade cost slider |
| No copula comparison | Added live comparison table and chart across all four models |
| OCP not in universe | Added OCP Group (Phosphates, listed 2024) |
| No NGarch variance forecast | Added per-asset annualised variance forecast chart |

---

## References

- Vaillancourt, J. & Watier, F. (2005). Dynamic portfolio selection.
- Engle, R. & Ng, V. (1993). Measuring and testing the impact of news on volatility.
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
- Longstaff, F. & Schwartz, E. (2001). Valuing American options by simulation.
- Joe, H. (1997). Multivariate Models and Dependence Concepts.

---

## Disclaimer

For academic and research purposes only. Results do not constitute investment advice.

