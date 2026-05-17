# Moroccan Portfolio Optimizer

**Dynamic Portfolio Optimization for the Casablanca Stock Exchange**  
*NGarch + Copula Methodology — Vaillancourt & Watier (2005)*

---

## Overview

This repository implements a dynamic mean-variance portfolio optimization framework
for Moroccan assets traded on the Bourse des Valeurs de Casablanca (BVC).

The methodology combines:

- **NGarch(1,1)** marginal volatility models for each asset, capturing leverage effects
  and asymmetric volatility responses to return shocks
- **Copula functions** (Gaussian, Student-t, Clayton, Gumbel) to model joint dependence
  between asset excess returns
- **Longstaff-Schwartz Monte Carlo** backward induction to compute conditional expectations
  for the recursive Bellman equations
- A **closed-form optimal weight** expression derived from the constrained variance
  minimization problem

The objective: find the sequence of monthly portfolio weights that minimizes the
variance of terminal wealth while achieving a target cumulative return c.

---

## Problem Statement

```
min   Var[ prod_{t=1}^{T} (1+r_t)^{-1} (1+R̃_t) ]
w_t

s.t.  E[ prod_{t=1}^{T} (1+r_t)^{-1} (1+R̃_t) ] = 1 + c
```

**Optimal weights (closed form):**

```
w_t = -(1 + λ_T / 2·W_t) · μ_{t-1}^T · V_{t-1}^{-1}
```

**Minimum achievable variance:**

```
Var_min = c² / (E[Σ τ_t] - 1)
```

---

## Asset Universe

| Ticker | Name                      | Sector      |
|--------|---------------------------|-------------|
| ATW    | Attijariwafa Bank         | Banking     |
| IAM    | Maroc Telecom             | Telecoms    |
| BCP    | Banque Centrale Populaire | Banking     |
| MNG    | Managem                   | Mining      |
| TQM    | TotalEnergies MM          | Energy      |
| CIH    | CIH Bank                  | Banking     |
| LBV    | Label Vie                 | Retail      |
| WAA    | Wafa Assurance            | Insurance   |
| MUT    | Mutandis                  | Consumer    |
| ADDH   | Addoha                    | Real Estate |

---

## Installation

```bash
git clone https://github.com/your-username/moroccan-portfolio-optimizer.git
cd moroccan-portfolio-optimizer

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The application launches at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Set `app.py` as the entry point
5. Click **Deploy**

No additional secrets or environment variables are required.

---

## Project Structure

```
moroccan-portfolio-optimizer/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .streamlit/
    └── config.toml      # Optional theme configuration
```

---

## Methodology Reference

| Component        | Reference                                         |
|------------------|---------------------------------------------------|
| Optimization     | Vaillancourt & Watier (2005)                      |
| NGarch model     | Engle & Ng (1993)                                 |
| Copula theory    | Sklar (1959); Joe (1997)                          |
| LSM algorithm    | Longstaff & Schwartz (2001)                       |
| Risk-free rate   | Bons du Trésor (BAM) — 3.5% annual                |

---

## Parameters

| Parameter         | Default     | Description                                    |
|-------------------|-------------|------------------------------------------------|
| Target return c   | 15%         | Annualized cumulative return objective         |
| Horizon T         | 12 months   | Investment and rebalancing horizon             |
| Paths M           | 5,000       | Number of LSM Monte Carlo paths                |
| Copula            | Student-t   | Joint dependence model                         |
| ν (Student-t)     | 5           | Degrees of freedom for tail thickness          |
| Risk-free rate    | 3.5% p.a.   | BAM Bons du Trésor benchmark rate              |

---

## Disclaimer

This tool is designed for academic and research purposes. Results do not constitute
investment advice. Simulated optimization performance does not guarantee future returns.

---

*Casablanca School of Finance / Quantitative Finance Research*
