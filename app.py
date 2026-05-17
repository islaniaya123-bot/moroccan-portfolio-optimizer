
**Software Stack**

| Component | Library | Purpose |
|-----------|---------|---------|
| Interface | Streamlit | Interactive dashboard framework |
| GARCH | arch | Conditional volatility estimation |
| Optimization | SciPy | Portfolio optimization algorithms |
| Statistics | NumPy, Pandas | Numerical computing and data manipulation |
| Visualization | Plotly | Interactive charts and graphs |
| Data | yfinance | Market data acquisition |

**Model Assumptions**

1. No transaction costs or market frictions
2. Constant risk-free rate over investment horizon
3. Short selling permitted (weights can be zero but not negative in this implementation)
4. Investor cannot influence market prices through trading activity
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 24px 0 16px 0;">
<div style="font-size: 12px; color: #4B5563;">
    Methodology: GARCH(1,1) + Copula Dependence | Prince (2007) HEC Montreal<br>
    Data Sources: MASI, MADEX, ATW, IAM, OCP | Risk-Free Rate: Bons du Trésor<br>
    For educational and research purposes only. Not investment advice.
</div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
main()
