"""
streamlit_app.py
----------------
Streamlit dashboard for the Smart Portfolio Optimizer.

- Select tickers, date range, risk-free rate, initial capital
- Build three portfolios: Equal-Weight, Inverse-Vol, Markowitz
- Display performance metrics, weights, and equity curves
"""

import numpy as np
import pandas as pd
import streamlit as st

from utilities import (
    download_price_data,
    compute_returns,
    compute_annualized_mean_returns,
    compute_annualized_cov_matrix,
    equal_weight_weights,
    inverse_vol_weights,
    markowitz_max_sharpe_weights,
    backtest_portfolio,
)

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Smart Portfolio Optimizer", layout="wide")

st.title("📈 Smart Portfolio Optimizer")
st.write(
    "Interactive dashboard to compare Equal-Weight, Inverse-Volatility and Markowitz portfolios."
)

# ------------------------------
# Sidebar – Parameters
# ------------------------------
st.sidebar.header("Configuration")

# Default tickers
default_tickers = "SPY, VEA, EEM, TLT, LQD, GLD"

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated):",
    value=default_tickers
)

start_date = st.sidebar.date_input("Start date:", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date:", value=pd.to_datetime("today"))

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, e.g. 0.02 = 2%)",
    min_value=0.0,
    max_value=0.2,
    value=0.02,
    step=0.005,
    format="%.3f"
)

gamma = st.sidebar.slider(
    "Risk aversion (γ) for Markowitz",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

# 🔹 Nouveau : capital initial monétaire
initial_capital = st.sidebar.number_input(
    "Initial portfolio value (e.g. 10 000)",
    min_value=0.0,
    value=10000.0,
    step=1000.0,
    format="%.2f"
)

run_button = st.sidebar.button("🚀 Run Optimization")

# ------------------------------
# Main logic
# ------------------------------

if run_button:
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) < 2:
        st.error("Please enter at least 2 valid tickers.")
    else:
        st.write(f"Selected tickers: **{', '.join(tickers)}**")

        # 1) Download data
        with st.spinner("Downloading market data..."):
            prices = download_price_data(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

        if prices.empty:
            st.error("No data downloaded. Please check tickers and date range.")
        else:
            st.subheader("📊 Price data (tail)")
            st.dataframe(prices.tail())

            # 2) Returns & stats
            returns = compute_returns(prices)
            mean_returns = compute_annualized_mean_returns(returns)
            cov_matrix = compute_annualized_cov_matrix(returns)

            st.subheader("📈 Annualized mean returns")
            st.dataframe(mean_returns.to_frame("Mean Return").style.format("{:.2%}"))

            # 3) Build portfolios
            n_assets = len(tickers)
            w_equal = equal_weight_weights(n_assets)
            w_inv_vol = inverse_vol_weights(returns)
            w_markowitz = markowitz_max_sharpe_weights(
                mean_returns, cov_matrix, risk_free_rate=risk_free_rate, gamma=gamma
            )

            # 4) Backtest (base 100)
            res_equal = backtest_portfolio(returns, w_equal, risk_free_rate=risk_free_rate)
            res_inv_vol = backtest_portfolio(returns, w_inv_vol, risk_free_rate=risk_free_rate)
            res_markowitz = backtest_portfolio(returns, w_markowitz, risk_free_rate=risk_free_rate)

            # 5) Metrics table
            results_df = pd.DataFrame([
                {
                    "Strategy": "Equal-Weight",
                    "Annual Return": res_equal["ann_return"],
                    "Annual Volatility": res_equal["ann_vol"],
                    "Sharpe": res_equal["sharpe"],
                },
                {
                    "Strategy": "Inverse-Vol",
                    "Annual Return": res_inv_vol["ann_return"],
                    "Annual Volatility": res_inv_vol["ann_vol"],
                    "Sharpe": res_inv_vol["sharpe"],
                },
                {
                    "Strategy": "Markowitz",
                    "Annual Return": res_markowitz["ann_return"],
                    "Annual Volatility": res_markowitz["ann_vol"],
                    "Sharpe": res_markowitz["sharpe"],
                },
            ])

            st.subheader("📊 Performance metrics (annualized)")
            st.dataframe(
                results_df.set_index("Strategy")
                .style.format("{:.2%}", subset=["Annual Return", "Annual Volatility"])
                .format("{:.2f}", subset=["Sharpe"])
            )

            # 6) Weights tables
            weights_df = pd.DataFrame(
                {
                    "Equal-Weight": w_equal,
                    "Inverse-Vol": w_inv_vol,
                    "Markowitz": w_markowitz,
                },
                index=tickers,
            )

            st.subheader("🧱 Portfolio weights")
            st.dataframe(weights_df.style.format("{:.2%}"))

            # 7) Equity curves (base 100)
            curves_base100 = pd.DataFrame({
                "Equal-Weight": res_equal["curve"],
                "Inverse-Vol": res_inv_vol["curve"],
                "Markowitz": res_markowitz["curve"],
            })

            st.subheader("📉 Portfolio value over time (base 100)")
            st.line_chart(curves_base100)

            # 8) 💰 Equity curves in monetary terms
            if initial_capital > 0:
                curves_monetary = curves_base100 * (initial_capital / 100.0)

                st.subheader(f"💰 Portfolio monetary value over time (initial = {initial_capital:,.2f})")
                st.line_chart(curves_monetary)
            else:
                st.info("Initial capital is 0, monetary curve is not displayed.")
else:
    st.info("Configure parameters on the left, then click **🚀 Run Optimization**.")
