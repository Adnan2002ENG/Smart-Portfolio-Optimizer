"""
app.py
------
Script principal du projet Smart Portfolio Optimizer.

- Définit les paramètres (tickers, période, taux sans risque)
- Utilise utilities.py pour :
    * télécharger les données
    * calculer les statistiques
    * construire les portefeuilles
    * backtester chaque stratégie
- Affiche les résultats et les courbes de performance.

À lancer avec :  python app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ==========================
# Paramètres du projet
# ==========================

TICKERS = [
    "SPY",   # ETF S&P 500
    "VEA",   # Developed Markets ex-US
    "EEM",   # Emerging Markets
    "TLT",   # US Long Term Bonds
    "LQD",   # Investment Grade Corp Bonds
    "GLD",   # Gold
]

START_DATE = "2015-01-01"
END_DATE = None  # None = jusqu'à aujourd'hui
RISK_FREE_RATE = 0.02  # 2% annuel (à adapter selon les conditions de marché)


def main():
    # --------------------------
    # 1) Données & statistiques
    # --------------------------
    print("Téléchargement des données de marché...")
    prices = download_price_data(TICKERS, START_DATE, END_DATE)
    print("Prix (dernieres lignes) :")
    print(prices.tail())

    returns = compute_returns(prices)
    mean_returns = compute_annualized_mean_returns(returns)
    cov_matrix = compute_annualized_cov_matrix(returns)

    print("\nRendements moyens annualisés :")
    print(mean_returns.round(4))

    # --------------------------
    # 2) Construction des portefeuilles
    # --------------------------
    n_assets = len(TICKERS)

    w_equal = equal_weight_weights(n_assets)
    w_inv_vol = inverse_vol_weights(returns)
    w_markowitz = markowitz_max_sharpe_weights(mean_returns, cov_matrix, RISK_FREE_RATE, gamma=0.5)

    print("\nPoids Equal-Weight :")
    print(dict(zip(TICKERS, np.round(w_equal, 4))))

    print("\nPoids Inverse-Vol :")
    print(dict(zip(TICKERS, np.round(w_inv_vol, 4))))

    print("\nPoids Markowitz :")
    print(dict(zip(TICKERS, np.round(w_markowitz, 4))))

    # --------------------------
    # 3) Backtest des stratégies
    # --------------------------
    res_equal = backtest_portfolio(returns, w_equal, risk_free_rate=RISK_FREE_RATE)
    res_inv_vol = backtest_portfolio(returns, w_inv_vol, risk_free_rate=RISK_FREE_RATE)
    res_markowitz = backtest_portfolio(returns, w_markowitz, risk_free_rate=RISK_FREE_RATE)

    # Tableau récapitulatif
    results_df = pd.DataFrame([
        {
            "Strategy": "Equal-Weight",
            "Annual Return": res_equal["ann_return"],
            "Annual Vol": res_equal["ann_vol"],
            "Sharpe": res_equal["sharpe"],
        },
        {
            "Strategy": "Inverse-Vol",
            "Annual Return": res_inv_vol["ann_return"],
            "Annual Vol": res_inv_vol["ann_vol"],
            "Sharpe": res_inv_vol["sharpe"],
        },
        {
            "Strategy": "Markowitz",
            "Annual Return": res_markowitz["ann_return"],
            "Annual Vol": res_markowitz["ann_vol"],
            "Sharpe": res_markowitz["sharpe"],
        },
    ])

    print("\nRésultats des stratégies (annualisés) :")
    print(results_df.set_index("Strategy").round(4))

    # --------------------------
    # 4) Visualisation
    # --------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(res_equal["curve"], label="Equal-Weight")
    plt.plot(res_inv_vol["curve"], label="Inverse-Vol")
    plt.plot(res_markowitz["curve"], label="Markowitz (Sharpe)")
    plt.title("Évolution de la valeur du portefeuille (base 100)")
    plt.xlabel("Date")
    plt.ylabel("Valeur du portefeuille")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
