"""
utilities.py
-------------
Fonctions utilitaires pour le projet Smart Portfolio Optimizer :

- Téléchargement des données de marché (robuste à plusieurs formats yfinance)
- Calcul des rendements & statistiques annualisées
- Construction de portefeuilles (Equal-Weight, Inverse-Vol)
- Optimisation de Markowitz (maximisation Sharpe approx)
- Backtest des portefeuilles
"""

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp

# Nombre approximatif de jours de bourse par an
TRADING_DAYS = 252


# ==========================
# Données & statistiques
# ==========================

def download_price_data(tickers, start, end=None):
    """
    Télécharge les prix pour une liste de tickers et renvoie un DataFrame
    avec une colonne par ticker (prix de clôture ajustés ou, à défaut, close).

    Gestion robuste des formats yfinance :
    - colonnes simples
    - colonnes MultiIndex (niveau 0 = 'Adj Close' / 'Close', niveau 1 = tickers)
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)

    if raw.empty:
        raise ValueError("Aucune donnée téléchargée. Vérifie les tickers et la période.")

    # ---- Cas 1 : colonnes MultiIndex ----
    if isinstance(raw.columns, pd.MultiIndex):
        # Exemple habituel : niveau 0 = ['Adj Close', 'Close', 'Open', ...]
        #                    niveau 1 = ['SPY', 'TLT', ...]
        level0 = raw.columns.get_level_values(0)

        if "Adj Close" in level0:
            data = raw["Adj Close"]  # DataFrame avec colonnes = tickers
        elif "Close" in level0:
            data = raw["Close"]
        else:
            raise KeyError(
                f"Colonnes disponibles (niveau 0) : {sorted(set(level0))}, "
                "mais ni 'Adj Close' ni 'Close' trouvés."
            )

    # ---- Cas 2 : colonnes simples ----
    else:
        cols = list(raw.columns)

        if "Adj Close" in cols:
            data = raw[["Adj Close"]].copy()
            # Si un seul ticker (string) passé, renommer la colonne avec ce ticker
            if isinstance(tickers, str):
                data.columns = [tickers]
        elif "Close" in cols:
            data = raw[["Close"]].copy()
            if isinstance(tickers, str):
                data.columns = [tickers]
        else:
            raise KeyError(
                f"Colonnes disponibles : {cols}, mais ni 'Adj Close' ni 'Close' trouvés."
            )

    data = data.dropna()
    return data


def compute_returns(price_df):
    """
    Calcule les rendements journaliers à partir des prix.
    """
    returns = price_df.pct_change().dropna()
    return returns


def compute_annualized_mean_returns(returns_df):
    """
    Rendement moyen annualisé de chaque actif.
    """
    return returns_df.mean() * TRADING_DAYS


def compute_annualized_cov_matrix(returns_df):
    """
    Matrice de covariance annualisée.
    """
    return returns_df.cov() * TRADING_DAYS


# ==========================
# Stratégies de portefeuille
# ==========================

def equal_weight_weights(n_assets):
    """
    Portefeuille à poids égaux.
    """
    return np.ones(n_assets) / n_assets


def inverse_vol_weights(returns_df):
    """
    Portefeuille inverse-vol (approximation simple du risk-parity).
    On prend 1 / volatilité annualisée comme poids brut, puis on normalise.
    """
    vol_ann = returns_df.std() * np.sqrt(TRADING_DAYS)
    inv_vol = 1.0 / vol_ann
    weights = inv_vol / inv_vol.sum()
    return weights.values  # numpy array


def markowitz_max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate=0.0, gamma=0.5):
    """
    Optimisation de Markowitz pour approx. maximiser le Sharpe ratio.

    On maximise :
        (mu^T w - risk_free_rate) - gamma * (w^T Σ w)

    sous contraintes :
        - somme des poids = 1
        - poids >= 0

    mean_returns : pandas Series (actifs -> rendement moyen annualisé)
    cov_matrix   : pandas DataFrame (matrice de covariance annualisée)
    """
    n = len(mean_returns)
    w = cp.Variable(n)

    mu = mean_returns.values
    Sigma = cov_matrix.values

    portfolio_return = mu @ w
    portfolio_variance = cp.quad_form(w, Sigma)

    objective = cp.Maximize(portfolio_return - risk_free_rate - gamma * portfolio_variance)

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimisation Markowitz a échoué. Vérifie les données ou les paramètres.")

    return np.array(w.value).flatten()


# ==========================
# Backtest & performance
# ==========================

def backtest_portfolio(returns_df, weights, risk_free_rate=0.02, initial_value=100.0):
    """
    Backtest d'un portefeuille à poids fixes.

    returns_df : DataFrame des rendements journaliers des actifs
    weights    : vecteur numpy des poids (somme = 1)
    risk_free_rate : taux sans risque annuel (pour Sharpe)
    initial_value  : valeur initiale du portefeuille

    Retourne un dict :
    {
      'curve': Series valeur portefeuille,
      'ann_return': float,
      'ann_vol': float,
      'sharpe': float
    }
    """
    # Rendement journalier du portefeuille
    port_returns = returns_df.values @ weights
    port_returns_series = pd.Series(port_returns, index=returns_df.index)

    # Courbe de valeur cumulée
    curve = (1 + port_returns_series).cumprod() * initial_value

    # Statistiques
    ann_return = port_returns_series.mean() * TRADING_DAYS
    ann_vol = port_returns_series.std() * np.sqrt(TRADING_DAYS)

    if ann_vol > 0:
        sharpe = (ann_return - risk_free_rate) / ann_vol
    else:
        sharpe = np.nan

    return {
        "curve": curve,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe
    }
