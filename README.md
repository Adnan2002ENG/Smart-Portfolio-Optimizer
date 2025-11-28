# Smart Portfolio Optimizer  
**Project 1 – Finance & Big Data Portfolio**  
**Python, Quantitative Finance, Optimization, Risk Management**

This project implements a complete portfolio optimization and backtesting engine using Python.  
It compares three widely used investment strategies:

1. **Equal-Weight Portfolio (EW)**
2. **Inverse-Volatility Portfolio (IV)**
3. **Markowitz Mean-Variance Optimization (Max Sharpe)**

The goal is to analyze how different allocation methods behave in terms of performance, risk, and stability across time.

---

## 🎯 Objective

This project answers a classic business question in asset management:

> **How should an investor allocate capital across several assets to maximize risk-adjusted performance?**

It simulates, compares, and visualizes the portfolio strategies using real market data (ETF prices), and evaluates them with professional metrics (Sharpe, volatility, drawdowns).

---

## 📊 Features

### ✔ Data Pipeline
- Live market data download with **yfinance**
- Daily returns computation  
- Annualized mean returns, volatility, and covariance matrix
- Clean and reproducible pipeline

### ✔ Portfolio Strategies Implemented
#### **1. Equal-Weight (EW)**
Simple allocation: all assets have the same weight.  
Benchmark portfolio.

#### **2. Inverse-Volatility (IV)**
Risk-based allocation that assigns more weight to stable assets  
and less weight to volatile assets.

#### **3. Markowitz Optimization (Max Sharpe)**
Convex optimization with **cvxpy**.  
Constraints:
- Weights ≥ 0  
- Sum(weights) = 1  
- Risk-free rate considered

---

## 📈 Backtesting & Risk Metrics

For each strategy, the system computes:

- **Annualized return**
- **Annualized volatility**
- **Sharpe ratio**
- **Cumulative portfolio value**
- **Risk-adjusted performance comparison**

The results allow you to evaluate which portfolio gives the best compromise between return and risk.

---

## 📉 Example Output

Example of performance table:

| Strategy       | Annual Return | Annual Vol | Sharpe |
|----------------|---------------|------------|--------|
| Equal-Weight   | 0.10          | 0.14       | 0.57   |
| Inverse-Vol    | 0.09          | 0.10       | 0.71   |
| Markowitz      | 0.14          | 0.16       | 0.75   |

(Results vary depending on dates and tickers.)

---

## 🧠 What This Project Demonstrates (NEOMA interview level)

- Strong understanding of **Modern Portfolio Theory (MPT)**  
- Ability to implement an optimization model used in real finance jobs  
- Knowledge of **risk management** concepts (volatility, covariance, Sharpe)  
- Ability to backtest investment strategies professionally  
- Technical mastery of Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `yfinance`
  - `cvxpy`

This project shows that you understand how to turn financial theory into a practical decision-making tool — a key skill for roles in **Asset Management, Quantitative Research, Risk, and Portfolio Analytics**.

---

## 🧱 Project Structure

