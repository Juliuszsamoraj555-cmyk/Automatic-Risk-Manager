import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

@st.cache_data(ttl=3600)
def get_data(tickers_tuple):
    return yf.download(list(tickers_tuple), period="3y")['Close']

def get_portfolio_stats(data_only):
    daily_rets = data_only.pct_change().dropna()
    monthly_rets = data_only.resample('ME').last().pct_change().dropna()
    monthly_vars = monthly_rets.quantile(0.05) * -1
    corr_matrix = monthly_rets.corr()
    return daily_rets, monthly_rets, monthly_vars, corr_matrix

def optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x):
    if opt_mode == "Bezpieczeństwo (VaR-First)":
        p = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko_val]
        target_w_raw = (1 / (monthly_vars ** p)) * (1 - corr_matrix.mean())
    else:
        # Sortino Ratio
        downside_std = monthly_rets[monthly_rets < 0].std() + 1e-6
        sortino = monthly_rets.mean() / downside_std
        target_w_raw = (sortino.clip(lower=0) ** {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko_val]) * (1 - corr_matrix.mean())

    target_w = target_w_raw / target_w_raw.sum()
    c_bounds = [(min_bounds[t], 1.0) for t in tickers]
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if limit_2x:
        constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})

    res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                   method='SLSQP', bounds=c_bounds, constraints=constraints)
    return res.x

def run_monte_carlo(data_only, wagi, kwota, adj_mc, market_params):
    n_sims = 3000
    nu = 4 # Fat Tails
    t_scale = np.sqrt((nu - 2) / nu)
    dt = 1/252
    
    log_rets = np.log(data_only / data_only.shift(1)).dropna()
    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
    
    results = {}
    for y in [5, 10]:
        days = y * 252
        paths = np.zeros((days, n_sims))
        curr = np.full(n_sims, float(kwota))
        
        t_beta = market_params.get('beta', 1.0) if adj_mc else 1.0
        p_alpha = market_params.get('alpha', 0.0) if adj_mc else 0.0
        
        for d in range(days):
            eps = np.random.standard_t(df=nu, size=n_sims) * t_scale
            if adj_mc:
                mu = (market_params['rf'] + t_beta * (market_params['rm'] - market_params['rf']) + p_alpha - 0.5 * (p_sigma**2)) * dt
                if d % 252 == 0: 
                    t_beta = t_beta * (1 - market_params['speed']) + 1.0 * market_params['speed']
            else:
                mu = (np.log(1 + data_only.pct_change().dropna().mean() @ wagi) * 252 - 0.5 * (p_sigma**2)) * dt
            
            curr *= np.exp(mu + p_sigma * eps * np.sqrt(dt))
            paths[d, :] = curr
        results[y] = paths
    return results
