import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

@st.cache_data(ttl=3600)
def get_data(tickers_tuple):
    return yf.download(list(tickers_tuple), period="3y")['Close']

def optimize_portfolio(tickers, data_only, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x):
    if opt_mode == "Bezpieczeństwo (VaR-First)":
        p = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko_val]
        target_w_raw = (1 / (monthly_vars ** p)) * (1 - corr_matrix.mean())
    else:
        sortino = monthly_rets_mean(data_only) / (downside_std(data_only) + 1e-6)
        target_w_raw = (sortino.clip(lower=0) ** {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko_val]) * (1 - corr_matrix.mean())

    target_w = target_w_raw / target_w_raw.sum()
    c_bounds = [(min_bounds[t], 1.0) for t in tickers]
    
    res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                   method='SLSQP', bounds=c_bounds,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + 
                   ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
    return res.x

def monthly_rets_mean(df):
    return df.resample('ME').last().pct_change().dropna().mean()

def downside_std(df):
    m_rets = df.resample('ME').last().pct_change().dropna()
    return m_rets[m_rets < 0].std()
