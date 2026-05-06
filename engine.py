import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

@st.cache_data(ttl=3600)
def get_data(tickers_tuple):
    """Pobiera dane historyczne zamknięcia dla zadanych tickerów."""
    return yf.download(list(tickers_tuple), period="3y")['Close']

def get_portfolio_stats(data_only):
    """Oblicza statystyki niezbędne do optymalizacji wag."""
    daily_rets = data_only.pct_change().dropna()
    # Wykorzystanie 'ME' (Month End) zgodnie z najnowszymi standardami pandas
    monthly_rets = data_only.resample('ME').last().pct_change().dropna()
    # Value at Risk (VaR) na poziomie 5% jako miara ryzyka
    monthly_vars = monthly_rets.quantile(0.05) * -1
    corr_matrix = monthly_rets.corr()
    return daily_rets, monthly_rets, monthly_vars, corr_matrix

def optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x):
    """
    Optymalizuje wagi portfela, dążąc do celu teoretycznego (VaR lub Sortino) 
    przy zachowaniu zadanych ograniczeń.
    """
    if opt_mode == "Bezpieczeństwo (VaR-First)":
        p = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko_val]
        # Im wyższy VaR, tym mniejsza waga. (1 - mean_corr) promuje dywersyfikację.
        target_w_raw = (1 / (monthly_vars ** p)) * (1 - corr_matrix.mean())
    else:
        # Optymalizacja pod kątem Sortino Ratio (zysk do ryzyka spadkowego)
        m_rets = monthly_rets.mean()
        d_std = monthly_rets[monthly_rets < 0].std() + 1e-6
        sortino = m_rets / d_std
        p = {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko_val]
        target_w_raw = (sortino.clip(lower=0) ** p) * (1 - corr_matrix.mean())

    # Normalizacja do 100%
    target_w = target_w_raw / target_w_raw.sum()
    c_bounds = [(min_bounds[t], 1.0) for t in tickers]
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if limit_2x:
        # Warunek: największa pozycja nie może być większa niż dwukrotność najmniejszej
        constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})

    # Minimalizacja różnicy kwadratów między wagami rzeczywistymi a celem
    res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                   method='SLSQP', bounds=c_bounds, constraints=constraints)
    return res.x

def run_monte_carlo(data_only, wagi, kwota, adj_mc, market_params):
    """
    Symulacja Monte Carlo wykorzystująca proces GBM z rozkładem t-Studenta (Fat Tails).
    """
    n_sims, nu, dt = 3000, 4, 1/252
    t_scale = np.sqrt((nu - 2) / nu) # Korekta skali dla zachowania wariancji rozkładu t
    
    log_rets = np.log(data_only / data_only.shift(1)).dropna()
    # Roczna zmienność portfela na bazie macierzy kowariancji
    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
    
    results = {}
    for y in [5, 10]:
        days = y * 252
        paths = np.zeros((days, n_sims))
        curr = np.full(n_sims, float(kwota))
        
        # Inicjalizacja parametrów rynkowych
        t_beta = market_params.get('beta', 1.0)
        p_alpha = market_params.get('alpha', 0.0)
        
        for d in range(days):
            # Generowanie szumu losowego (grube ogony zwiększają realizm krachów)
            eps = np.random.standard_t(df=nu, size=n_sims) * t_scale
            
            if adj_mc:
                # MODEL SKORYGOWANY (CAPM + dryft)
                # Dryft = (stopa wolna od ryzyka + beta * premia rynkowa + alpha - korekta zmienności)
                mu = (market_params['rf'] + t_beta * (market_params['rm'] - market_params['rf']) + p_alpha - 0.5 * (p_sigma**2)) * dt
                
                # Powrót Bety do średniej (1.0) w czasie (Mean Reversion)
                if d % 252 == 0 and d > 0: 
                    t_beta = t_beta * (1 - market_params['speed']) + 1.0 * market_params['speed']
            else:
                # MODEL STANDARDOWY (HISTORYCZNY)
                # Obliczanie średniego historycznego zwrotu geometrycznego
                hist_mu = (data_only.pct_change().dropna().mean() @ wagi) * 252
                mu = (np.log(1 + hist_mu) - 0.5 * (p_sigma**2)) * dt
            
            # Przejście na następny dzień (Geometryczny Ruch Browna)
            curr *= np.exp(mu + p_sigma * eps * np.sqrt(dt))
            paths[d, :] = curr
        
        # Statystyki końcowe
        final = paths[-1, :]
        med = np.median(final)
        
        results[y] = {
            'paths': paths,
            'stats': pd.DataFrame({
                "Metryka": [
                    "95. Percentyl ", 
                    "Mediana ", 
                    "5. Percentyl ", 
                    "Prawd. straty", 
                    "CAGR "
                ],
                "Wartość": [
                    f"{np.percentile(final, 95):,.0f} PLN",
                    f"{med:,.0f} PLN",
                    f"{np.percentile(final, 5):,.0f} PLN",
                    f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%",
                    f"{((med / kwota)**(1/y) - 1)*100:.2f}%"
                ]
            })
        }
    return results
