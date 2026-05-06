import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

@st.cache_data(ttl=3600)
def get_data(tickers_tuple):
    """Pobiera dane historyczne dla podanych tickerów."""
    data = yf.download(list(tickers_tuple), period="3y")['Close']
    return data

def get_portfolio_stats(data_only):
    """Oblicza podstawowe statystyki portfela."""
    daily_rets = data_only.pct_change().dropna()
    # 'ME' to aktualne oznaczenie dla Month End w pandas
    monthly_rets = data_only.resample('ME').last().pct_change().dropna()
    monthly_vars = monthly_rets.quantile(0.05) * -1
    corr_matrix = monthly_rets.corr()
    return daily_rets, monthly_rets, monthly_vars, corr_matrix

def optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x):
    """
    Optymalizuje wagi portfela na podstawie wybranego trybu:
    - Bezpieczeństwo (VaR-First)
    - Efektywność (Sortino Ratio)
    """
    if opt_mode == "Bezpieczeństwo (VaR-First)":
        p = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko_val]
        # Cel: Odwrotność VaR ważona korelacyjnie
        target_w_raw = (1 / (monthly_vars ** p)) * (1 - corr_matrix.mean())
    else:
        # Sortino Ratio (zysk do ryzyka spadkowego)
        downside_std = monthly_rets[monthly_rets < 0].std() + 1e-6
        sortino = monthly_rets.mean() / downside_std
        p = {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko_val]
        target_w_raw = (sortino.clip(lower=0) ** p) * (1 - corr_matrix.mean())

    # Normalizacja wag wstępnych
    target_w = target_w_raw / target_w_raw.sum()
    
    # Ograniczenia (Bounds)
    c_bounds = [(min_bounds[t], 1.0) for t in tickers]
    
    # Warunek: Suma wag = 100%
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Opcjonalny warunek: Limit 2x (największa pozycja nie większa niż 2x najmniejsza)
    if limit_2x:
        constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})

    # Minimalizacja różnicy kwadratów od celu teoretycznego przy zachowaniu więzów
    res = minimize(
        lambda w: np.sum((w - target_w.values)**2), 
        target_w.values, 
        method='SLSQP', 
        bounds=c_bounds, 
        constraints=constraints
    )
    return res.x

def run_monte_carlo(data_only, wagi, kwota, adj_mc, market_params):
    """
    Przeprowadza symulację Monte Carlo z uwzględnieniem grubych ogonów (rozkład t-Studenta).
    """
    n_sims, nu, dt = 3000, 4, 1/252
    t_scale = np.sqrt((nu - 2) / nu) # Skalowanie dla zachowania wariancji
    
    log_rets = np.log(data_only / data_only.shift(1)).dropna()
    # Roczna zmienność portfela
    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
    
    results = {}
    for y in [5, 10]:
        days = y * 252
        paths = np.zeros((days, n_sims))
        curr = np.full(n_sims, float(kwota))
        
        t_beta = market_params.get('beta', 1.0) if adj_mc else 1.0
        p_alpha = market_params.get('alpha', 0.0) if adj_mc else 0.0
        
        for d in range(days):
            # Szum z grubymi ogonami
            eps = np.random.standard_t(df=nu, size=n_sims) * t_scale
            
            if adj_mc:
                # Model CAPM z dryftem
                # $\mu = (r_f + \beta(r_m - r_f) + \alpha - 0.5\sigma^2)dt$
                mu = (market_params['rf'] + t_beta * (market_params['rm'] - market_params['rf']) + p_alpha - 0.5 * (p_sigma**2)) * dt
                
                # Mean reversion bety (powrót do średniej rynkowej co rok)
                if d % 252 == 0 and d > 0: 
                    t_beta = t_beta * (1 - market_params['speed']) + 1.0 * market_params['speed']
            else:
                # Model oparty na średniej historycznej
                mu = (np.log(1 + data_only.pct_change().dropna().mean() @ wagi) * 252 - 0.5 * (p_sigma**2)) * dt
            
            curr *= np.exp(mu + p_sigma * eps * np.sqrt(dt))
            paths[d, :] = curr
        
        # Wyliczanie statystyk końcowych
        final = paths[-1, :]
        med = np.median(final)
        
        results[y] = {
            'paths': paths,
            'stats': pd.DataFrame({
                "Metryka": [
                    "95. Percentyl ", 
                    "Mediana ", 
                    "5. Percentyl (", 
                    "Prawdopodobieństwo straty", 
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
