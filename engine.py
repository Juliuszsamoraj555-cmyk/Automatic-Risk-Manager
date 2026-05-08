import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
@st.cache_data(ttl=3600)
def get_final_data(tickers_tuple, target_ccy):
    # 1. Pobieranie danych (zdejmujemy strefę czasową .tz_localize(None))
    raw_data = yf.download(list(tickers_tuple), period="3y")['Close'].tz_localize(None)
    fx_data = yf.download("USDPLN=X", period="3y")['Close'].tz_localize(None)
    
    # Upewniamy się, że fx_data to Series (czasem yfinance zwraca DataFrame)
    if isinstance(fx_data, pd.DataFrame):
        fx_data = fx_data.iloc[:, 0]
    
    normalized_df = raw_data.copy()
    
    # Jeśli mamy tylko jeden ticker, raw_data może być Series - zamieniamy na DataFrame
    if isinstance(normalized_df, pd.Series):
        normalized_df = normalized_df.to_frame()

    # 3. Przeliczanie dynamiczne
    for ticker_name in normalized_df.columns:
        try:
            t_obj = yf.Ticker(ticker_name)
            native_ccy = t_obj.fast_info['currency']
        except:
            native_ccy = 'USD'
            
        if native_ccy == "USD" and target_ccy == "PLN":
            # Używamy .multiply dla pewności wyrównania dat
            normalized_df[ticker_name] = normalized_df[ticker_name].multiply(fx_data, axis=0)
        elif native_ccy == "PLN" and target_ccy == "USD":
            # Używamy .divide
            normalized_df[ticker_name] = normalized_df[ticker_name].divide(fx_data, axis=0)
            
    # Usuwamy wiersze, gdzie nie ma kompletu danych (np. luki w święta)
    return normalized_df.dropna()

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
    if "VaR-First" in opt_mode:
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

def run_monte_carlo(data_only, wagi, kwota, tickers, adj_mc, rf_rate, mkt_ret, alpha_ret, beta_speed, betas, alphas, target_ccy, mc_labels):
    n_sims, dt = 3000, 1/252
    log_rets = np.log(data_only / data_only.shift(1)).dropna()
    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
    daily_rets = data_only.pct_change().dropna()
    results = {}

    for y in [5, 10]:
        days = y * 252
        paths = np.zeros((days, n_sims))
        curr = np.full(n_sims, float(kwota))
        
        if adj_mc:
            # --- TRYB: FAT TAILS ENGINE (SKORYGOWANY) ---
            nu = 4 
            t_scale = np.sqrt((nu - 2) / nu)
            p_beta = np.sum([betas[t] * wagi[idx] for idx, t in enumerate(tickers)])
            p_alpha = np.sum([alphas[t] * wagi[idx] for idx, t in enumerate(tickers)]) * (alpha_ret / 100)
            t_beta = p_beta
            
            for d in range(days):
                # Rozkład t-Studenta dla "Grubych Ogonów"
                epsilon = np.random.standard_t(df=nu, size=n_sims) * t_scale
                mu = (rf_rate + t_beta * (mkt_ret - rf_rate) + p_alpha - 0.5 * (p_sigma**2)) * dt
                curr *= np.exp(mu + p_sigma * epsilon * np.sqrt(dt))
                paths[d, :] = curr
                if d % 252 == 0: 
                    t_beta = t_beta * (1 - beta_speed) + 1.0 * beta_speed
        else:
            # --- TRYB: STANDARDOWY (ZWYKŁY MONTE CARLO) ---
            mu = (np.sum(daily_rets.mean() * wagi) * 252 - 0.5 * (p_sigma**2)) * dt
            for d in range(days):
                epsilon = np.random.normal(0, 1, n_sims)
                curr *= np.exp(mu + p_sigma * epsilon * np.sqrt(dt))
                paths[d, :] = curr
        
        final = paths[-1, :]
        med = np.median(final)
        results[y] = {
            'paths': paths,
            'stats': pd.DataFrame({
                "Metryka": mc_labels, # Silnik mówi: 'Wstaw tu to, co mi podasz w pudełku'
                "Wartość": [
                    f"{np.percentile(final, 95):,.0f} {target_ccy}",   # 1. 95th
                    f"{np.percentile(final, 75):,.0f} {target_ccy}",   # 2. Q3 (NOWE)
                    f"{med:,.0f} {target_ccy}",                        # 3. Mediana
                    f"{np.percentile(final, 25):,.0f} {target_ccy}",   # 4. Q1 (NOWE)
                    f"{np.percentile(final, 5):,.0f} {target_ccy}",    # 5. 5th
                    f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%",  # 6. Prawd. straty
                    f"{((med / kwota)**(1/y) - 1)*100:.2f}%"           # 7. CAGR
                ]
            })
        }
    return results
def run_backtest(data_only, wagi, benchmark_data):
    """
    Oblicza historyczną wydajność portfela na podstawie wyliczonych wag.
    """
    # 1. Stopy zwrotu z aktywów w portfelu
    rets = data_only.pct_change().dropna()
    
    # 2. Ważona stopa zwrotu portfela (dzień po dniu)
    port_rets = rets.dot(wagi)
    
    # 3. Skumulowana stopa zwrotu (1.0 = start)
    port_cum = (1 + port_rets).cumprod()
    
    # 4. Skumulowana stopa zwrotu benchmarku (np. SPY)
    bench_rets = benchmark_data.pct_change().dropna()
    # Wyrównujemy indeksy, żeby benchmark i portfel miały te same dni
    bench_cum = (1 + bench_rets[port_cum.index]).cumprod()
    
    return port_cum, bench_cum
