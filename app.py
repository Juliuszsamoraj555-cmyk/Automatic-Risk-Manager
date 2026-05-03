import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button {
        width: 100%; background-color: #238636 !important; color: white !important;
        border-radius: 8px; font-weight: 700; height: 3.5em; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR
st.title("🛡️ Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Ustawienia")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD",
        help="""
        **Jak wpisywać symbole?**
        System pobiera dane z Yahoo Finance. 
        * **USA:** Sam ticker (np. `AAPL`).
        * **Polska:** Dodaj `.WA` (np. `ALE.WA`).
        * **Krypto:** Dodaj `-USD` (np. `BTC-USD`).
        """
    )
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    
    opt_mode = st.radio(
        "Tryb Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        index=0,
        help="""
        **Bezpieczeństwo (VaR):** Skupia się na minimalizacji strat w najgorszych scenariuszach. Wybiera najbardziej stabilne spółki.
        **Efektywność (Sortino):** Szuka najlepszego zysku w stosunku do ryzyka spadków. Docenia spółki, które rosną gwałtownie, ale rzadko zaliczają głębokie 'doły'.
        """
    )

    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    # --- SKORYGOWANA SYMULACJA Z POPRAWKĄ NA "SENS" ---
    st.divider()
    adj_mc = st.checkbox(
        "Skorygowana symulacja Monte Carlo", 
        value=False,
        help="Włącza model CAPM (Capital Asset Pricing Model) skorygowany o Alfę spółek."
    )
    
    if adj_mc:
        with st.expander("📈 Parametry Korekty Rynkowej", expanded=True):
            rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
            mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
            alpha_retention = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30, 
                                        help="Ile % historycznej przewagi spółki nad rynkiem utrzyma się w symulacji. 0% = wynik rynkowy, 100% = pełna powtórka sukcesu.")
            beta_speed = st.slider("Szybkość wygasania Bety:", 0.0, 0.2, 0.05)

    st.divider()
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada 2x:** Algorytm pilnuje, aby największa pozycja w portfelu była maksymalnie dwa razy większa niż najmniejsza. Zapobiega to dominacji jednej spółki i chroni przed ryzykiem specyficznym.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Co to robi?**
        To matematyczna 'wróżba' oparta na faktach. System przeprowadza **10 000 wirtualnych rzutów kostką**, tworząc tysiące alternatywnych scenariuszy przyszłości dla Twojego portfela.
        """
    )
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Analizowanie danych...'):
            fetch_tickers = tickers + (["SPY"] if adj_mc else [])
            data_raw = yf.download(fetch_tickers, period="3y")['Close']
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            # Pobieranie i separacja benchmarku
            if adj_mc:
                spy_rets = data_raw["SPY"].pct_change().dropna()
                stock_data = data_raw[tickers]
                
                # Obliczanie Alfy i Bety
                betas = {}
                alphas = {}
                spy_annual_ret = (1 + spy_rets.mean())**252 - 1
                
                for t in tickers:
                    t_rets = stock_data[t].pct_change().dropna()
                    combined = pd.concat([t_rets, spy_rets], axis=1).dropna()
                    cov = np.cov(combined.iloc[:,0], combined.iloc[:,1])[0,1]
                    var = np.var(combined.iloc[:,1])
                    b = cov / var
                    betas[t] = b
                    
                    # Historyczny zwrot roczny spółki
                    hist_annual_ret = (1 + t_rets.mean())**252 - 1
                    # Alfa historyczna = Wynik - [Rf + Beta * (Rm_hist - Rf)]
                    capm_hist = rf_rate + b * (spy_annual_ret - rf_rate)
                    alphas[t] = hist_annual_ret - capm_hist
                
                data_only = stock_data
            else:
                data_only = data_raw[tickers] if "SPY" in data_raw.columns else data_raw

            df_daily_rets = data_only.pct_change().dropna()
            df_monthly_rets = data_only.resample('ME').last().pct_change().dropna()
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()

        # OPTYMALIZACJA
        if opt_mode == "Bezpieczeństwo (VaR-First)":
            penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
            target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - corr_matrix.mean())
        else:
            mean_ret = df_monthly_rets.mean()
            downside_std = df_monthly_rets[df_monthly_rets < 0].std()
            sortino_ratios = mean_ret / (downside_std + 1e-6)
            power_map = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            target_w_raw = (sortino_ratios.clip(lower=0) ** power_map[ryzyko]) * (1 - corr_matrix.mean())

        target_w = target_w_raw / target_w_raw.sum()
        res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                       method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), 
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + 
                       ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
        wagi_finalne = res.x

        # WYNIKI
        tabs = st.tabs(["📈 Portfel", "🔮 Symulacja Monte Carlo", "🔗 Korelacje"])

        with tabs[0]:
            st.subheader(f"Rekomendowana alokacja ({opt_mode})")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi_finalne * monthly_vars).sum()
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystyczna miara ryzyka straty miesięcznej.")
            c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}", help="Mierzy jak bardzo aktywa poruszają się w tym samym kierunku.")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Szacowana miesięczna strata przy kapitale {kwota:,.0f} PLN.")
            
            df_wynik = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi_finalne * 100, 'Kwota': wagi_finalne * kwota})
            if adj_mc: df_wynik['Beta'] = [betas[t] for t in tickers]
            st.dataframe(df_wynik.sort_values(by='Udział (%)', ascending=False).style.format({
                'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'
            }), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"Symulacja Monte Carlo - 10,000 symulacji ({opt_mode})")
                st.info("""**Ważna informacja:** Symulacja Monte Carlo bazuje na zmienności historycznej i statystyce. 
                        Pamiętaj, że wyniki historyczne nie są gwarancją przyszłych zysków.""")
                
                n_sims, dt = 10000, 1/252
                log_rets = np.log(data_only / data_only.shift(1)).dropna()
                port_sigma_annual = np.sqrt(np.dot(wagi_finalne.T, np.dot(log_rets.cov().values, wagi_finalne))) * np.sqrt(252)
                
                col_a, col_b = st.columns(2)
                plt.style.use("dark_background")

                for i, (years, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    days = years * 252
                    paths = np.zeros((days, n_sims))
                    current_prices = np.full(n_sims, float(kwota))
                    
                    if adj_mc:
                        # Obliczanie ważonej Bety i ważonej Alfy portfela
                        p_beta = np.sum([betas[t] * wagi_finalne[idx] for idx, t in enumerate(tickers)])
                        p_alpha = np.sum([alphas[t] * wagi_finalne[idx] for idx, t in enumerate(tickers)])
                        # Utrzymanie części Alfy zgodnie z suwakiem
                        adj_alpha = p_alpha * (alpha_retention / 100)
                        
                        temp_beta = p_beta
                        for d in range(days):
                            # CAPM + Utrzymana Alfa - Volatility Drag
                            expected_annual = rf_rate + temp_beta * (mkt_ret - rf_rate) + adj_alpha
                            mu_adj = (expected_annual - 0.5 * (port_sigma_annual**2)) * dt
                            current_prices *= np.exp(mu_adj + port_sigma_annual * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                            paths[d, :] = current_prices
                            if d % 252 == 0: temp_beta = temp_beta * (1 - beta_speed) + 1.0 * beta_speed
                    else:
                        hist_mu = np.sum(df_daily_rets.mean() * wagi_finalne) * 252
                        mu_adj = (hist_mu - 0.5 * (port_sigma_annual**2)) * dt
                        for d in range(days):
                            current_prices *= np.exp(mu_adj + port_sigma_annual * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                            paths[d, :] = current_prices

                    final_v = paths[-1, :]
                    mediana = np.median(final_v)
                    
                    stats_df = pd.DataFrame({
                        "Metryka": ["95. Percentyl", "3. Kwartyl (75%)", "Mediana", "1. Kwartyl (25%)", "5. Percentyl", "Szansa na stratę kapitału", "Średni roczny zwrot (CAGR)"],
                        "Wartość": [
                            f"{np.percentile(final_v, 95):,.2f}", 
                            f"{np.percentile(final_v, 75):,.2f}", 
                            f"{mediana:,.2f}", 
                            f"{np.percentile(final_v, 25):,.2f}", 
                            f"{np.percentile(final_v, 5):,.2f}", 
                            f"{(np.sum(final_v < kwota) / n_sims) * 100:.1f}%",
                            f"{((mediana / kwota)**(1/years) - 1)*100:.2f}%"
                        ]
                    })

                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        st.table(stats_df)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)

    except Exception as e:
        st.error(f"Błąd: {e}")
