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
    
    # --- NOWA SEKCJA: PARAMETRY CAPM ---
    with st.expander("📈 Zaawansowane parametry CAPM/GBM"):
        rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
        mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
        beta_speed = st.slider("Szybkość wygasania Bety:", 0.0, 0.2, 0.05, 
                               help="Jak szybko Beta spółki dąży do 1.0 (rynku) wraz z upływem lat.")

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
        
        **Dlaczego to ważne?**
        Zamiast jednej linii 'zysku', widzisz cały wachlarz możliwości – od bardzo optymistycznych po kryzysowe. Pozwala to realnie ocenić **szansę na stratę** oraz zrozumieć, jak szeroki jest zakres niepewności w inwestowaniu.
        """
    )
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Pobieranie danych i obliczanie parametrów CAPM...'):
            # Pobieramy tickers + SPY jako benchmark
            all_tickers = tickers + ["SPY"]
            data_raw = yf.download(all_tickers, period="3y")['Close']
            
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            # Oddzielamy benchmark
            spy_data = data_raw["SPY"]
            data_raw = data_raw[tickers]
            
            df_daily_rets = data_raw.pct_change().dropna()
            spy_rets = spy_data.pct_change().dropna()
            
            # Obliczanie Bety dla każdej spółki
            betas = {}
            for t in tickers:
                cov = np.cov(df_daily_rets[t], spy_rets)[0, 1]
                var = np.var(spy_rets)
                betas[t] = cov / var

            df_monthly_rets = data_raw.resample('ME').last().pct_change().dropna()
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()
            avg_corr_each = corr_matrix.mean()

        # OPTYMALIZACJA
        if opt_mode == "Bezpieczeństwo (VaR-First)":
            penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
            target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        else:
            mean_ret = df_monthly_rets.mean()
            downside_std = df_monthly_rets[df_monthly_rets < 0].std()
            sortino_ratios = mean_ret / (downside_std + 1e-6)
            power_map = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            target_w_raw = (sortino_ratios.clip(lower=0) ** power_map[ryzyko]) * (1 - avg_corr_each)

        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        wagi_finalne = res.x

        # WYNIKI
        tabs = st.tabs(["📈 Portfel", "🔮 Symulacja Monte Carlo", "🔗 Korelacje"])

        with tabs[0]:
            st.subheader(f"Rekomendowana alokacja ({opt_mode})")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi_finalne * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystyczna miara ryzyka straty miesięcznej.")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}", help="Mierzy jak bardzo aktywa poruszają się w tym samym kierunku.")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Szacowana miesięczna strata przy kapitale {kwota:,.0f} PLN.")
            
            st.divider()
            df_wynik = pd.DataFrame({
                'Ticker': tickers,
                'Beta': [betas[t] for t in tickers],
                'Udział (%)': wagi_finalne * 100,
                'Kwota': wagi_finalne * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_wynik.style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"Symulacja Monte Carlo - 10,000 symulacji ({opt_mode})")
                st.info("**Tryb zaawansowany:** Symulacja wykorzystuje model GBM (Geometric Brownian Motion) skorygowany o CAPM i volatility drag.")
                
                # Przygotowanie parametrów GBM dla portfela
                n_sims = 10000
                dt = 1/252
                
                # Obliczamy Betę portfela
                port_beta = np.sum([betas[t] * wagi_finalne[i] for i, t in enumerate(tickers)])
                
                # Obliczamy zmienność dzienną (logarytmiczną) i roczną
                log_returns = np.log(data_raw / data_raw.shift(1)).dropna()
                cov_matrix = log_returns.cov().values
                port_sigma_annual = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix, wagi_finalne))) * np.sqrt(252)

                col_a, col_b = st.columns(2)
                plt.style.use("dark_background")

                for i, (years, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    days = years * 252
                    
                    # Inicjalizacja ścieżek
                    paths = np.zeros((days, n_sims))
                    current_prices = np.full(n_sims, float(kwota))
                    
                    # Symulacja krok po kroku (dla mean-reversion bety)
                    temp_beta = port_beta
                    for d in range(days):
                        # 1. CAPM: E(Ri)
                        expected_return_annual = rf_rate + temp_beta * (mkt_ret - rf_rate)
                        
                        # 2. Drift Adjustment: mu_adj = E(Ri) - 0.5 * sigma^2
                        # Przeliczamy na skalę kroku czasowego dt
                        mu_adj = (expected_return_annual - 0.5 * (port_sigma_annual**2)) * dt
                        
                        # 3. Randomness (epsilon)
                        epsilon = np.random.normal(0, 1, n_sims)
                        
                        # 4. GBM Formula
                        current_prices *= np.exp(mu_adj + port_sigma_annual * epsilon * np.sqrt(dt))
                        paths[d, :] = current_prices
                        
                        # 5. Beta smoothing (raz na rok symulacji)
                        if d % 252 == 0:
                            temp_beta = temp_beta * (1 - beta_speed) + 1.0 * beta_speed
                    
                    final_v = paths[-1, :]
                    mediana = np.median(final_v)
                    stats = {
                        "95. Percentyl": np.percentile(final_v, 95),
                        "3. Kwartyl (75%)": np.percentile(final_v, 75),
                        "Mediana": mediana,
                        "1. Kwartyl (25%)": np.percentile(final_v, 25),
                        "5. Percentyl": np.percentile(final_v, 5),
                        "Szansa na stratę": f"{(np.sum(final_v < kwota) / n_sims) * 100:.1f}%",
                        "Średni roczny zwrot (CAGR)": f"{((mediana / kwota)**(1/years) - 1)*100:.2f}%"
                    }

                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        st.table(pd.DataFrame(stats.items(), columns=["Metryka", "Wartość"]))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)
                plt.style.use('default')

        with tabs[2]:
            st.subheader("Mapa Korelacji")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Coś poszło nie tak: {e}")
