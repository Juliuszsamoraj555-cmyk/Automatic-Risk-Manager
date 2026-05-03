Faktycznie! W ferworze walki z błędami serwera i "wygładzaniem" kodu, zgubiłem tę kluczową funkcję, która pozwalała Jankowi zdecydować, czy chce czekać na skomplikowane obliczenia, czy tylko zobaczyć podział portfela.

Dziękuję za czujność – przywracamy tę opcję. Poniżej znajduje się kompletna, profesjonalna wersja kodu, która łączy nowoczesny wygląd (SaaS look), zaawansowaną optymalizację VaR z limitem 2x oraz opcjonalną, zoptymalizowaną pamięciowo symulację Monte Carlo (10 000 prób).

Wprowadziłem też funkcję, która automatycznie czyści strukturę danych z Yahoo Finance (squeeze), bo ostatnio ich format płata figle algorytmom.

Pełny, poprawiony kod app.py (Z opcją Monte Carlo)
Python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY (Musi być pierwsza!)
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (Profesjonalny SaaS Look)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    
    /* Karty metryk na górze */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    /* Stylizacja panelu bocznego */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Zielony, profesjonalny przycisk */
    .stButton > button {
        width: 100%;
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 700;
        height: 3.5em;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR (Konfiguracja)
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534360.png", width=60)
    st.header("⚙️ Ustawienia")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek (ticker):", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True, help="Największa pozycja będzie max 2x większa od najmniejszej.")
    
    # --- TUTAJ JEST PRZYWRÓCONA OPCJA ---
    run_mc = st.checkbox("Wykonaj symulacje Monte Carlo", value=True, help="Uruchamia zaawansowane projekcje 5/10 lat (10,000 prób). Może wydłużyć czas analizy.")
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Agregowanie danych rynkowych...'):
            # Pobieranie Close i czyszczenie struktury (Squeeze)
            data_raw = yf.download(tickers, period="3y")['Close']
            df_returns = data_raw.pct_change().dropna().squeeze()
            
            # Konwersja na miesięczne
            df_monthly = df_returns.resample('ME').add(1).prod().sub(1)
            
            # Statystyki do VaR i korelacji (z dopasowaniem nazw)
            monthly_vars = df_monthly.quantile(0.05) * -1
            corr_matrix = df_monthly.corr()
            avg_corr_each = corr_matrix.mean()

        # --- OPTYMALIZACJA WAG ---
        penalty_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
        penalty = penalty_map.get(ryzyko)
        
        # Formuła: Odwrotność VaR skorygowana o korelację
        target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        target_w = target_w_raw / target_w_raw.sum()

        # Matematyczny solver dla limitu 2x
        def objective(w): return np.sum((w - target_w.values)**2)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=constraints)
        wagi_finalne = res.x

        # --- TABS: ORGANIZACJA WYNIKÓW ---
        tab_list = ["📈 Portfel i Ryzyko"]
        if run_mc: tab_list.append("🔮 Projekcje 5/10 Lat")
        tab_list.append("🔗 Mapa Korelacji")
        
        tabs = st.tabs(tab_list)

        # TAB 1: PORTFEL
        with tabs[0]:
            st.subheader("Rekomendowany podział kapitału")
            
            c1, c2, c3 = st.columns(3)
            port_var = (wagi_finalne * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Monthly VaR (95%)", f"{port_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}")
            c3.metric("Ryzyko (PLN)", f"{port_var * kwota:,.2f}")
            
            st.divider()
            df_wynik = pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': wagi_finalne * 100,
                'Kwota': wagi_finalne * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_wynik.style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

        # TAB 2: MONTE CARLO (OPCJONALNE)
        if run_mc:
            with tabs[1]:
                st.subheader("Projekcje długoterminowe (10,000 symulacji)")
                st.info("Poniższa analiza pokazuje wachlarz możliwych scenariuszy na podstawie historycznej zmienności.")
                
                # Silnik Monte Carlo (Zoptymalizowany pod pamięć RAM)
                cov_matrix_daily = df_returns.cov()
                port_mean_daily = np.sum(df_returns.mean() * wagi_finalne)
                port_std_daily = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix_daily, wagi_finalne)))
                n_sims = 10000
                years = [5, 10]
                
                col_a, col_b = st.columns(2)
                plt.style.use("dark_background")

                for i, (y, label) in enumerate(zip(years, ["5 Lat", "10 Lat"])):
                    days = y * 252
                    # Symulacja 10k ścieżek
                    sim_rets = np.random.normal(port_mean_daily, port_std_daily, (days, n_sims))
                    sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                    
                    final_v = sim_paths[-1, :]
                    mediana = np.median(final_v)
                    cagr = (mediana / kwota)**(1/y) - 1
                    p95, p5 = np.percentile(final_v, 95), np.percentile(final_v, 5)
                    szansa_straty = (np.sum(final_v < kwota) / n_sims) * 100

                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {label}")
                        st.table(pd.DataFrame({
                            "Metryka": ["Mediana (Bazowy)", "Scenariusz Optymistyczny", "Scenariusz Pesymistyczny", "Zwrot CAGR", "Szansa straty"],
                            "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{szansa_straty:.1f}%"]
                        }))
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Rysujemy próbkę 100 linii
                        ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.06, linewidth=0.7)
                        ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2.5, label='Mediana')
                        ax.set_ylim(np.percentile(final_v, 1)*0.7, np.percentile(final_v, 99)*1.3)
                        ax.set_title(f"Rozpiętość scenariuszy ({label})")
                        ax.grid(True, alpha=0.15)
                        st.pyplot(fig)
                plt.style.use('default')

        # TAB 3: KORELACJE (Teraz dynamiczna w zależności od MC)
        with tabs[-1]:
            st.subheader("Mapa Korelacji Aktywów")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Coś poszło nie tak podczas obliczeń: {e}")
        st.info("Logi serwera mogą zawierać więcej szczegółów.")
