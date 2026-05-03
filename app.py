import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (SaaS Look)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button {
        width: 100%;
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 700;
        height: 3.5em;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Ustawienia")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek (ticker):", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    run_mc = st.checkbox("Wykonaj symulacje Monte Carlo", value=True)
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Pobieranie i analizowanie danych...'):
            # Pobieranie danych (3 lata historyczne)
            data_raw = yf.download(tickers, period="3y")['Close']
            
            # Naprawa struktury MultiIndex (jeśli występuje)
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            # Zwroty dzienne (do Monte Carlo)
            df_daily_rets = data_raw.pct_change().dropna()
            
            # Zwroty miesięczne (do VaR i Optymalizacji) - NAPRAWIONA SKŁADNIA
            df_monthly_rets = data_raw.resample('ME').last().pct_change().dropna()
            
            # Statystyki ryzyka
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()
            avg_corr_each = corr_matrix.mean()

        # --- OPTYMALIZACJA WAG ---
        penalty_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
        penalty = penalty_map.get(ryzyko)
        
        # Formuła: Ryzyko skorygowane o korelację
        target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        # Rozwiązanie matematyczne
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=constraints)
        wagi_finalne = res.x

        # --- WYNIKI (TABS) ---
        tab_titles = ["📈 Portfel i Ryzyko"]
        if run_mc: tab_titles.append("🔮 Projekcje 5/10 Lat")
        tab_titles.append("🔗 Mapa Korelacji")
        
        tabs = st.tabs(tab_titles)

        # TAB 1: PORTFEL
        with tabs[0]:
            st.subheader("Rekomendowana alokacja")
            c1, c2, c3 = st.columns(3)
            port_var = (wagi_finalne * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Monthly VaR (95%)", f"{port_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}")
            c3.metric("Ryzyko (PLN)", f"{port_var * kwota:,.2f}")
            
            df_wynik = pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': wagi_finalne * 100,
                'Kwota': wagi_finalne * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_wynik.style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

        # TAB 2: MONTE CARLO
        if run_mc:
            with tabs[1]:
                st.subheader("Projekcje długoterminowe (10,000 symulacji)")
                
                # Obliczanie parametrów portfela
                cov_matrix = df_daily_rets.cov()
                port_mean = np.sum(df_daily_rets.mean() * wagi_finalne)
                port_std = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix, wagi_finalne)))
                
                n_sims = 10000
                plt.style.use("dark_background")
                col_a, col_b = st.columns(2)

                for i, (y, label) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    days = y * 252
                    sim_rets = np.random.normal(port_mean, port_std, (days, n_sims))
                    sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                    
                    final_v = sim_paths[-1, :]
                    mediana = np.median(final_v)
                    cagr = (mediana / kwota)**(1/y) - 1
                    
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {label}")
                        st.table(pd.DataFrame({
                            "Metryka": ["Mediana", "95% Optymizm", "5% Pesymizm", "CAGR"],
                            "Wartość": [f"{mediana:,.2f}", f"{np.percentile(final_v, 95):,.2f}", f"{np.percentile(final_v, 5):,.2f}", f"{cagr*100:.2f}%"]
                        }))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2)
                        st.pyplot(fig)
                plt.style.use('default')

        # TAB 3: KORELACJE
        with tabs[-1]:
            st.subheader("Mapa Korelacji Aktywów")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Coś poszło nie tak podczas obliczeń: {e}")
