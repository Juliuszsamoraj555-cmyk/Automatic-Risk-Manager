import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Wymuszenie stabilnego backendu dla serwera
matplotlib.use('Agg')

# 1. KONFIGURACJA STRONY (Musi być pierwsza!)
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (SaaS Look)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #e6edf3; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stButton > button { width: 100%; background-color: #238636 !important; color: white !important; border-radius: 8px; font-weight: 700; height: 3.5em; border: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Konfiguracja")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek (ticker):", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    n_sims = 10000 # Stała liczba symulacji dla precyzji
    analizuj = st.button("URUCHOM ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Pobieranie i analizowanie danych...'):
            # Pobieranie Close i czyszczenie struktury
            raw_data = yf.download(tickers, period="3y")['Close']
            
            # Naprawa problemu MultiIndex w yfinance
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(-1)
            
            if raw_data.empty:
                st.error("Nie udało się pobrać danych. Sprawdź symbole spółek.")
                st.stop()
            
            daily_rets = raw_data.pct_change().dropna()
            monthly_rets = raw_data.resample('ME').last().pct_change().dropna()
            
            # Miesięczny VaR (5%)
            m_vars = monthly_rets.quantile(0.05) * -1
            corr_m = monthly_rets.corr()
            avg_c = corr_m.mean()

        # --- OPTYMALIZACJA ---
        penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
        target_w_raw = (1 / (m_vars ** penalty)) * (1 - avg_c)
        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        wagi = res.x

        # --- TABS ---
        t1, t2, t3 = st.tabs(["📈 Portfel", "🔮 Projekcje", "🔗 Korelacje"])

        with t1:
            st.subheader("Rekomendowana alokacja")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi * m_vars).sum()
            mask = np.triu(np.ones_like(corr_m, dtype=bool), k=1)
            m_corr_val = corr_m.where(mask).stack().mean()
            
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{m_corr_val:.2f}")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
            
            df_res = pd.DataFrame({
                'Ticker': m_vars.index,
                'Udział (%)': wagi * 100,
                'Kwota (Waluta)': wagi * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_res.style.format({'Udział (%)': '{:.2f}%', 'Kwota (Waluta)': '{:,.2f}'}), hide_index=True, use_container_width=True)

        with t2:
            st.subheader(f"Symulacja Monte Carlo ({n_sims} scenariuszy)")
            
            # Statystyki portfela
            p_mean = np.sum(daily_rets.mean() * wagi)
            p_std = np.sqrt(np.dot(wagi.T, np.dot(daily_returns.cov(), wagi)))
            
            col_5, col_10 = st.columns(2)
            plt.style.use("dark_background")

            for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                days = y * 252
                # Symulacja portfela
                s_rets = np.random.normal(p_mean, p_std, (days, n_sims))
                s_paths = kwota * np.cumprod(1 + s_rets, axis=0)
                
                final_v = s_paths[-1, :]
                mediana = np.median(final_v)
                cagr = (mediana / kwota)**(1/y) - 1
                
                with (col_5 if i == 0 else col_10):
                    st.write(f"#### Prognoza na {lbl}")
                    st.table(pd.DataFrame({
                        "Metryka": ["Mediana", "95% Optymizm", "5% Pesymizm", "CAGR"],
                        "Wartość": [f"{mediana:,.2f}", f"{np.percentile(final_v, 95):,.2f}", f"{np.percentile(final_v, 5):,.2f}", f"{cagr*100:.2f}%"]
                    }))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Próbka do wykresu
                    ax.plot(s_paths[:, :100], color='skyblue', alpha=0.06, linewidth=0.7)
                    ax.plot(np.median(s_paths, axis=1), color='white', linewidth=2)
                    ax.set_title(f"Scenariusze: {lbl}")
                    st.pyplot(fig)
            plt.style.use('default')

        with t3:
            st.subheader("Mapa powiązań (Korelacje)")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_m, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas obliczeń: {e}")
        st.info("Logi serwera mogą zawierać więcej szczegółów.")

else:
    st.info("👈 Skonfiguruj parametry i kliknij 'URUCHOM PEŁNĄ ANALIZĘ'.")
