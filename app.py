import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# --- 2. PROFESSIONAL CSS UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    .stButton > button { width: 100%; background-color: #238636 !important; color: white !important; border-radius: 8px; font-weight: 700; height: 3.5em; border: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Konfiguracja")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek:", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    n_sims = st.slider("Liczba symulacji MC:", 1000, 10000, 5000)
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# --- 4. GŁÓWNA LOGIKA ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Pobieranie danych rynkowych...'):
            raw_data = yf.download(tickers, period="3y")['Close']
            
            if raw_data.empty:
                st.error("Nie udało się pobrać danych dla podanych symboli. Sprawdź, czy tickery są poprawne.")
                st.stop()
                
            daily_rets = raw_data.pct_change().dropna()
            monthly_rets = raw_data.resample('ME').last().pct_change().dropna()
            
            monthly_vars = monthly_rets.quantile(0.05) * -1
            corr_matrix = monthly_rets.corr()
            avg_corr = corr_matrix.mean()

        # Optymalizacja
        risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
        penalty = risk_map.get(ryzyko)
        target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr)
        target_weights = target_weights_raw / target_weights_raw.sum()

        def objective(w): return np.sum((w - target_weights.values)**2)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_weights.values, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in range(len(tickers))), constraints=constraints)
        final_weights = res.x

        # --- WYNIKI ---
        t1, t2, t3 = st.tabs(["📈 Portfel", "🔮 Projekcje", "🔗 Korelacje"])

        with t1:
            c1, c2, c3 = st.columns(3)
            port_var = (final_weights * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            m_corr = corr_matrix.where(mask).stack().mean()
            c1.metric("Monthly VaR (95%)", f"{port_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{m_corr:.2f}")
            c3.metric("Ryzyko (PLN)", f"{port_var * kwota:,.2f}")
            
            st.dataframe(pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': final_weights * 100,
                'Kwota': final_weights * kwota
            }).sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

        with t2:
            st.subheader(f"Symulacja Monte Carlo ({n_sims} prób)")
            # Stabilizacja macierzy kowariancji (fix dla LinAlgError)
            cov_matrix = daily_rets.cov().values + np.eye(len(tickers)) * 1e-8
            port_mean = np.sum(daily_rets.mean() * final_weights)
            port_std = np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights)))
            
            col_5, col_10 = st.columns(2)
            plt.style.use("dark_background")
            for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                days = y * 252
                sim_rets = np.random.normal(port_mean, port_std, (days, n_sims))
                sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                final_v = sim_paths[-1, :]
                mediana = np.median(final_v)
                cagr = (mediana / kwota)**(1/y) - 1
                
                with (col_5 if i == 0 else col_10):
                    st.write(f"#### Prognoza {lbl}")
                    st.table(pd.DataFrame({"Metryka": ["Mediana", "95% Optymizm", "5% Pesymizm", "CAGR"], "Wartość": [f"{mediana:,.2f}", f"{np.percentile(final_v, 95):,.2f}", f"{np.percentile(final_v, 5):,.2f}", f"{cagr*100:.2f}%"]}))
                    fig, ax = plt.subplots()
                    ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.05)
                    ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2)
                    st.pyplot(fig)
            plt.style.use('default')

        with t3:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd: {e}")
        st.info("Sprawdź, czy tickery spółek są poprawne i czy masz połączenie z internetem.")
