import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY (Musi być pierwsza!)
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (SaaS Look)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
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
    n_sims = st.slider("Liczba symulacji Monte Carlo:", 1000, 10000, 10000)
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Pobieranie danych i optymalizacja VaR...'):
            raw_data = yf.download(tickers, period="3y")['Close']
            if raw_data.empty: st.stop()
            
            daily_rets = raw_data.pct_change().dropna()
            monthly_rets = raw_data.resample('ME').last().pct_change().dropna()
            
            monthly_vars = monthly_returns = monthly_rets.quantile(0.05) * -1
            corr_matrix = monthly_rets.corr()
            avg_corr = corr_matrix.mean()

        # --- OPTYMALIZACJA ---
        penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
        target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr)
        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        final_weights = res.x

        # --- WYNIKI (TABS) ---
        t1, t2, t3 = st.tabs(["📈 Portfel i Ryzyko", "🔮 Projekcje 5/10 Lat", "🔗 Mapa Korelacji"])

        with t1:
            st.subheader("Rekomendowana alokacja kapitału")
            c1, c2, c3 = st.columns(3)
            p_var = (final_weights * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Monthly VaR (95%)", f"{p_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}")
            c3.metric("Ryzyko (Kwota)", f"{p_var * kwota:,.2f}")
            
            df_res = pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': final_weights * 100,
                'Kwota (Waluta)': final_weights * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_res.style.format({'Udział (%)': '{:.2f}%', 'Kwota (Waluta)': '{:,.2f}'}), hide_index=True, use_container_width=True)

        with t2:
            st.subheader(f"Projekcje Monte Carlo ({n_sims} scenariuszy)")
            
            # Parametry portfela do symulacji (Fix na pamięć RAM)
            cov_matrix = daily_rets.cov().values
            port_mean = np.sum(daily_rets.mean() * final_weights)
            port_std = np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights)))
            
            col_5, col_10 = st.columns(2)
            plt.style.use("dark_background")

            for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                days = y * 252
                # Symulacja 10k ścieżek
                sim_rets = np.random.normal(port_mean, port_std, (days, n_sims))
                sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                
                final_v = sim_paths[-1, :]
                mediana = np.median(final_v)
                p95, p5 = np.percentile(final_v, 95), np.percentile(final_v, 5)
                cagr = (mediana / kwota)**(1/y) - 1
                chance_loss = (np.sum(final_v < kwota) / n_sims) * 100

                with (col_5 if i == 0 else col_10):
                    st.write(f"#### Prognoza na {lbl}")
                    st.table(pd.DataFrame({
                        "Metryka": ["Mediana", "95% (Optymizm)", "5% (Pesymizm)", "Zwrot CAGR", "Szansa na stratę"],
                        "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{chance_loss:.1f}%"]
                    }))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.06, linewidth=0.7)
                    ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2.5, label='Mediana')
                    ax.set_ylim(np.percentile(final_v, 1)*0.7, np.percentile(final_v, 99)*1.3)
                    ax.set_title(f"Wachlarz scenariuszy ({lbl})")
                    ax.grid(True, alpha=0.15)
                    st.pyplot(fig)
            plt.style.use('default')

        with t3:
            st.subheader("Powiązania historyczne aktywów")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

        st.caption("⚠️ Analiza Monte Carlo oparta na danych historycznych. Nie gwarantuje przyszłych zysków.")

    except Exception as e:
        st.error(f"Błąd krytyczny: {e}")
        st.info("Spróbuj odświeżyć stronę lub sprawdź tickery spółek.")
