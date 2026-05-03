import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. KONFIGURACJA ---
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button { width: 100%; background-color: #238636 !important; color: white !important; border-radius: 8px; font-weight: 600; height: 3.5em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Konfiguracja")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek:", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000)
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    analizuj = st.button("GENERUJ ANALIZĘ")

# --- 3. ANALIZA ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('📊 Mielenie danych rynkowych...'):
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_returns = raw_data.pct_change().dropna()
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # Optymalizacja
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
    target_weights = target_weights_raw / target_weights_raw.sum()

    def objective(weights): return np.sum((weights - target_weights.values)**2)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if limit_2x:
        cons.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    res = minimize(objective, target_weights.values, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in tickers), constraints=cons)
    final_weights = res.x

    # --- TABS ---
    tab1, tab2 = st.tabs(["📈 Portfel", "🔮 Projekcje 5/10 Lat"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        portfel_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.where(mask).stack().mean()
        
        c1.metric("Monthly VaR (95%)", f"{portfel_var*100:.2f}%")
        c2.metric("Średnia Korelacja", f"{mean_corr:.2f}")
        c3.metric("Ryzyko (Waluta)", f"{portfel_var * kwota:,.2f}")

        st.dataframe(pd.DataFrame({
            'Ticker': monthly_vars.index,
            'Udział': final_weights * 100,
            'Kwota': final_weights * kwota
        }).sort_values(by='Udział', ascending=False).style.format({'Udział': '{:.2f}%', 'Kwota': '{:,.2f}'}), 
        hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Projekcje Monte Carlo (5,000 symulacji)")
        
        # Optymalizacja pamięci: liczymy parametry portfela zamiast każdej spółki osobno w MC
        cov_matrix = daily_returns.cov().values
        # Roczna zmienność i zwrot portfela (uproszczone dla stabilności MC)
        port_mean = np.sum(daily_returns.mean() * final_weights)
        port_std = np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights)))
        
        col_a, col_b = st.columns(2)
        plt.style.use("dark_background")

        for i, (years, label) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
            n_sims = 5000
            n_days = years * 252
            
            # Generowanie zwrotów portfela (Memory efficient)
            sim_returns = np.random.normal(port_mean, port_std, (n_days, n_sims))
            sim_paths = kwota * np.cumprod(1 + sim_returns, axis=0)
            
            final_vals = sim_paths[-1, :]
            mediana = np.median(final_vals)
            p95, p5 = np.percentile(final_vals, 95), np.percentile(final_vals, 5)
            cagr = (mediana / kwota)**(1/years) - 1
            chance_loss = (np.sum(final_vals < kwota) / n_sims) * 100

            with (col_a if i == 0 else col_b):
                st.write(f"### Horyzont {label}")
                st.table(pd.DataFrame({
                    "Metryka": ["Mediana", "95% (Optymizm)", "5% (Pesymizm)", "Zwrot CAGR", "Szansa na stratę"],
                    "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{chance_loss:.1f}%"]
                }))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                # Wyświetlamy 50 przykładowych ścieżek
                ax.plot(sim_paths[:, :50], color='skyblue', alpha=0.1, linewidth=0.8)
                ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2, label='Mediana')
                ax.set_title(f"Wachlarz scenariuszy: {label}")
                ax.set_ylabel("Wartość (PLN)")
                ax.grid(True, alpha=0.1)
                st.pyplot(fig)

        st.caption("⚠️ Symulacja Monte Carlo oparta na zmienności historycznej. Dane nie gwarantują przyszłych zysków.")
