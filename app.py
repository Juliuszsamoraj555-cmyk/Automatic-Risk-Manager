import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Automatic Risk Manager Pro",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. CUSTOM CSS (Lifting Graficzny) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button {
        width: 100%;
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        height: 3.5em !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
st.title("🛡️ Automatic Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Konfiguracja")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek:", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000)
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    
    st.info("💡 Symulacja Monte Carlo zostanie przeprowadzona na **10 000** scenariuszy dla maksymalnej precyzji statystycznej.")
    analizuj = st.button("GENERUJ RAPORT")

# --- 4. LOGIKA ANALITYCZNA ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('📊 Pobieranie danych i uruchamianie 10,000 symulacji...'):
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_returns = raw_data.pct_change().dropna()
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # Optymalizacja wag
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

    # --- 5. TABS ---
    tab1, tab2 = st.tabs(["📈 Portfel i Ryzyko", "🔮 Projekcje 5/10 Lat"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        portfel_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.where(mask).stack().mean()
        
        c1.metric("Monthly VaR (95%)", f"{portfel_var*100:.2f}%")
        c2.metric("Średnia Korelacja", f"{mean_corr:.2f}")
        c3.metric("Ryzyko (Waluta)", f"{portfel_var * kwota:,.2f}")

        st.divider()
        wynik_df = pd.DataFrame({
            'Ticker': monthly_vars.index,
            'VaR Spółki': [f"{v*100:.2f}%" for v in monthly_vars],
            'Udział': final_weights * 100,
            'Kwota': final_weights * kwota
        }).sort_values(by='Udział', ascending=False)
        st.dataframe(wynik_df.style.format({'Udział': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Projekcje Długoterminowe (Metoda Monte Carlo)")
        
        # PARAMETRY MC
        n_sims = 10000  # ZWIĘKSZONA LICZBA SYMULACJI
        cov_matrix = daily_returns.cov()
        L = np.linalg.cholesky(cov_matrix)
        mean_daily = daily_returns.mean().values
        
        def run_mc(n_years):
            days = n_years * 252
            Z = np.random.normal(size=(days, n_sims, len(tickers)))
            shocks = Z @ L.T 
            daily_sim_rets = mean_daily + shocks
            port_rets = daily_sim_rets @ final_weights
            return kwota * np.cumprod(1 + port_rets, axis=0)

        res_5 = run_sim_5 = run_mc(5)
        res_10 = run_sim_10 = run_mc(10)

        col_a, col_b = st.columns(2)
        plt.style.use("dark_background")

        for i, (data, label, years) in enumerate(zip([res_5, res_10], ["5 Lat", "10 Lat"], [5, 10])):
            final_vals = data[-1, :]
            mediana = np.median(final_vals)
            p95, p5 = np.percentile(final_vals, 95), np.percentile(final_vals, 5)
            cagr = (mediana / kwota)**(1/years) - 1
            chance_loss = (np.sum(final_vals < kwota) / n_sims) * 100
            
            with (col_a if i == 0 else col_b):
                st.write(f"### Horyzont {label}")
                st.table(pd.DataFrame({
                    "Metryka": ["Mediana", "95. Percentyl (Optymizm)", "5. Percentyl (Pesymizm)", "Średni zwrot (CAGR)", "Szansa na stratę"],
                    "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{chance_loss:.1f}%"]
                }))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                # Rysujemy tylko 100 losowych ścieżek dla wydajności UI
                sampled_indices = np.random.choice(n_sims, 100, replace=False)
                ax.plot(data[:, sampled_indices], color='skyblue', alpha=0.07, linewidth=0.8)
                ax.plot(np.median(data, axis=1), color='white', linewidth=2.5, label='Mediana')
                ax.set_ylim(np.percentile(data[-1,:], 1)*0.8, np.percentile(data[-1,:], 99)*1.2)
                ax.set_title(f"Symulacja {label} (Próbka 100 z 10,000 ścieżek)")
                ax.grid(True, alpha=0.15)
                st.pyplot(fig)

        st.warning("⚠️ **Zastrzeżenie:** Powyższa analiza jest symulacją Monte Carlo opartą na danych historycznych (3Y lookback). Rynek finansowy jest nieprzewidywalny. Wyniki historyczne nie gwarantują identycznych rezultatów w przyszłości. Ryzyko straty kapitału leży po stronie inwestora.")

else:
    st.info("👈 Wprowadź dane i kliknij przycisk, aby rozpocząć analizę ryzyka.")
