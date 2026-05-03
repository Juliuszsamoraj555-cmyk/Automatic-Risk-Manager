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

# --- 2. PROFESSIONAL CSS UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    
    /* Karty metryk */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Professional Green Button */
    .stButton > button {
        width: 100%;
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        height: 3.5em !important;
        border: none !important;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR INPUTS ---
st.title("🛡️ Automatic Risk Manager Pro")
st.caption("Advanced Portfolio Optimization & Monte Carlo Predictive Engine")

with st.sidebar:
    st.header("⚙️ Konfiguracja")
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Symbole spółek (ticker):", default_tickers)
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    
    st.divider()
    st.write("🔧 **Silnik Analityczny**")
    n_sims = st.slider("Liczba symulacji MC:", 1000, 10000, 5000)
    
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# --- 4. GŁÓWNA LOGIKA OBLICZEŃ ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('📊 Pobieranie danych rynkowych i obliczanie macierzy ryzyka...'):
        # Dane 3-letnie
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_rets = raw_data.pct_change().dropna()
        monthly_rets = raw_data.resample('ME').last().pct_change().dropna()
        
        # Statystyki do optymalizacji
        monthly_vars = monthly_rets.quantile(0.05) * -1
        corr_matrix = monthly_rets.corr()
        avg_corr = corr_matrix.mean()

    # --- OPTYMALIZACJA WAG (Brain) ---
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    
    # Formuła: Odwrotność VaR skorygowana o korelacje
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr)
    target_weights = target_weights_raw / target_weights_raw.sum()

    # Solver matematyczny dla limitu 2x
    def objective(w): return np.sum((w - target_weights.values)**2)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if limit_2x:
        constraints.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
    
    bounds = tuple((0.01, 1.0) for _ in range(len(tickers)))
    res = minimize(objective, target_weights.values, method='SLSQP', bounds=bounds, constraints=constraints)
    final_weights = res.x

    # --- 5. INTERFEJS WYNIKÓW (TABS) ---
    tab1, tab2, tab3 = st.tabs(["📈 Alokacja Portfela", "🔮 Projekcje 5/10 Lat", "🔗 Mapa Korelacji"])

    with tab1:
        st.subheader("Rekomendowany podział kapitału")
        
        # Górne metryki
        c1, c2, c3 = st.columns(3)
        port_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        m_corr = corr_matrix.where(mask).stack().mean()
        
        c1.metric("Monthly VaR (95%)", f"{port_var*100:.2f}%")
        c2.metric("Średnia Korelacja", f"{m_corr:.2f}")
        c3.metric("Ryzyko (Kwota)", f"{port_var * kwota:,.2f}")

        st.divider()
        
        # Tabela
        df_final = pd.DataFrame({
            'Ticker': monthly_vars.index,
            'VaR Spółki': [f"{v*100:.2f}%" for v in monthly_vars],
            'Udział (%)': final_weights * 100,
            'Kwota Inwestycji': final_weights * kwota
        }).sort_values(by='Udział (%)', ascending=False)
        
        st.dataframe(df_final.style.format({'Udział (%)': '{:.2f}%', 'Kwota Inwestycji': '{:,.2f}'}), 
                     hide_index=True, use_container_width=True)

    with tab2:
        st.subheader(f"Symulacja Monte Carlo ({n_sims} scenariuszy)")
        
        # Silnik Monte Carlo (Zoptymalizowany pod pamięć)
        cov_matrix = daily_rets.cov().values
        port_mean = np.sum(daily_rets.mean() * final_weights)
        port_std = np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights)))
        
        col_5, col_10 = st.columns(2)
        plt.style.use("dark_background")

        for i, (y, label) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
            days = y * 252
            # Generujemy ścieżki (Dni x Symulacje)
            sim_rets = np.random.normal(port_mean, port_std, (days, n_sims))
            sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
            
            # Statystyki końcowe
            final_v = sim_paths[-1, :]
            mediana = np.median(final_v)
            p95, p5 = np.percentile(final_v, 95), np.percentile(final_v, 5)
            cagr = (mediana / kwota)**(1/y) - 1
            szansa_straty = (np.sum(final_v < kwota) / n_sims) * 100

            with (col_5 if i == 0 else col_10):
                st.write(f"#### Prognoza na {label}")
                st.table(pd.DataFrame({
                    "Metryka": ["Mediana (Bazowy)", "Scenariusz Optymistyczny", "Scenariusz Pesymistyczny", "Zwrot Średnioroczny (CAGR)", "Szansa na stratę"],
                    "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{szansa_straty:.1f}%"]
                }))
                
                # Wykres "Wachlarz"
                fig, ax = plt.subplots(figsize=(10, 6))
                # Próbka 100 linii do rysowania
                ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.05, linewidth=0.7)
                ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2.5, label='Mediana')
                
                # Dynamiczne skalowanie osi Y
                ax.set_ylim(np.percentile(final_v, 1)*0.7, np.percentile(final_v, 99)*1.3)
                ax.set_title(f"Rozpiętość scenariuszy ({label})")
                ax.grid(True, alpha=0.15, linestyle='--')
                st.pyplot(fig)

        st.warning("⚠️ **Nota prawna:** Analiza Monte Carlo oparta jest na zmienności historycznej. Giełda nie jest powtarzalna – wyniki symulacji mają charakter wyłącznie edukacyjny i nie stanowią gwarancji zysku.")

    with tab3:
        st.subheader("Mapa Korelacji Historycznych")
        fig_c, ax_c = plt.subplots(figsize=(12, 7))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
        st.pyplot(fig_c)

else:
    st.info("👈 Skonfiguruj parametry i kliknij przycisk, aby uruchomić model.")
    st.image("https://images.unsplash.com/photo-1611974717537-488439d4371f?auto=format&fit=crop&q=80&w=1200", use_container_width=True)
