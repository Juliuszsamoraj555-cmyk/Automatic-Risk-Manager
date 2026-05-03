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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Lifting Graficzny) ---
st.markdown("""
    <style>
    /* Import czcionki Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Stylizacja kart metryk */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Stylizacja bocznego paska */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Przycisk Analizy */
    .stButton > button {
        width: 100%;
        background-color: #238636 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        height: 3em !important;
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.4);
    }

    /* Tabele */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    
    /* Nagłówki */
    h1, h2, h3 {
        color: #e6edf3;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIKA I SIDEBAR ---
st.title("🛡️ Automatic Risk Manager Pro")
st.caption("Inteligentna optymalizacja portfela oparta na algorytmach Value at Risk")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534360.png", width=80) # Opcjonalna ikonka
    st.header("Konfiguracja")
    
    default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
    tickers_input = st.text_input("Twoje spółki (ticker):", default_tickers)
    kwota = st.number_input("Kwota do zainwestowania:", value=25000, min_value=100)
    
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    run_mc = st.checkbox("Symulacje długoterminowe", value=True)
    
    analizuj = st.button("URUCHOM ANALIZĘ")

# --- 4. GŁÓWNA LOGIKA APLIKACJI ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('⏳ Agregowanie danych rynkowych...'):
        # Pobieranie danych
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_returns = raw_data.pct_change().dropna()
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        
        # Statystyki ryzyka
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # --- OPTYMALIZACJA (Algorytm Risk-First) ---
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
    target_weights = target_weights_raw / target_weights_raw.sum()

    def objective(weights): 
        return np.sum((weights - target_weights.values)**2)

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if limit_2x:
        cons.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    res = minimize(objective, target_weights.values, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in tickers), constraints=cons)
    final_weights = res.x

    # --- 5. INTERFEJS WYNIKÓW (ZAKŁADKI) ---
    tab1, tab2, tab3 = st.tabs(["📊 Alokacja i Ryzyko", "🔮 Projekcje 5/10 Lat", "🔗 Analiza Powiązań"])

    with tab1:
        st.subheader("Rekomendowany podział kapitału")
        
        # Metryki w kolumnach
        m1, m2, m3 = st.columns(3)
        portfel_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.where(mask).stack().mean()
        
        m1.metric("Monthly VaR (95%)", f"{portfel_var*100:.2f}%")
        m2.metric("Średnia Korelacja", f"{mean_corr:.2f}")
        m3.metric("Ryzyko w Walucie", f"{portfel_var * kwota:,.2f}")

        st.divider()
        
        # Tabela alokacji
        wynik_df = pd.DataFrame({
            'Ticker': monthly_vars.index,
            'VaR Spółki': [f"{v*100:.2f}%" for v in monthly_vars],
            'Udział w Portfelu': final_weights * 100,
            'Kwota Inwestycji': final_weights * kwota
        }).sort_values(by='Udział w Portfelu', ascending=False)
        
        st.dataframe(wynik_df.style.format({'Udział w Portfelu': '{:.2f}%', 'Kwota Inwestycji': '{:,.2f}'}), 
                     hide_index=True, use_container_width=True)

    with tab2:
        if run_mc:
            st.subheader("Projekcje Monte Carlo")
            st.info("Poniższe wykresy prezentują 100 losowych scenariuszy przebiegu wartości Twojego kapitału.")
            
            # Silnik Monte Carlo
            cov_matrix = daily_returns.cov()
            L = np.linalg.cholesky(cov_matrix)
            n_sims = 1000
            mean_daily = daily_returns.mean().values
            
            def run_sim(years):
                days = years * 252
                Z = np.random.normal(size=(days, n_sims, len(tickers)))
                shocks = Z @ L.T 
                daily_sim_rets = mean_daily + shocks
                port_rets = daily_sim_rets @ final_weights
                return kwota * np.cumprod(1 + port_rets, axis=0)

            res_5 = run_sim(5)
            res_10 = run_sim(10)
            
            # Tabele i Wykresy (Wachlarze)
            c5, c10 = st.columns(2)
            plt.style.use("dark_background")
            
            for i, (data, label, years) in enumerate(zip([res_5, res_10], ["5 Lat", "10 Lat"], [5, 10])):
                final_vals = data[-1, :]
                mediana = np.median(final_vals)
                p95, p5 = np.percentile(final_vals, 95), np.percentile(final_vals, 5)
                cagr = (mediana / kwota)**(1/years) - 1
                
                with (c5 if i == 0 else c10):
                    st.write(f"### Horyzont {label}")
                    st.table(pd.DataFrame({
                        "Metryka": ["Mediana", "Scenariusz Optymistyczny", "Scenariusz Pesymistyczny", "Średni zwrot (CAGR)"],
                        "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%"]
                    }))
                    
                    # Wykres wachlarzowy
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sampled = data[:, np.random.choice(n_sims, 100, replace=False)]
                    ax.plot(sampled, color='skyblue', alpha=0.07, linewidth=0.8)
                    ax.plot(np.median(data, axis=1), color='white', linewidth=2.5, label='Mediana')
                    ax.set_ylim(np.percentile(data[-1,:], 1)*0.8, np.percentile(data[-1,:], 99)*1.2)
                    ax.set_title(f"Wachlarz scenariuszy: {label}")
                    ax.grid(True, alpha=0.15)
                    st.pyplot(fig)
            
            plt.style.use('default')
            st.warning("⚠️ Powyższa analiza oparta jest na danych historycznych. Giełda jest nieprzewidywalna – wyniki symulacji nie gwarantują rzeczywistych zysków.")

    with tab3:
        st.subheader("Mapa Korelacji Aktywów")
        st.write("Wizualizacja powiązań między spółkami. Im czerwieńszy kolor, tym większe prawdopodobieństwo, że spółki spadną w tym samym czasie.")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

else:
    # Ekran startowy (zanim Janek kliknie przycisk)
    st.info("👈 Skonfiguruj swój portfel w panelu bocznym i kliknij 'Analizuj i Optymalizuj', aby zobaczyć magię.")
    st.image("https://images.unsplash.com/photo-1611974717537-488439d4371f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_container_width=True)
