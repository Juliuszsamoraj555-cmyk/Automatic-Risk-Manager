import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
# --- CUSTOM CSS (Lifting Graficzny) ---
st.markdown("""
    <style>
    /* Główny font i tło */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Stylizacja bocznego paska */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #30363d;
    }

    /* Stylizacja kart ze statystykami */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Przyciski */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #238636;
        color: white;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #2ea043;
        border: none;
    }
    </style>
    """, unsafe_allow_index=True)

# --- Konfiguracja strony ---
st.set_page_config(page_title="Automatic Risk Manager Pro", layout="wide")

st.title("🛡️ Automatic Risk Manager Pro")
st.markdown("Profesjonalna optymalizacja portfela i projekcje długoterminowe.")

# --- SIDEBAR ---
st.sidebar.header("Ustawienia Portfela")
default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
tickers_input = st.sidebar.text_input("Spółki (oddzielone przecinkiem):", default_tickers)
kwota = st.sidebar.number_input("Kwota inwestycji:", value=25000)
ryzyko = st.sidebar.select_slider("Poziom Ryzyka:", options=['low', 'medium', 'high'], value='medium')
limit_2x = st.sidebar.checkbox("Zastosuj limit 2x (Dywersyfikacja)", value=True)
run_mc = st.sidebar.checkbox("Uruchom Projekcje Długoterminowe", value=True)

if st.sidebar.button("Analizuj i Optymalizuj"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Pobieranie danych...'):
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_returns = raw_data.pct_change().dropna()
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        
        # Obliczenia ryzyka i korelacji
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # --- OPTYMALIZACJA WAG ---
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

    # --- SEKCJA 1: WYNIKI ALOKACJI ---
    st.header("📊 Twoja Zoptymalizowana Alokacja")
    col1, col2 = st.columns([2, 1])

    with col1:
        wynik_df = pd.DataFrame({
            'Spółka': monthly_vars.index,
            'Monthly VaR (5%)': [f"{v*100:.2f}%" for v in monthly_vars],
            'Procent Portfela': final_weights * 100,
            'Kwota (Waluta)': final_weights * kwota
        }).sort_values(by='Procent Portfela', ascending=False)
        st.dataframe(wynik_df.style.format({'Procent Portfela': '{:.2f}%', 'Kwota (Waluta)': '{:,.2f}'}), 
                     hide_index=True, use_container_width=True)

    with col2:
        st.subheader("📉 Statystyki Ryzyka")
        portfel_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr_val = corr_matrix.where(mask).stack().mean()
        st.metric("Monthly Value At Risk (95%)", f"{portfel_var*100:.2f}%")
        st.metric("Średnia Korelacja Portfela", f"{mean_corr_val:.2f}")

    # --- SEKCJA 2: MONTE CARLO ---
    if run_mc:
        st.divider()
        st.header("🔮 Projekcje Długoterminowe (Monte Carlo)")
        
        # Macierz Cholesky'ego do korelacji szumu
        cov_matrix = daily_returns.cov()
        L = np.linalg.cholesky(cov_matrix)
        
        n_sims = 1000
        years = [5, 10]
        mean_daily = daily_returns.mean().values
        
        # POPRAWIONA FUNKCJA SYMULACJI (bez einsum)
        def run_sim_paths(n_years):
            days = n_years * 252
            Z = np.random.normal(size=(days, n_sims, len(tickers)))
            # Zastosowanie korelacji: (days, sims, tickers) @ (tickers, tickers)
            correlated_shocks = Z @ L.T 
            daily_sim_rets = mean_daily + correlated_shocks
            # Ważenie zwrotów: (days, sims, tickers) dot (tickers,) -> (days, sims)
            port_daily_rets = daily_sim_rets @ final_weights
            return kwota * np.cumprod(1 + port_daily_rets, axis=0)

        # Tabele statystyk
        c_5y, c_10y = st.columns(2)
        sim_data = {}

        for i, y in enumerate(years):
            paths = run_sim_paths(y)
            sim_data[y] = paths
            final_vals = paths[-1, :]
            
            mediana = np.median(final_vals)
            p95 = np.percentile(final_vals, 95)
            p5 = np.percentile(final_vals, 5)
            cagr = (mediana / kwota)**(1/y) - 1
            chance_loss = (np.sum(final_vals < kwota) / n_sims) * 100
            
            stats_df = pd.DataFrame({
                "Metryka": ["Mediana (Scenariusz bazowy)", "95. Percentyl (Optymistyczny)", "5. Percentyl (Pesymistyczny)", "Średnioroczny zwrot (CAGR)", "Szansa na stratę"],
                "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{chance_loss:.1f}%"]
            })
            
            with (c_5y if i == 0 else c_10y):
                st.subheader(f"📅 Prognoza na {y} lat")
                st.table(stats_df)

        # Wykresy wachlarzowe (zgodnie z Twoim wzorem)
        st.subheader("💡 Symulacja przebiegu wartości portfela")
        plt.style.use("dark_background")
        cp1, cp2 = st.columns(2)
        
        for i, y in enumerate(years):
            fig, ax = plt.subplots(figsize=(10, 5))
            paths = sim_data[y]
            # Próbka 100 linii dla czytelności
            sampled = paths[:, np.random.choice(n_sims, 100, replace=False)]
            ax.plot(sampled, color='skyblue', alpha=0.05, linewidth=0.5)
            ax.plot(np.median(paths, axis=1), color='white', linewidth=2, label='Mediana')
            
            # Skalowanie osi Y, żeby nie ucinało percentyli
            ax.set_ylim(np.percentile(paths[-1,:], 1) * 0.8, np.percentile(paths[-1,:], 99) * 1.2)
            ax.set_title(f"Horyzont {y} lat")
            ax.grid(True, alpha=0.2)
            
            with (cp1 if i == 0 else cp2):
                st.pyplot(fig)
        
        plt.style.use('default')

    # --- MAPA KORELACJI ---
    st.divider()
    st.subheader("🔗 Mapa Korelacji Aktywów")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    st.caption("Powyższa analiza jest symulacją Monte Carlo opartą na danych historycznych. Nie gwarantuje ona zysków w przyszłości.")
