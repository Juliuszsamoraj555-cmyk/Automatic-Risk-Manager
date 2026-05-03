import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# Konfiguracja strony
st.set_page_config(page_title="Automatic Risk Manager Pro", layout="wide")

st.title("🛡️ Automatic Risk Manager Pro")
st.markdown("Profesjonalne zarządzanie ryzykiem: Optymalizacja VaR + Symulacje Długoterminowe")

# --- SIDEBAR: PANEL UŻYTKOWNIKA ---
st.sidebar.header("⚙️ Ustawienia Portfela")
tickers_input = st.sidebar.text_input("Wpisz spółki (oddzielone przecinkiem):", "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD")
kwota = st.sidebar.number_input("Kwota inwestycji (PLN/USD):", value=25000, step=1000)
ryzyko = st.sidebar.select_slider("Poziom Ryzyka:", options=['low', 'medium', 'high'], value='medium')

st.sidebar.subheader("Opcje Zaawansowane")
limit_2x = st.sidebar.checkbox("Zastosuj limit 2x (Dywersyfikacja)", value=True)
run_mc = st.sidebar.checkbox("Uruchom Symulację Monte Carlo", value=True)

if st.sidebar.button("🚀 Analizuj i Optymalizuj"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Pobieranie i przetwarzanie danych rynkowych...'):
        # Pobieranie danych (3 lata)
        raw_data = yf.download(tickers, period="3y")['Close']
        
        # Dane dzienne (do Monte Carlo i Cholesky'ego)
        daily_returns = raw_data.pct_change().dropna()
        
        # Dane miesięczne (do VaR i optymalizacji)
        monthly_data = raw_data.resample('ME').last()
        monthly_returns = monthly_data.pct_change().dropna()
        
        # Obliczenia bazowe (monthly_vars zachowuje tickery jako index)
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # --- SILNIK OPTYMALIZACJI ---
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    
    # Wyliczamy wagi idealne (target) na podstawie Twojej formuły
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
    target_weights = target_weights_raw / target_weights_raw.sum()

    # Optymalizacja matematyczna
    def objective(weights): 
        return np.sum((weights - target_weights)**2)

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if limit_2x:
        constraints.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    bounds = tuple((0.01, 1.0) for _ in range(len(tickers)))
    res = minimize(objective, target_weights.values, method='SLSQP', bounds=bounds, constraints=constraints)
    final_weights = res.x

    # --- SEKCJA 1: WYNIKI ALOKACJI ---
    col_results, col_stats = st.columns([2, 1])

    with col_results:
        st.subheader("📊 Optymalna Alokacja Portfela")
        wynik_df = pd.DataFrame({
            'Spółka': monthly_vars.index,
            'Procent (%)': final_weights * 100,
            'Kwota (Waluta)': final_weights * kwota
        }).sort_values(by='Procent (%)', ascending=False)
        
        st.dataframe(wynik_df.style.format({'Procent (%)': '{:.2f}%', 'Kwota (Waluta)': '{:,.2f}'}), 
                     hide_index=True, use_container_width=True)

    with col_stats:
        st.subheader("📉 Statystyki Ryzyka")
        portfel_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr_val = corr_matrix.where(mask).stack().mean()

        st.metric("Monthly Value At Risk (95%)", f"{portfel_var*100:.2f}%")
        st.metric("Średnia Korelacja Portfela", f"{mean_corr_val:.2f}")
        st.write(f"Wartość narażona na ryzyko miesięczne: **{portfel_var * kwota:,.2f}**")

    # --- SEKCJA 2: MONTE CARLO ---
    if run_mc:
        st.divider()
        st.header("🔮 Projekcje Długoterminowe (Monte Carlo)")
        
        st.subheader("🧬 Macierz Cholesky'ego")
        st.info("Algorytm dekompozycji Cholesky'ego pozwala symulować przyszłość z zachowaniem historycznych korelacji między spółkami.")
        
        cov_matrix_daily = daily_returns.cov()
        L = np.linalg.cholesky(cov_matrix_daily)
        
        fig_chol, ax_chol = plt.subplots(figsize=(10, 4))
        sns.heatmap(L, xticklabels=monthly_vars.index, yticklabels=monthly_vars.index, 
                    annot=True, fmt=".3f", cmap="YlGnBu", ax=ax_chol, annot_kws={"size": 7})
        st.pyplot(fig_chol)

        # Obliczenia symulacji
        n_sims = 1000
        mean_daily = daily_returns.mean().values
        
        def run_sim(years):
            days = years * 252
            # Szum losowy: (dni, symulacje, spółki)
            Z = np.random.normal(size=(days, n_sims, len(tickers)))
            # Poprawione mnożenie macierzy szumu przez korelacje
            correlated_shocks = Z @ L.T 
            daily_sim_rets = mean_daily + correlated_shocks
            port_daily_rets = np.dot(daily_sim_rets, final_weights)
            return kwota * np.prod(1 + port_daily_rets, axis=0)

        results_5y = run_sim(5)
        results_10y = run_sim(10)

        col_5y, col_10y = st.columns(2)

        for i, (y_data, y_label) in enumerate(zip([results_5y, results_10y], ["5 lat", "10 lat"])):
            mediana = np.median(y_data)
            p95 = np.percentile(y_data, 95)
            p5 = np.percentile(y_data, 5)
            chance_loss = (np.sum(y_data < kwota) / n_sims) * 100
            
            stats_df = pd.DataFrame({
                "Metryka": ["Mediana (Scenariusz bazowy)", "95. Percentyl (Optymistyczny)", "5. Percentyl (Pesymistyczny)", "Szansa na stratę kapitału"],
                "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{chance_loss:.1f}%"]
            })
            
            with (col_5y if i == 0 else col_10y):
                st.subheader(f"📅 Prognoza na {y_label}")
                st.table(stats_df)
                
                fig_h, ax_h = plt.subplots(figsize=(6, 3))
                sns.histplot(y_data, bins=30, kde=True, color=("orange" if i==0 else "green"), ax=ax_h)
                ax_h.axvline(kwota, color='red', linestyle='--', label="Start")
                ax_h.set_title(f"Rozkład kapitału po {y_label}")
                st.pyplot(fig_h)

    # --- SEKCJA 3: KORELACJE ---
    st.divider()
    st.subheader("🔗 Mapa Korelacji Aktywów")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
    plt.xticks(rotation=45)
    st.pyplot(fig_corr)

    st.caption("Analiza oparta na danych historycznych z 36 miesięcy. Symulacje nie stanowią gwarancji przyszłych wyników.")
