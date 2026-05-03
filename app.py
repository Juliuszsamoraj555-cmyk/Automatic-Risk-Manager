import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Automatic Risk Manager Pro", layout="wide")

st.title("🛡️ Automatic Risk Manager Pro")
st.markdown("Optymalizacja VaR + Symulacja Monte Carlo")

# --- SIDEBAR ---
st.sidebar.header("Ustawienia")
tickers_input = st.sidebar.text_input("Spółki:", "AAPL, MSFT, TSLA, NVDA, WMT, PG, JNJ")
kwota = st.sidebar.number_input("Kwota inwestycji:", value=25000)
ryzyko = st.sidebar.select_slider("Poziom Ryzyka:", options=['low', 'medium', 'high'], value='medium')
limit_2x = st.sidebar.checkbox("Zastosuj limit 2x", value=True)
run_mc = st.sidebar.checkbox("Uruchom Symulację Monte Carlo", value=True)

if st.sidebar.button("Analizuj Portfel"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Przetwarzanie danych...'):
        # Pobieramy dane (potrzebujemy dziennych stóp zwrotu do Monte Carlo)
        raw_data = yf.download(tickers, period="3y")['Close']
        daily_returns = raw_data.pct_change().dropna()
        
        # Dane miesięczne do VaR (tak jak wcześniej)
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        monthly_vars = monthly_returns.quantile(0.05) * -1
        corr_matrix = monthly_returns.corr()
        avg_corr_each = corr_matrix.mean()

    # --- OPTYMALIZACJA ---
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
    target_weights = target_weights_raw / target_weights_raw.sum()

    def objective(weights): return np.sum((weights - target_weights)**2)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if limit_2x:
        cons.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    res = minimize(objective, target_weights, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in tickers), constraints=cons)
    final_weights = res.x

    # --- WYNIKI GŁÓWNE ---
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 Optymalna Alokacja")
        df_results = pd.DataFrame({'Spółka': tickers, 'Procent': final_weights*100, 'Kwota': final_weights*kwota}).sort_values(by='Procent', ascending=False)
        st.dataframe(df_results.style.format({'Procent': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

    with c2:
        st.subheader("📉 Ryzyko")
        port_var = (final_weights * monthly_vars).sum()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        st.metric("Monthly VaR (95%)", f"{port_var*100:.2f}%")
        st.metric("Średnia Korelacja", f"{corr_matrix.where(mask).stack().mean():.2f}")

    # --- SYMULACJA MONTE CARLO ---
    if run_mc:
        st.divider()
        st.subheader("🔮 Symulacja Monte Carlo (Prognoza na 252 dni)")
        
        # Parametry symulacji
        n_sims = 500
        n_days = 252
        
        # Średnie i kowariancja z danych dziennych
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        
        # Generowanie losowych ścieżek (Geometryczne Ruchy Browna)
        # Tworzymy macierz losową uwzględniającą korelacje (Cholesky)
        L = np.linalg.cholesky(cov_matrix)
        
        sim_results = np.zeros((n_days, n_sims))
        
        for i in range(n_sims):
            Z = np.random.normal(size=(n_days, len(tickers)))
            daily_sim_returns = mean_returns.values + np.dot(Z, L.T)
            # Portfel to suma ważona zwrotów spółek
            portfolio_sim_returns = np.dot(daily_sim_returns, final_weights)
            # Skumulowany zwrot
            sim_results[:, i] = kwota * np.cumprod(1 + portfolio_sim_returns)

        # Wykresy Monte Carlo
        mc_col1, mc_col2 = st.columns(2)
        
        with mc_col1:
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            ax_mc.plot(sim_results, color='skyblue', alpha=0.1)
            ax_mc.plot(np.mean(sim_results, axis=1), color='red', label='Średnia ścieżka')
            ax_mc.set_title("500 możliwych przyszłości portfela")
            ax_mc.set_ylabel("Wartość portfela (PLN)")
            st.pyplot(fig_mc)
            
        with mc_col2:
            final_values = sim_results[-1, :]
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            sns.histplot(final_values, kde=True, ax=ax_hist, color='navy')
            ax_hist.axvline(kwota, color='red', linestyle='--', label='Kwota startowa')
            ax_hist.set_title("Rozkład wartości portfela po roku")
            st.pyplot(fig_hist)
            
        # Statystyki z symulacji
        prob_profit = np.sum(final_values > kwota) / n_sims * 100
        st.write(f"Prawdopodobieństwo zysku po roku: **{prob_profit:.1f}%**")
        st.write(f"Średnia przewidywana wartość: **{np.mean(final_values):,.2f} PLN**")

    # Mapa korelacji na dole
    st.divider()
    st.subheader("🔗 Mapa Korelacji")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
    st.pyplot(fig_corr)
