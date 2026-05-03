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
   # --- SYMULACJA MONTE CARLO (Horyzont 5 i 10 lat) ---
    if run_mc:
        st.divider()
        st.header("🔮 Projekcje Długoterminowe (Monte Carlo)")

        # 1. WIZUALIZACJA CHOLESKY'EGO
        st.subheader("🧬 Macierz Cholesky'ego (Struktura powiązań)")
        st.info("Poniższa macierz pokazuje, jak algorytm 'rozumie' wspólne ryzyko Twoich spółek. To te wartości sterują losem symulacji.")
        
        cov_matrix = daily_returns.cov()
        # Obliczamy macierz L (dolnotrójkątną)
        L = np.linalg.cholesky(cov_matrix)
        
        fig_chol, ax_chol = plt.subplots(figsize=(10, 4))
        sns.heatmap(L, xticklabels=tickers, yticklabels=tickers, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax_chol)
        st.pyplot(fig_chol)

        # 2. SILNIK SYMULACJI
        n_sims = 1000
        years = [5, 10]
        stats_list = []

        mean_daily = daily_returns.mean().values
        
        # Funkcja do przeprowadzania symulacji dla danego horyzontu
        def run_simulation(n_years):
            days = n_years * 252
            # Generujemy wszystkie losowe zwroty na raz dla szybkości
            Z = np.random.normal(size=(days, n_sims, len(tickers)))
            # Aplikujemy korelacje (macierz L) do szumu losowego
            correlated_shocks = np.einsum('jk,ikl->ijl', L, Z)
            # Dodajemy średni dryf (returns)
            daily_sim_rets = mean_daily + correlated_shocks
            # Ważymy zwroty wagami portfela
            port_daily_rets = np.dot(daily_sim_rets, final_weights)
            # Obliczamy końcową wartość (skumulowany iloczyn)
            final_vals = kwota * np.prod(1 + port_daily_rets, axis=0)
            return final_vals

        # Obliczenia dla 5 i 10 lat
        results = {y: run_simulation(y) for y in years}

        # 3. PREZENTACJA TABEL (Side by Side)
        col_5y, col_10y = st.columns(2)

        for i, y in enumerate(years):
            final_vals = results[y]
            
            # Statystyki
            mediana = np.median(final_vals)
            p95 = np.percentile(final_vals, 95)
            p5 = np.percentile(final_vals, 5)
            chance_loss = (np.sum(final_vals < kwota) / n_sims) * 100
            
            # Tabela
            data_stats = {
                "Metryka": ["Mediana (Scenariusz bazowy)", "95. Percentyl (Optymistyczny)", "5. Percentyl (Pesymistyczny)", "Szansa na stratę kapitału"],
                "Wartość": [f"{mediana:,.2f} PLN", f"{p95:,.2f} PLN", f"{p5:,.2f} PLN", f"{chance_loss:.1f}%"]
            }
            
            with (col_5y if i == 0 else col_10y):
                st.subheader(f"📅 Po {y} latach")
                st.table(pd.DataFrame(data_stats))
                
                # Mały wykres rozkładu pod tabelką
                fig_h, ax_h = plt.subplots(figsize=(6, 3))
                sns.histplot(final_vals, bins=30, kde=True, color=("orange" if i==0 else "green"), ax=ax_h)
                ax_h.axvline(kwota, color='red', linestyle='--')
                ax_h.set_title(f"Rozkład wyników po {y} latach")
                st.pyplot(fig_h)

        st.caption("Uwaga: Symulacja zakłada, że historyczna średnia i korelacja utrzymają się w przyszłości.")
       
