import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# --- Konfiguracja strony ---
st.set_page_config(page_title="Automatic Risk Manager Pro", layout="wide")

st.title("🛡️ Automatic Risk Manager Pro")
st.markdown("Zoptymalizuj swój portfel na podstawie matematycznych modeli ryzyka (VaR) i korelacji.")

# --- SIDEBAR (Panel Janka) ---
st.sidebar.header("Ustawienia Portfela")

# Definicja 15 spółek na podstawie Twojego screena (Domyślne)
default_tickers = "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD"
tickers_input = st.sidebar.text_input("Spółki (oddzielone przecinkiem):", default_tickers)

# Kwota inwestycji na podstawie screena (Domyślne)
kwota = st.sidebar.number_input("Kwota inwestycji (Waluta):", value=25000)

# Suwaki kontrolujące algorytm
ryzyko = st.sidebar.select_slider("Poziom Ryzyka:", options=['low', 'medium', 'high'], value='medium')
limit_2x = st.sidebar.checkbox("Zastosuj limit 2x (Dywersyfikacja)", value=True, help="Największa pozycja będzie max 2x większa od najmniejszej.")
run_mc = st.sidebar.checkbox("Uruchom Projekcje Długoterminowe", value=True)

if st.sidebar.button("Analizuj i Optymalizuj"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Przetwarzanie danych rynkowych...'):
        # Pobieramy dane wejściowe (DANE DZIENNE na 3 LATA)
        raw_data = yf.download(tickers, period="3y")['Close']
        
        # Obliczamy dzienne stopy zwrotu
        daily_returns = raw_data.pct_change().dropna()
        
        # Konwertujemy na dane miesięczne (do VaR i optymalizacji)
        monthly_returns = raw_data.resample('ME').last().pct_change().dropna()
        
        # OBLICZANIE VAAR I KORELACJI (Faza 1: Baza)
        # monthly_vars to Series, gdzie indeksami są nazwy spółek
        monthly_vars = monthly_returns.quantile(0.05) * -1
        
        # Liczymy macierz korelacji
        corr_matrix = monthly_returns.corr()
        
        # Liczymy średnią korelację każdej spółki z pozostałymi
        avg_corr_each = corr_matrix.mean()

    # --- LOGIKA OBLICZANIA WAG (Twoja formuła) ---
    # Logika poziomów ryzyka: Zmieniamy potęgę kary za VaR
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    
    # Krok A: Waga bazowa: Odwrotność VaR (Im większy VaR, tym mniejsza waga)
    # Krok B: Korekta korelacji: Mnożymy przez (1 - średnia korelacja)
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
    
    # Normalizacja do 100%
    target_weights = target_weights_raw / target_weights_raw.sum()

    # --- OPTYMALIZACJA MATEMATYCZNA (uwzględnia limit 2x) ---
    def objective(weights): 
        # Minimalizujemy sumę kwadratów różnic między wagami a celem
        return np.sum((weights - target_weights.values)**2)

    # Ograniczenia podstawowe
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # DODATKOWE OGRANICZENIE JANKA: Max Pozycja <= 2x Min Pozycja
    if limit_2x:
        # Constraint: 2*min - max >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    # Uruchamiamy optymalizator SLSQP (z minimu 1% na spółkę)
    res = minimize(objective, target_weights.values, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in tickers), constraints=cons)
    final_weights = res.x

    # --- SEKCJA 1: WYNIKI ALOKACJI ---
    st.header("📊 Twoja Zoptymalizowana Alokacja")
    col1, col2 = st.columns([2, 1])

    with col1:
        # Tabela wyników (Sortowana od największej pozycji)
        wynik_df = pd.DataFrame({
            'Spółka': monthly_vars.index,  # Bierzemy nazwy z wyliczeń
            'Monthly VaR (5%)': [f"{v*100:.2f}%" for v in monthly_vars],
            'Śr. Korelacja': [f"{c:.2f}" for c in avg_corr_each],
            'Procent Portfela': final_weights * 100,
            'Kwota (Waluta)': final_weights * kwota
        }).sort_values(by='Procent Portfela', ascending=False)
        
        st.dataframe(wynik_df.style.format({'Procent Portfela': '{:.2f}%', 'Kwota (Waluta)': '{:,.2f}'}), 
                     hide_index=True, use_container_width=True)

    with col2:
        # Sekcja statystyk ogólnych
        st.subheader("📉 Statystyki Ryzyka")
        
        # Monthly VaR Portfela (Ważona suma VaR-ów spółek)
        portfel_var = (final_weights * monthly_vars).sum()
        
        # Obliczanie średniej korelacji całego portfela (unikalne pary)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr_val = corr_matrix.where(mask).stack().mean()

        st.metric("Monthly Value At Risk (95%)", f"{portfel_var*100:.2f}%")
        st.metric("Średnia Korelacja Portfela", f"{mean_corr_val:.2f}")
        st.write(f"Wartość narażona na ryzyko miesięczne: **{portfel_var * kwota:,.2f}**")

    # --- SEKCJA 2: PROJEKCJE DŁUGOTERMINOWE ---
    if run_mc:
        st.divider()
        st.header("🔮 Projekcje Długoterminowe wartości Twojego kapitału")
        
        # Obliczamy macierz Cholesky'ego z danych dziennych (Klucz do symulacji)
        cov_matrix = daily_returns.cov()
        L = np.linalg.cholesky(cov_matrix)
        
        n_sims = 1000 # Liczba symulacji
        years = [5, 10]
        results = {}

        # Silnik symulacji (Geometria Browna z zachowaniem korelacji)
        mean_daily = daily_returns.mean().values
        
        def run_sim_paths(n_years):
            days = n_years * 252
            Z = np.random.normal(size=(days, n_sims, len(tickers)))
            correlated_shocks = np.einsum('jk,ikl->ijl', L, Z)
            daily_sim_rets = mean_daily + correlated_shocks
            port_daily_rets = np.dot(daily_sim_rets, final_weights)
            # Zwracamy pełne ścieżki cenowe
            return kwota * np.cumprod(1 + port_daily_rets, axis=0)

        # 1. Obliczenia Side-by-Side (5 i 10 lat)
        c_5y, c_10y = st.columns(2)
        
        for i, y in enumerate(years):
            final_vals = run_sim_paths(y)[-1, :] # Bierzemy tylko ostatni dzień
            
            # Obliczenia statystyk (Percentyle i Mediana)
            mediana = np.median(final_vals)
            p95 = np.percentile(final_vals, 95)
            p5 = np.percentile(final_vals, 5)
            chance_loss = (np.sum(final_vals < kwota) / n_sims) * 100
            
            # Obliczenia CAGR (Średnioroczna stopa zwrotu) liczone z MEDIANY
            cagr = (mediana / kwota)**(1/y) - 1
            
            # Tabela statystyk
            data_stats = {
                "Metryka": ["Mediana (Scenariusz bazowy)", "95. Percentyl (Optymistyczny)", "5. Percentyl (Pesymistyczny)", "Średnioroczna stopa zwrotu ( CAGR)", "Szansa na stratę kapitału"],
                "Wartość": [f"{mediana:,.2f}", f"{p95:,.2f}", f"{p5:,.2f}", f"{cagr*100:.2f}%", f"{chance_loss:.1f}%"]
            }
            
            with (c_5y if i == 0 else c_10y):
                st.subheader(f"📅 Prognoza na {y} lat")
                st.table(pd.DataFrame(data_stats))

        # 2. WYKRES WACHLARZA SCENARIUSZY (Side-by-Side)
        st.subheader("💡 Symulacja przebiegu wartości portfela w czasie")
        st.markdown("Poniższe wykresy prezentują 100 losowych scenariuszy przebiegu wartości Twoich pieniędzy w czasie. Czarna, grubsza linia reprezentuje medianę (scenariusz bazowy). Im szerszy 'wachlarz' linii, tym większa niepewność na rynku.")
        
        c_p1, c_p2 = st.columns(2)
        plt.style.use("dark_background") # Ciemny motyw jak w referencji
        
        for i, (y, data_paths) in enumerate(zip(years, [run_sim_paths(5), run_sim_paths(10)])):
            fig_f, ax_f = plt.subplots(figsize=(10, 5))
            
            # Bierzemy próbkę 100 symulacji, żeby nie zamulić przeglądarki
            n_plot_sims = 100
            sampled_paths = data_paths[:, np.random.choice(n_sims, n_plot_sims, replace=False)]
            
            # Rysujemy wachlarz ( cienkie, jasne linie)
            time = np.arange(len(data_paths))
            ax_f.plot(time, sampled_paths, color='skyblue', alpha=0.03, linewidth=0.5)
            
            # Rysujemy medianę (gruba czarna linia - na ciemnym tle)
            median_path = np.median(data_paths, axis=1)
            ax_f.plot(time, median_path, color='white', linewidth=2, label='Mediana (Scenariusz bazowy)')
            
            # Ograniczamy widok, żeby nie ucinało percentyli (Percentyle 5 i 95 ostatniego dnia)
            last_day = data_paths[-1, :]
            ax_f.set_ylim(np.percentile(last_day, 1) * 0.9, np.percentile(last_day, 99) * 1.1)
            
            ax_f.set_title(f"Rozpiętość scenariuszy giełdowych ({y} lat)")
            ax_f.set_ylabel("Wartość portfela (Waluta)")
            ax_f.set_xlabel("Dni giełdowe")
            ax_f.grid(True, linestyle='--', alpha=0.3)
            ax_f.legend(loc='upper left', fontsize='small')
            
            with (c_p1 if i == 0 else c_p2):
                st.pyplot(fig_f)
                
        # Powrót do domyślnego stylu, żeby nie psuć reszty strony
        plt.style.use('default') 

    # --- SEKCJA 3: KORELACJE NA DOLE ---
    st.divider()
    st.subheader("🔗 Mapa Korelacji Twoich Aktywów")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 8})
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Zastrzeżenie Prawne ---
    st.caption("Ważne zastrzeżenie: Powyższe symulacje są analizą typu Monte Carlo opartą wyłącznie na historycznych danych giełdowych. Historyczne wyniki rynkowe nie są gwarancją osiągnięcia podobnych rezultatów w przyszłości. Rynek finansowy cechuje się nieprzewidywalnością i ryzykiem straty zainwestowanego kapitału. Nie należy traktować tych prognoz jako porady inwestycyjnej.")
