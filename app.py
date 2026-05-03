import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# 1. KONFIGURACJA STRONY (Musi być pierwsza!)
# Wczytujemy ikonę z pliku image_8.png
v_alpha_icon = Image.open('image_8.png')

st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon=v_alpha_icon, layout="wide")

# 2. DESIGN CSS (SaaS Look - Minimalistyczny, bez emoji)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button {
        width: 100%; background-color: #238636 !important; color: white !important;
        border-radius: 8px; font-weight: 700; height: 3.5em; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR (Z nowym logotypem zamiast emoji)
# Umieszczamy logotyp nad tytułem
st.sidebar.image(v_alpha_icon, width=100)
st.title("Risk Manager Pro")

with st.sidebar:
    # Usunięto emoji z nagłówka
    st.header("Ustawienia")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD",
        help="""
        **Jak wpisywać symbole?**
        System pobiera dane bezpośrednio z serwerów Yahoo Finance. 
        * **USA:** Należy wpisać sam ticker, na przykład AAPL, TSLA lub MSFT.
        * **Polska (GPW):** Należy dodać rozszerzenie .WA, na przykład ALE.WA lub PKO.WA.
        * **Niemcy:** Należy dodać rozszerzenie .DE, na przykład BMW.DE.
        * **Kryptowaluty:** Należy dodać przyrostek -USD, na przykład BTC-USD.
        * **Surowce:** Należy użyć symboli kontraktów terminowych (np. GC=F dla złota).
        
        Dokładny symbol można zweryfikować na stronie finance.yahoo.com.
        """
    )
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    # Usunięto emoji z opisów
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada proporcji 2x:** Algorytm pilnuje, aby największa pozycja w portfelu była maksymalnie dwa razy większa niż najmniejsza. Zapobiega to dominacji jednej spółki i chroni przed ryzykiem specyficznym.
        """
    )
    
    # Usunięto emoji z opisów
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Metodologia:** System przeprowadza 10 000 wirtualnych rzutów kostką, tworząc tysiące alternatywnych scenariuszy przyszłości dla Twojego portfela.
        **Zastosowanie:** Pozwala to realnie ocenić szansę na stratę oraz zrozumieć, jak szeroki jest zakres niepewności w inwestowaniu.
        """
    )
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        # Usunięto emoji z tekstu spinnera
        with st.spinner('Analizowanie danych rynkowych...'):
            data_raw = yf.download(tickers, period="3y")['Close']
            # Jeśli tickers jest listą, Close ma MultiIndex (Date, Ticker). Potrzebujemy Ticker.
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            # Obliczanie stóp zwrotu i parametrów
            df_daily_rets = data_raw.pct_change().dropna()
            df_monthly_rets = data_raw.resample('ME').last().pct_change().dropna()
            
            # Miesięczny VaR (percentyl 5%) dla każdej spółki
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            
            # Macierz korelacji i średnia korelacja każdej spółki z resztą
            corr_matrix = df_monthly_rets.corr()
            avg_corr_each = corr_matrix.mean()

        # --- NOWY SILNIK OPTYMALIZACJI ---
        # Cel: Znaleźć wagi 'w', które minimalizują różnicę z wagami docelowymi 'target_w'

        # Wagi docelowe: Karzemy za ryzyko (VaR) podniesione do potęgi profilu ryzyka
        # Power zwiększa nacisk na bezpieczeństwo dla profili low/medium.
        penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
        target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        
        # Normalizacja wag (suma = 100%)
        target_w = target_w_raw / target_w_raw.sum()

        # Funkcja celu do minimalizacji (suma kwadratów różnic)
        def objective(w):
            return np.sum((w - target_w.values)**2)

        # Ograniczenia
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} # Suma wag = 1
        ]
        
        # Jeśli włączona, dodaj zasadę dywersyfikacji 2x (największy <= 2*najmniejszy)
        if limit_2x:
            # 2 * min(w) - max(w) >= 0 <=> max(w) <= 2 * min(w)
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        # Uruchomienie optymalizacji (Bounds [0.01, 1.0] - min 1% na spółkę)
        # Początkowy punkt startowy (wagi równe)
        w0 = np.full(len(tickers), 1/len(tickers))
        res = minimize(objective, w0, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        
        # Wagi finalne
        wagi_finalne = res.x

        # WYNIKI
        # Usunięto emoji z nazw kart
        t_names = ["Portfel"]
        if run_mc: t_names.append("Projekcje")
        t_names.append("Korelacje")
        tabs = st.tabs(t_names)

        with tabs[0]:
            # Usunięto emoji z nagłówka
            st.subheader("Rekomendowana alokacja")
            
            c1, c2, c3 = st.columns(3)
            # Miesięczny VaR portfela (ważona średnia VaR-ów spółek)
            p_var = (wagi_finalne * monthly_vars).sum()
            
            # Średnia korelacja całego portfela
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            # Usunięto emojis z nazw metryk
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Z prawdopodobieństwem 95%, strata w ciągu jednego miesiąca nie przekroczy tej wartości w normalnych warunkach rynkowych.")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}", help="Miara współzależności ruchów aktywów. Im niższa, tym lepsza dywersyfikacja.")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Przeliczenie miesięcznego VaR na kwotę przy kapitale {kwota:,.0f} PLN.")
            
            st.divider()
            df_wynik = pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': wagi_finalne * 100,
                'Kwota': wagi_finalne * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            
            st.dataframe(df_wynik.style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                # Usunięto emoji z nagłówka
                st.subheader("Projekcje 10,000 symulacji")
                
                # Parametry GBM dla portfela
                cov_matrix = df_daily_rets.cov()
                p_mean = np.sum(df_daily_rets.mean() * wagi_finalne)
                # Sigma portfela (z uwzględnieniem korelacji)
                p_std = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix, wagi_finalne)))
                
                # Przeprowadzenie symulacji
                n_sims = 10000
                plt.style.use("dark_background")
                
                col_a, col_b = st.columns(2)

                for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    # Obliczenia GBM dla horyzontu czasowego
                    days = y * 252 # 252 dni sesyjne w roku
                    sim_rets = np.random.normal(p_mean, p_std, (days, n_sims))
                    sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                    
                    final_v = sim_paths[-1, :]
                    mediana = np.median(final_v)
                    
                    # Obliczenie CAGR (średnioroczny zwrot)
                    # CAGR = (wartość finalna / startowa)^(1/lata) - 1
                    cagr = (mediana / kwota)**(1/y) - 1
                    
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        # Tabela z percentylami i CAGR
                        st.table(pd.DataFrame({
                            "Metryka": ["Mediana", "95% Optymizm", "5% Pesymizm", "CAGR"],
                            "Wartość": [f"{mediana:,.2f}", f"{np.percentile(final_v, 95):,.2f}", f"{np.percentile(final_v, 5):,.2f}", f"{cagr*100:.2f}%"]
                        }))
                        
                        # Wykres z wybranymi ścieżkami (np. 100 pierwszych dla czytelności)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Używamy koloru #238636 (zielony z motywu) dla ścieżek
                        ax.plot(sim_paths[:, :100], color='#238636', alpha=0.06)
                        # Mediana białą, grubszą linią
                        ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)
                
                plt.style.use('default') # Przywróć domyślny styl dla innych kart

        with tabs[-1]:
            # Usunięto emoji z nagłówka
            st.subheader("Mapa Korelacji")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        # Usunięto emoji z tekstu błędu
        st.error(f"Coś poszło nie tak podczas analizy: {e}")
