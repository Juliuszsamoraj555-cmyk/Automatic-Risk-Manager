import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="📊", layout="wide")

# 2. DESIGN CSS (Clean & Tech Look)
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
        text-transform: uppercase; letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR
st.title("RISK MANAGER PRO")
with st.sidebar:
    st.subheader("KONFIGURACJA")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD",
        help="""
        **Instrukcja wprowadzania symboli:**
        
        System pobiera dane bezpośrednio z serwerów Yahoo Finance. 
        * **Rynek USA:** Należy wpisać sam ticker, na przykład AAPL, TSLA lub MSFT.
        * **Giełda Papierów Wartościowych w Warszawie:** Należy dodać rozszerzenie .WA, na przykład ALE.WA lub PKO.WA.
        * **Rynek niemiecki:** Należy dodać rozszerzenie .DE, na przykład BMW.DE.
        * **Kryptowaluty:** Należy dodać przyrostek -USD, na przykład BTC-USD.
        * **Surowce:** Należy użyć symboli kontraktów terminowych, na przykład GC=F dla złota.
        
        Dokładny symbol można zweryfikować na stronie finance.yahoo.com.
        """
    )
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    st.divider()
    
    opt_mode = st.radio(
        "Tryb Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        index=0,
        help="""
        **Bezpieczeństwo (VaR):** Priorytetem jest minimalizacja strat w scenariuszach o niskim prawdopodobieństwie wystąpienia. Model wybiera aktywa o najwyższej stabilności.
        
        **Efektywność (Sortino):** Model dąży do uzyskania najwyższej stopy zwrotu w relacji do ryzyka spadków. Premiuje aktywa o silnym trendzie wzrostowym przy jednoczesnym ograniczaniu zmienności ujemnej.
        """
    )

    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    st.divider()
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada proporcji 2x:** Algorytm zapewnia, że alokacja w największą pozycję portfelu nie przekroczy dwukrotności alokacji w pozycję najmniejszą.
        
        **Cel stosowania:**
        Mechanizm ten zapobiega nadmiernej koncentracji kapitału w pojedynczym aktywie. Ogranicza to ryzyko specyficzne, czyli prawdopodobieństwo poniesienia znacznych strat wynikających z nieprzewidzianych zdarzeń dotyczących konkretnego emitenta.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Metodologia:**
        Jest to zaawansowana analiza statystyczna. System generuje 10 000 alternatywnych ścieżek cenowych, opierając się na parametrach rozkładu prawdopodobieństwa Twojego portfela.
        
        **Zastosowanie:**
        Zamiast prezentacji pojedynczego wyniku, model ukazuje pełny zakres możliwych rezultatów inwestycyjnych. Pozwala to precyzyjnie oszacować prawdopodobieństwo straty oraz zrozumieć skalę niepewności w założonym horyzoncie czasowym.
        """
    )
    
    adj_mc = False
    if run_mc:
        adj_mc = st.checkbox(
            "Skorygowana symulacja Monte Carlo", 
            value=False,
            help="Aktywuje model CAPM uwzględniający historyczną Alfę. Model koryguje wyniki o wpływ zmienności na stopę zwrotu, co zwiększa realizm prognozy w długim terminie."
        )
        
        if adj_mc:
            with st.expander("PARAMETRY CAPM / GBM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_retention = st.slider("Utrzymanie Alfy (%):", 0, 100, 30, 
                                            help="Wartość określająca, jaka część historycznej przewagi spółki nad rynkiem zostanie uwzględniona w prognozie.")
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05, 
                                       help="Parametr określający tempo, w jakim współczynnik Beta portfela dąży do wartości rynkowej równej 1.0.")

    st.divider()
    analizuj = st.button("URUCHOM ANALIZĘ SYSTEMOWĄ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('PRZETWARZANIE DANYCH RYNKOWYCH...'):
            fetch_tickers = tickers + (["SPY"] if adj_mc else [])
            data_raw = yf.download(fetch_tickers, period="3y")['Close']
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            if adj_mc:
                spy_rets = data_raw["SPY"].pct_change().dropna()
                stock_data = data_raw[tickers]
                betas, alphas = {}, {}
                spy_annual_ret = (1 + spy_rets.mean())**252 - 1
                for t in tickers:
                    t_rets = stock_data[t].pct_change().dropna()
                    combined = pd.concat([t_rets, spy_rets], axis=1).dropna()
                    b = np.cov(combined.iloc[:,0], combined.iloc[:,1])[0,1] / np.var(combined.iloc[:,1])
                    betas[t] = b
                    alphas[t] = ((1 + t_rets.mean())**252 - 1) - (rf_rate + b * (spy_annual_ret - rf_rate))
                data_only = stock_data
            else:
                data_only = data_raw[tickers] if "SPY" in data_raw.columns else data_raw

            df_daily_rets = data_only.pct_change().dropna()
            df_monthly_rets = data_only.resample('ME').last().pct_change().dropna()
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()

        # OPTYMALIZACJA
        if opt_mode == "Bezpieczeństwo (VaR-First)":
            penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
            target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - corr_matrix.mean())
        else:
            sortino = df_monthly_rets.mean() / (df_monthly_rets[df_monthly_rets < 0].std() + 1e-6)
            target_w_raw = (sortino.clip(lower=0) ** {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko]) * (1 - corr_matrix.mean())

        target_w = target_w_raw / target_w_raw.sum()
        res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), 
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
        wagi = res.x

        # WYNIKI
        tabs = st.tabs(["STRUKTURA PORTFELA", "SYMULACJA MONTE CARLO", "MACIERZ KORELACJI", "METODOLOGIA"])

        with tabs[0]:
            st.subheader(f"REKOMENDOWANA ALOKACJA: {opt_mode.upper()}")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi * monthly_vars).sum()
            c1.metric(
                "Miesięczny VaR (95%)", 
                f"{p_var*100:.2f}%", 
                help="Wskaźnik określający, że z prawdopodobieństwem 95% miesięczna strata nie przekroczy wskazanej wartości procentowej w normalnych warunkach rynkowych."
            )
            c2.metric(
                "Średnia Korelacja", 
                f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}",
                help="Miara statystyczna określająca stopień współzależności ruchów cenowych aktywów. Wartości bliskie zeru wskazują na wysoką dywersyfikację."
            )
            c3.metric(
                "Ryzyko (PLN)", 
                f"{p_var * kwota:,.2f}",
                help=f"Wartość VaR przeliczona na kwotę pieniężną w relacji do Twojego kapitału początkowego wynoszącego {kwota:,.0f} PLN."
            )
            
            st.divider()
            df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
            if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
            st.dataframe(df_out.sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"SYMULACJA MONTE CARLO: 10,000 SCENARIUSZY")
                st.info("Symulacja opiera się na analizie statystycznej danych historycznych. Wyniki nie stanowią gwarancji przyszłych stóp zwrotu.")
                
                n_sims, dt = 10000, 1/252
                log_rets = np.log(data_only / data_only.shift(1)).dropna()
                p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
                
                col_a, col_b = st.columns(2)
                plt.style.use("dark_background")

                for i, (y, lbl) in enumerate(zip([5, 10], ["5 LAT", "10 LAT"])):
                    days = y * 252
                    paths = np.zeros((days, n_sims))
                    curr = np.full(n_sims, float(kwota))
                    
                    if adj_mc:
                        p_beta = np.sum([betas[t] * wagi[idx] for idx, t in enumerate(tickers)])
                        p_alpha = np.sum([alphas[t] * wagi[idx] for idx, t in enumerate(tickers)]) * (alpha_retention / 100)
                        t_beta = p_beta
                        for d in range(days):
                            mu = (rf_rate + t_beta * (mkt_ret - rf_rate) + p_alpha - 0.5 * (p_sigma**2)) * dt
                            curr *= np.exp(mu + p_sigma * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                            paths[d, :] = curr
                            if d % 252 == 0: t_beta = t_beta * (1 - beta_speed) + 1.0 * beta_speed
                    else:
                        mu = (np.sum(df_daily_rets.mean() * wagi) * 252 - 0.5 * (p_sigma**2)) * dt
                        for d in range(days):
                            curr *= np.exp(mu + p_sigma * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                            paths[d, :] = curr

                    final = paths[-1, :]
                    res_df = pd.DataFrame({
                        "Metryka": ["95. Percentyl", "3. Kwartyl (75%)", "Mediana", "1. Kwartyl (25%)", "5. Percentyl", "Prawdopodobieństwo straty", "CAGR"],
                        "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 75):,.2f}", f"{np.median(final):,.2f}", f"{np.percentile(final, 25):,.2f}", 
                                    f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((np.median(final) / kwota)**(1/y) - 1)*100:.2f}%"]
                    })
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### PERSPEKTYWA: {lbl}")
                        st.table(res_df)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(paths[:, :100], color='#238636', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                        st.pyplot(fig)
                plt.style.use('default')

        with tabs[2]:
            st.subheader("MACIERZ KORELACJI AKTYWÓW")
            fig_c, ax_c = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

        with tabs[3]:
            st.header("METODOLOGIA OBLICZEŃ")
            with st.expander("ALGORYTM OPTYMALIZACJI PORTFELA"):
                st.markdown("""
                Model stosuje precyzyjne techniki alokacji w zależności od zdefiniowanej strategii:
                - **Strategia Bezpieczeństwa (VaR)**: Alokacja bazuje na minimalizacji potencjalnej straty maksymalnej przy uwzględnieniu korelacji.
                - **Strategia Efektywności (Sortino)**: Optymalizacja pod kątem maksymalizacji stopy zwrotu przy jednoczesnej minimalizacji odchylenia ujemnego.
                """)
            with st.expander("MODEL SYMULACJI (GBM I CAPM)"):
                st.markdown("""
                W trybie skorygowanym system wykorzystuje proces **Geometrycznych Ruchów Browna (GBM)** zintegrowany z modelem wyceny aktywów kapitałowych. Model uwzględnia:
                - **Dryf skorygowany o zmienność**: Redukcja oczekiwanej stopy zwrotu o wpływ wariancji na wyniki długoterminowe.
                - **Konwergencję współczynnika Beta**: Proces stopniowego dostosowania ryzyka systematycznego portfela do średniej rynkowej.
                """)

    except Exception as e:
        st.error(f"Wystąpił błąd podczas przetwarzania danych: {e}")
