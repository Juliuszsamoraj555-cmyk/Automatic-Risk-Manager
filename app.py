import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. KONFIGURACJA STRONY
try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="Valpha Portfolio Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="Valpha Portfolio Manager", layout="wide")

# 2. DESIGN CSS (SaaS Tech Look - Bez emoji)
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
    .disclaimer-red {
        background-color: #1c2128; border-left: 5px solid #d73a49; padding: 15px;
        border-radius: 8px; margin-bottom: 25px; font-size: 0.85em; color: #adbac7; line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. PEŁNY DISCLAIMER PRAWNY (BEZ SKRÓTÓW)
st.markdown("""
    <div class="disclaimer-red">
        <strong>WAŻNE INFORMACJE PRAWNE ORAZ ZASTRZEŻENIA</strong><br>
        Niniejsza aplikacja ma charakter wyłącznie informacyjny oraz edukacyjny i nie stanowi rekomendacji inwestycyjnej ani porady finansowej w rozumieniu Rozporządzenia Ministra Finansów z dnia 19 października 2005 r. w sprawie informacji stanowiących rekomendacje dotyczące instrumentów finansowych lub ich emitentów. 
        Inwestowanie na rynkach kapitałowych oraz w kryptowaluty wiąże się z wysokim ryzykiem utraty części lub całości kapitału. Wszelkie symulacje, w tym modele Monte Carlo oraz prognozy CAPM, bazują na danych historycznych i zaawansowanych algorytmach statystycznych, które nie stanowią gwarancji osiągnięcia podobnych wyników w przyszłości. 
        Autor narzędzia nie ponosi żadnej odpowiedzialności za decyzje inwestycyjne podjęte na podstawie danych generowanych przez system. Pamiętaj, że wyniki historyczne nie są wyznacznikiem przyszłych zysków. Przed podjęciem jakichkolwiek działań na rynku skonsultuj się z licencjonowanym doradcą inwestycyjnym.
    </div>
    """, unsafe_allow_html=True)

# 4. SIDEBAR - KONFIGURACJA I SUBSKRYPCJA
with st.sidebar:
    try:
        st.image(v_alpha_icon, width=100)
    except:
        pass
    st.title("Valpha Portfolio Manager")
    
    st.subheader("STATUS SUBSKRYPCJI")
    license_key = st.text_input("Klucz licencyjny PRO:", type="password", help="Wprowadź swój unikalny klucz, aby odblokować zaawansowane modele Monte Carlo oraz nielimitowaną liczbę spółek w portfelu.")
    is_pro = (license_key == "PRO2024")
    
    if is_pro:
        st.success("WERSJA PRO AKTYWNA")
    else:
        st.warning("WERSJA FREE (LIMIT: 5 SPÓŁEK)")

    st.divider()
    st.subheader("KONFIGURACJA PORTFELA")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, NVDA, TSLA, AMZN",
        help="""
        **Instrukcja wprowadzania symboli:**
        
        System pobiera dane bezpośrednio z serwerów Yahoo Finance. 
        * **Rynek USA:** Wpisuj sam ticker (np. AAPL, TSLA, MSFT).
        * **Polska (GPW):** Dodaj rozszerzenie .WA (np. ALE.WA, PKO.WA, CDR.WA).
        * **Rynek niemiecki:** Dodaj .DE (np. BMW.DE, ADS.DE).
        * **Kryptowaluty:** Dodaj -USD (np. BTC-USD, ETH-USD).
        * **Surowce:** Użyj symboli kontraktów terminowych (np. GC=F dla złota, CL=F dla ropy).
        
        Dokładny symbol możesz sprawdzić na finance.yahoo.com.
        """
    )
    
    kwota = st.number_input("Kapitał początkowy (PLN):", value=25000, step=1000)
    
    st.divider()
    opt_mode = st.radio(
        "Model Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        help="""
        **Bezpieczeństwo (VaR):** Model koncentruje się na minimalizacji potencjalnych strat w najgorszych scenariuszach rynkowych. Wybiera aktywa o najwyższej stabilności i najniższej korelacji.
        
        **Efektywność (Sortino):** Model szuka najlepszego stosunku zysku do ryzyka spadków (downside risk). Premiuje aktywa, które rosną stabilnie, ale rzadko zaliczają gwałtowne załamania ceny.
        """
    )
    
    ryzyko_val = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada proporcji 2x:** Algorytm pilnuje, aby największa pozycja w portfelu była maksymalnie dwa razy większa niż pozycja najmniejsza.
        
        **Dlaczego to ważne?** Zapobiega to tzw. dominacji jednego aktywa. Nawet jeśli model uzna daną spółkę za idealną, limit wymusza rozłożenie kapitału, chroniąc Cię przed ryzykiem specyficznym – czyli nagłym upadkiem jednej firmy z przyczyn pozarynkowych.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Co to robi?** System przeprowadza 10 000 wirtualnych symulacji przyszłości dla Twojego portfela. 
        
        **Zastosowanie:** Zamiast jednej linii zysku, widzisz cały wachlarz możliwości – od skrajnie pesymistycznych po bardzo optymistyczne. Pozwala to realnie zrozumieć statystyczne prawdopodobieństwo straty kapitału w czasie.
        """
    )
    
    adj_mc = False
    if run_mc:
        label_adj = "Skorygowana symulacja Monte Carlo"
        if not is_pro: label_adj += " (Wymaga PRO)"
        
        adj_mc = st.checkbox(
            label_adj, 
            value=False, 
            disabled=not is_pro,
            help="Włącza model CAPM skorygowany o historyczną Alfę. Uwzględnia wpływ zmienności na stopę zwrotu (volatility drag) oraz pozwala na symulowanie przewagi rynkowej spółek (Alfa retention)."
        )
        
        if adj_mc and is_pro:
            with st.expander("PARAMETRY RYNKOWE CAPM / GBM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_ret = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30, help="Jaki procent historycznej przewagi spółki nad rynkiem utrzyma się w przyszłości.")
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05, help="Tempo, w jakim Beta spółki dąży do średniej rynkowej (1.0).")

    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ SYSTEMOWĄ")

# 5. LOGIKA ANALIZY I OBLICZEŃ
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not is_pro and len(tickers) > 5:
        st.error(f"Wykryto {len(tickers)} spółek. Wersja darmowa obsługuje maksymalnie 5 pozycji. Usuń nadmiarowe symbole lub aktywuj wersję PRO.")
    else:
        try:
            with st.spinner('Pobieranie i analizowanie danych rynkowych...'):
                fetch_list = tickers + (["SPY"] if adj_mc else [])
                data = yf.download(fetch_list, period="3y")['Close']
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(-1)
                
                if adj_mc:
                    spy_rets = data["SPY"].pct_change().dropna()
                    stock_data = data[tickers]
                    betas, alphas = {}, {}
                    spy_annual = (1 + spy_rets.mean())**252 - 1
                    for t in tickers:
                        t_rets = stock_data[t].pct_change().dropna()
                        comb = pd.concat([t_rets, spy_rets], axis=1).dropna()
                        b = np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1])
                        betas[t] = b
                        hist_ret = (1 + t_rets.mean())**252 - 1
                        alphas[t] = hist_ret - (rf_rate + b * (spy_annual - rf_rate))
                    data_only = stock_data
                else:
                    data_only = data[tickers] if "SPY" in data.columns else data

                daily_rets = data_only.pct_change().dropna()
                monthly_rets = data_only.resample('ME').last().pct_change().dropna()
                monthly_vars = monthly_rets.quantile(0.05) * -1
                corr_matrix = monthly_rets.corr()

            # --- OPTYMALIZACJA ---
            if opt_mode == "Bezpieczeństwo (VaR-First)":
                p = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko_val]
                target_w_raw = (1 / (monthly_vars ** p)) * (1 - corr_matrix.mean())
            else:
                sortino = monthly_rets.mean() / (monthly_rets[monthly_rets < 0].std() + 1e-6)
                target_w_raw = (sortino.clip(lower=0) ** {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko_val]) * (1 - corr_matrix.mean())

            target_w = target_w_raw / target_w_raw.sum()
            res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                           method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), 
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + 
                           ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
            wagi = res.x

            # --- PREZENTACJA WYNIKÓW ---
            tabs = st.tabs(["Struktura Portfela", "Symulacja Monte Carlo", "Macierz Korelacji", "Metodologia"])

            with tabs[0]:
                st.subheader(f"Rekomendowana alokacja ({opt_mode})")
                c1, c2, c3 = st.columns(3)
                p_var = (wagi * monthly_vars).sum()
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystycznie istnieje tylko 5% szansy, że w ciągu jednego miesiąca portfel straci więcej niż ten procent. Jest to miara normalnego ryzyka rynkowego.")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}", help="Określa, jak bardzo spółki poruszają się w parze. Blisko 0 oznacza świetną dywersyfikację.")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Twój miesięczny VaR przeliczony na kwotę przy kapitale {kwota:,.0f} PLN.")
                
                st.divider()
                df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
                if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
                st.dataframe(df_out.sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True, use_container_width=True)

            if run_mc:
                with tabs[1]:
                    st.subheader("Symulacja Monte Carlo - 10,000 symulacji")
                    st.info("""**Ważna informacja:** Symulacja bazuje na analizie statystycznej zmienności historycznej. 
                            Wyniki nie stanowią gwarancji przyszłych zysków, a realne warunki rynkowe mogą odbiegać od symulowanych scenariuszy.""")
                    
                    n_sims, dt = 10000, 1/252
                    log_rets = np.log(data_only / data_only.shift(1)).dropna()
                    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
                    
                    col_a, col_b = st.columns(2)
                    plt.style.use("dark_background")

                    for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                        days = y * 252
                        paths = np.zeros((days, n_sims))
                        curr = np.full(n_sims, float(kwota))
                        
                        if adj_mc:
                            p_beta = np.sum([betas[t] * wagi[idx] for idx, t in enumerate(tickers)])
                            p_alpha = np.sum([alphas[t] * wagi[idx] for idx, t in enumerate(tickers)]) * (alpha_ret / 100)
                            t_beta = p_beta
                            for d in range(days):
                                mu = (rf_rate + t_beta * (mkt_ret - rf_rate) + p_alpha - 0.5 * (p_sigma**2)) * dt
                                curr *= np.exp(mu + p_sigma * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                                paths[d, :] = curr
                                if d % 252 == 0: t_beta = t_beta * (1 - beta_speed) + 1.0 * beta_speed
                        else:
                            mu = (np.sum(daily_rets.mean() * wagi) * 252 - 0.5 * (p_sigma**2)) * dt
                            for d in range(days):
                                curr *= np.exp(mu + p_sigma * np.random.normal(0, 1, n_sims) * np.sqrt(dt))
                                paths[d, :] = curr

                        final = paths[-1, :]
                        med = np.median(final)
                        res_df = pd.DataFrame({
                            "Metryka": ["95. Percentyl (Optymizm)", "3. Kwartyl (75%)", "Mediana (Statystyczny wynik)", "1. Kwartyl (25%)", "5. Percentyl (Pesymizm)", "Prawdopodobieństwo straty", "CAGR (Roczny zwrot)"],
                            "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 75):,.2f}", f"{med:,.2f}", f"{np.percentile(final, 25):,.2f}", 
                                        f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((med / kwota)**(1/y) - 1)*100:.2f}%"]
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
                st.subheader("Macierz korelacji między aktywami")
                fig_c, ax_c = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
                st.pyplot(fig_c)

            with tabs[3]:
                st.header("Metodologia obliczeń i algorytmy")
                
                with st.expander("1. Optymalizacja wag portfela", expanded=True):
                    st.markdown("""
                    Algorytm wyznacza optymalny udział każdej spółki w oparciu o wybraną strategię:
                    
                    **A. Model VaR-First (Bezpieczeństwo)**
                    Wagi są wyznaczane na podstawie odwrotności Wartości Zagrożonej (VaR) oraz średniej korelacji:
                    $$W_i \\propto \\frac{1 - \\bar{\\rho}_i}{VaR_i^p}$$
                    Gdzie $p$ to wykładnik kary za ryzyko zależny od Twojego profilu.
                    
                    **B. Model Sortino (Efektywność)**
                    Maksymalizacja zysku w stosunku do odchylenia standardowego strat (downside deviation):
                    $$W_i \\propto \\left(\\frac{R_i - R_f}{\\sigma_{downside}}\\right)^p \\cdot (1 - \\bar{\\rho}_i)$$
                    """)
                
                with st.expander("2. Skorygowana Symulacja Monte Carlo (GBM + CAPM)"):
                    st.markdown("""
                    Model generuje tysiące alternatywnych ścieżek cenowych przy użyciu procesu Geometrycznego Ruchu Browna (GBM):
                    
                    **Krok 1: Wyznaczenie Oczekiwanej Stopy Zwrotu (CAPM + Alfa)**
                    Stosujemy model wyceny aktywów kapitałowych wzbogacony o historyczną przewagę (Alfę):
                    $$E(R_i) = R_f + \\beta_i(E(R_m) - R_f) + \\alpha \\cdot \\text{retention}$$
                    
                    **Krok 2: Korekta Dryfu (Volatility Drag)**
                    Wariancja obniża realną medianę kapitału w czasie, co uwzględniamy w parametrze $\\mu$:
                    $$\\mu_{adj} = E(R_i) - \\frac{1}{2}\\sigma^2$$
                    
                    **Krok 3: Generowanie ścieżek cenowych**
                    Dla każdego kroku czasowego $\\Delta t$ wyliczamy nową cenę:
                    $$P_{t+1} = P_t \\cdot e^{(\\mu_{adj} \\Delta t + \\sigma \\epsilon \\sqrt{\\Delta t})}$$
                    Gdzie $\\epsilon$ to zmienna losowa z rozkładu normalnego $N(0,1)$.
                    """)

        except Exception as e:
            st.error(f"Błąd krytyczny systemu: {e}")
