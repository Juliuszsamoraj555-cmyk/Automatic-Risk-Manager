import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS
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

# 3. SIDEBAR
st.title("🛡️ Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Ustawienia")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD",
        help="""
        **Jak wpisywać symbole?**
        System pobiera dane z Yahoo Finance. 
        * **USA:** Sam ticker (np. `AAPL`).
        * **Polska:** Dodaj `.WA` (np. `ALE.WA`).
        * **Krypto:** Dodaj `-USD` (np. `BTC-USD`).
        """
    )
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    st.divider()
    
    opt_mode = st.radio(
        "Tryb Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        index=0,
        help="""
        **Bezpieczeństwo (VaR):** Skupia się na minimalizacji strat w najgorszych scenariuszach. Wybiera najbardziej stabilne spółki.
        **Efektywność (Sortino):** Szuka najlepszego zysku w stosunku do ryzyka spadków. Docenia spółki, które rosną gwałtownie, ale rzadko zaliczają głębokie 'doły'.
        """
    )

    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    st.divider()
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada 2x:** Algorytm pilnuje, aby największa pozycja w portfelu była maksymalnie dwa razy większa niż najmniejsza. 
        Zapobiega to dominacji jednej spółki i chroni przed ryzykiem specyficznym.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Co to robi?**
        To matematyczna 'wróżba' oparta na faktach. System przeprowadza **10 000 wirtualnych rzutów kostką**, tworząc tysiące scenariuszy przyszłości dla Twojego portfela.
        """
    )
    
    adj_mc = False
    if run_mc:
        adj_mc = st.checkbox(
            "Skorygowana symulacja Monte Carlo", 
            value=False,
            help="Włącza model CAPM skorygowany o Alfę spółek. Zamiast czystej historii, pozwala uwzględnić przewagę rynkową liderów."
        )
        
        if adj_mc:
            with st.expander("📈 Parametry CAPM/GBM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_retention = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30, 
                                            help="Ile % historycznej przewagi spółki nad rynkiem utrzyma się w symulacji.")
                beta_speed = st.slider("Szybkość wygasania Bety:", 0.0, 0.2, 0.05, 
                                       help="Symuluje 'starzenie się' spółki – jej Beta z czasem dąży do 1.0.")

    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Analizowanie danych...'):
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
            mean_ret = df_monthly_rets.mean()
            downside_std = df_monthly_rets[df_monthly_rets < 0].std()
            sortino = mean_ret / (downside_std + 1e-6)
            target_w_raw = (sortino.clip(lower=0) ** {'low': 0.5, 'medium': 1.0, 'high': 1.5}[ryzyko]) * (1 - corr_matrix.mean())

        target_w = target_w_raw / target_w_raw.sum()
        res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), 
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
        wagi = res.x

        # WYNIKI
        t_names = ["📈 Portfel", "🔮 Symulacja Monte Carlo", "🔗 Korelacje", "🧠 Metodologia"]
        tabs = st.tabs(t_names)

        with tabs[0]:
            st.subheader(f"Rekomendowana alokacja ({opt_mode})")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi * monthly_vars).sum()
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystyczna miara ryzyka straty miesięcznej.")
            c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Szacowana miesięczna strata przy kapitale {kwota:,.0f} PLN.")
            
            df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
            if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
            st.dataframe(df_out.sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"Symulacja Monte Carlo - 10,000 symulacji ({opt_mode})")
                st.info("""**Ważna informacja:** Symulacja Monte Carlo bazuje na zmienności historycznej i statystyce. 
                        Pamiętaj, że wyniki historyczne nie są gwarancją przyszłych zysków.""")
                
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
                        "Metryka": ["95. Percentyl", "3. Kwartyl (75%)", "Mediana", "1. Kwartyl (25%)", "5. Percentyl", "Szansa na stratę", "Zwrot (CAGR)"],
                        "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 75):,.2f}", f"{np.median(final):,.2f}", f"{np.percentile(final, 25):,.2f}", 
                                    f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((np.median(final) / kwota)**(1/y) - 1)*100:.2f}%"]
                    })
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        st.table(res_df)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)
                plt.style.use('default')

        with tabs[2]:
            st.subheader("Mapa Korelacji Między Aktywami")
            fig_c, ax_c = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

        # --- NOWA ZAKŁADKA: METODOLOGIA ---
        with tabs[3]:
            st.header("🧠 Metodologia Obliczeń")
            
            with st.expander("1. Optymalizacja Wag Portfela", expanded=True):
                st.markdown("""
                W zależności od wybranego trybu, algorytm stosuje jedną z dwóch zaawansowanych technik alokacji:
                
                **A. Tryb Bezpieczeństwa (VaR-First)**
                Wagi są wyznaczane na podstawie odwrotności Wartości Zagrożonej (VaR) oraz średniej korelacji spółki z resztą portfela:
                $$W_i \\propto \\frac{1 - \\bar{\\rho}_i}{VaR_i^p}$$
                Gdzie:
                - $VaR_i$: Miesięczna strata historyczna (percentyl 5%).
                - $\\bar{\\rho}_i$: Średnia korelacja danej spółki z pozostałymi komponentami.
                - $p$: Parametr profilu ryzyka (Kara za zmienność).
                
                **B. Tryb Efektywności (Sortino)**
                Wagi są optymalizowane pod kątem maksymalizacji stosunku zysku do zmienności ujemnej:
                $$W_i \\propto \\left(\\frac{R_i - R_f}{\\sigma_{downside}}\\right)^p \\cdot (1 - \\bar{\\rho}_i)$$
                Model ten premiuje aktywa, które rosną stabilnie, nie karząc ich za gwałtowne skoki cen w górę.
                """)

            with st.expander("2. Skorygowana Symulacja Monte Carlo (CAPM + GBM)"):
                st.markdown("""
                W trybie zaawansowanym stosujemy model **Geometric Brownian Motion (GBM)** zintegrowany z modelem wyceny aktywów kapitałowych (**CAPM**).
                
                **Krok 1: Wyznaczenie Oczekiwanej Stopy Zwrotu**
                Zamiast ufać tylko historii, wyliczamy zwrot na podstawie ryzyka systematycznego (Bety):
                $$E(R_i) = R_f + \\beta_i \\cdot (E(R_m) - R_f) + \\alpha \\cdot \\text{retention}$$
                
                **Krok 2: Korekta o Volatility Drag**
                W statystyce długoterminowej zmienność obniża medianę kapitału. Korygujemy dryf symulacji o połowę wariancji:
                $$\\mu_{adj} = E(R_i) - \\frac{1}{2}\\sigma^2$$
                
                **Krok 3: Generowanie ścieżek cenowych**
                Cena w każdym kolejnym kroku czasowym $\\Delta t$ (dziennym) wyliczana jest wzorem:
                $$P_{t+1} = P_t \\cdot e^{(\\mu_{adj} \\cdot \\Delta t + \\sigma \\cdot \\epsilon \\cdot \\sqrt{\\Delta t})}$$
                Gdzie $\\epsilon$ jest liczbą losową z rozkładu normalnego $N(0, 1)$.
                
                **Krok 4: Mean Reversion Bety**
                Symulujemy dojrzewanie spółek. Im dłuższy horyzont czasowy, tym Beta portfela szybciej dąży do średniej rynkowej ($1.0$):
                $$\\beta_{t+1} = \\beta_t \\cdot (1 - \\text{speed}) + 1.0 \\cdot \\text{speed}$$
                """)

            st.success("Modele matematyczne są aktualizowane dynamicznie na podstawie danych rynkowych z ostatnich 3 lat.")

    except Exception as e:
        st.error(f"Błąd: {e}")
