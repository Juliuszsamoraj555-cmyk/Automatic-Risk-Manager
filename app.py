To świetny kierunek. Dzięki temu Janek nie czuje się przytłoczony matematyką na start, ale jeśli chce wejść w rolę analityka "Quant", może jednym kliknięciem aktywować zaawansowany model.

Wprowadziłem zmiany tak, aby domyślnie system działał na czystych danych historycznych (tak jak wcześniej), a sekcja Skorygowana symulacja Monte Carlo pojawiała się jako dodatkowa opcja w panelu bocznym.

Co się zmieniło?
Checkbox "Skorygowana symulacja Monte Carlo": Aktywuje on model CAPM/GBM. Jeśli jest wyłączony, symulacja opiera się na prostym dryfie historycznym.

Inteligentne pobieranie danych: Benchmark (SPY) jest pobierany tylko wtedy, gdy Janek zaznaczy opcję korekty, co przyspiesza działanie standardowej analizy.

Zachowanie wszystkich opisów: Wszystkie dymki pomocy (?) i ostrzeżenia pozostały na swoich miejscach.

Kompletny i Zaktualizowany Kod app.py
Python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (SaaS Look)
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
        * **USA:** Wpisuj sam ticker (np. `AAPL`, `TSLA`, `MSFT`).
        * **Polska (GPW):** Dodaj `.WA` (np. `ALE.WA`, `PKO.WA`).
        * **Niemcy:** Dodaj `.DE` (np. `BMW.DE`).
        * **Kryptowaluty:** Dodaj `-USD` (np. `BTC-USD`).
        * **Złoto/Surowce:** Użyj symboli kontraktów (np. `GC=F` dla złota).
        
        Wyszukaj ticker na *finance.yahoo.com*, jeśli nie jesteś pewien.
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
    
    # --- SEKCJA: SKORYGOWANA SYMULACJA MONTE CARLO ---
    st.divider()
    adj_mc = st.checkbox(
        "Skorygowana symulacja Monte Carlo", 
        value=False,
        help="Włącza model CAPM (Capital Asset Pricing Model). Zamiast średniej historycznej, symulacja użyje oczekiwanej stopy zwrotu wynikającej z ryzyka rynkowego spółki (Bety)."
    )
    
    if adj_mc:
        with st.expander("📈 Parametry CAPM/GBM", expanded=True):
            rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
            mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
            beta_speed = st.slider("Szybkość wygasania Bety:", 0.0, 0.2, 0.05, 
                                   help="Symuluje 'starzenie się' spółki – jej Beta z czasem dąży do 1.0.")

    st.divider()
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada 2x:** Algorytm pilnuje, aby największa pozycja w portfelu była maksymalnie dwa razy większa niż najmniejsza.
        
        **W jakim celu?**
        Zapobiega to tzw. 'dominacji' jednej spółki. Nawet jeśli model uzna jakąś firmę za bardzo bezpieczną, limit ten wymusza rozłożenie kapitału na pozostałe aktywa. Chroni Cię to przed **ryzykiem specyficznym** – czyli sytuacją, w której jedna firma nagle upada z przyczyn, których nie widać w statystykach.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        **Co to robi?**
        To matematyczna 'wróżba' oparta na faktach. System przeprowadza **10 000 wirtualnych rzutów kostką**, tworząc tysiące alternatywnych scenariuszy przyszłości dla Twojego portfela.
        
        **Dlaczego to ważne?**
        Zamiast jednej linii 'zysku', widzisz cały wachlarz możliwości – od bardzo optymistycznych po kryzysowe. Pozwala to realnie ocenić **szansę na stratę** oraz zrozumieć, jak szeroki jest zakres niepewności w inwestowaniu.
        """
    )
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Analizowanie danych rynkowych...'):
            # Pobieranie danych
            fetch_tickers = tickers + (["SPY"] if adj_mc else [])
            data_raw = yf.download(fetch_tickers, period="3y")['Close']
            
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            # Logika CAPM (tylko jeśli wybrano korektę)
            betas = {}
            if adj_mc:
                spy_rets = data_raw["SPY"].pct_change().dropna()
                stock_data = data_raw[tickers]
                for t in tickers:
                    t_rets = stock_data[t].pct_change().dropna()
                    # Synchronizacja dat
                    combined = pd.concat([t_rets, spy_rets], axis=1).dropna()
                    cov = np.cov(combined.iloc[:,0], combined.iloc[:,1])[0,1]
                    var = np.var(combined.iloc[:,1])
                    betas[t] = cov / var
                data_only = stock_data
            else:
                data_only = data_raw[tickers] if "SPY" in data_raw.columns else data_raw

            df_daily_rets = data_only.pct_change().dropna()
            df_monthly_rets = data_only.resample('ME').last().pct_change().dropna()
            
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()
            avg_corr_each = corr_matrix.mean()

        # OPTYMALIZACJA
        if opt_mode == "Bezpieczeństwo (VaR-First)":
            penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
            target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        else:
            mean_ret = df_monthly_rets.mean()
            downside_std = df_monthly_rets[df_monthly_rets < 0].std()
            sortino_ratios = mean_ret / (downside_std + 1e-6)
            power_map = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            target_w_raw = (sortino_ratios.clip(lower=0) ** power_map[ryzyko]) * (1 - avg_corr_each)

        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        wagi_finalne = res.x

        # WYNIKI
        tabs = st.tabs(["📈 Portfel", "🔮 Symulacja Monte Carlo", "🔗 Korelacje"])

        with tabs[0]:
            st.subheader(f"Rekomendowana alokacja ({opt_mode})")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi_finalne * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystyczna miara ryzyka straty miesięcznej.")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}", help="Mierzy jak bardzo aktywa poruszają się w tym samym kierunku.")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Szacowana miesięczna strata przy kapitale {kwota:,.0f} PLN.")
            
            st.divider()
            display_df = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi_finalne * 100, 'Kwota': wagi_finalne * kwota})
            if adj_mc: display_df['Beta'] = [betas[t] for t in tickers]
            
            st.dataframe(display_df.sort_values(by='Udział (%)', ascending=False).style.format({
                'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'
            }), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"Symulacja Monte Carlo - 10,000 symulacji ({opt_mode})")
                st.info(f"**Tryb:** {'Skorygowany (CAPM/GBM)' if adj_mc else 'Standardowy (Historyczny)'}. "
                        "Symulacja bazuje na zmienności historycznej i statystyce. Wyniki historyczne nie gwarantują przyszłych zysków.")
                
                n_sims = 10000
                dt = 1/252
                log_returns = np.log(data_only / data_only.shift(1)).dropna()
                cov_matrix = log_returns.cov().values
                port_sigma_annual = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix, wagi_finalne))) * np.sqrt(252)
                
                if adj_mc:
                    port_beta = np.sum([betas[t] * wagi_finalne[i] for i, t in enumerate(tickers)])

                col_a, col_b = st.columns(2)
                plt.style.use("dark_background")

                for i, (years, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    days = years * 252
                    paths = np.zeros((days, n_sims))
                    current_prices = np.full(n_sims, float(kwota))
                    
                    if adj_mc:
                        temp_beta = port_beta
                        for d in range(days):
                            mu_adj = (rf_rate + temp_beta * (mkt_ret - rf_rate) - 0.5 * (port_sigma_annual**2)) * dt
                            epsilon = np.random.normal(0, 1, n_sims)
                            current_prices *= np.exp(mu_adj + port_sigma_annual * epsilon * np.sqrt(dt))
                            paths[d, :] = current_prices
                            if d % 252 == 0: temp_beta = temp_beta * (1 - beta_speed) + 1.0 * beta_speed
                    else:
                        # Standardowy model (historyczny dryf)
                        hist_mu = np.sum(df_daily_rets.mean() * wagi_finalne) * 252
                        mu_adj = (hist_mu - 0.5 * (port_sigma_annual**2)) * dt
                        for d in range(days):
                            epsilon = np.random.normal(0, 1, n_sims)
                            current_prices *= np.exp(mu_adj + port_sigma_annual * epsilon * np.sqrt(dt))
                            paths[d, :] = current_prices

                    final_v = paths[-1, :]
                    mediana = np.median(final_v)
                    stats = {
                        "95. Percentyl": np.percentile(final_v, 95),
                        "3. Kwartyl (75%)": np.percentile(final_v, 75),
                        "Mediana": mediana,
                        "1. Kwartyl (25%)": np.percentile(final_v, 25),
                        "5. Percentyl": np.percentile(final_v, 5),
                        "Szansa na stratę": f"{(np.sum(final_v < kwota) / n_sims) * 100:.1f}%",
                        "Średni roczny zwrot (CAGR)": f"{((mediana / kwota)**(1/years) - 1)*100:.2f}%"
                    }

                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        st.table(pd.DataFrame(stats.items(), columns=["Metryka", "Wartość"]))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)
                plt.style.use('default')

        with tabs[2]:
            st.subheader("Mapa Korelacji")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")
