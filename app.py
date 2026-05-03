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
    st.set_page_config(page_title="Risk Manager Pro", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="Risk Manager Pro", layout="wide")
    st.error("Błąd: Nie znaleziono pliku logo 'image_8.png'. Upewnij się, że plik graficzny jest w głównym katalogu na GitHubie.")

# 2. DESIGN CSS (SaaS Tech Look)
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
    .disclaimer-box {
        background-color: #1c2128; border-left: 5px solid #d73a49; padding: 15px;
        border-radius: 8px; margin-bottom: 25px; font-size: 0.85em; color: #adbac7;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. NOTA PRAWNA
st.markdown("""
    <div class="disclaimer-box">
        <strong>WAŻNE INFORMACJE PRAWNE</strong><br>
        Niniejsza aplikacja ma charakter wyłącznie informacyjny oraz edukacyjny i nie stanowi rekomendacji inwestycyjnej ani porady finansowej w rozumieniu przepisów prawa. 
        Inwestowanie na rynkach kapitałowych wiąże się z ryzykiem utraty części lub całości kapitału. Autor narzędzia nie ponosi odpowiedzialności za decyzje inwestycyjne podjęte 
        na podstawie wyświetlanych danych. Przed podjęciem jakichkolwiek działań skonsultuj się z licencjonowanym doradcą finansowym.
    </div>
    """, unsafe_allow_html=True)

# 4. SIDEBAR - SUBSKRYPCJA I KONFIGURACJA
with st.sidebar:
    try:
        st.image(v_alpha_icon, width=80)
    except:
        pass
    st.title("RISK MANAGER PRO")
    
    st.subheader("STATUS SUBSKRYPCJI")
    license_key = st.text_input("Klucz licencyjny PRO:", type="password")
    is_pro = (license_key == "PRO2024") # Twój klucz dostępu
    
    if is_pro:
        st.success("WERSJA PRO AKTYWNA")
    else:
        st.warning("WERSJA FREE (LIMIT: 5 SPÓŁEK)")

    st.divider()
    st.subheader("PARAMETRY WEJŚCIOWE")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, NVDA, TSLA, AMZN",
        help="Wpisuj symbole oddzielone przecinkiem. Giełda Polska: .WA, USA: sam ticker."
    )
    
    kwota = st.number_input("Kapitał początkowy (PLN):", value=25000, step=1000)
    
    st.divider()
    opt_mode = st.radio(
        "Model Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        help="VaR skupia się na stabilności. Sortino szuka zysku przy ograniczaniu spadków."
    )
    
    ryzyko_val = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    
    run_mc = st.checkbox("Wykonaj symulacje Monte Carlo", value=True)
    
    adj_mc = False
    if run_mc:
        label_adj = "Skorygowana symulacja Monte Carlo"
        if not is_pro:
            label_adj += " (Dostępne w PRO)"
        
        adj_mc = st.checkbox(label_adj, value=False, disabled=not is_pro)
        
        if adj_mc and is_pro:
            with st.expander("PARAMETRY CAPM / GBM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_ret = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30)
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05)

    st.divider()
    analizuj = st.button("URUCHOM ANALIZĘ SYSTEMOWĄ")

# 5. LOGIKA ANALIZY
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # Walidacja PRO/FREE
    can_proceed = True
    if not is_pro and len(tickers) > 5:
        st.error(f"Wykryto {len(tickers)} spółek. Wersja darmowa obsługuje maksymalnie 5. Aktywuj wersję PRO.")
        can_proceed = False
    
    if can_proceed:
        try:
            with st.spinner('Pobieranie i przetwarzanie danych...'):
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

            # Optymalizacja
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

            # TABS
            tabs = st.tabs(["STRUKTURA PORTFELA", "SYMULACJA MONTE CARLO", "KORELACJE", "METODOLOGIA"])

            with tabs[0]:
                st.subheader(f"REKOMENDOWANA ALOKACJA: {opt_mode.upper()}")
                c1, c2, c3 = st.columns(3)
                p_var = (wagi * monthly_vars).sum()
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
                
                df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
                if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
                st.dataframe(df_out.sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True, use_container_width=True)

            if run_mc:
                with tabs[1]:
                    st.subheader("SYMULACJA MONTE CARLO (10,000 ŚCIEŻEK)")
                    st.info("Symulacja wykorzystuje model Geometric Brownian Motion (GBM).")
                    
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
                            "Metryka": ["95. Percentyl", "3. Kwartyl (75%)", "Mediana", "1. Kwartyl (25%)", "5. Percentyl", "Szansa na stratę", "CAGR"],
                            "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 75):,.2f}", f"{med:,.2f}", f"{np.percentile(final, 25):,.2f}", 
                                        f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((med / kwota)**(1/y) - 1)*100:.2f}%"]
                        })
                        with (col_a if i == 0 else col_b):
                            st.write(f"#### PERSPEKTYWA: {lbl}")
                            st.table(res_df)
                            fig, ax = plt.subplots()
                            ax.plot(paths[:, :100], color='#238636', alpha=0.06)
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                            st.pyplot(fig)
                    plt.style.use('default')

            with tabs[2]:
                st.subheader("MACIERZ KORELACJI AKTYWÓW")
                fig_c, ax_c = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
                st.pyplot(fig_c)

            with tabs[3]:
                st.header("🧠 METODOLOGIA OBLICZEŃ")
                with st.expander("1. OPTYMALIZACJA PORTFELA", expanded=True):
                    st.markdown("""
                    **Model VaR-First**: Alokacja odwrotnie proporcjonalna do ryzyka i korelacji:
                    $$W_i \\propto \\frac{1 - \\bar{\\rho}_i}{VaR_i^p}$$
                    **Model Sortino**: Maksymalizacja zysku w stosunku do zmienności spadkowej:
                    $$W_i \\propto \\left(\\frac{R_i - R_f}{\\sigma_{downside}}\\right)^p \\cdot (1 - \\bar{\\rho}_i)$$
                    """)
                with st.expander("2. SYMULACJA MONTE CARLO (GBM + CAPM)"):
                    st.markdown("""
                    **Oczekiwana Stopa Zwrotu (CAPM + Alfa)**:
                    $$E(R_i) = R_f + \\beta_i(E(R_m) - R_f) + \\alpha_{adj}$$
                    **Korekta Dryfu (Volatility Drag)**:
                    $$\\mu_{adj} = E(R_i) - \\frac{1}{2}\\sigma^2$$
                    **Równanie Geometrycznych Ruchów Browna**:
                    $$P_{t+1} = P_t \\cdot e^{(\\mu_{adj} \\Delta t + \\sigma \\epsilon \\sqrt{\\Delta t})}$$
                    """)

        except Exception as e:
            st.error(f"Błąd systemu: {e}")
