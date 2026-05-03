import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="📊", layout="wide")

# 2. DESIGN CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px;
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .stButton > button {
        width: 100%; background-color: #238636 !important; color: white !important;
        border-radius: 8px; font-weight: 700; height: 3.5em; border: none;
    }
    .disclaimer-box {
        background-color: #1c2128; border-left: 5px solid #d73a49; padding: 15px;
        border-radius: 8px; margin-bottom: 25px; font-size: 0.85em; color: #adbac7;
    }
    .pro-badge {
        background-color: #f1e05a; color: black; padding: 2px 6px; 
        border-radius: 4px; font-size: 0.7em; font-weight: bold; vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. NOTA PRAWNA
st.markdown("""
    <div class="disclaimer-box">
        <strong>WAŻNE INFORMACJE PRAWNE</strong><br>
        Niniejsza aplikacja ma charakter informacyjny i edukacyjny. Nie stanowi porady inwestycyjnej. 
        Inwestowanie wiąże się z ryzykiem utraty kapitału.
    </div>
    """, unsafe_allow_html=True)

# 4. SIDEBAR - SUBSKRYPCJA I KONFIGURACJA
st.title("RISK MANAGER PRO")
with st.sidebar:
    st.subheader("STATUS SUBSKRYPCJI")
    # Prosta weryfikacja klucza (możesz tu wpisać swój sekretny kod)
    license_key = st.text_input("Wprowadź klucz PRO:", type="password", help="Wprowadź klucz, aby odblokować nielimitowane spółki i zaawansowane modele.")
    is_pro = license_key == "PRO2024" # Przykładowy klucz
    
    if is_pro:
        st.success("Wersja PRO aktywna")
    else:
        st.warning("Wersja FREE (Limit 5 spółek)")

    st.divider()
    st.subheader("KONFIGURACJA")
    
    tickers_raw = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, ALE.WA, BTC-USD",
        help="Wpisz symbole oddzielone przecinkiem."
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    opt_mode = st.radio(
        "Tryb Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"]
    )
    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    st.divider()
    run_mc = st.checkbox("Wykonaj symulacje Monte Carlo", value=True)
    
    # Podfunkcja MC - Widoczna tylko jeśli MC zaznaczone
    adj_mc = False
    if run_mc:
        label_adj = "Skorygowana symulacja Monte Carlo"
        if not is_pro:
            label_adj += " (PRO)"
        
        adj_mc = st.checkbox(label_adj, value=False, disabled=not is_pro)
        
        if adj_mc and is_pro:
            with st.expander("PARAMETRY CAPM / GBM", expanded=True):
                rf_rate = st.number_input("Rf %:", value=4.0) / 100
                mkt_ret = st.number_input("Rm %:", value=10.0) / 100
                alpha_retention = st.slider("Utrzymanie Alfy %:", 0, 100, 30)
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05)

    st.divider()
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    
    analizuj = st.button("URUCHOM ANALIZĘ SYSTEMOWĄ")

# 5. WERYFIKACJA LIMITÓW PRZED ANALIZĄ
can_run = True
if not is_pro:
    if len(tickers) > 5:
        st.error(f"Wykryto {len(tickers)} spółek. Wersja darmowa obsługuje maksymalnie 5. Usuń nadmiarowe symbole lub aktywuj wersję PRO.")
        can_run = False
    if adj_mc:
        st.error("Skorygowana symulacja Monte Carlo dostępna jest tylko w wersji PRO.")
        can_run = False

# 6. GŁÓWNA LOGIKA
if analizuj and can_run:
    try:
        with st.spinner('PRZETWARZANIE...'):
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

        # TABS
        tabs = st.tabs(["STRUKTURA", "MONTE CARLO", "KORELACJE", "METODOLOGIA"])

        with tabs[0]:
            st.subheader(f"ALOKACJA: {opt_mode.upper()}")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi * monthly_vars).sum()
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
            c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
            
            df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
            if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
            st.dataframe(df_out.sort_values(by='Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), hide_index=True)

        if run_mc:
            with tabs[1]:
                st.subheader("SYMULACJA 10,000 SCENARIUSZY")
                n_sims, dt = 10000, 1/252
                log_rets = np.log(data_only / data_only.shift(1)).dropna()
                p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
                
                col_a, col_b = st.columns(2)
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
                        "Metryka": ["95. Percentyl", "Mediana", "5. Percentyl", "Prawdopodobieństwo straty", "CAGR"],
                        "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.median(final):,.2f}", f"{np.percentile(final, 5):,.2f}", 
                                    f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((np.median(final) / kwota)**(1/y) - 1)*100:.2f}%"]
                    })
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### {lbl}")
                        st.table(res_df)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(paths[:, :100], color='#238636', alpha=0.06)
                        ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                        st.pyplot(fig)

        with tabs[2]:
            fig_c, ax_c = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

        with tabs[3]:
            st.header("METODOLOGIA")
            st.markdown("Szczegółowy opis algorytmów VaR, Sortino oraz modeli GBM/CAPM zastosowanych w aplikacji.")

    except Exception as e:
        st.error(f"Błąd: {e}")
