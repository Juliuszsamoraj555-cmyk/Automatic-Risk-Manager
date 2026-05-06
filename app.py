import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import styles
import database
import engine
# 2. TO MUSI BYĆ PIERWSZA KOMENDA STREAMLIT W PLIKU!
try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="vAlpha Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="vAlpha Manager", layout="wide")

# --- INICJALIZACJA ---
styles.apply_custom_css()
supabase = database.init_supabase()

try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="vAlpha Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="vAlpha Manager", layout="wide")

# --- LOGOWANIE ---
if 'user' not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚀 vAlpha Terminal")
        t1, t2 = st.tabs(["Logowanie", "Rejestracja"])
        with t1:
            m = st.text_input("E-mail")
            p = st.text_input("Hasło", type="password")
            if st.button("Zaloguj"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": m, "password": p})
                    st.session_state.user = res
                    st.rerun()
                except: st.error("Błąd logowania.")
    st.stop()

user_email = st.session_state.user.user.email
days_left = database.get_pro_days(supabase, user_email)
is_pro = days_left > 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("vAlpha Manager")
    if is_pro: st.success(f"💎 PRO: {days_left} dni")
    else: st.warning("🆓 FREE"); st.link_button("🚀 KUP PRO", f"https://buy.stripe.com/7sYbJ1fft827aVRbPud3i03?prefilled_email={user_email}")
    
    tickers_input = st.text_input("Tickery:", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał (PLN):", value=25000)
    opt_mode = st.radio("Model:", ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"])
    ryzyko_val = st.select_slider("Ryzyko:", options=['low', 'medium', 'high'])
    limit_2x = st.checkbox("Limit dywersyfikacji (2x)", value=True)
    constraints_input = st.text_input("📍 Min. udział (PRO)", placeholder="NVDA:10", disabled=not is_pro)
    
    run_mc = st.checkbox("Symulacje Monte Carlo", value=True)
    adj_mc = st.checkbox("Fat Tails Engine", value=False, disabled=not is_pro)
    
    market_params = {'rf': 0.04, 'rm': 0.10, 'alpha': 0.3, 'speed': 0.05}
    if adj_mc and is_pro:
        with st.expander("PARAMETRY RYNKOWE", expanded=False):
            market_params['rf'] = st.number_input("Rf %:", value=4.0) / 100
            market_params['rm'] = st.number_input("Rm %:", value=10.0) / 100
            market_params['alpha'] = st.slider("Alfa %:", 0, 100, 30) / 100
            market_params['speed'] = st.slider("Beta Speed:", 0.0, 0.2, 0.05)

    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# --- ANALIZA ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not is_pro and len(tickers) > 5:
        st.error("Wersja FREE obsługuje do 5 spółek.")
        st.stop()

    min_bounds = {t: 0.01 for t in tickers}
    if constraints_input and is_pro:
        for p in constraints_input.split(','):
            try:
                t, v = p.split(':')
                tk = t.strip().upper()
                if tk in min_bounds: min_bounds[tk] = float(v)/100
            except: pass

    with st.spinner('Analizowanie...'):
        try:
            data = engine.get_data(tuple(tickers + (["SPY"] if adj_mc else [])))
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
            
            data_only = data[tickers]
            daily_rets, monthly_rets, monthly_vars, corr_matrix = engine.get_portfolio_stats(data_only)
            
            if adj_mc:
                spy_rets = data["SPY"].pct_change().dropna()
                betas = []
                for t in tickers:
                    comb = pd.concat([daily_rets[t], spy_rets], axis=1).dropna()
                    b = np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1])
                    betas.append(b)
                market_params['beta'] = np.mean(betas)

            wagi = engine.optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x)

            # TABY Z WYNIKAMI
            t1, t2, t3, t4 = st.tabs(["Portfel", "Monte Carlo", "Korelacja", "Metodologia"])
            
            with t1:
                st.subheader("Optymalna Struktura")
                p_var = (wagi * monthly_vars).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Miesięczny VaR", f"{p_var*100:.2f}%")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.0f}")
                st.dataframe(pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi*100, 'PLN': wagi*kwota}), use_container_width=True)

            with t2:
                if run_mc:
                    mc_results = engine.run_monte_carlo(data_only, wagi, kwota, adj_mc, market_params)
                    col_a, col_b = st.columns(2)
                    for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                        paths = mc_results[y]
                        final = paths[-1, :]
                        with (col_a if i==0 else col_b):
                            st.write(f"#### {lbl}")
                            st.write(f"Mediana: **{np.median(final):,.0f} PLN**")
                            fig, ax = plt.subplots()
                            ax.plot(paths[:, :50], alpha=0.1, color='#238636')
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                            plt.style.use("dark_background")
                            st.pyplot(fig)

            with t3:
                fig_c, ax_c = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_c)
                st.pyplot(fig_c)
                
            with t4:
                st.write("vAlpha Engine wykorzystuje rozkład t-Studenta (Fat Tails) dla realizmu rynkowego.")

        except Exception as e:
            st.error(f"Błąd analizy: {e}")
