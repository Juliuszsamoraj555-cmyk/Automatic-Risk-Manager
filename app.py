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
    
    # DEFINIUJEMY DOMYŚLNE WARTOŚCI (żeby zawsze istniały)
    rf_rate = 0.04
    mkt_ret = 0.10
    alpha_ret = 30
    beta_speed = 0.05

    if adj_mc and is_pro:
        with st.expander("PARAMETRY RYNKOWE", expanded=False):
            rf_rate = st.number_input("Rf % (Stopa wolna od ryzyka):", value=4.0) / 100
            mkt_ret = st.number_input("Rm % (Oczekiwany zwrot rynku):", value=10.0) / 100
            alpha_ret = st.slider("Alfa % (Utrzymanie przewagi):", 0, 100, 30)
            beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05)

    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# --- 5. LOGIKA ANALIZY (w app.py) ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # 1. Sprawdzenie limitu wersji FREE
    if not is_pro and len(tickers) > 5:
        st.error("Wersja FREE obsługuje do 5 spółek. Twoja lista ma więcej pozycji.")
        st.stop()

    # 2. Parsowanie limitów (Smart Constraints)
    min_bounds = {t: 0.01 for t in tickers}
    if constraints_input and is_pro:
        for p in constraints_input.split(','):
            try:
                t, v = p.split(':')
                tk = t.strip().upper()
                if tk in min_bounds: min_bounds[tk] = float(v)/100
            except: st.warning(f"Błędny format limitu: {p}")
        if sum(min_bounds.values()) > 1.0:
            st.error("Suma minimalnych udziałów przekracza 100%!")
            st.stop()

    # 3. PROCES ANALIZY
    with st.spinner('Pobieranie danych rynkowych i przeliczanie...'):
        try:
            # POBIERANIE DANYCH (Tu powstaje zmienna 'data')
            fetch_list = tickers + (["SPY"] if adj_mc else [])
            data = engine.get_data(tuple(fetch_list))
            
            if data is None or data.empty:
                st.error("Nie udało się pobrać danych dla podanych tickerów.")
                st.stop()
            
            # Naprawa MultiIndex (jeśli yfinance go wygenerował)
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(-1)
            
            # Wyciąganie danych tylko dla wybranych spółek (bez SPY)
            data_only = data[tickers]
            
            # Pobieranie statystyk z silnika
            daily_rets, monthly_rets, monthly_vars, corr_matrix = engine.get_portfolio_stats(data_only)
            
            # Obliczanie Bety i Alfy (tylko dla Fat Tails Engine)
            betas, alphas = {}, {}
            if adj_mc:
                spy_rets = data["SPY"].pct_change().dropna()
                spy_annual = (1 + spy_rets.mean())**252 - 1
                for t in tickers:
                    t_rets = data_only[t].pct_change().dropna()
                    comb = pd.concat([t_rets, spy_rets], axis=1).dropna()
                    b = np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1])
                    betas[t] = b
                    hist_ret = (1 + t_rets.mean())**252 - 1
                    alphas[t] = hist_ret - (rf_rate + b * (spy_annual - rf_rate))

            # Optymalizacja wag (engine.py)
            wagi = engine.optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x)

            # --- WYŚWIETLANIE WYNIKÓW (Taby) ---
            t1, t2, t3, t4 = st.tabs(["Struktura Portfela", "Monte Carlo", "Korelacja", "Metodologia"])
            
            with t1:
                st.subheader("Rekomendowana Alokacja")
                p_var = (wagi * monthly_vars).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.0f} PLN")
                
                df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi*100, 'PLN': wagi*kwota})
                st.dataframe(df_out.sort_values('Udział (%)', ascending=False).style.format({
                    'Udział (%)': '{:.2f}%', 'PLN': '{:,.2f} PLN'
                }), use_container_width=True, hide_index=True)

            with t2:
                if run_mc:
                    mc_data = engine.run_monte_carlo(
                        data_only, wagi, kwota, tickers, adj_mc, 
                        rf_rate, mkt_ret, alpha_ret, beta_speed, betas, alphas
                    )
                    # ... (tutaj kod wykresów - ten co miałeś, bo jest dobry) ...
                    # UWAGA: Upewnij się, że kod wykresów jest wcięty pod "if run_mc:"
                    col_a, col_b = st.columns(2)
                    plt.style.use("dark_background")
                    for i, (y, lbl) in enumerate(zip([5, 10], ["5 LAT", "10 LAT"])):
                        paths = mc_data[y]['paths']
                        with (col_a if i==0 else col_b):
                            st.write(f"#### {lbl}")
                            st.table(mc_data[y]['stats'])
                            fig, ax = plt.subplots()
                            ax.plot(paths[:, :50], alpha=0.15, color='#238636')
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                            ax.set_facecolor('#0d1117')
                            fig.patch.set_facecolor('#0d1117')
                            st.pyplot(fig)

            with t3:
                fig_c, ax_c = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_c)
                fig_c.patch.set_facecolor('#0d1117')
                st.pyplot(fig_c)

        
                
            with t4:
                st.header("Metodologia vAlpha Engine")
                st.write("""
                * **Model ryzyka:** Wykorzystujemy proces stochastyczny z rozkładem t-Studenta (Fat Tails), co pozwala lepiej szacować ryzyko krachów rynkowych niż standardowy rozkład normalny.
                * **Optymalizacja:** Algorytm SLSQP minimalizuje odchylenie od celu (VaR lub Sortino) przy zachowaniu Twoich restrykcji.
                * **Dostosowanie rynkowe:** Przy włączonej opcji 'Adjust MC', symulacja uwzględnia dryft wynikający z Bety portfela oraz historycznej Alfy.
                """)

        except Exception as e:
            st.error(f"Błąd analizy: Proszę sprawdzić poprawność tickerów. Szczegóły: {e}")
