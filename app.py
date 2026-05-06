import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 1. IMPORTY TWOICH MODUŁÓW
import styles
import database
import engine

# 2. KONFIGURACJA STRONY (Musi być pierwszą komendą Streamlit!)
try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="vAlpha Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="vAlpha Manager", layout="wide")

# 3. INICJALIZACJA STYLÓW I BAZY
styles.apply_custom_css()
supabase = database.init_supabase()

# --- 4. LOGIKA DOSTĘPU (FREEMIUM) ---
# Sprawdzamy stan sesji. Jeśli nie ma 'user', ustawiamy flagi na False zamiast blokować stronę.
is_logged_in = 'user' in st.session_state

if is_logged_in:
    user_email = st.session_state.user.user.email
    days_left = database.get_pro_days(supabase, user_email)
    is_pro = days_left > 0
else:
    user_email = "Gość"
    days_left = -1
    is_pro = False

# --- SIDEBAR (LOGOWANIE + KONFIGURACJA) ---
with st.sidebar:
    st.title("vAlpha Manager")
    
    if not is_logged_in:
        # SEKCJA DLA GOŚCIA
        st.info("Zaloguj się, aby odblokować zaawansowane modele.")
        with st.popover("🔑 Zaloguj / Rejestracja", use_container_width=True):
            tab_l, tab_r = st.tabs(["Logowanie", "Rejestracja"])
            with tab_l:
                m = st.text_input("E-mail", key="l_mail")
                p = st.text_input("Hasło", type="password", key="l_pw")
                if st.button("Zaloguj", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": m, "password": p})
                        st.session_state.user = res
                        st.rerun()
                    except: st.error("Błąd danych.")
            with tab_r:
                rm = st.text_input("E-mail", key="r_mail")
                rp = st.text_input("Hasło", type="password", key="r_pw")
                if st.button("Załóż konto", use_container_width=True):
                    try:
                        supabase.auth.sign_up({"email": rm, "password": rp})
                        st.success("Konto utworzone! Potwierdź maila.")
                    except: st.error("Błąd rejestracji.")
    else:
        # SEKCJA DLA ZALOGOWANEGO
        st.write(f"Witaj: **{user_email}**")
        if is_pro:
            st.success(f"💎 STATUS: PRO ({days_left} dni)")
        else:
            st.warning("🆓 STATUS: FREE")
            st.link_button("🚀 ODBLOKUJ PRO", f"https://buy.stripe.com/7sYbJ1fft827aVRbPud3i03?prefilled_email={user_email}")
        if st.button("Wyloguj", use_container_width=True):
            del st.session_state.user
            st.rerun()

    st.divider()
    
    # --- INPUTY STANDARDOWE ---
    tickers_input = st.text_input("Tickery:", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał (PLN):", value=25000)
    opt_mode = st.radio("Model:", ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"])
    ryzyko_val = st.select_slider("Ryzyko:", options=['low', 'medium', 'high'])
    limit_2x = st.checkbox("Limit dywersyfikacji (2x)", value=True)
    
    # --- BLOKADA FUNKCJI PRO (UI) ---
    label_min = "📍 Min. udział (PRO) 🔒" if not is_pro else "📍 Min. udział (PRO) 💎"
    constraints_input = st.text_input(label_min, placeholder="NVDA:10", disabled=not is_pro)
    
    st.divider()
    run_mc = st.checkbox("Symulacje Monte Carlo", value=True)
    
    label_adj = "Fat Tails Engine 🔒" if not is_pro else "Fat Tails Engine 💎"
    adj_mc_checkbox = st.checkbox(label_adj, value=False)
    
    # Bezpiecznik: adj_mc musi być False, jeśli nie ma PRO
    adj_mc = False
    rf_rate, mkt_ret, alpha_ret, beta_speed = 0.04, 0.10, 30.0, 0.05

    if adj_mc_checkbox:
        if is_pro:
            adj_mc = True
            with st.expander("PARAMETRY RYNKOWE", expanded=False):
                rf_rate = st.number_input("Rf %:", value=4.0) / 100
                mkt_ret = st.number_input("Rm %:", value=10.0) / 100
                alpha_ret = st.slider("Alfa %:", 0, 100, 30)
                beta_speed = st.slider("Beta Speed:", 0.0, 0.2, 0.05)
        else:
            st.warning("Ta funkcja wymaga konta PRO.")

    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# --- LOGIKA ANALIZY (FREEMIUM ENFORCEMENT) ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # 1. Limit 5 spółek dla darmowych
    if not is_pro and len(tickers) > 5:
        st.error("Wersja FREE obsługuje do 5 spółek.")
        st.stop()

    # 2. Inicjalizacja limitów (constraints tylko dla PRO)
    min_bounds = {t: 0.01 for t in tickers}
    if constraints_input and is_pro:
        for p in constraints_input.split(','):
            try:
                tk, v = p.split(':')
                tk = tk.strip().upper()
                if tk in min_bounds: min_bounds[tk] = float(v)/100
            except: pass

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
