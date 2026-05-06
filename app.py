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
    
    market_params = {'rf': 0.04, 'rm': 0.10, 'alpha': 0.3, 'speed': 0.05}
    if adj_mc and is_pro:
        with st.expander("PARAMETRY RYNKOWE", expanded=False):
            market_params['rf'] = st.number_input("Rf %:", value=4.0) / 100
            market_params['rm'] = st.number_input("Rm %:", value=10.0) / 100
            market_params['alpha'] = st.slider("Alfa %:", 0, 100, 30) / 100
            market_params['speed'] = st.slider("Beta Speed:", 0.0, 0.2, 0.05)

    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# --- ANALIZA (w app.py) ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    # ... (logika limitów bez zmian) ...

    with st.spinner('Analizowanie...'):
        try:
            # Pobieramy dane (SPY tylko jeśli adj_mc jest True)
            fetch_list = tickers + (["SPY"] if adj_mc else [])
            data = engine.get_data(tuple(fetch_list))
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
            
            data_only = data[tickers]
            daily_rets, monthly_rets, monthly_vars, corr_matrix = engine.get_portfolio_stats(data_only)
            
            # Przygotowujemy zmienne dla silnika (nawet puste, jeśli nie ma adj_mc)
            betas, alphas = {}, {}
            # Domyślne wartości, żeby funkcja nie "wybuchła"
            r_f, m_r, a_r, b_s = 0.0, 0.0, 0.0, 0.0 

            if adj_mc:
                r_f, m_r, a_r, b_s = rf_rate, mkt_ret, alpha_ret, beta_speed
                spy_rets = data["SPY"].pct_change().dropna()
                spy_annual = (1 + spy_rets.mean())**252 - 1
                for t in tickers:
                    t_rets = data_only[t].pct_change().dropna()
                    comb = pd.concat([t_rets, spy_rets], axis=1).dropna()
                    b = np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1])
                    betas[t] = b
                    hist_ret = (1 + t_rets.mean())**252 - 1
                    alphas[t] = hist_ret - (r_f + b * (spy_annual - r_f))

            wagi = engine.optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x)

            # --- WYŚWIETLANIE TABÓW ---
            t1, t2, t3, t4 = st.tabs(["Struktura Portfela", "Monte Carlo", "Korelacja", "Metodologia"])
            
            with t1:
                st.subheader("Rekomendowana Alokacja")
                p_var = (wagi * monthly_vars).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.0f} PLN")
                
                df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi*100, 'PLN': wagi*kwota})
                st.dataframe(
                    df_out.sort_values('Udział (%)', ascending=False).style.format({
                        'Udział (%)': '{:.2f}%', 
                        'PLN': '{:,.2f} PLN'
                    }), 
                    use_container_width=True, 
                    hide_index=True
                )

           with t2:
                if run_mc:
                    # 1. WYWOŁANIE SILNIKA
                    # Przekazujemy komplet danych. Jeśli adj_mc jest False, r_f itd. są zerami, 
                    # a silnik sam przełączy się na tryb Standard (Gauss).
                    mc_data = engine.run_monte_carlo(
                        data_only, wagi, kwota, tickers, adj_mc, 
                        r_f, m_r, a_r, b_s, betas, alphas
                    )
                    
                    st.subheader("Symulacja Ścieżek Portfela")
                    if adj_mc:
                        st.info("🚀 Tryb: **Fat Tails Engine** (CAPM + Rozkład t-Studenta)")
                    else:
                        st.info("📊 Tryb: **Standard Monte Carlo** (Historyczny dryf + Rozkład Gaussa)")
                    
                    col_a, col_b = st.columns(2)
                    plt.style.use("dark_background") # Gwarantuje ciemny motyw wykresów
                    
                    for i, (y, lbl) in enumerate(zip([5, 10], ["PERSPEKTYWA: 5 LAT", "PERSPEKTYWA: 10 LAT"])):
                        paths = mc_data[y]['paths']
                        stats_df = mc_data[y]['stats']
                        
                        with (col_a if i == 0 else col_b):
                            st.write(f"#### {lbl}")
                            
                            # Tabela statystyk (Percentyle, CAGR, Prawd. straty)
                            st.table(stats_df)
                            
                            # Generowanie wykresu
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Wizualizacja chmury (50 losowych ścieżek dla czytelności)
                            ax.plot(paths[:, :50], alpha=0.15, color='#238636')
                            
                            # Linia mediany (Biała, grubsza)
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=3, label="Mediana")
                            
                            # Stylizacja techniczna
                            ax.set_facecolor('#0d1117')
                            fig.patch.set_facecolor('#0d1117')
                            ax.set_ylabel("Wartość portfela (PLN)")
                            ax.set_xlabel("Dni handlowe")
                            ax.grid(True, alpha=0.1, linestyle='--')
                            
                            st.pyplot(fig)
                            
                else:
                    st.info("Symulacje Monte Carlo są wyłączone. Możesz je włączyć w panelu bocznym.")
            with t3:
                st.subheader("Macierz Korelacji Składników")
                fig_c, ax_c = plt.subplots(figsize=(10, 8))
                # Wykorzystanie seaborn dla czytelnej macierzy korelacji
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c, cbar=True)
                fig_c.patch.set_facecolor('#0d1117')
                ax_c.set_title("Korelacja Miesięcznych Stóp Zwrotu", color="white")
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
