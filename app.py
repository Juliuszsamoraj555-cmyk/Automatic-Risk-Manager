import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os
from supabase import create_client, Client

# --- 1. KONFIGURACJA SUPABASE (ZABEZPIECZONA) ---
# Pobieramy surowe dane z Environment Variables
url_raw = os.environ.get("SUPABASE_URL")
key_raw = os.environ.get("SUPABASE_KEY")

# Mechanizm czyszczenia adresu URL (usuwa /rest/v1, spacje i końcowe ukośniki)
if url_raw:
    url = url_raw.split("/rest/v1")[0].strip().rstrip("/")
else:
    url = None

key = key_raw.strip() if key_raw else None

# Sprawdzenie czy klucze w ogóle istnieją
if not url or not key:
    st.error("BŁĄD: Brak kluczy Supabase w ustawieniach Environment na Renderze!")
    st.stop()

# Inicjalizacja klienta
try:
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"Błąd inicjalizacji bazy danych: {e}")
    st.stop()

# --- 2. CACHE DANYCH ---
@st.cache_data(ttl=3600)
def get_data_cached(tickers_tuple):
    return yf.download(list(tickers_tuple), period="3y")['Close']

# --- 3. KONFIGURACJA STRONY ---
try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="Valpha Portfolio Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="Valpha Portfolio Manager", layout="wide")

# --- 4. DESIGN CSS ---
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

# --- 5. FUNKCJE POMOCNICZE ---
def check_pro_status(email):
    try:
        res = supabase.table("profiles").select("is_pro").eq("email", email).single().execute()
        return res.data['is_pro'] if res.data else False
    except:
        return False

# --- 6. LOGIKA LOGOWANIA / EKRAN STARTOWY ---
if 'user' not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚀 vAlpha Terminal")
        st.write("Zaloguj się, aby uzyskać dostęp do profesjonalnych narzędzi zarządzania ryzykiem.")
        
        tab1, tab2 = st.tabs(["Logowanie", "Rejestracja"])
        
        with tab1:
            login_email = st.text_input("E-mail", key="login_email")
            login_password = st.text_input("Hasło", type="password", key="login_pw")
            if st.button("Zaloguj się", key="login_btn"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": login_email, "password": login_password})
                    st.session_state.user = res
                    st.rerun()
                except Exception as e:
                    st.error("Nieprawidłowy e-mail lub hasło.")

        with tab2:
            reg_email = st.text_input("Podaj e-mail", key="reg_email")
            reg_password = st.text_input("Ustaw hasło (min. 6 znaków)", type="password", key="reg_pw")
            if st.button("Załóż darmowe konto", key="reg_btn"):
                try:
                    supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                    st.success("Konto utworzone! Sprawdź skrzynkę e-mail, aby potwierdzić rejestrację.")
                except Exception as e:
                    st.error(f"Błąd rejestracji: {e}")
    st.stop()

# --- 7. DANE ZALOGOWANEGO UŻYTKOWNIKA ---
user_email = st.session_state.user.user.email
is_pro = check_pro_status(user_email)

# --- 8. SIDEBAR ---
with st.sidebar:
    st.title("vAlpha Manager")
    st.write(f"Zalogowany: **{user_email}**")
    
    if is_pro:
        st.success("💎 STATUS: PRO")
    else:
        st.warning("🆓 STATUS: FREE")
        if st.button("🚀 ODBLOKUJ WERSJĘ PRO"):
            st.info("Przekierowanie do płatności Stripe (Wkrótce)")

    if st.button("Wyloguj"):
        supabase.auth.sign_out()
        del st.session_state.user
        st.rerun()

    st.divider()
    st.subheader("KONFIGURACJA PORTFELA")
    tickers_input = st.text_input("Symbole spółek (ticker):", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał początkowy (PLN):", value=25000, step=1000)
    opt_mode = st.radio("Model Optymalizacji:", ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"])
    ryzyko_val = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Wymuś dywersyfikację (Limit 2x)", value=True)
    run_mc = st.checkbox("Wykonaj symulacje Monte Carlo", value=True)
    
    adj_mc = False
    if run_mc:
        label_adj = "Skorygowana symulacja Monte Carlo"
        if not is_pro: label_adj += " (Wymaga PRO)"
        adj_mc = st.checkbox(label_adj, value=False, disabled=not is_pro)
        
        if adj_mc and is_pro:
            with st.expander("PARAMETRY RYNKOWE CAPM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_ret = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30)
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05)

    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ SYSTEMOWĄ")

# --- 9. LOGIKA ANALIZY ---
st.markdown('<div class="disclaimer-red"><strong>WAŻNE:</strong> Aplikacja edukacyjna. Inwestowanie wiąże się z ryzykiem.</div>', unsafe_allow_html=True)

if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not is_pro and len(tickers) > 5:
        st.error(f"Wersja darmowa obsługuje max. 5 spółek. Aktywuj PRO dla większej liczby.")
    else:
        try:
            with st.spinner('Analizowanie danych...'):
                fetch_list = tickers + (["SPY"] if adj_mc else [])
                data = get_data_cached(tuple(fetch_list))
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(-1)
                
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

            tabs = st.tabs(["Struktura", "Monte Carlo", "Korelacja", "Metodologia"])
            with tabs[0]:
                c1, c2, c3 = st.columns(3)
                p_var = (wagi * monthly_vars).sum()
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
                st.dataframe(pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota}), hide_index=True)

            if run_mc:
                with tabs[1]:
                    n_sims, dt = 3000, 1/252
                    p_sigma = np.sqrt(np.dot(wagi.T, np.dot(np.log(data_only / data_only.shift(1)).dropna().cov().values, wagi))) * np.sqrt(252)
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
                        with (col_a if i == 0 else col_b):
                            st.write(f"#### PERSPEKTYWA: {lbl}")
                            st.metric("Mediana", f"{np.median(paths[-1, :]):,.2f} PLN")
                            fig, ax = plt.subplots()
                            ax.plot(paths[:, :50], color='#238636', alpha=0.1)
                            ax.plot(np.median(paths, axis=1), color='white')
                            st.pyplot(fig)
            with tabs[2]:
                fig_c, ax_c = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_c)
                st.pyplot(fig_c)
        except Exception as e:
            st.error(f"Błąd: {e}")
