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
import datetime

# --- 1. PANCERNA KONFIGURACJA SUPABASE ---
url_raw = os.environ.get("SUPABASE_URL")
key_raw = os.environ.get("SUPABASE_KEY")

if url_raw:
    url = url_raw.split("/rest/v1")[0].strip().rstrip("/")
else:
    url = None
key = key_raw.strip() if key_raw else None

if not url or not key:
    st.error("KRYTYCZNY BŁĄD: Brak kluczy Supabase w Environment Variables!")
    st.stop()

try:
    supabase: Client = create_client(url, key)
except Exception as e:
    st.error(f"Błąd połączenia z bazą: {e}")
    st.stop()

# --- 2. OPTYMALIZACJA: CACHE DATA ---
@st.cache_data(ttl=3600)
def get_data_cached(tickers_tuple):
    return yf.download(list(tickers_tuple), period="3y")['Close']

# --- 3. KONFIGURACJA STRONY ---
try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="Valpha Portfolio Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="Valpha Portfolio Manager", layout="wide")

# --- 4. PEŁNY DESIGN CSS ---
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

# --- 5. LOGIKA DOSTĘPU CZASOWEGO ---
def get_pro_info(email):
    try:
        res = supabase.table("profiles").select("pro_until").eq("email", email).single().execute()
        if res.data and res.data['pro_until']:
            pro_until = datetime.datetime.fromisoformat(res.data['pro_until'].replace('Z', '+00:00'))
            now = datetime.datetime.now(datetime.timezone.utc)
            if pro_until > now:
                delta = pro_until - now
                return delta.days + 1
        return -1
    except:
        return -1

# --- 6. LOGIKA LOGOWANIA ---
if 'user' not in st.session_state:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🚀 vAlpha Terminal")
        tab1, tab2 = st.tabs(["Logowanie", "Rejestracja"])
        with tab1:
            l_email = st.text_input("E-mail", key="l_mail")
            l_pw = st.text_input("Hasło", type="password", key="l_pw")
            if st.button("Zaloguj się"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": l_email, "password": l_pw})
                    st.session_state.user = res
                    st.rerun()
                except: st.error("Błąd logowania.")
        with tab2:
            r_email = st.text_input("Twój e-mail", key="r_mail")
            r_pw = st.text_input("Hasło", type="password", key="r_pw")
            if st.button("Załóż konto"):
                try:
                    supabase.auth.sign_up({"email": r_email, "password": r_pw})
                    st.success("Konto utworzone! Potwierdź maila.")
                except Exception as e: st.error(f"Błąd: {e}")
    st.stop()

# --- 7. STATUS UŻYTKOWNIKA ---
user_email = st.session_state.user.user.email
days_left = get_pro_info(user_email)
is_pro = days_left > 0

# --- 8. DISCLAIMER ---
st.markdown('<div class="disclaimer-red"><strong>WAŻNE:</strong> System edukacyjny. Nie stanowi porady inwestycyjnej.</div>', unsafe_allow_html=True)

# --- 9. SIDEBAR ---
with st.sidebar:
    try: st.image(v_alpha_icon, width=100)
    except: pass
    st.title("Valpha Manager")
    st.write(f"Witaj: **{user_email}**")
    
    if is_pro:
        st.success(f"💎 STATUS: PRO ({days_left} dni)")
    else:
        st.warning("🆓 STATUS: FREE (Limit: 5 spółek)")
        st.link_button("🚀 ODBLOKUJ PRO", f"https://buy.stripe.com/7sYbJ1fft827aVRbPud3i03?prefilled_email={user_email}")

    if st.button("Wyloguj"):
        supabase.auth.sign_out()
        del st.session_state.user
        st.rerun()

    st.divider()
    tickers_input = st.text_input("Tickery (np. AAPL, NVDA):", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał (PLN):", value=25000)
    
    opt_mode = st.radio("Model Optymalizacji:", ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"])
    ryzyko_val = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Limit dywersyfikacji (2x)", value=True)

    st.write("---")
    label_min = "📍 Min. udział (np. NVDA:10) 🔒" if not is_pro else "📍 Min. udział (np. NVDA:10)"
    constraints_input = st.text_input(label_min, placeholder="TICKER:PROCENT", disabled=not is_pro)

    st.divider()
    run_mc = st.checkbox("Symulacje Monte Carlo", value=True)
    adj_mc = False
    if run_mc:
        label_adj = "Fat Tails Engine 🔒" if not is_pro else "Fat Tails Engine"
        adj_mc = st.checkbox(label_adj, value=False, disabled=not is_pro)
        if adj_mc and is_pro:
            with st.expander("PARAMETRY RYNKOWE", expanded=False):
                rf_rate = st.number_input("Rf %:", value=4.0) / 100
                mkt_ret = st.number_input("Rm %:", value=10.0) / 100
                alpha_ret = st.slider("Alfa %:", 0, 100, 30)
                beta_speed = st.slider("Beta Speed:", 0.0, 0.2, 0.05)

    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# --- 10. GŁÓWNA LOGIKA ANALIZY ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # 1. Logika limitów (Smart Constraints)
    min_bounds = {t: 0.01 for t in tickers}
    if constraints_input and is_pro:
        parts = [p.strip() for p in constraints_input.split(',')]
        for p in parts:
            try:
                t_name, val = p.split(':')
                t_name = t_name.strip().upper()
                if t_name in min_bounds:
                    min_bounds[t_name] = float(val) / 100
            except: st.warning(f"Błąd formatu: {p}")
        if sum(min_bounds.values()) > 1.0:
            st.error("Suma limitów > 100%!")
            st.stop()

    if not is_pro and len(tickers) > 5:
        st.error("Wersja FREE obsługuje do 5 spółek.")
    else:
        try:
            with st.spinner('Przeliczanie danych...'):
                # Pobieranie danych
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
                        alphas[t] = ((1 + t_rets.mean())**252 - 1) - (rf_rate + b * (spy_annual - rf_rate))
                    data_only = stock_data
                else:
                    data_only = data[tickers]

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
                c_bounds = [(min_bounds[t], 1.0) for t in tickers]
                
                res = minimize(lambda w: np.sum((w - target_w.values)**2), target_w.values, 
                               method='SLSQP', bounds=c_bounds,
                               constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] + 
                               ([{'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)}] if limit_2x else []))
                wagi = res.x

                # WIDOK WYNIKÓW
                tabs = st.tabs(["Struktura Portfela", "Symulacja Monte Carlo", "Korelacja", "Metodologia"])

                with tabs[0]:
                    st.subheader(f"Rekomendowana alokacja ({opt_mode})")
                    c1, c2, c3 = st.columns(3)
                    p_var = (wagi * monthly_vars).sum()
                    c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%")
                    c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}")
                    c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
                    
                    df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
                    if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
                    st.dataframe(df_out.sort_values('Udział (%)', ascending=False), use_container_width=True, hide_index=True)

                with tabs[1]:
                    if run_mc:
                        st.subheader("Symulacja Monte Carlo - Fat Tails Engine")
                        n_sims, dt = 3000, 1/252
                        nu = 4 # Stopnie swobody
                        t_scale = np.sqrt((nu - 2) / nu)
                        log_rets = np.log(data_only / data_only.shift(1)).dropna()
                        p_sigma = np.sqrt(np.dot(wagi.T, np.dot(log_rets.cov().values, wagi))) * np.sqrt(252)
                        
                        col_a, col_b = st.columns(2)
                        for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                            days = y * 252
                            paths = np.zeros((days, n_sims))
                            curr = np.full(n_sims, float(kwota))
                            
                            p_alpha = np.sum([alphas[t] * wagi[idx] for idx, t in enumerate(tickers)]) * (alpha_ret / 100) if adj_mc else 0
                            t_beta = np.sum([betas[t] * wagi[idx] for idx, t in enumerate(tickers)]) if adj_mc else 1
                            
                            for d in range(days):
                                eps = np.random.standard_t(df=nu, size=n_sims) * t_scale
                                if adj_mc:
                                    mu = (rf_rate + t_beta * (mkt_ret - rf_rate) + p_alpha - 0.5 * (p_sigma**2)) * dt
                                    if d % 252 == 0: t_beta = t_beta * (1 - beta_speed) + 1.0 * beta_speed
                                else:
                                    mu = (np.sum(daily_rets.mean() * wagi) * 252 - 0.5 * (p_sigma**2)) * dt
                                
                                curr *= np.exp(mu + p_sigma * eps * np.sqrt(dt))
                                paths[d, :] = curr
                            
                            final = paths[-1, :]
                            med = np.median(final)
                            res_df = pd.DataFrame({
                                "Metryka": ["Mediana", "95. Percentyl", "5. Percentyl (Ryzyko)", "Prawd. straty"],
                                "Wartość": [f"{med:,.2f}", f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%"]
                            })
                            with (col_a if i == 0 else col_b):
                                st.write(f"#### Perspektywa: {lbl}")
                                st.table(res_df)
                                fig, ax = plt.subplots()
                                ax.plot(paths[:, :50], color='#238636', alpha=0.1)
                                ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                                plt.style.use("dark_background")
                                st.pyplot(fig)
                    else: st.info("Symulacje MC są wyłączone.")

                with tabs[2]:
                    fig_c, ax_c = plt.subplots(figsize=(10, 7))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
                    st.pyplot(fig_c)

                with tabs[3]:
                    st.header("Metodologia vAlpha Engine")
                    with st.expander("1. Optymalizacja", expanded=True):
                        st.markdown("$$W_i \\propto \\frac{1 - \\bar{\\rho}_i}{VaR_i^p}$$")
                    with st.expander("2. Fat Tails Monte Carlo", expanded=True):
                        st.markdown("""Używamy **rozkladu t-Studenta** ($$\\nu=4$$) zamiast Gaussa, aby uwzględnić krachy rzędu -20%.""")

        except Exception as e: st.error(f"Błąd systemu: {e}")
