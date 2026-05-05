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

# --- 5. LOGIKA DOSTĘPU CZASOWEGO (PRO NA 30 DNI) ---
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
        st.write("Zaloguj się do profesjonalnego systemu zarządzania ryzykiem.")
        tab1, tab2 = st.tabs(["Logowanie", "Rejestracja"])
        with tab1:
            l_email = st.text_input("E-mail", key="l_mail")
            l_pw = st.text_input("Hasło", type="password", key="l_pw")
            if st.button("Zaloguj się"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": l_email, "password": l_pw})
                    st.session_state.user = res
                    st.rerun()
                except: st.error("Błąd logowania. Sprawdź dane.")
        with tab2:
            r_email = st.text_input("Twój e-mail", key="r_mail")
            r_pw = st.text_input("Hasło (min. 6 znaków)", type="password", key="r_pw")
            if st.button("Załóż konto"):
                try:
                    supabase.auth.sign_up({"email": r_email, "password": r_pw})
                    st.success("Konto utworzone! Potwierdź maila.")
                except Exception as e: st.error(f"Błąd: {e}")
    st.stop()

# --- 7. LOGIKA PO ZALOGOWANIU ---
user_email = st.session_state.user.user.email
days_left = get_pro_info(user_email)
is_pro = days_left > 0

# --- 8. PEŁNY DISCLAIMER PRAWNY ---
st.markdown("""
    <div class="disclaimer-red">
        <strong>WAŻNE INFORMACJE PRAWNE ORAZ ZASTRZEŻENIA</strong><br>
        Niniejsza aplikacja ma charakter wyłącznie informacyjny oraz edukacyjny i nie stanowi rekomendacji inwestycyjnej ani porady finansowej...
    </div>
    """, unsafe_allow_html=True)

# --- 9. SIDEBAR (Z AUTOMATYCZNYM LINKIEM) ---
with st.sidebar:
    try: st.image(v_alpha_icon, width=100)
    except: pass
    st.title("Valpha Manager")
    st.write(f"Zalogowany: **{user_email}**")
    
    if is_pro:
        st.success(f"💎 STATUS: PRO (Zostało {days_left} dni)")
    else:
        st.warning("🆓 STATUS: FREE (Limit: 5 spółek)")
        # --- NOWOŚĆ: Automatyczne przekazywanie maila do Stripe ---
        stripe_base_url = "https://buy.stripe.com/7sYbJ1fft827aVRbPud3i03"
        stripe_url_with_email = f"{stripe_base_url}?prefilled_email={user_email}"
        st.link_button("🚀 ODBLOKUJ PRO NA 30 DNI (25 PLN)", stripe_url_with_email)

    if st.button("Wyloguj"):
        supabase.auth.sign_out()
        del st.session_state.user
        st.rerun()

    st.divider()
    # ... reszta konfiguracji portfela (kod bez zmian jak wyżej) ...
    tickers_input = st.text_input("Symbole spółek (ticker):", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał początkowy (PLN):", value=25000)
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
            with st.expander("PARAMETRY RYNKOWE CAPM / GBM", expanded=True):
                rf_rate = st.number_input("Stopa wolna od ryzyka (Rf %):", value=4.0) / 100
                mkt_ret = st.number_input("Oczekiwany zwrot rynku (Rm %):", value=10.0) / 100
                alpha_ret = st.slider("Utrzymanie przewagi (Alfa %):", 0, 100, 30)
                beta_speed = st.slider("Szybkość stabilizacji Bety:", 0.0, 0.2, 0.05)

    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ SYSTEMOWĄ")
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not is_pro and len(tickers) > 5:
        st.error(f"Wersja FREE obsługuje do 5 spółek. Twoja lista ma {len(tickers)} pozycji.")
    else:
        try:
            with st.spinner('Analizowanie danych rynkowych...'):
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
                else: data_only = data[tickers] if "SPY" in data.columns else data

                daily_rets = data_only.pct_change().dropna()
                monthly_rets = data_only.resample('ME').last().pct_change().dropna()
                monthly_vars = monthly_rets.quantile(0.05) * -1
                corr_matrix = monthly_rets.corr()

            # OPTYMALIZACJA WAG
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

            tabs = st.tabs(["Struktura Portfela", "Symulacja Monte Carlo", "Macierz Korelacji", "Metodologia"])

            with tabs[0]:
                st.subheader(f"Rekomendowana alokacja ({opt_mode})")
                c1, c2, c3 = st.columns(3)
                p_var = (wagi * monthly_vars).sum()
                c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Istnieje 5% szans, że w miesiącu portfel straci więcej niż ten procent.")
                c2.metric("Średnia Korelacja", f"{corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean():.2f}", help="Blisko 0 oznacza świetną dywersyfikację.")
                c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help="Miesięczny VaR przeliczony na kwotę.")
                st.divider()
                df_out = pd.DataFrame({'Ticker': tickers, 'Udział (%)': wagi * 100, 'Kwota': wagi * kwota})
                if adj_mc: df_out['Beta'] = [betas[t] for t in tickers]
                st.dataframe(df_out.sort_values('Udział (%)', ascending=False).style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}', 'Beta': '{:.2f}'}), use_container_width=True, hide_index=True)
                if run_mc:
                   with tabs[1]:
                    st.subheader("Symulacja Monte Carlo - 3,000 symulacji (Fat Tails Edition)")
                    st.info("Model: Rozkład t-Studenta (df=4). Uwzględnia ryzyko 'grubych ogonów' (krachy -20% raz na 5 lat).")
                    
                    n_sims, dt = 3000, 1/252
                    nu = 4  # Parametr Fat Tails (im mniejszy, tym cięższe ogony)
                    # Korekta skali, aby wariancja rozkładu t-Studenta wynosiła 1 (do poprawnego skalowania sigmą)
                    t_scale = np.sqrt((nu - 2) / nu) 

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
                            p_alpha = np.sum([alphas[t] * wagi[idx] for idx, t in enumerate(tickers)]) * (alpha_ret / 100)
                            t_beta = p_beta
                            for d in range(days):
                                # ZAMIANA: np.random.normal -> np.random.standard_t
                                epsilon = np.random.standard_t(df=nu, size=n_sims) * t_scale
                                mu = (rf_rate + t_beta * (mkt_ret - rf_rate) + p_alpha - 0.5 * (p_sigma**2)) * dt
                                curr *= np.exp(mu + p_sigma * epsilon * np.sqrt(dt))
                                paths[d, :] = curr
                                if d % 252 == 0: t_beta = t_beta * (1 - beta_speed) + 1.0 * beta_speed
                        else:
                            mu = (np.sum(daily_rets.mean() * wagi) * 252 - 0.5 * (p_sigma**2)) * dt
                            for d in range(days):
                                # ZAMIANA: np.random.normal -> np.random.standard_t
                                epsilon = np.random.standard_t(df=nu, size=n_sims) * t_scale
                                curr *= np.exp(mu + p_sigma * epsilon * np.sqrt(dt))
                                paths[d, :] = curr

                        final = paths[-1, :]
                        med = np.median(final)
                        res_df = pd.DataFrame({
                            "Metryka": ["95. Percentyl (Optymizm)", "3. Kwartyl (75%)", "Mediana (Statystyczny wynik)", "1. Kwartyl (25%)", "5. Percentyl (Pesymizm)", "Prawdopodobieństwo straty", "CAGR (Roczny zwrot)"],
                            "Wartość": [f"{np.percentile(final, 95):,.2f}", f"{np.percentile(final, 75):,.2f}", f"{med:,.2f}", f"{np.percentile(final, 25):,.2f}", f"{np.percentile(final, 5):,.2f}", f"{(np.sum(final < kwota) / n_sims) * 100:.1f}%", f"{((med / kwota)**(1/y) - 1)*100:.2f}%"]
                        })
                        with (col_a if i == 0 else col_b):
                            st.write(f"#### PERSPEKTYWA: {lbl}")
                            st.table(res_df)
                            fig, ax = plt.subplots()
                            ax.plot(paths[:, :50], color='#238636', alpha=0.1)
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
                            st.pyplot(fig)

            with tabs[2]:
                st.subheader("Macierz korelacji między aktywami")
                fig_c, ax_c = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
                st.pyplot(fig_c)

            with tabs[3]:
                st.header("Metodologia obliczeń i algorytmy")
                
                with st.expander("1. Optymalizacja wag portfela", expanded=True):
                    st.markdown("""
                    **Model VaR-First (Bezpieczeństwo)**: $W_i \\propto \\frac{1 - \\bar{\rho}_i}{VaR_i^p}$
                    
                    **Model Sortino (Efektywność)**: $W_i \\propto \\left(\\frac{R_i - R_f}{\\sigma_{downside}}\\right)^p \\cdot (1 - \\bar{\rho}_i)$
                    
                    *Algorytm minimalizuje różnicę między wagami surowymi a docelowymi, uwzględniając korelacje między aktywami.*
                    """)

                with st.expander("2. Skorygowana Symulacja Monte Carlo (Fat Tails Engine)", expanded=True):
                    st.markdown("""
                    **Model Rozkładu**: $t$-Studenta ($\\\\nu=4$)
                    
                    W odróżnieniu od standardowych modeli opartych na rozkładzie Gaussa, vAlpha wykorzystuje **rozkład $t$-Studenta**. Pozwala to na modelowanie tzw. **"grubych ogonów" (Fat Tails)**, czyli zdarzeń ekstremalnych, które na giełdzie występują znacznie częściej, niż zakłada tradycyjna statystyka (np. krachy -20%).
                    
                    **Kluczowe równania**:
                    1. **CAPM + Alfa**: $E(R_i) = R_f + \\beta_i(E(R_m) - R_f) + \\alpha \\cdot \\text{retention}$
                    2. **Korekta Dryfu**: $\\mu_{adj} = E(R_i) - \\frac{1}{2}\\sigma^2$
                    3. **Geometryczny Ruch Browna (Fat Tails)**: $P_{t+1} = P_t \\cdot e^{(\\mu_{adj} \\Delta t + \\sigma \\cdot \\epsilon_t \\sqrt{\\Delta t})}$
                    
                    *Gdzie $\\epsilon_t$ jest losowane z rozkładu t-Studenta, co symuluje realizm rynkowy (korekty rzędu -20% raz na 5 lat).*
                    """)
        except Exception as e: st.error(f"Błąd: {e}")
