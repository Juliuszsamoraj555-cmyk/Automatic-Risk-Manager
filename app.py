import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import translations
from PIL import Image
from engine import normalize_to_target_currency

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
# --- 1. LOGIKA WYBORU JĘZYKA ---
if 'lang' not in st.session_state:
    st.session_state.lang = "PL" # Domyślnie polski

with st.sidebar:
    # 1. Profesjonalny przełącznik języka (płaski, poziomy)
    lang_choice = st.radio(
        "Language",
        options=["PL", "EN"],
        index=0 if st.session_state.lang == "PL" else 1,
        horizontal=True,
        label_visibility="collapsed" # Ukrywamy etykietę "Language"
    )
    
    # Obsługa zmiany (tylko jeśli użytkownik kliknie)
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()
        
    L = translations.LANGS[st.session_state.lang]
    st.markdown("<br>", unsafe_allow_html=True) # Mały odstęp
    st.title("vAlpha Manager")
    st.write("---")
    current_currency = L["currency"]
    
    
    if not is_logged_in:
        # --- SEKCJA DLA GOŚCIA ---
        st.info(L["login_info"])
        
        with st.popover(L["auth_popover"], use_container_width=True):
            tab_l, tab_r = st.tabs([L["tab_login"], L["tab_register"]])
            
            # WSZYSTKO PONIŻEJ MUSI BYĆ WCIĘTE W RAMACH POPOVERA
            with tab_l:
                st.text_input(L["email"], key="l_mail")
                st.text_input(L["password"], type="password", key="l_pw")
                
                if st.button("Zaloguj", use_container_width=True):
                    email_val = st.session_state.get("l_mail")
                    pass_val = st.session_state.get("l_pw")
                    
                    if email_val and pass_val:
                        try:
                            res = supabase.auth.sign_in_with_password({
                                "email": email_val, 
                                "password": pass_val
                            })
                            st.session_state.user = res
                            st.success(L["msg_logged"])
                            st.rerun()
                        except Exception as e:
                            st.error(L["msg_auth_error"])
                    else:
                       st.warning(L["msg_fill_fields"])
            with tab_r:
                st.text_input(L["email"], key="r_mail")     # Używamy L["email"]
                st.text_input(L["password"], type="password", key="r_pw") # Używamy L["password"]
                
                if st.button(L["btn_register"], use_container_width=True):
                    email_reg = st.session_state.get("r_mail")
                    pass_reg = st.session_state.get("r_pw")
                    
                    if email_reg and pass_reg:
                        try:
                            supabase.auth.sign_up({
                                "email": email_reg, 
                                "password": pass_reg
                            })
                            st.success(L["msg_reg_success"])
                        except:
                           st.error(L["msg_reg_error"])
                    else:
                        st.warning(L["msg_fill_reg"])
    
    else:
        # --- SEKCJA DLA ZALOGOWANEGO ---
        # Ten blok musi być wyrównany do 'if not is_logged_in'
        st.write(L["welcome"].format(user_email))
        
        if is_pro:
          st.success(L["status_pro"].format(days_left))
        else:
            st.warning(L["status_free"])
            st.link_button(
            L["btn_unlock_pro"],
            f"https://buy.stripe.com/7sYbJ1fft827aVRbPud3i03?prefilled_email={user_email}"
        ) # <--- TEGO NAWIASU BRAKOWAŁO
        if st.button(L["btn_logout"], use_container_width=True):
            if "user" in st.session_state:
                del st.session_state.user
            st.rerun()


    st.divider()
    
    # --- INPUTY STANDARDOWE ---
    tickers_input = st.text_input(
    L["tickers_label"], 
    value="AAPL, MSFT, NVDA, TSLA, AMZN", 
    help=L["tickers_help"]
)
    
    kwota = st.number_input(
        L["capital_label"].format(current_currency), 
        value=25000 if current_currency == "PLN" else 5000
    )
    opt_mode = st.radio(
    L["opt_model_label"], 
    L["opt_model_options"],
    help=L["opt_model_help"]
)
    # 1. Suwak Ryzyka z mapowaniem
ryzyko_display = st.select_slider(
    L["risk_label"], 
    options=L["risk_options"],
    value=L["risk_options"][1] # Domyślnie środkowa opcja (Średnie/Medium)
)

# Tłumaczymy wybrany tekst z powrotem na język silnika (low/medium/high)
risk_map = {
    L["risk_options"][0]: "low",
    L["risk_options"][1]: "medium",
    L["risk_options"][2]: "high"
}
ryzyko_val = risk_map[ryzyko_display]

# 2. Checkbox dywersyfikacji
limit_2x = st.checkbox(
    L["limit_2x_label"], 
    value=True, 
    help=L["limit_2x_help"]
)

    
    # --- BLOKADA FUNKCJI PRO (UI) ---
suffix = L["min_weight_locked"] if not is_pro else ""
label_min = L["min_weight_label"] + suffix

# 2. Wyświetlamy pole tekstowe
constraints_input = st.text_input(
    label_min, 
    placeholder="NVDA:10", 
    disabled=not is_pro,
    help=L["min_weight_help"]
)
    
    
st.divider()
run_mc = st.checkbox(
    L["run_mc_label"], 
    value=True, 
    help=L["run_mc_help"]
)
suffix_adj = L["valpha_engine_locked"] if not is_pro else ""
label_adj = L["valpha_engine_label"] + suffix_adj

adj_mc_checkbox = st.checkbox(
    label_adj, 
    value=False, 
    help=L["valpha_engine_help"]
)
adj_mc = False
rf_rate, mkt_ret, alpha_ret, beta_speed = 0.04, 0.10, 30.0, 0.05
if adj_mc_checkbox:
    if is_pro:
            adj_mc = True
            with st.expander(L["expander_market"], expanded=False):
                rf_rate = st.number_input(
                    L["rf_label"], 
                    value=4.0, 
                    help=L["rf_help"]
                ) / 100
                
                mkt_ret = st.number_input(
                    L["rm_label"], 
                    value=10.0, 
                    help=L["rm_help"]
                ) / 100
                
                alpha_ret = st.slider(
                    L["alpha_label"], 
                    0, 100, 30, 
                    help=L["alpha_help"]
                )
                
                beta_speed = st.slider(
                    L["beta_speed_label"], 
                    0.0, 0.2, 0.05, 
                    help=L["beta_speed_help"]
                )
    else:
            st.warning(L["msg_pro_required"])

    analizuj = st.button(L["btn_run_analysis"], use_container_width=True)

# --- LOGIKA ANALIZY (FREEMIUM ENFORCEMENT) ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # 1. Limit 5 spółek dla darmowych
    if not is_pro and len(tickers) > 5:
        st.error(L["err_free_limit"])
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
    with st.spinner(L["spinner_loading"]): # Użyj tłumaczenia dla spinnera
        try:
            fetch_list = tickers + (["SPY"] if adj_mc else [])
            
            # 1. POBIERANIE SUROWYCH DANYCH
            raw_data = engine.get_data(tuple(fetch_list))
            
            # 2. NORMALIZACJA WALUTOWA (DODAJ TO!)
            # Tutaj zamieniamy np. Apple z USD na PLN (lub odwrotnie)
            data = engine.normalize_to_target_currency(raw_data, current_currency)
            
            if data is None or data.empty:
                st.error(L["error_no_data"])
                st.stop()
            
            # Dalsza część używa już 'data', która jest w jednej walucie
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(-1)
            
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
            t1, t2, t3, t4 = st.tabs(L["tabs"])
            
            with t1:
                st.subheader(L["t1_subheader"])
                p_var = (wagi * monthly_vars).sum()
    
                c1, c2, c3 = st.columns(3)
    
    # METRYKA 1: VaR
    c1.metric(L["metric_var_label"], f"{p_var*100:.2f}%")
    c1.caption(L["metric_var_caption"])
                
                # METRYKA 2: Korelacja
avg_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
c2.metric(L["metric_corr_label"], f"{avg_corr:.2f}")
c2.caption(L["metric_corr_caption"])

# METRYKA 3: Ryzyko kwotowe (Dynamiczna waluta)
c3.metric(
    L["metric_risk_val_label"].format(current_currency), 
    f"{p_var * kwota:,.0f} {current_currency}"
)
c3.caption(L["metric_risk_val_caption"])

# TABELA: Udziały w portfelu
df_out = pd.DataFrame({
    L['col_ticker']: tickers, 
    L['col_share']: wagi * 100, 
    current_currency: wagi * kwota
})

st.dataframe(
    df_out.sort_values(L['col_share'], ascending=False).style.format({
        L['col_share']: '{:.2f}%', 
        current_currency: f'{{:,.2f}} {current_currency}'
    }), 
    use_container_width=True, 
    hide_index=True
)

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
    st.header(L["t4_header"])
    st.markdown(L["t4_text"])

        except Exception as e:
    st.error(f"{L['err_analysis_main']}")
    with st.expander(L["err_details"]):
        st.code(str(e))
