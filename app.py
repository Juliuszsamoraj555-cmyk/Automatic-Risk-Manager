import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import translations
import plotly.graph_objects as go
import streamlit.components.v1 as components
from PIL import Image
from engine import get_final_data
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
def inject_ga():
    ga_id = "G-KCCV2DZKM5"  # <--- WPISZ SWÓJ IDENTYFIKATOR
    ga_js = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{ga_id}');
        </script>
    """
    components.html(ga_js, height=0)

inject_ga()

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


if 'lang' not in st.session_state:
    st.session_state.lang = "PL"
L = translations.LANGS[st.session_state.lang]
current_currency = L["currency"]

if "risk_accepted" not in st.session_state:
    st.session_state.risk_accepted = False
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
        
    
    
    # 4. LOGIKA DISCLAIMERA (Znikający boks)
    
        
        
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
    
    


# --- 2. LOGIKA DISCLAIMERA NA ŚRODKU EKRANU ---
if not st.session_state.risk_accepted:
    st.markdown("<br><br>", unsafe_allow_html=True) # Odstęp od góry strony
    
    # Usunęliśmy .sidebar - komunikat pojawi się na głównym ekranie
    st.warning(f"## {L['risk_header']}\n\n{L['risk_text']}")
    
    # Centrujemy przycisk akceptacji, używając pustych kolumn bocznych
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        if st.button(L["btn_accept_risk"], use_container_width=True, type="primary"):
            st.session_state.risk_accepted = True
            st.rerun()
            
    # Zatrzymujemy kod tutaj - dopóki nie klikną, nie zobaczą nic poniżej
    st.stop()

# --- 3. EKRAN GŁÓWNY (Pojawia się po kliknięciu "Akceptuję") ---
st.title("vAlpha Manager")
st.write("---")

st.header("Konfiguracja Portfela")
col1, col2 = st.columns(2)
with col1:
    tickers_input = st.text_input(
    L["tickers_label"], 
    value="AAPL, MSFT, NVDA, TSLA, AMZN", 
    help=L["tickers_help"]
)
    
    kwota = st.number_input(
    L["capital_label"].format(current_currency), 
    value=25000 if current_currency == "PLN" else 5000
    )
with col2:
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
st.divider()
col3, col4 = st.columns(2)
with col3:
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
with col4:
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

    with st.spinner(L["spinner_loading"]):
        try:
            # ZMIANA: Zawsze dodajemy SPY do listy, bo jest potrzebny do Backtestingu (benchmark)
            # list(set(...)) usuwa duplikaty, gdyby użytkownik sam wpisał SPY
            fetch_list = list(set(tickers + ["SPY"]))
        
            #1. POBIERANIE DANYCH (Silnik sam przelicza waluty)
            data = engine.get_final_data(tuple(fetch_list), current_currency)
        
            if data is None or data.empty:
                st.error(L["error_no_data"])
                st.stop()
        
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(-1)
        
        # Dane tylko dla Twoich spółek do optymalizacji
            data_only = data[tickers]
        
        # 2. STATYSTYKI I OPTYMALIZACJA WAG
            daily_rets, monthly_rets, monthly_vars, corr_matrix = engine.get_portfolio_stats(data_only)
        
        # Obliczanie Bety i Alfy (potrzebne do Fat Tails Engine)
            betas, alphas = {}, {}
        # SPY musi być w data, bo dodaliśmy go do fetch_list
            spy_rets = data["SPY"].pct_change().dropna()
            spy_annual = (1 + spy_rets.mean())**252 - 1
        
            for t in tickers:
                t_rets = data_only[t].pct_change().dropna()
                comb = pd.concat([t_rets, spy_rets], axis=1).dropna()
                b = np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1])
                betas[t] = b
                hist_ret = (1 + t_rets.mean())**252 - 1
                alphas[t] = hist_ret - (rf_rate + b * (spy_annual - rf_rate))

        # GŁÓWNY WYNIK: Optymalizacja wag
            wagi = engine.optimize_weights(tickers, monthly_rets, monthly_vars, corr_matrix, opt_mode, ryzyko_val, min_bounds, limit_2x)

        # --- NOWOŚĆ: WYWOŁANIE BACKTESTINGU ---
        # data["SPY"] służy jako tło do porównania wyników
            port_cum, bench_cum = engine.run_backtest(data_only, wagi, data["SPY"])

        # --- 4. WYŚWIETLANIE WYNIKÓW (Taby) ---
        # Dodajemy L["tab_backtest"] do listy tabów
            t1, t2, t3, t4, t5 = st.tabs(L["tabs"] + [L["tab_backtest"]])
            
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
                    # Wywołujemy silnik z nowymi argumentami (waluta i etykiety z L)
                    mc_data = engine.run_monte_carlo(
                        data_only, wagi, kwota, tickers, adj_mc, 
                        rf_rate, mkt_ret, alpha_ret, beta_speed, betas, alphas,
                        target_ccy=current_currency,
                        mc_labels=L["mc_metrics"]
                    )

                    col_a, col_b = st.columns(2)
                    plt.style.use("dark_background")
                    
                    # Pobieramy etykiety lat z tłumaczeń (np. L["mc_5y"], L["mc_10y"])
                    years_labels = [L.get("mc_5y", "5 LAT"), L.get("mc_10y", "10 LAT")]

                    for i, (y, lbl) in enumerate(zip([5, 10], years_labels)):
                        paths = mc_data[y]['paths']
                        
                        with (col_a if i == 0 else col_b):
                            st.markdown(f"### {lbl}")
                            
                            # Wyświetlamy tabelę z 7 metrykami (Q1, Q3 itd.)
                            st.table(mc_data[y]['stats'])
                            
                            # Wykres
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Rysujemy 50 przykładowych ścieżek
                            ax.plot(paths[:, :50], alpha=0.1, color='#238636')
                            
                            # Linia mediany (grubsza)
                            ax.plot(np.median(paths, axis=1), color='white', linewidth=2, label="Mediana")
                            
                            # Estetyka wykresu
                            ax.set_facecolor('#0d1117')
                            fig.patch.set_facecolor('#0d1117')
                            ax.set_ylabel(current_currency)
                            ax.set_xlabel(L.get("mc_days", "Dni handlowe"))
                            
                            st.pyplot(fig)

            with t3:
                fig_c, ax_c = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_c)
                fig_c.patch.set_facecolor('#0d1117')
                st.pyplot(fig_c)

        
                
            with t4:
                st.header(L["t4_header"])
                st.markdown(L["t4_text"])
            with t5:
                st.subheader(L["backtest_header"])
    
                # Tworzymy profesjonalny wykres Plotly
                fig = go.Figure()

                # Linia Portfela (Grubsza, neonowy turkus jak na zdjęciu)
                fig.add_trace(go.Scatter(
                x=port_cum.index, 
                y=(port_cum - 1) * 100,
                mode='lines',
                name=L["backtest_port_label"],
                line=dict(color='#00d4ff', width=3),
                hovertemplate='%{y:.2f}%'
                ))

                # Linia Benchmarku (Cieńsza, szara/przerywana)
                fig.add_trace(go.Scatter(
                x=bench_cum.index, 
                y=(bench_cum - 1) * 100,
                mode='lines',
                name=L["backtest_bench_label"],
                line=dict(color='rgba(255, 255, 255, 0.4)', width=1.5, dash='dash'),
                hovertemplate='%{y:.2f}%'
            ))
        
            # Stylistyka "Professional Dark" nawiązująca do image_28725d.png
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="MAX")
                            ]),
                            bgcolor="rgba(0,0,0,0)", activecolor="#00d4ff"
                        )
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        ticksuffix="%", side="right"
                    )
                )
                    # Wyświetlenie interaktywnego wykresu
                st.plotly_chart(fig, use_container_width=True)
            
                # Statystyki pod wykresem
                final_return = (port_cum.iloc[-1] - 1) * 100
                bench_return = (bench_cum.iloc[-1] - 1) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric(L["backtest_port_label"], f"{final_return:.2f}%")
                col2.metric(L["backtest_bench_label"], f"{bench_return:.2f}%")
                col3.metric("Alpha", f"{final_return - bench_return:.2f}%", 
                            delta=f"{final_return - bench_return:.2f}%")

        except Exception as e:  # <--- KLUCZOWY MOMENT: tutaj nazywamy błąd literką 'e'
            st.error(L["err_generic"]) # Twoja ogólna wiadomość o błędzie
            with st.expander(L["err_details"]): # Rozwijane szczegóły
                st.code(str(e))


# --- STOPKA (FOOTER) ---
st.write("<br><br>", unsafe_allow_html=True)
st.divider()

foot_col1, foot_col2, foot_col3 = st.columns([2, 1, 1])

with foot_col1:
    st.markdown(f"**vAlpha Manager © 2026**")
    st.caption(L["footer_disclaimer"])

with foot_col2:
    st.markdown(f"**{L['footer_legal']}**")
    
    # Pobieramy aktualny język ("PL" lub "EN")
    current_lang = st.session_state.lang

    @st.dialog(L["footer_terms"], width="large")
    def show_terms():
        # Dynamiczne wczytanie: regulamin_PL.md lub regulamin_EN.md
        with open(f"regulamin_{current_lang}.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())

    @st.dialog(L["footer_privacy"], width="large")
    def show_privacy():
        # Dynamiczne wczytanie: polityka_PL.md lub polityka_EN.md
        with open(f"polityka_{current_lang}.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())

    # Wyświetlamy przyciski-linki
    if st.button(L["footer_terms"], variant="link"):
        show_terms()
    if st.button(L["footer_privacy"], variant="link"):
        show_privacy()

with foot_col3:
    st.markdown(f"**{L['footer_support']}**")
    st.markdown(f"[{L['footer_contact']}](mailto:support@valpha.pl)")
