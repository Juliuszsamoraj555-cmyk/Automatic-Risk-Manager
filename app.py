import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. KONFIGURACJA STRONY
st.set_page_config(page_title="Automatic Risk Manager Pro", page_icon="🛡️", layout="wide")

# 2. DESIGN CSS (SaaS Look)
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
    }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR (Z PEŁNYMI OPISAMI)
st.title("🛡️ Risk Manager Pro")
with st.sidebar:
    st.header("⚙️ Ustawienia")
    
    tickers_input = st.text_input(
        "Symbole spółek (ticker):", 
        "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD",
        help="""
        **Jak wpisywać symbole?**
        System pobiera dane z Yahoo Finance. 
        * **USA:** Sam ticker (np. `AAPL`).
        * **Polska:** Dodaj `.WA` (np. `ALE.WA`).
        * **Niemcy:** Dodaj `.DE` (np. `BMW.DE`).
        * **Kryptowaluty:** Dodaj `-USD` (np. `BTC-USD`).
        """
    )
    
    kwota = st.number_input("Kapitał początkowy:", value=25000, step=1000)
    
    st.divider()
    
    opt_mode = st.radio(
        "Tryb Optymalizacji:",
        ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        index=0,
        help="""
        **Bezpieczeństwo (VaR):** Skupia się na minimalizacji strat w najgorszych scenariuszach. Wybiera najbardziej stabilne spółki.
        **Efektywność (Sortino):** Szuka najlepszego zysku w stosunku do ryzyka spadków. Docenia spółki, które rosną gwałtownie, ale rzadko zaliczają głębokie 'doły'.
        """
    )

    ryzyko = st.select_slider("Profil Ryzyka:", options=['low', 'medium', 'high'], value='medium')
    
    limit_2x = st.checkbox(
        "Wymuś dywersyfikację (Limit 2x)", 
        value=True,
        help="""
        **Zasada 2x:** Algorytm pilnuje, aby największa pozycja była max 2x większa od najmniejszej. Zapobiega to dominacji jednej spółki i chroni przed ryzykiem specyficznym.
        """
    )
    
    run_mc = st.checkbox(
        "Wykonaj symulacje Monte Carlo", 
        value=True,
        help="""
        System przeprowadza **10 000 wirtualnych rzutów kostką**, tworząc tysiące scenariuszy przyszłości. Pozwala to ocenić szansę na stratę oraz zrozumieć zakres niepewności.
        """
    )
    
    st.divider()
    analizuj = st.button("URUCHOM PEŁNĄ ANALIZĘ")

# 4. GŁÓWNA LOGIKA
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    try:
        with st.spinner('📊 Analizowanie danych...'):
            data_raw = yf.download(tickers, period="3y")['Close']
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(-1)
            
            df_daily_rets = data_raw.pct_change().dropna()
            df_monthly_rets = data_raw.resample('ME').last().pct_change().dropna()
            
            monthly_vars = df_monthly_rets.quantile(0.05) * -1
            corr_matrix = df_monthly_rets.corr()
            avg_corr_each = corr_matrix.mean()

        # SILNIK OPTYMALIZACJI
        if opt_mode == "Bezpieczeństwo (VaR-First)":
            penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
            target_w_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr_each)
        else:
            mean_ret = df_monthly_rets.mean()
            downside_std = df_monthly_rets[df_monthly_rets < 0].std()
            sortino_ratios = mean_ret / (downside_std + 1e-6)
            power_map = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
            target_w_raw = (sortino_ratios.clip(lower=0) ** power_map[ryzyko]) * (1 - avg_corr_each)

        target_w = target_w_raw / target_w_raw.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        wagi_finalne = res.x

        # WYNIKI
        t_names = ["📈 Portfel"]
        if run_mc: t_names.append("🔮 Projekcje")
        t_names.append("🔗 Korelacje")
        tabs = st.tabs(t_names)

        with tabs[0]:
            st.subheader(f"Rekomendowana alokacja ({opt_mode})")
            c1, c2, c3 = st.columns(3)
            p_var = (wagi_finalne * monthly_vars).sum()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_c = corr_matrix.where(mask).stack().mean()
            
            c1.metric("Miesięczny VaR (95%)", f"{p_var*100:.2f}%", help="Statystyczna miara ryzyka straty miesięcznej.")
            c2.metric("Średnia Korelacja", f"{mean_c:.2f}", help="Mierzy jak bardzo aktywa poruszają się w tym samym kierunku.")
            c3.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}", help=f"Szacowana miesięczna strata przy kapitale {kwota:,.0f} PLN.")
            
            st.divider()
            df_wynik = pd.DataFrame({
                'Ticker': monthly_vars.index,
                'Udział (%)': wagi_finalne * 100,
                'Kwota': wagi_finalne * kwota
            }).sort_values(by='Udział (%)', ascending=False)
            st.dataframe(df_wynik.style.format({'Udział (%)': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True, use_container_width=True)

        if run_mc:
            with tabs[1]:
                st.subheader(f"Projekcje 10,000 symulacji ({opt_mode})")
                cov_matrix = df_daily_rets.cov()
                p_mean = np.sum(df_daily_rets.mean() * wagi_finalne)
                p_std = np.sqrt(np.dot(wagi_finalne.T, np.dot(cov_matrix, wagi_finalne)))
                
                n_sims = 10000
                plt.style.use("dark_background")
                col_a, col_b = st.columns(2)

                for i, (y, lbl) in enumerate(zip([5, 10], ["5 Lat", "10 Lat"])):
                    days = y * 252
                    sim_rets = np.random.normal(p_mean, p_std, (days, n_sims))
                    sim_paths = kwota * np.cumprod(1 + sim_rets, axis=0)
                    final_v = sim_paths[-1, :]
                    
                    # OBLICZENIA NOWYCH METRYK
                    mediana = np.median(final_v)
                    p95 = np.percentile(final_v, 95)
                    q3 = np.percentile(final_v, 75)
                    q1 = np.percentile(final_v, 25)
                    p5 = np.percentile(final_v, 5)
                    chance_loss = (np.sum(final_v < kwota) / n_sims) * 100
                    cagr = (mediana / kwota)**(1/y) - 1
                    
                    with (col_a if i == 0 else col_b):
                        st.write(f"#### Prognoza {lbl}")
                        st.table(pd.DataFrame({
                            "Metryka": [
                                "Mediana (Scenariusz bazowy)", 
                                "95% Optymizm (Hossa)", 
                                "3. Kwartyl (75% szans na mniej niż)", 
                                "1. Kwartyl (25% szans na mniej niż)", 
                                "5% Pesymizm (Bessa)", 
                                "Szansa na stratę kapitału",
                                "Średni roczny zwrot (CAGR)"
                            ],
                            "Wartość": [
                                f"{mediana:,.2f}", 
                                f"{p95:,.2f}", 
                                f"{q3:,.2f}", 
                                f"{q1:,.2f}", 
                                f"{p5:,.2f}", 
                                f"{chance_loss:.1f}%",
                                f"{cagr*100:.2f}%"
                            ]
                        }))
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(sim_paths[:, :100], color='skyblue', alpha=0.06)
                        ax.plot(np.median(sim_paths, axis=1), color='white', linewidth=2.5)
                        st.pyplot(fig)
                plt.style.use('default')

        with tabs[-1]:
            st.subheader("Mapa Korelacji")
            fig_c, ax_c = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_c)
            st.pyplot(fig_c)

    except Exception as e:
        st.error(f"Coś poszło nie tak: {e}")
