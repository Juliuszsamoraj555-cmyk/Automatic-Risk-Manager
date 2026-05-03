import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# MUST BE FIRST: Konfiguracja strony
st.set_page_config(page_title="Risk Manager Pro", layout="wide")

# --- CSS UI ---
st.markdown("""
    <style>
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stButton > button { width: 100%; background-color: #238636 !important; color: white !important; font-weight: 700; height: 3em; border: none; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Automatic Risk Manager Pro")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ustawienia")
    tickers_raw = st.text_input("Spółki:", "AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, V, JPM, JNJ, WMT, PG, MA, UNH, HD")
    kwota = st.number_input("Kapitał:", value=25000)
    ryzyko = st.select_slider("Ryzyko:", options=['low', 'medium', 'high'], value='medium')
    limit_2x = st.checkbox("Limit 2x", value=True)
    analizuj = st.button("URUCHOM ANALIZĘ")

# --- ANALIZA ---
if analizuj:
    tickers = [t.strip().upper() for t in tickers_raw.split(',')]
    
    try:
        with st.spinner('Pobieranie danych...'):
            df = yf.download(tickers, period="3y")['Close']
            if df.empty:
                st.error("Błąd: Nie udało się pobrać danych z Yahoo Finance.")
                st.stop()
            
            rets_d = df.pct_change().dropna()
            rets_m = df.resample('ME').last().pct_change().dropna()
            
            v_m = rets_m.quantile(0.05) * -1
            corr = rets_m.corr()
            avg_c = corr.mean()

        # Optymalizacja
        penalty = {'low': 2.0, 'medium': 1.0, 'high': 0.5}[ryzyko]
        target_w = (1 / (v_m ** penalty)) * (1 - avg_c)
        target_w = target_w / target_w.sum()

        def objective(w): return np.sum((w - target_w.values)**2)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        if limit_2x:
            cons.append({'type': 'ineq', 'fun': lambda w: 2 * np.min(w) - np.max(w)})
        
        res = minimize(objective, target_w.values, method='SLSQP', bounds=[(0.01, 1.0)]*len(tickers), constraints=cons)
        weights = res.x

        # WYNIKI
        t1, t2 = st.tabs(["📊 Portfel", "🔮 Projekcja"])
        
        with t1:
            c1, c2 = st.columns(2)
            p_var = (weights * v_m).sum()
            c1.metric("Miesięczny VaR", f"{p_var*100:.2f}%")
            c2.metric("Ryzyko (PLN)", f"{p_var * kwota:,.2f}")
            
            out_df = pd.DataFrame({'Ticker': v_m.index, 'Udział': weights*100, 'Kwota': weights*kwota})
            st.dataframe(out_df.sort_values('Udział', ascending=False).style.format({'Udział': '{:.2f}%', 'Kwota': '{:,.2f}'}), hide_index=True)

        with t2:
            st.subheader("Symulacja Monte Carlo (5,000 prób)")
            p_mean = np.sum(rets_d.mean() * weights)
            p_std = np.sqrt(np.dot(weights.T, np.dot(rets_d.cov(), weights)))
            
            # Symulacja 5 lat
            sims = np.random.normal(p_mean, p_std, (252*5, 5000))
            paths = kwota * np.cumprod(1 + sims, axis=0)
            
            final = paths[-1, :]
            med = np.median(final)
            st.write(f"Przewidywana mediana po 5 latach: **{med:,.2f} PLN**")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            plt.style.use('dark_background')
            ax.plot(paths[:, :50], color='skyblue', alpha=0.1)
            ax.plot(np.median(paths, axis=1), color='white', linewidth=2)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Coś poszło nie tak: {e}")
