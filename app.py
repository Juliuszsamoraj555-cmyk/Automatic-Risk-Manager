import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Automatic Risk Manager", layout="wide")

st.title("🛡️ Automatic Risk Manager")
st.markdown("Zoptymalizuj swój portfel na podstawie matematycznych modeli ryzyka (VaR).")

# --- SIDEBAR (Panel Janka) ---
st.sidebar.header("Ustawienia Portfela")
tickers_input = st.sidebar.text_input("Wpisz spółki (oddzielone przecinkiem):", "AAPL, MSFT, TSLA, NVDA, WMT, PG, JNJ")
kwota = st.sidebar.number_input("Kwota inwestycji:", value=100000)
ryzyko = st.sidebar.select_slider("Poziom Ryzyka:", options=['low', 'medium', 'high'], value='medium')
limit_2x = st.sidebar.checkbox("Zastosuj limit 2x (Dywersyfikacja)", value=True)

if st.sidebar.button("Analizuj i Optymalizuj"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner('Pobieranie danych rynkowych...'):
        data = yf.download(tickers, period="3y")['Close']
        returns = data.resample('ME').last().pct_change().dropna()
        monthly_vars = returns.quantile(0.05) * -1
        avg_corr = returns.corr().mean()

    # --- LOGIKA OBLICZEŃ ---
    risk_map = {'low': 2.0, 'medium': 1.0, 'high': 0.5}
    penalty = risk_map.get(ryzyko)
    target_weights_raw = (1 / (monthly_vars ** penalty)) * (1 - avg_corr)
    target_weights = target_weights_raw / target_weights_raw.sum()

    # Optymalizacja z limitem
    def objective(weights): return np.sum((weights - target_weights)**2)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if limit_2x:
        cons.append({'type': 'ineq', 'fun': lambda x: 2 * np.min(x) - np.max(x)})
    
    res = minimize(objective, target_weights, method='SLSQP', bounds=tuple((0.01, 1.0) for _ in tickers), constraints=cons)
    final_weights = res.x

    # --- PREZENTACJA WYNIKÓW ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Twoja Alokacja")
        wynik_df = pd.DataFrame({
            'Spółka': monthly_vars.index,
            'Procent': final_weights * 100,
            'Kwota (PLN)': final_weights * kwota
        }).sort_values(by='Procent', ascending=False)
        
        st.dataframe(wynik_df.style.format({'Procent': '{:.2f}%', 'Kwota (PLN)': '{:,.2f}'}))

    with col2:
        st.subheader("📉 Statystyki")
        portfel_var = (final_weights * monthly_vars).sum()
        st.metric("Monthly VaR Portfela", f"{portfel_var*100:.2f}%")
        st.write(f"Maksymalna strata miesięczna: **{portfel_var * kwota:,.2f} PLN**")

    # Wykres korelacji
    st.subheader("🔗 Mapa Korelacji (Dlaczego takie wagi?)")
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
