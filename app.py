import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Importy naszych modułów
import styles
import database
import engine

# 1. Inicjalizacja
supabase = database.init_supabase()
styles.apply_custom_css()

try:
    v_alpha_icon = Image.open('image_8.png')
    st.set_page_config(page_title="Valpha Portfolio Manager", page_icon=v_alpha_icon, layout="wide")
except:
    st.set_page_config(page_title="Valpha Portfolio Manager", layout="wide")

# 2. Logowanie (uproszczone w app.py dla widoczności)
if 'user' not in st.session_state:
    # ... (tutaj Twój dotychczasowy kod logowania/rejestracji) ...
    st.stop()

user_email = st.session_state.user.user.email
days_left = database.get_pro_days(supabase, user_email)
is_pro = days_left > 0

# 3. Sidebar UI
with st.sidebar:
    st.title("Valpha Manager")
    if is_pro: st.success(f"💎 PRO: {days_left} dni")
    else: st.warning("🆓 FREE"); st.link_button("🚀 ODBLOKUJ", "URL_STRIPE")
    
    tickers_input = st.text_input("Spółki:", "AAPL, MSFT, NVDA, TSLA, AMZN")
    kwota = st.number_input("Kapitał:", value=25000)
    opt_mode = st.radio("Model:", ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"])
    ryzyko_val = st.select_slider("Ryzyko:", options=['low', 'medium', 'high'])
    limit_2x = st.checkbox("Limit dywersyfikacji", value=True)
    constraints_input = st.text_input("📍 Min. udział (PRO)", disabled=not is_pro)
    run_mc = st.checkbox("Monte Carlo", value=True)
    adj_mc = st.checkbox("Fat Tails Engine", value=False, disabled=not is_pro)
    analizuj = st.button("🚀 URUCHOM ANALIZĘ")

# 4. Logika główna
if analizuj:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # Obsługa limitów
    min_bounds = {t: 0.01 for t in tickers}
    if constraints_input and is_pro:
        for p in constraints_input.split(','):
            try:
                t, v = p.split(':')
                if t.strip().upper() in min_bounds: min_bounds[t.strip().upper()] = float(v)/100
            except: pass

    # Pobieranie i obliczenia
    with st.spinner('Analiza w toku...'):
        data = engine.get_data(tuple(tickers + (["SPY"] if adj_mc else [])))
        # ... (tutaj reszta logiki przeliczania, która wywołuje engine.optimize_portfolio) ...
        # (Wszystkie taby z wykresami zostają tutaj w app.py)
