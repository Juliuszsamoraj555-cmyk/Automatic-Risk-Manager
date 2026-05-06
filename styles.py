import streamlit as st

def apply_custom_css():
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
