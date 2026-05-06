import os
import datetime
import streamlit as st
from supabase import create_client, Client

def init_supabase():
    url_raw = os.environ.get("SUPABASE_URL")
    key_raw = os.environ.get("SUPABASE_KEY")
    if not url_raw or not key_raw:
        st.error("Brak kluczy Supabase!")
        st.stop()
    url = url_raw.split("/rest/v1")[0].strip().rstrip("/")
    return create_client(url, key_raw.strip())

def get_pro_days(supabase, email):
    try:
        res = supabase.table("profiles").select("pro_until").eq("email", email).single().execute()
        if res.data and res.data['pro_until']:
            pro_until = datetime.datetime.fromisoformat(res.data['pro_until'].replace('Z', '+00:00'))
            now = datetime.datetime.now(datetime.timezone.utc)
            if pro_until > now:
                return (pro_until - now).days + 1
        return -1
    except:
        return -1
