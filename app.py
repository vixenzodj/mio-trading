import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V41", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE BLINDATO ---
def fast_engine_v41(df, spot, r=0.045):
    if df.empty: return df
    s = float(spot)
    k = df['strike'].values
    # Pulizia IV e DTE per evitare divisioni per zero
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    t = np.where(df['dte_years'].values <= 0, 1e-9, df['dte_years'].values)
    oi = df['openInterest'].values
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Dollar Gamma Formula: 0.5 * Gamma * S^2 * 0.01
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    is_call = (df['type'].values == 'call')
    mult = np.where(is_call, 1, -1)
    
    # Creazione nuove colonne in modo sicuro
    df['Gamma'] = gamma * mult
    df['Vanna'] = vanna * mult
    df['Charm'] = charm * mult
    return df

# --- SIDEBAR: LOGICA GEXBOT ---
st.sidebar.header("🕹️ GEXBOT ENGINE V41")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        all_exps = t_obj.options
        today = datetime.now()
        
        # Mappatura DTE
        exp_list = []
        for e in all_exps:
            d = datetime.strptime(e, '%Y-%m-%d')
            dte = (d - today).days + 1
            exp_list.append({'date': e, 'dte': dte})
        
        df_exp_map = pd.DataFrame(exp_list)
        
        # Selezione multipla DTE (Gexbot Style)
        dte_labels = [f"{row['dte']} DTE ({row['date']})" for _, row in df_exp_map.iterrows()]
        selected_labels = st.sidebar.multiselect("SCADENZE (DTE
