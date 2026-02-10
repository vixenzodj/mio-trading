import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V33", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE PURO (NUMPY) ---
def compute_greeks_numpy(df, s, t, r=0.045):
    # Trasformiamo in array per evitare errori di indice duplicate
    k = df['strike'].to_numpy()
    v = df['impliedVolatility'].to_numpy()
    v = np.where(v <= 0, 1e-9, v)
    oi = df['openInterest'].to_numpy()
    types = df['type'].to_numpy()
    t = max(t, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    # Theta
    is_call = (types == 'call')
    theta_part1 = -(s * pdf * v) / (2 * np.sqrt(t))
    theta_part2 = r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))
    theta = (theta_part1 - theta_part2) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    return pd.DataFrame({
        'strike': k,
        'Gamma': gamma * mult,
        'Vanna': vanna * mult,
        'Vega': vega,
        'Theta': theta
    })

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V33")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        
        # Logica Step automatica
        def_idx = 5 if "NDX" in t_str or "SPX" in t_str else 2
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 2, 5, 10, 25, 50, 100, 250], index=def_idx)
        zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 15, 5)
        main_metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Vega', 'Theta'])

        hist = t_obj.history(period='2d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- PULIZIA DEFINITIVA ---
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            # Reset index totale prima dell'unione
            df_raw = pd.concat([c, p], axis=0, ignore_index=True)
            
            # Raggruppamento per eliminare i duplicati di
