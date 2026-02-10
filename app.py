import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V32", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def fast_greeks(df, s, t, r=0.045):
    # Calcolo su array numpy puri per evitare conflitti di indici
    k = df['strike'].values
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    oi = df['openInterest'].values
    types = df['type'].values
    t = max(t, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    is_call = (types == 'call')
    theta_part1 = -(s * pdf * v) / (2 * np.sqrt(t))
    theta_part2 = r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))
    theta = (theta_part1 - theta_part2) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    # Restituiamo un nuovo dataframe con indice pulito
    return pd.DataFrame({
        'strike': k,
        'Gamma': gamma * mult,
        'Vanna': vanna * mult,
        'Vega': vega,
        'Theta': theta
    }).reset_index(drop=True)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V32")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        def_idx = 5 if "NDX" in t_str or "SPX" in t_str else 2
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 2, 5, 10, 25, 50, 100, 250], index=def_idx)
        zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 15, 5)
        main_metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Vega', 'Theta'])

        hist = t_obj.history(period='2d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- PULIZIA RADICALE ---
            c = chain.calls[['strike', 'impliedVolatility', 'openInterest']].copy()
            p = chain.puts[['strike', 'impliedVolatility', 'openInterest']].copy()
            c['type'], p['type'] = 'call', 'put'
            
            # Uniamo e raggruppiamo subito per eliminare duplicati nativi di Yahoo
            df_raw = pd.concat([c, p], ignore_index=True)
            df_raw = df_raw.groupby(['strike', 'type'], as_index=False).first()
            
            # Calcolo Greche
            df_greeks = fast_greeks(df_raw, spot, t_yrs)
            
            # Calcolo livelli chiave su dati aggregati per strike
            df_total_strike = df_greeks.groupby('strike', as_index=False).sum().sort_values('strike').reset_index(drop=True)
            
            # Zero Gamma (su indice garantito unico)
            df_total_strike['cum_gamma'] = df_total_strike['Gamma'].cumsum()
            z_gamma_idx = df_total_strike['cum_gamma'].abs().idxmin()
            z_gamma = df_total_strike.loc[z_gamma_idx, 'strike']

            # Aggregazione visiva (Binning)
            df_total_strike['bin'] = np.floor(df_total_strike['strike'] / strike_step) * strike_step
            df_plot = df_total_strike.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'})
            
            # Zoom
            limit = (spot * zoom_pct) / 100
            df_view = df_plot[(df_plot['strike'] >= spot - limit) & (df_plot['strike'] <= spot + limit)].copy()
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- INTERFACCIA ---
            st.markdown(f"## 🏛️ {active_t} Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPOT", f"{spot:.2f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.0f}")
            m3.metric("CALL WALL", f"{call_wall:.0f}")
            
            net_g = df_total_strike['Gamma'].sum()
            b_txt, b_col = ("BULLISH", "#00ff00") if net_g > 0 else ("BEARISH", "#ff4444")
            m4.markdown(f"<div style='text-align:center; padding:8px; border-radius:5px; border:1px solid {b_col}; color:{b_col}'><b>BIAS: {b_txt}</b></div>", unsafe_allow_html=True)

            # --- PLOT ---
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h',
                marker_color=['#00ff00' if x > 0 else '#00aaff' for x in df_view[main_metric]],
                width=strike_step * 0.8
            ))
            
            # Linee
            for val, col, txt in zip([spot, z_gamma, call_wall, put_wall], 
                                     ['cyan', 'yellow', 'red', 'green'], 
                                     ['SPOT', '0G', 'CWALL', 'PWALL']):
                fig.add_hline(y=val, line_color=col, line_dash="dash" if col=='yellow' else "solid", annotation_text=txt)

            fig.update_layout(
