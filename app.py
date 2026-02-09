import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Institutional Multi-Timeframe Terminal", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

def fix_ticker(symbol):
    s = symbol.upper().strip()
    if s in ["NDX", "SPX", "RUT", "VIX"]: return f"^{s}"
    return s

def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
        r = 0.045
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0]*5)

@st.cache_data(ttl=60)
def analyze_timeframe(t_obj, spot, exp_idx):
    try:
        exps = t_obj.options
        sel_exp = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        chain = t_obj.option_chain(sel_exp)
        c_res = chain.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p_res = chain.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g_net = c_res[0].sum() - p_res[0].sum()
        v_net = c_res[1].sum() - p_res[1].sum()
        
        if g_net > 0 and v_net > 0: return "LONG", "#00ff00", "Alta Probabilità"
        elif g_net < 0 and v_net < 0: return "SHORT", "#ff4444", "Alta Probabilità"
        elif g_net < 0: return "SHORT", "#ffaa00", "Debole/Volatile"
        else: return "LONG", "#00aaff", "Debole/Accumulo"
    except: return "N/D", "#555555", "Dati Assenti"

@st.cache_data(ttl=60)
def load_main_data(symbol, exp_idx, zoom_val):
    ticker_str = fix_ticker(symbol)
    t_obj = yf.Ticker(ticker_str)
    hist = t_obj.history(period='1d')
    if hist.empty: return None, None, None, None, None
    spot = hist['Close'].iloc[-1]
    exps = t_obj.options
    sel_exp = exps[min(exp_idx, len(exps)-1)]
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    chain = t_obj.option_chain(sel_exp)
    c_res = chain.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    p_res = chain.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    df = pd.DataFrame({'strike': chain.calls['strike']})
    df['Gamma'], df['Vanna'], df['Charm'] = c_res[0]-p_res[0], c_res[1]-p_res[1], c_res[2]-p_res[2]
    df['Vega'], df['Theta'] = c_res[3]+p_res[3], c_res[4]+p_res[4]
    l, u = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
    return spot, df[(df['strike']>=l) & (df['strike']<=u)], sel_exp, df, t_obj

# --- INTERFACCIA ---
st.sidebar.header("🕹️ TERMINAL CONTROL")
input_ticker = st.sidebar.text_input("TICKER (es. NDX, SPX, NVDA)", "NDX").upper()
zoom = st.sidebar.slider("ZOOM %", 1, 30, 5)
metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

res = load_main_data(input_ticker, 0, zoom)
if res[0]:
    spot, df_z, exp_d, df_f, t_obj = res
    
    # --- MATRICE DI INTELLIGENZA (IL MOTORE CHE VOLEVI) ---
    st.subheader("📡 Multi-Timeframe Market Bias")
    m1, m2, m3 = st.columns(3)
    
    # Calcolo dei 3 stadi
    b_scalp, c_scalp, p_scalp = analyze_timeframe(t_obj, spot, 0) # 0DTE
    b_intra, c_intra, p_intra = analyze_timeframe(t_obj, spot, 2) # Weekly
    b_swing, c_swing, p_swing = analyze_timeframe(t_obj, spot, 5) # Monthly
    
    m1.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; border:2px solid {c_scalp};'><h3 style='margin:0;'>SCALPING (Breve)</h3><h1 style='color:{c_scalp};'>{b_scalp}</h1><p>{p_scalp}</p></div>", unsafe_allow_html=True)
    m2.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; border:2px solid {c_intra};'><h3 style='margin:0;'>INTRADAY (Medio)</h3><h1 style='color:{c_intra};'>{b_intra}</h1><p>{p_intra}</p></div>", unsafe_allow_html=True)
    m3.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; border:2px solid {c_swing};'><h3 style='margin:0;'>SWING (Lungo)</h3><h1 style='color:{c_swing};'>{b_swing}</h1><p>{p_swing}</p></div>", unsafe_allow_html=True)

    # GRAFICO
    st.divider()
    z_flip = df_z.loc[df_z[metric].abs().idxmin(), 'strike']
    max_v = df_z[metric].abs().max()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[metric], orientation='h', marker_color=np.where(df_z[metric]>=0, '#00ff00', '#00aaff')))
    fig.add_hline(y=spot, line_color="cyan", annotation_text=f"LIVE: {spot:.2f}")
    fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
    fig.update_layout(template="plotly_dark", height=600, title=f"PROFILO {metric} - {input_ticker} (FOCUS BREVE TERMINE)")
    st.plotly_chart(fig, use_container_width=True)

    # TUTTE LE METRICHE
    st.subheader("📊 Greche Totali (Asset Global Exposure)")
    cols = st.columns(5)
    for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
        val = df_f[m].sum()
        cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")
