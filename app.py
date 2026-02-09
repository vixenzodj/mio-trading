import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe GEX Pro", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# --- FIX TICKER (NDX -> ^NDX) ---
def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    if s in ["NDX", "SPX", "RUT", "VIX"]: return f"^{s}"
    return s

# --- MOTORE GRECHE ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
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
    except: return pd.Series([0.0]*5)

# --- ANALISI BIAS ---
def analyze_tf(t_obj, spot, exp_idx):
    try:
        exps = t_obj.options
        if not exps: return "N/D", "#555", "No Options"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g, v = c[0].sum() - p[0].sum(), c[1].sum() - p[1].sum()
        if g > 0 and v > 0: return "LONG", "#00ff00", "High Prob"
        if g < 0 and v < 0: return "SHORT", "#ff4444", "High Prob"
        return "NEUTRAL", "#ffff00", "Volatile"
    except: return "ERR", "#555", "Data Error"

# --- SIDEBAR ---
st.sidebar.header("🕹️ TERMINAL")
raw_ticker = st.sidebar.text_input("TICKER (es. NDX, SPX, NVDA)", "NDX")
zoom = st.sidebar.slider("ZOOM %", 1, 30, 5)
metric = st.sidebar.radio("GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

# --- LOGICA PRINCIPALE ---
t_str = fix_ticker(raw_ticker)

if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        
        if hist.empty:
            st.warning("In attesa di dati validi da Yahoo Finance...")
        else:
            spot = hist['Close'].iloc[-1]
            
            # --- SEZIONE BIAS ---
            st.subheader(f"📡 Multi-Timeframe Bias: {raw_ticker.upper()}")
            m1, m2, m3 = st.columns(3)
            with st.spinner('Calcolando Bias...'):
                b1, c1, p1 = analyze_tf(t_obj, spot, 0)
                b2, c2, p2 = analyze_tf(t_obj, spot, 2)
                b3, c3, p3 = analyze_tf(t_obj, spot, 5)
            
            m1.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c1}; border-radius:10px;'><h4>SCALP (0DTE)</h4><h2 style='color:{c1};'>{b1}</h2><small>{p1}</small></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c2}; border-radius:10px;'><h4>INTRA (Weekly)</h4><h2 style='color:{c2};'>{b2}</h2><small>{p2}</small></div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c3}; border-radius:10px;'><h4>SWING (Monthly)</h4><h2 style='color:{c3};'>{b3}</h2><small>{p3}</small></div>", unsafe_allow_html=True)

            # --- GRAFICO FOCUS ---
            st.divider()
            exps = t_obj.options
            # Carichiamo la scadenza 0DTE per il grafico principale
            ch_main = t_obj.option_chain(exps[0])
            t_yrs_m = max((datetime.strptime(exps[0], '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            
            c_m = ch_main.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs_m), axis=1)
            p_m = ch_main.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs_m), axis=1)
            
            df = pd.DataFrame({'strike': ch_main.calls['strike']})
            df['Gamma'], df['Vanna'], df['Charm'] = c_m[0]-p_m[0], c_m[1]-p_m[1], c_m[2]-p_m[2]
            df['Vega'], df['Theta'] = c_m[3]+p_m[3], c_m[4]+p_m[4]
            
            # Filtro Zoom
            l, u = spot * (1 - zoom/100), spot * (1 + zoom/100)
            df_z = df[(df['strike']>=l) & (df['strike']<=u)]
            
            z_flip = df_z.loc[df_z[metric].abs().idxmin(), 'strike']
            max_v = df_z[metric].abs().max()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[metric], orientation='h', marker_color=np.where(df_z[metric]>=0, '#00ff00', '#00aaff')))
            fig.add_hline(y=spot, line_color="cyan", annotation_text="LIVE PRICE")
            fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
            fig.update_layout(template="plotly_dark", height=600, title=f"Profilo {metric} - {raw_ticker}")
            st.plotly_chart(fig, use_container_width=True)

            # --- METRICHE TOTALI ---
            st.subheader("📊 Greche Totali (Full Chain)")
            cols = st.columns(5)
            for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
                total = df[m].sum()
                cols[i].metric(f"Total {m}", f"{total/1e6:.2f}M" if abs(total)>1e5 else f"{total:.2f}")

    except Exception as e:
        st.error(f"Errore nel caricamento: {e}")
else:
    st.info("Inserisci un ticker nella barra laterale per iniziare.")
