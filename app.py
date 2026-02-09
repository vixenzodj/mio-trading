import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Institutional GEX Dashboard", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT", "VIX", "DJI"] else s

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

def analyze_tf(t_obj, spot, exp_idx):
    try:
        exps = t_obj.options
        if not exps: return "N/D", "#555"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: (norm.pdf((np.log(spot/r['strike']) + (0.045 + 0.5 * r['impliedVolatility']**2) * t_yrs) / (r['impliedVolatility'] * np.sqrt(t_yrs))) / (spot * r['impliedVolatility'] * np.sqrt(t_yrs))) * r['openInterest'] * 100, axis=1)
        p = ch.puts.apply(lambda r: (norm.pdf((np.log(spot/r['strike']) + (0.045 + 0.5 * r['impliedVolatility']**2) * t_yrs) / (r['impliedVolatility'] * np.sqrt(t_yrs))) / (spot * r['impliedVolatility'] * np.sqrt(t_yrs))) * r['openInterest'] * 100, axis=1)
        g_net = c.sum() - p.sum()
        return ("LONG", "#00ff00") if g_net > 0 else ("SHORT", "#ff4444")
    except: return "ERR", "#555"

# --- SIDEBAR ---
st.sidebar.header("🕹️ TERMINAL SETTINGS")
sel_ticker = st.sidebar.selectbox("ASSET", ["CERCA..."] + sorted(TICKER_LIST))
manual_t = st.sidebar.text_input("MANUAL TICKER", "")
active_t = manual_t if manual_t else (sel_ticker if sel_ticker != "CERCA..." else "NDX")
trade_mode = st.sidebar.selectbox("TIMEFRAME", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
zoom_val = st.sidebar.slider("ZOOM %", 1, 30, 5)
main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

t_str = fix_ticker(active_t)
if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = hist['Close'].iloc[-1]
            exps = t_obj.options
            idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
            sel_exp = exps[min(idx, len(exps)-1)]
            
            # --- CALCOLO DATI ---
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            ch = t_obj.option_chain(sel_exp)
            c_m = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            p_m = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike']})
            df['Gamma'], df['Vanna'], df['Charm'] = c_m[0]-p_m[0], c_m[1]-p_m[1], c_m[2]-p_m[2]
            df['Vega'], df['Theta'] = c_m[3]+p_m[3], c_m[4]+p_m[4]
            
            # IDENTIFICAZIONE MURI (WALLS)
            call_wall_strike = df.loc[df['Gamma'].idxmax(), 'strike']
            put_wall_strike = df.loc[df['Gamma'].idxmin(), 'strike']

            # --- HEADER BIAS ---
            st.subheader(f"📡 Radar: {active_t.upper()}")
            m1, m2, m3 = st.columns(3)
            b1, c1 = analyze_tf(t_obj, spot, 0)
            b2, c2 = analyze_tf(t_obj, spot, 2)
            b3, c3 = analyze_tf(t_obj, spot, 5)
            m1.metric("SCALP", b1, delta_color="normal")
            m2.metric("INTRA", b2, delta_color="normal")
            m3.metric("SWING", b3, delta_color="normal")

            # --- GRAFICO DINAMICO ---
            l, u = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            df_z = df[(df['strike']>=l) & (df['strike']<=u)]
            z_flip = df_z.loc[df_z[main_metric].abs().idxmin(), 'strike']
            
            fig = go.Figure()
            # Barre Metrica
            fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[main_metric], orientation='h', 
                                 marker_color=np.where(df_z[main_metric]>=0, '#00ff00', '#00aaff'), name=main_metric))
            # Linea Prezzo Live
            fig.add_hline(y=spot, line_color="cyan", line_width=3, annotation_text=f"SPOT: {spot:.2f}", annotation_position="top right")
            # Linea Zero Flip
            fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
            # CALL WALL & PUT WALL
            fig.add_hline(y=call_wall_strike, line_color="red", line_width=4, annotation_text=f"CALL WALL: {call_wall_strike}", annotation_font_color="red")
            fig.add_hline(y=put_wall_strike, line_color="green", line_width=4, annotation_text=f"PUT WALL: {put_wall_strike}", annotation_font_color="green")

            fig.update_layout(template="plotly_dark", height=750, title=f"Profilo {main_metric} - Scadenza: {sel_exp}", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- RIEPILOGO METRICHE ---
            st.divider()
            cols = st.columns(5)
            for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
                val = df[m].sum()
                cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")

    except Exception as e: st.error(f"Seleziona un asset o attendi... {e}")
