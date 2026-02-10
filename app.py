import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX Intelligence Terminal V7", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA", "AAPL", "IBIT", "MSTR"]

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
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
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g_net = c[0].sum() - p[0].sum()
        return ("LONG", "#00ff00") if g_net > 0 else ("SHORT", "#ff4444")
    except: return "ERR", "#555"

# --- SIDEBAR ---
st.sidebar.header("🕹️ TERMINAL CONTROL")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
trade_mode = st.sidebar.selectbox("TIMEFRAME", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
strike_step = st.sidebar.selectbox("STEP STRIKE", [5, 10, 25, 50, 100], index=3)
num_levels = st.sidebar.slider("LIVELLI VISIBILI", 10, 80, 40)
main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

t_str = fix_ticker(active_t)
try:
    t_obj = yf.Ticker(t_str)
    hist = t_obj.history(period='1d')
    if not hist.empty:
        spot = hist['Close'].iloc[-1]
        exps = t_obj.options
        idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
        sel_exp = exps[min(idx, len(exps)-1)]
        
        # --- CALCOLO CORE ---
        t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel_exp)
        c_vals = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p_vals = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        
        df = pd.DataFrame({'strike': ch.calls['strike']})
        df['Gamma'], df['Vanna'], df['Charm'] = c_vals[0]-p_vals[0], c_vals[1]-p_vals[1], c_vals[2]-p_vals[2]
        df['Vega'], df['Theta'] = c_vals[3]+p_vals[3], c_vals[4]+p_vals[4]

        # --- BIAS RADAR ---
        st.subheader(f"📡 Trend Context: {active_t} @ {spot:.2f}")
        r1, r2, r3 = st.columns(3)
        b1, c1 = analyze_tf(t_obj, spot, 0)
        b2, c2 = analyze_tf(t_obj, spot, 2)
        b3, c3 = analyze_tf(t_obj, spot, 5)
        r1.markdown(f"<div style='text-align:center;border:2px solid {c1};padding:10px;border-radius:10px;'>SCALP: <b style='color:{c1};'>{b1}</b></div>", unsafe_allow_html=True)
        r2.markdown(f"<div style='text-align:center;border:2px solid {c2};padding:10px;border-radius:10px;'>INTRA: <b style='color:{c2};'>{b2}</b></div>", unsafe_allow_html=True)
        r3.markdown(f"<div style='text-align:center;border:2px solid {c3};padding:10px;border-radius:10px;'>SWING: <b style='color:{c3};'>{b3}</b></div>", unsafe_allow_html=True)

        # --- LOGICA GRANULARITÀ (FIXED) ---
        df['strike_bin'] = (df['strike'] / strike_step).round() * strike_step
        df_grouped = df.groupby('strike_bin')[['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']].sum().reset_index()
        df_grouped.rename(columns={'strike_bin': 'strike'}, inplace=True)

        # Filtraggio distanza dallo spot
        df_grouped['abs_dist'] = (df_grouped['strike'] - spot).abs()
        df_plot = df_grouped.sort_values('abs_dist').head(num_levels).sort_values('strike')

        # Livelli Chiave
        call_wall = df.loc[df['Gamma'].idxmax(), 'strike']
        put_wall = df.loc[df['Gamma'].idxmin(), 'strike']
        z_flip = df_plot.loc[df_plot[main_metric].abs().idxmin(), 'strike']

        # --- GRAFICO MIRROR ---
        fig = go.Figure()
        colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
        
        fig.add_trace(go.Bar(
            y=df_plot['strike'].astype(str), 
            x=df_plot[main_metric], 
            orientation='h', 
            marker_color=colors,
            text=[f"{v/1e3:.1f}k" for v in df_plot[main_metric]],
            textposition='outside'
        ))

        # Linee Orizzontali
        fig.add_hline(y=str(float(call_wall)), line_color="red", line_width=3, annotation_text=f"CALL WALL: {call_wall}")
        fig.add_hline(y=str(float(put_wall)), line_color="#00ff00", line_width=3, annotation_text=f"PUT WALL: {put_wall}")
        fig.add_hline(y=str(float(z_flip)), line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")

        fig.update_layout(
            template="plotly_dark", height=900,
            title=f"PROFILO {main_metric.upper()} - {active_t} ({sel_exp})",
            xaxis=dict(title=f"Net {main_metric} Exposure", zerolinecolor="white"),
            yaxis=dict(title="STRIKE PRICE", autorange="reversed", type='category'),
            bargap=0.2
        )
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Errore nel calcolo: {e}")
