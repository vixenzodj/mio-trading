import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Professional GEX Terminal V6", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA", "AAPL", "IBIT", "MSTR"]

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE GRECHE ---
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
        if not exps: return "N/D", "#555", "No Data"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g_net = c[0].sum() - p[0].sum()
        return ("LONG", "#00ff00") if g_net > 0 else ("SHORT", "#ff4444")
    except: return "ERR", "#555"

# --- SIDEBAR ---
st.sidebar.header("🎯 GEX SETTINGS")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
trade_mode = st.sidebar.selectbox("TIMEFRAME", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
strike_step = st.sidebar.selectbox("GRANULARITÀ (Step)", [5, 10, 25, 50, 100, 250], index=3)
num_levels = st.sidebar.slider("LIVELLI VISIBILI", 10, 60, 30)
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
        
        # --- CALCOLO ---
        t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel_exp)
        c_res = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p_res = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        
        df = pd.DataFrame({'strike': ch.calls['strike']})
        df['Gamma'], df['Vanna'], df['Charm'] = c_res[0]-p_res[0], c_res[1]-p_res[1], c_res[2]-p_res[2]
        df['Vega'], df['Theta'] = c_res[3]+p_res[3], c_res[4]+p_res[4]

        # --- RADAR BIAS (RIPRISTINATO) ---
        st.subheader(f"📡 Radar: {active_t} @ {spot:.2f}")
        r1, r2, r3 = st.columns(3)
        b1, c1 = analyze_tf(t_obj, spot, 0)
        b2, c2 = analyze_tf(t_obj, spot, 2)
        b3, c3 = analyze_tf(t_obj, spot, 5)
        r1.markdown(f"<div style='text-align:center;border:2px solid {c1};padding:10px;border-radius:10px;'>SCALP: <b style='color:{c1};'>{b1}</b></div>", unsafe_allow_html=True)
        r2.markdown(f"<div style='text-align:center;border:2px solid {c2};padding:10px;border-radius:10px;'>INTRA: <b style='color:{c2};'>{b2}</b></div>", unsafe_allow_html=True)
        r3.markdown(f"<div style='text-align:center;border:2px solid {c3};padding:10px;border-radius:10px;'>SWING: <b style='color:{c3};'>{b3}</b></div>", unsafe_allow_html=True)

        # --- LOGICA GRANULARITÀ (MOLTO IMPORTANTE) ---
        # Arrotondiamo gli strike allo step scelto e raggruppiamo la liquidità
        df['strike_group'] = (df['strike'] / strike_step).round() * strike_step
        df_grouped = df.groupby('strike_group').sum().reset_index()
        df_grouped = df_grouped.rename(columns={'strike_group': 'strike'})

        # Filtro intorno allo Spot
        df_grouped['dist'] = abs(df_grouped['strike'] - spot)
        df_plot = df_grouped.sort_values('dist').head(num_levels).sort_values('strike')

        # Muri reali (su catena completa)
        call_wall = df.loc[df['Gamma'].idxmax(), 'strike']
        put_wall = df.loc[df['Gamma'].idxmin(), 'strike']
        z_flip = df_plot.loc[df_plot[main_metric].abs().idxmin(), 'strike']

        # --- GRAFICO ---
        fig = go.Figure()
        colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
        
        # Trasformiamo strike in stringa per avere barre equispaziate e leggibili
        fig.add_trace(go.Bar(y=df_plot['strike'].astype(str), x=df_plot[main_metric], 
                             orientation='h', marker_color=colors))

        # Linee di Livello (usiamo l'indice della stringa per posizionarle correttamente)
        fig.add_hline(y=str(float(call_wall)), line_color="red", line_width=3, annotation_text=f"CALL WALL: {call_wall}")
        fig.add_hline(y=str(float(put_wall)), line_color="#00ff00", line_width=3, annotation_text=f"PUT WALL: {put_wall}")
        fig.add_hline(y=str(float(z_flip)), line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")

        fig.update_layout(template="plotly_dark", height=800, 
                          yaxis=dict(title="STRIKE", autorange="reversed", type='category'),
                          title=f"PROFILO {main_metric} - {active_t} (Step: {strike_step})")
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Errore: {e}")
