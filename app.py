import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Multi-Strategy GEX Terminal")

# --- TRADING MODE SELECTOR (SIDEBAR) ---
st.sidebar.header("🕹️ STRATEGY MODE")
trading_mode = st.sidebar.selectbox(
    "Seleziona Tipo di Trading",
    ["SCALPING (0DTE Focus)", "INTRADAY (Weekly Focus)", "SWING (Monthly/Institutional)"]
)

# Configurazione dinamica in base al modo scelto
if trading_mode == "SCALPING (0DTE Focus)":
    refresh_rate = 60000 # 1 minuto
    default_zoom = 2
    expiry_auto_select = 0 # La più vicina
    st_autorefresh(interval=refresh_rate, key="scalp_refresh")
elif trading_mode == "INTRADAY (Weekly Focus)":
    refresh_rate = 300000 # 5 minuti
    default_zoom = 7
    expiry_auto_select = 1 # Prossima settimana
    st_autorefresh(interval=refresh_rate, key="intra_refresh")
else: # SWING
    refresh_rate = 900000 # 15 minuti
    default_zoom = 15
    expiry_auto_select = 3 # Scadenza mensile tipica
    st_autorefresh(interval=refresh_rate, key="swing_refresh")

# --- MOTORE DI CALCOLO (OTTIMIZZATO) ---
def get_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
        d1 = (np.log(s/k) + (0.04 + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        pdf = norm.pdf(d1)
        gamma = pdf / (s * v * np.sqrt(t_yrs))
        vanna = (pdf * d1) / v
        charm = (pdf * ( (0.04/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) ))
        return pd.Series([gamma * oi * 100, vanna * oi, charm * oi, s * pdf * np.sqrt(t_yrs) * oi, 0])
    except: return pd.Series([0]*5)

@st.cache_data(ttl=60)
def get_market_data(ticker, exp_idx, zoom_pct):
    t_obj = yf.Ticker(ticker)
    hist = t_obj.history(period='1d')
    if hist.empty: return None
    spot = hist['Close'].iloc[-1]
    sel_exp = t_obj.options[exp_idx]
    dt_exp = datetime.strptime(sel_exp, '%Y-%m-%d')
    t_yrs = max((dt_exp - datetime.now()).days, 0.5) / 365
    opts = t_obj.option_chain(sel_exp)
    
    c_grk = opts.calls.apply(lambda r: get_greeks(r, spot, t_yrs), axis=1)
    p_grk = opts.puts.apply(lambda r: get_greeks(r, spot, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': opts.calls['strike']})
    df['Gamma'] = c_grk[0] - p_grk[0]
    df['Vanna'] = c_grk[1] - p_grk[1]
    df['Charm'] = c_grk[2] - p_grk[2]
    
    l_bound, u_bound = spot * (1 - zoom_pct/100), spot * (1 + zoom_pct/100)
    return spot, df[(df['strike'] >= l_bound) & (df['strike'] <= u_bound)], sel_exp, df

# --- INTERFACCIA ---
st.title(f"🏛️ {trading_mode}")
ticker = st.sidebar.text_input("SYMBOL", "QQQ").upper()
try:
    exps = yf.Ticker(ticker).options
    exp_idx = st.sidebar.selectbox("SCADENZA", range(len(exps)), index=min(expiry_auto_select, len(exps)-1), format_func=lambda x: exps[x])
    zoom = st.sidebar.slider("ZOOM AREA %", 1, 40, default_zoom)
    metric_sel = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'])
except: st.stop()

data = get_market_data(ticker, exp_idx, zoom)
if data:
    spot, df_plot, exp_date, df_full = data
    zero_flip = df_plot.loc[df_plot[metric_sel].abs().idxmin(), 'strike']
    
    # GRAFICO
    max_val = df_plot[metric_sel].abs().max()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot['strike'], x=df_plot[metric_sel], orientation='h',
        marker_color=np.where(df_plot[metric_sel] >= 0, '#00ff00', '#00aaff'),
    ))
    
    fig.add_hline(y=spot, line_color="cyan", annotation_text="SPOT")
    fig.add_hline(y=zero_flip, line_dash="dash", line_color="yellow", annotation_text=f"ZERO {metric_sel}")

    fig.update_layout(template="plotly_dark", height=800, xaxis=dict(range=[-max_val*1.1, max_val*1.1]))
    st.plotly_chart(fig, use_container_width=True)

    # INFO BOX OPERATIVA
    st.subheader("💡 Analisi Strategica")
    if trading_mode == "SCALPING (0DTE Focus)":
        st.write("Target: Movimenti rapidi. Monitora lo **Zero Gamma** (Giallo) come punto di rottura per accelerazioni improvvise.")
    elif trading_mode == "INTRADAY (Weekly Focus)":
        st.write("Target: Range giornalieri. Le barre più lunghe indicano dove il prezzo probabilmente 'stagnerà' durante la sessione.")
    else:
        st.write("Target: Grandi Istituzioni. Questi muri rappresentano coperture pesanti che durano settimane; cercate inversioni di trend primario qui.")
