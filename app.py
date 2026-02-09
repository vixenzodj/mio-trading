import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="PRO Visual Analyzer")
st_autorefresh(interval=600000, key="datarefresh")

def calculate_all_greeks(row, spot, t, r=0.04):
    s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
    if v <= 0 or t <= 0: return pd.Series([0]*5, index=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    pdf = norm.pdf(d1)
    gamma = pdf / (s * v * np.sqrt(t))
    vanna = (pdf * d1) / v
    charm = (pdf * ( (r/(v*np.sqrt(t))) - (d1/(2*t)) ))
    vega = s * pdf * np.sqrt(t)
    theta = -(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d1 - v * np.sqrt(t))
    return pd.Series([gamma*oi, vanna*oi, charm*oi, vega*oi, theta*oi], index=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

@st.cache_data(ttl=600)
def get_data_engine(symbol, exp_idx, noise_pct):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    all_exps = t_obj.options
    sel_exp = all_exps[exp_idx]
    opts = t_obj.option_chain(sel_exp)
    
    t_days = (datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days
    t_yrs = max(t_days, 0.5) / 365
    
    calls, puts = opts.calls.copy(), opts.puts.copy()
    
    # Range dinamico basato sul nuovo filtro potenziato
    l_lim, u_lim = price * (1 - noise_pct/100), price * (1 + noise_pct/100)
    
    # Calcolo Walls sui dati filtrati per evitare gli strike "fantasma" a 600
    f_calls = calls[(calls['strike'] >= price) & (calls['strike'] <= u_lim)]
    f_puts = puts[(puts['strike'] <= price) & (puts['strike'] >= l_lim)]
    
    cw = f_calls.loc[f_calls['openInterest'].idxmax(), 'strike'] if not f_calls.empty else price
    pw = f_puts.loc[f_puts['openInterest'].idxmax(), 'strike'] if not f_puts.empty else price
    
    c_grk = calls.apply(lambda r: calculate_all_greeks(r, price, t_yrs), axis=1)
    p_grk = puts.apply(lambda r: calculate_all_greeks(r, price, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': calls['strike']})
    for m in ['Gamma', 'Vanna', 'Charm']: df[m] = c_grk[m] - p_grk[m]
    for m in ['Vega', 'Theta']: df[m] = c_grk[m] + p_grk[m]
    
    return price, df, sel_exp, cw, pw, l_lim, u_lim

# --- UI ---
st.sidebar.header("🔍 Radar Sensibilità")
ticker = st.sidebar.selectbox("Asset", ['QQQ', 'SPY', 'NVDA', 'AAPL', 'TSLA'])
t_obj_side = yf.Ticker(ticker)
exps = t_obj_side.options
exp_idx = st.sidebar.selectbox("Scadenza", range(len(exps)), format_func=lambda x: exps[x])

# Filtro potenziato fino al 50%
noise_zoom = st.sidebar.slider("Ampiezza Analisi (%)", 1, 50, 10)
metric = st.sidebar.selectbox("Metrica", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

try:
    spot, df, date, cw, pw, low, high = get_data_engine(ticker, exp_idx, noise_zoom)
    
    df_plot = df[(df['strike'] >= low) & (df['strike'] <= high)].copy()
    
    # Calcolo spessore dinamico: più zoommi, più le barre diventano "cicciotte"
    bar_width = 0.05 if noise_zoom < 5 else 0.2

    fig = go.Figure()

    # Rappresentazione fisica delle barre
    fig.add_trace(go.Bar(
        y=df_plot['strike'],
        x=df_plot[metric],
        orientation='h',
        marker_color=np.where(df_plot[metric] >= 0, '#00ff88', '#ff4444'),
        name=metric,
        # Questo rende le barre più o meno "massicce" visivamente
        marker_line_width=1,
        marker_line_color="rgba(255,255,255,0.2)"
    ))

    # Linee Muri con spessore maggiorato
    fig.add_hline(y=cw, line_dash="dot", line_color="#00ff88", line_width=5,
                  annotation_text=f"🔴 CALL WALL: {cw}", annotation_position="top right")
    fig.add_hline(y=pw, line_dash="dot", line_color="#ff4444", line_width=5,
                  annotation_text=f"🟢 PUT WALL: {pw}", annotation_position="bottom right")
    fig.add_hline(y=spot, line_color="cyan", line_width=2,
                  annotation_text=f"📍 PREZZO ATTUALE: {spot:.2f}")

    fig.update_layout(
        template="plotly_dark", height=900,
        title=f"STRUTTURA FISICA {metric.upper()} - {ticker}",
        yaxis=dict(title="STRIKE", range=[low, high], autorange=False, gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(title="VOLUME ESPOSIZIONE", autorange=True, gridcolor='rgba(255,255,255,0.1)'),
        # Bargap gestisce quanto le barre sono vicine tra loro (più basso = più spesse)
        bargap=bar_width 
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Errore: {e}")
