import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Professional Trading Terminal")
st_autorefresh(interval=900000, key="datarefresh")

# --- MOTORE DI CALCOLO ---
def calculate_all_greeks(row, spot, t=1/365, r=0.04):
    s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
    if v <= 0 or t <= 0: return pd.Series([0]*5, index=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = pdf / (s * v * np.sqrt(t))
    vanna = (pdf * d1) / v
    charm = (pdf * ( (r/(v*np.sqrt(t))) - (d1/(2*t)) ))
    vega = s * pdf * np.sqrt(t)
    theta = -(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2)

    return pd.Series([gamma*oi, vanna*oi, charm*oi, vega*oi, theta*oi], 
                     index=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

def format_number(num):
    """Rende i numeri leggibili per un trader"""
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.2f} K"
    return f"{num:.2f}"

@st.cache_data(ttl=900)
def get_full_data(symbol):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    exp = t_obj.options[0]
    opts = t_obj.option_chain(exp)
    
    calls, puts = opts.calls.copy(), opts.puts.copy()
    c_greeks = calls.apply(lambda r: calculate_all_greeks(r, price), axis=1)
    p_greeks = puts.apply(lambda r: calculate_all_greeks(r, price), axis=1)
    
    full_df = pd.DataFrame({'strike': calls['strike']})
    for m in ['Gamma', 'Vanna', 'Charm']:
        full_df[m] = c_greeks[m] - p_greeks[m]
    for m in ['Vega', 'Theta']:
        full_df[m] = c_greeks[m] + p_greeks[m]
    
    # Identificazione Walls (Strike con OI massimo)
    call_wall = calls.loc[calls['openInterest'].idxmax(), 'strike']
    put_wall = puts.loc[puts['openInterest'].idxmax(), 'strike']
        
    return price, full_df, exp, call_wall, put_wall

# --- INTERFACCIA ---
st.title("🏹 ABS GEX & GREEKS PROFILE")

ticker = st.sidebar.selectbox("Asset", ['QQQ', 'SPY', 'NVDA', 'TSLA', 'AAPL'], index=0)
metrica = st.sidebar.selectbox("Metrica", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

try:
    spot, df, exp, c_wall, p_wall = get_full_data(ticker)
    df_plot = df[(df['strike'] > spot * 0.90) & (df['strike'] < spot * 1.10)]

    # --- GRAFICO PRO ---
    fig = go.Figure()
    colors = {'Gamma': '#00ff88', 'Vanna': '#ffcc00', 'Charm': '#ff00ff', 'Vega': '#00aaff', 'Theta': '#ff4444'}
    
    fig.add_trace(go.Bar(y=df_plot['strike'], x=df_plot[metrica], orientation='h',
                         marker_color=colors[metrica], name=metrica))

    # Linea SPOT
    fig.add_hline(y=spot, line_dash="dash", line_color="cyan", line_width=2,
                 annotation_text=f"SPOT: {spot:.2f}", annotation_position="bottom right")
    
    # Linea CALL WALL
    fig.add_hline(y=c_wall, line_dash="dot", line_color="#00ff88", line_width=3,
                 annotation_text=f"MAJOR CALL WALL: {c_wall}", annotation_position="top right")
    
    # Linea PUT WALL
    fig.add_hline(y=p_wall, line_dash="dot", line_color="#ff4444", line_width=3,
                 annotation_text=f"MAJOR PUT WALL: {p_wall}", annotation_position="bottom right")
    
    fig.update_layout(template="plotly_dark", height=800, barmode='relative',
                      title=f"{ticker} - {metrica} Straddle Profile (Expiry: {exp})")
    
    st.plotly_chart(fig, use_container_width=True)

    # --- METRICHE LEGGIBILI ---
    st.divider()
    cols = st.columns(5)
    metrics_list = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
    for i, m in enumerate(metrics_list):
        total = df[m].sum()
        cols[i].metric(f"Total {m}", format_number(total))

    st.info(f"💡 **Analisi Operativa:** Lo Spot è a {spot:.2f}. Il Call Wall a {c_wall} funge da calamita/resistenza, mentre il Put Wall a {p_wall} è il supporto principale dei Market Maker.")

except Exception as e:
    st.error(f"Errore: {e}")
