import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Strike Intelligence Terminal")
st_autorefresh(interval=900000, key="datarefresh")

# --- MOTORE DI CALCOLO ---
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

def format_num(num):
    if abs(num) >= 1_000_000: return f"{num / 1_000_000:.2f}M"
    if abs(num) >= 1_000: return f"{num / 1_000:.2f}K"
    return f"{num:.2f}"

@st.cache_data(ttl=600)
def get_data_engine(symbol, exp_idx, noise):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    all_exps = t_obj.options
    sel_exp = all_exps[exp_idx]
    opts = t_obj.option_chain(sel_exp)
    
    t_days = (datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days
    t_yrs = max(t_days, 0.5) / 365
    
    calls, puts = opts.calls.copy(), opts.puts.copy()
    
    # Filtro Rumore
    l_lim, u_lim = price * (1 - noise/100), price * (1 + noise/100)
    r_calls = calls[(calls['strike'] >= price) & (calls['strike'] <= u_lim)]
    r_puts = puts[(puts['strike'] <= price) & (puts['strike'] >= l_lim)]
    
    c_wall = r_calls.loc[r_calls['openInterest'].idxmax(), 'strike'] if not r_calls.empty else price
    p_wall = r_puts.loc[r_puts['openInterest'].idxmax(), 'strike'] if not r_puts.empty else price
    
    c_grk = calls.apply(lambda r: calculate_all_greeks(r, price, t_yrs), axis=1)
    p_grk = puts.apply(lambda r: calculate_all_greeks(r, price, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': calls['strike']})
    for m in ['Gamma', 'Vanna', 'Charm']: df[m] = c_grk[m] - p_grk[m]
    for m in ['Vega', 'Theta']: df[m] = c_grk[m] + p_grk[m]
    
    return price, df, sel_exp, c_wall, p_wall

# --- INTERFACCIA ---
st.sidebar.header("⚙️ Parametri Muri")
ticker = st.sidebar.selectbox("Asset", ['QQQ', 'SPY', 'NVDA', 'AAPL', 'TSLA'])
t_obj_side = yf.Ticker(ticker)
exps = t_obj_side.options
exp_idx = st.sidebar.selectbox("Scadenza", range(len(exps)), format_func=lambda x: exps[x])
noise = st.sidebar.slider("Filtro Rumore (%)", 5, 25, 10)
metric = st.sidebar.selectbox("Visualizza", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

try:
    spot, df, date, cw, pw = get_data_engine(ticker, exp_idx, noise)
    
    # Preparazione dati Grafico
    df['dist_pct'] = ((df['strike'] / spot) - 1) * 100
    df_plot = df[(df['strike'] > spot * 0.92) & (df['strike'] < spot * 1.08)]

    fig = go.Figure()

    # Barre con Hover personalizzato
    fig.add_trace(go.Bar(
        y=df_plot['strike'],
        x=df_plot[metric],
        orientation='h',
        marker_color='#00ff88' if df_plot[metric].sum() > 0 else '#ff4444',
        hovertemplate="<b>Strike: %{y}</b><br>" +
                      "Valore: %{x}<br>" +
                      "Distanza Spot: %{customdata:.2f}%<extra></extra>",
        customdata=df_plot['dist_pct']
    ))

    # Linee Muri con Etichette Prezzo
    fig.add_hline(y=cw, line_dash="dot", line_color="#00ff88", line_width=3,
                  annotation_text=f"CALL WALL: {cw}", annotation_position="top right")
    fig.add_hline(y=pw, line_dash="dot", line_color="#ff4444", line_width=3,
                  annotation_text=f"PUT WALL: {pw}", annotation_position="bottom right")
    fig.add_hline(y=spot, line_color="cyan", line_width=2,
                  annotation_text=f"SPOT: {spot:.2f}", annotation_position="bottom left")

    fig.update_layout(
        template="plotly_dark", height=800,
        title=f"STRUTTURA {metric.upper()} - {ticker} ({date})",
        yaxis=dict(title="STRIKE PRICE", side="left", tickformat=".2f"),
        xaxis=dict(title=f"ESPOSIZIONE {metric}"),
        hovermode="y unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recap Metriche
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Spot", f"{spot:.2f}")
    c2.metric("Call Wall Strike", cw)
    c3.metric("Put Wall Strike", pw)

except Exception as e:
    st.error(f"Errore nel caricamento dati: {e}")
