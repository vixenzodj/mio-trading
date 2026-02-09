import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Advanced Greeks Terminal")
st_autorefresh(interval=900000, key="datarefresh")

# --- MOTORE DI CALCOLO GRECHE ---
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

# --- LOGICA DATI ---
@st.cache_data(ttl=900)
def get_full_data(symbol):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    exp = t_obj.options[0]
    opts = t_obj.option_chain(exp)
    
    calls = opts.calls.copy()
    puts = opts.puts.copy()
    
    c_greeks = calls.apply(lambda r: calculate_all_greeks(r, price), axis=1)
    p_greeks = puts.apply(lambda r: calculate_all_greeks(r, price), axis=1)
    
    # Net Exposure (Call - Put per GEX/Vanna/Charm, Somma per Vega/Theta)
    full_df = pd.DataFrame({'strike': calls['strike']})
    for m in ['Gamma', 'Vanna', 'Charm']:
        full_df[m] = c_greeks[m] - p_greeks[m]
    for m in ['Vega', 'Theta']:
        full_df[m] = c_greeks[m] + p_greeks[m]
        
    return price, full_df, exp

# --- INTERFACCIA ---
st.title("🏛️ Professional Greeks Profile Terminal")

ticker = st.sidebar.selectbox("Seleziona Asset", ['QQQ', 'SPY', 'IWM', 'NVDA', 'TSLA', 'AAPL'], index=0)
metrica = st.sidebar.selectbox("Seleziona Metrica Grafico", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

try:
    spot, df, exp = get_full_data(ticker)
    
    # Range Strike (+/- 15% dallo spot)
    df_plot = df[(df['strike'] > spot * 0.85) & (df['strike'] < spot * 1.15)]

    # --- GRAFICO A BARRE (SPOTGAMMA STYLE) ---
    fig = go.Figure()
    
    # Colore dinamico in base alla metrica
    colors = {'Gamma': '#00ff88', 'Vanna': '#ffcc00', 'Charm': '#ff00ff', 'Vega': '#00aaff', 'Theta': '#ff4444'}
    
    fig.add_trace(go.Bar(
        y=df_plot['strike'], 
        x=df_plot[metrica], 
        orientation='h',
        marker_color=colors[metrica],
        name=metrica
    ))

    # Linee di supporto/resistenza visive
    fig.add_hline(y=spot, line_dash="dash", line_color="white", annotation_text=f"SPOT: {spot:.2f}")
    
    fig.update_layout(
        template="plotly_dark", height=900,
        title=f"ABS {metrica.upper()} PROFILE - {ticker} (Exp: {exp})",
        xaxis_title=f"Esposizione Netta {metrica}",
        yaxis_title="Strike Price",
        bargap=0.05
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- TABELLA RIASSUNTIVA ---
    st.divider()
    st.subheader(f"📈 Riepilogo Totale {ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Gamma", f"{df['Gamma'].sum():.2e}")
    c2.metric("Net Vanna", f"{df['Vanna'].sum():.2e}")
    c3.metric("Net Charm", f"{df['Charm'].sum():.2e}")
    c4.metric("Total Vega", f"{df['Vega'].sum():.2e}")
    c5.metric("Total Theta", f"{df['Theta'].sum():.2e}")

except Exception as e:
    st.error(f"Errore: {e}")
