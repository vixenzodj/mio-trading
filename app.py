import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Advanced Trading Terminal")
st_autorefresh(interval=900000, key="datarefresh")

# --- FUNZIONI MATEMATICHE ---
def calculate_all_greeks(row, spot, t, r=0.04):
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

    return pd.Series([gamma*oi, vanna*oi, charm*oi, vega*oi, theta*oi], index=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

def format_number(num):
    if abs(num) >= 1_000_000: return f"{num / 1_000_000:.2f} M"
    if abs(num) >= 1_000: return f"{num / 1_000:.2f} K"
    return f"{num:.2f}"

# --- LOGICA DI SCARICO DATI INTELLIGENTE ---
def get_smart_data(symbol, expiry_index, noise_filter):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    
    # Selezione della scadenza scelta dall'utente
    all_expiries = t_obj.options
    selected_expiry = all_expiries[expiry_index]
    opts = t_obj.option_chain(selected_expiry)
    
    # Calcolo tempo alla scadenza (T)
    from datetime import datetime
    t_days = (datetime.strptime(selected_expiry, '%Y-%m-%d') - datetime.now()).days
    t_years = max(t_days, 0.5) / 365 # 0.5 giorni minimo per evitare divisioni per zero
    
    calls, puts = opts.calls.copy(), opts.puts.copy()
    
    # FILTRO RUMORE DINAMICO: l'utente sceglie la % di distanza dallo spot
    lower_limit = price * (1 - noise_filter/100)
    upper_limit = price * (1 + noise_filter/100)
    
    # Cerchiamo i muri solo nell'area "sensata" scelta dall'utente
    relevant_calls = calls[(calls['strike'] >= price) & (calls['strike'] <= upper_limit)]
    relevant_puts = puts[(puts['strike'] <= price) & (puts['strike'] >= lower_limit)]
    
    # Se il filtro è troppo stretto, prendiamo il più vicino possibile allo spot
    c_wall = relevant_calls.loc[relevant_calls['openInterest'].idxmax(), 'strike'] if not relevant_calls.empty else calls.loc[(calls['strike']-price).abs().idxmin(), 'strike']
    p_wall = relevant_puts.loc[relevant_puts['openInterest'].idxmax(), 'strike'] if not relevant_puts.empty else puts.loc[(puts['strike']-price).abs().idxmin(), 'strike']
    
    # Calcolo Greche
    c_greeks = calls.apply(lambda r: calculate_all_greeks(r, price, t_years), axis=1)
    p_greeks = puts.apply(lambda r: calculate_all_greeks(r, price, t_years), axis=1)
    
    df = pd.DataFrame({'strike': calls['strike']})
    for m in ['Gamma', 'Vanna', 'Charm']: df[m] = c_greeks[m] - p_greeks[m]
    for m in ['Vega', 'Theta']: df[m] = c_greeks[m] + p_greeks[m]
        
    return price, df, selected_expiry, c_wall, p_wall

# --- INTERFACCIA ---
st.title("🏹 Smart Greeks & Walls Terminal")

# Sidebar controlli
st.sidebar.header("Impostazioni Analisi")
target_ticker = st.sidebar.selectbox("Asset", ['QQQ', 'SPY', 'NVDA', 'TSLA', 'AAPL'], index=0)

# Scelta Scadenza
ticker_obj = yf.Ticker(target_ticker)
expiries = ticker_obj.options
expiry_choice = st.sidebar.selectbox("Scadenza (0DTE è la prima)", range(len(expiries)), format_func=lambda x: expiries[x])

# Slider Rumore
noise_val = st.sidebar.slider("Filtro Rumore Strike (%)", 5, 30, 15, help="Ignora strike troppo lontani dal prezzo attuale")

selected_metric = st.sidebar.selectbox("Metrica Grafico", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

try:
    spot, full_df, exp_date, call_wall, put_wall = get_smart_data(target_ticker, expiry_choice, noise_val)
    
    # Grafico
    df_plot = full_df[(full_df['strike'] > spot * 0.9) & (full_df['strike'] < spot * 1.1)]
    fig = go.Figure()
    
    m_colors = {'Gamma': '#00ff88', 'Vanna': '#ffcc00', 'Charm': '#ff00ff', 'Vega': '#00aaff', 'Theta': '#ff4444'}
    
    fig.add_trace(go.Bar(y=df_plot['strike'], x=df_plot[selected_metric], orientation='h', marker_color=m_colors[selected_metric]))
    
    # Linee Muri e Spot
    fig.add_hline(y=spot, line_dash="dash", line_color="cyan", annotation_text=f"SPOT: {spot:.2f}")
    fig.add_hline(y=call_wall, line_dash="dot", line_color="#00ff88", line_width=3, annotation_text="CALL WALL", annotation_position="top right")
    fig.add_hline(y=put_wall, line_dash="dot", line_color="#ff4444", line_width=3, annotation_text="PUT WALL", annotation_position="bottom right")
    
    fig.update_layout(template="plotly_dark", height=700, title=f"{target_ticker} {selected_metric} Profile - {exp_date}")
    st.plotly_chart(fig, use_container_width=True)

    # Metriche in basso
    st.divider()
    cols = st.columns(5)
    for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
        cols[i].metric(f"Total {m}", format_number(full_df[m].sum()))

except Exception as e:
    st.error(f"Seleziona un'altra scadenza o ticker. Errore: {e}")
