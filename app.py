import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V13", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE GRECHE ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        r, d1 = 0.045, (np.log(s/k) + (0.045 + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2, pdf = d1 - v * np.sqrt(t_yrs), norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0.0]*5)

def calculate_bias_score(df):
    # Sintesi Istituzionale: Gamma (1.0) + Vanna (0.5) + Charm (0.5)
    score = (df['Gamma'].sum() * 1.0) + (df['Vanna'].sum() * 0.5) + (df['Charm'].sum() * 0.5)
    if score > 500: return "STRONG BULLISH", "#00ff00"
    if score > 0: return "LEANING BULLISH", "#adff2f"
    if score < -500: return "STRONG BEARISH", "#ff4444"
    return "LEANING BEARISH", "#ffaa00"

def find_zero_gamma(df, spot):
    # Cerca il punto in cui il Gamma Netto attraversa lo zero più vicino allo Spot
    df_sorted = df.sort_values('strike')
    df_sorted['gamma_sign'] = np.sign(df_sorted['Gamma'])
    idx_changes = np.where(df_sorted['gamma_sign'].diff().fillna(0) != 0)[0]
    
    if len(idx_changes) == 0: return spot
    
    # Trova l'incrocio più vicino al prezzo attuale
    closest_strike = df_sorted.iloc[idx_changes[np.abs(df_sorted.iloc[idx_changes]['strike'] - spot).argmin()]]['strike']
    return float(closest_strike)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE CONTROL")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    
    # SCELTA SCADENZA MANUALE
    sel_exp = st.sidebar.selectbox("SCADENZA (EXPIRY)", exps)
    
    strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
    num_levels = st.sidebar.slider("ZOOM STRIKE", 10, 150, 60)
    main_metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            
            # --- CALCOLO ---
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.2) / 365
            ch = t_obj.option_chain(sel_exp)
            
            c_v = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            p_v = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike'].astype(float)})
            df['Gamma'], df['Vanna'], df['Charm'] = c_v[0]-p_v[0], c_v[1]-p_v[1], c_v[2]-p_v[2]
            df['Vega'] = c_v[3] + p_v[3]

            # LIVELLI CHIAVE
            z_gamma = find_zero_gamma(df, spot)
            call_wall = float(df.loc[df['Gamma'].idxmax(), 'strike'])
            put_wall = float(df.loc[df['Gamma'].idxmin(), 'strike'])

            # --- DASHBOARD ISTITUZIONALE ---
            st.markdown(f"## 🏛️ Dealer Positioning: {active_t} | Expiry: {sel_exp}")
            
            bias_text, bias_color = calculate_bias_score(df)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div style='text-align:center;border:2px solid {bias_color};padding:15px;border-radius:10px;'><h4>INSTITUTIONAL BIAS</h4><h2 style='color:{bias_color};'>{bias_text}</h2></div>", unsafe_allow_html=True)
            m2.metric("SPOT PRICE", f"{spot:.2f}")
            m3.metric("ZERO GAMMA (FLIP)", f"{z_gamma:.0f}", f"{spot-z_gamma:.2f} pts")
            m4.metric("VOLATILITY WALLS", f"C: {call_wall:.0f}", f"P: {put_wall:.0f}")

            # --- GRAFICO ---
            df['bin'] = (df['strike'] / strike_step).round() * strike_step
            df_p = df.groupby('bin')[['Gamma', 'Vanna', 'Charm', 'Vega']].sum().reset_index()
            df_p.rename(columns={'bin': 'strike'}, inplace=True)
            
            # Zoom dinamico intorno allo spot
            df_p = df_p[(df_p['strike'] >= spot - (strike_step * num_levels/2)) & (df_p['strike'] <= spot + (strike_step * num_levels/2))]

            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_p[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_p['strike'], x=df_p[main_metric], orientation='h', 
                marker_color=colors, width=strike_step*0.75
            ))

            # Linee Operative
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text="CALL WALL (MAX RESISTANCE)")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL (MAX SUPPORT)")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA FLIP")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"LIVE SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=900,
                yaxis=dict(title="STRIKE", gridcolor="#333", autorange=True, tickformat=".0f"),
                xaxis=dict(title=f"Net {main_metric} Exposure (Dealer Hedge Requirement)", zerolinecolor="white"),
                bargap=0
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore nel caricamento dei dati per la scadenza {sel_exp}: {e}")
