import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V12", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE GRECHE AVANZATO ---
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
    # Calcolo del Bias basato sulla pressione combinata delle Greche
    # Normalizziamo i pesi: Gamma (fondamentale), Vanna (flusso IV), Charm (decadimento tempo)
    score = (df['Gamma'].sum() * 1.0) + (df['Vanna'].sum() * 0.5) + (df['Charm'].sum() * 0.5)
    if score > 1000: return "STRONG LONG", "#00ff00"
    if score > 0: return "LONG BIAS", "#adff2f"
    if score < -1000: return "STRONG SHORT", "#ff4444"
    return "SHORT BIAS", "#ffaa00"

def find_zero_gamma(df, spot):
    # Ordiniamo per strike e cerchiamo il cambio di segno più vicino allo spot
    df_sorted = df.sort_values('strike')
    df_sorted['gamma_sign'] = np.sign(df_sorted['Gamma'])
    # Trova dove il segno cambia
    sign_change = df_sorted.index[df_sorted['gamma_sign'].diff() != 0].tolist()
    
    # Filtriamo l'incrocio più vicino al prezzo attuale
    zero_gamma_strike = df_sorted.loc[df_sorted['strike'].sub(spot).abs().idxmin(), 'strike']
    for idx in sign_change:
        if idx == 0: continue
        current_strike = df_sorted.loc[idx, 'strike']
        if abs(current_strike - spot) < abs(zero_gamma_strike - spot):
            zero_gamma_strike = current_strike
    return float(zero_gamma_strike)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE CONTROL")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
trade_mode = st.sidebar.selectbox("TIMEFRAME", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
strike_step = st.sidebar.selectbox("STEP STRIKE", [5, 10, 25, 50, 100, 250], index=3)
num_levels = st.sidebar.slider("ZOOM LIVELLI", 10, 100, 50)
main_metric = st.sidebar.radio("VISUALIZZA NEL GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega'])

t_str = fix_ticker(active_t)
if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            
            # --- SELEZIONE SCADENZA ---
            exps = t_obj.options
            idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
            sel_exp = exps[min(idx, len(exps)-1)]
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            ch = t_obj.option_chain(sel_exp)
            
            # --- ELABORAZIONE DATI ---
            c_v = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            p_v = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike'].astype(float)})
            df['Gamma'], df['Vanna'], df['Charm'] = c_v[0]-p_v[0], c_v[1]-p_v[1], c_v[2]-p_v[2]
            df['Vega'] = c_v[3] + p_v[3]

            # LIVELLI CHIAVE
            z_gamma = find_zero_gamma(df, spot)
            call_wall = float(df.loc[df['Gamma'].idxmax(), 'strike'])
            put_wall = float(df.loc[df['Gamma'].idxmin(), 'strike'])

            # --- DASHBOARD SUPERIORE ---
            st.markdown(f"## 🛰️ {active_t} Intelligence Dashboard")
            b_col, c_col, p_col, z_col = st.columns([2, 1, 1, 1])
            
            # Calcolo Bias per il timeframe selezionato
            bias_text, bias_color = calculate_bias_score(df)
            
            b_col.markdown(f"<div style='text-align:center;border:2px solid {bias_color};padding:15px;border-radius:10px;'><h4>TF BIAS: {trade_mode}</h4><h2 style='color:{bias_color};'>{bias_text}</h2></div>", unsafe_allow_html=True)
            c_col.metric("CALL WALL", f"{call_wall:.0f}")
            p_col.metric("PUT WALL", f"{put_wall:.0f}")
            z_col.metric("ZERO GAMMA", f"{z_gamma:.0f}", delta=f"{spot-z_gamma:.2f}", delta_color="inverse")

            # --- GRAFICO PROFILO ---
            # Binning per granularità
            df['bin'] = (df['strike'] / strike_step).round() * strike_step
            df_p = df.groupby('bin')[['Gamma', 'Vanna', 'Charm', 'Vega']].sum().reset_index()
            df_p.rename(columns={'bin': 'strike'}, inplace=True)
            df_p = df_p[(df_p['strike'] >= spot - (strike_step * num_levels/2)) & (df_p['strike'] <= spot + (strike_step * num_levels/2))]

            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_p[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_p['strike'], x=df_p[main_metric], orientation='h', 
                marker_color=colors, width=strike_step*0.7
            ))

            # Linee dinamiche (ora precise e basate sullo spot)
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA FLIP")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=850,
                yaxis=dict(title="STRIKE PRICE", gridcolor="#333", tickformat=".0f"),
                xaxis=dict(title=f"Net {main_metric} Exposure", zerolinecolor="white"),
                title=f"Struttura Liquidità {active_t} - {sel_exp}"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore Analisi: {e}")
