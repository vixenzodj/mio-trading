import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V34", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE (Sostituisce il lento calc_greeks_pro) ---
def fast_engine(df, spot, t_yrs, r=0.045):
    # Trasformiamo in array numpy per velocità e per evitare errori di indice
    s = float(spot)
    k = df['strike'].to_numpy()
    v = np.where(df['impliedVolatility'].to_numpy() <= 0, 1e-9, df['impliedVolatility'].to_numpy())
    oi = df['openInterest'].to_numpy()
    types = df['type'].to_numpy()
    t = max(t_yrs, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Calcolo Greche
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    # Theta (correzione per tipo opzione)
    is_call = (types == 'call')
    theta_part1 = -(s * pdf * v) / (2 * np.sqrt(t))
    theta_part2 = r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))
    theta = (theta_part1 - theta_part2) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    return pd.DataFrame({
        'strike': k,
        'Gamma': gamma * mult,
        'Vanna': vanna * mult,
        'Charm': charm * mult,
        'Vega': vega,
        'Theta': theta
    })

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V34")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE (Granularità)", [1, 5, 10, 25, 50, 100, 250], index=4)
        num_levels = st.sidebar.slider("ZOOM AREA PREZZO (Punti)", 100, 2500, 1000)
        main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # UNIONE E PULIZIA RADICALE (Risolve il problema labels)
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            df_raw = pd.concat([c, p], ignore_index=True)
            
            # Raggruppiamo subito gli strike duplicati per sommare OI
            df_raw = df_raw.groupby(['strike', 'type'], as_index=False).agg({'impliedVolatility': 'mean', 'openInterest': 'sum'})
            
            # Eseguiamo i calcoli
            df_res = fast_engine(df_raw, spot, t_yrs)
            
            # ZERO GAMMA (Cumulativo)
            df_total = df_res.groupby('strike', as_index=False).sum().sort_values('strike')
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']

            # BINNING PER GRAFICO
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'})
            df_plot_zoom = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # BIAS
            gamma_net = df_total['Gamma'].sum()
            vanna_net = df_total['Vanna'].sum()
            if gamma_net > 0 and vanna_net > 0: bias, b_col = "ULTRA BULLISH", "#00ff00"
            elif gamma_net < 0 and vanna_net < 0: bias, b_col = "ULTRA BEARISH", "#ff4444"
            else: bias, b_col = "NEUTRAL / TRANSITION", "#ffff00"

            # DASHBOARD
            st.markdown(f"## 🏛️ {active_t} Terminal | Spot: {spot:.2f}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("SPOT PRICE", f"{spot:.2f}")
            d2.metric("ZERO GAMMA", f"{z_gamma:.0f}", f"{z_gamma-spot:.0f} pts")
            d3.metric("CALL WALL", f"{call_wall:.0f}")
            d4.markdown(f"<div style='border:2px solid {b_col}; padding:10px; border-radius:10px; text-align:center;'>BIAS ATTUALE<br><b style='color:{b_col}; font-size:22px;'>{bias}</b></div>", unsafe_allow_html=True)

            # GRAFICO
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_plot_zoom['strike'], x=df_plot_zoom[main_metric], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in df_plot_zoom[main_metric]],
                width=strike_step * 0.8,
                text=[f"{v/1e6:.1f}M" if abs(v)>1e5 else "" for v in df_plot_zoom[main_metric]], textposition='outside'
            ))

            fig.add_hline(y=call_wall, line_color="red", annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")

            fig.update_layout(template="plotly_dark", height=800, yaxis=dict(dtick=strike_step))
            st.plotly_chart(fig, use_container_width=True)

            # TABELLA (PULITA)
            st.markdown("### 📊 Livelli Chiave")
