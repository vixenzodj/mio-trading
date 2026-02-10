import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V36", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE (Senza indici Pandas) ---
def fast_engine(df, spot, t_yrs, r=0.045):
    s = float(spot)
    # Convertiamo subito in array numpy per ignorare le etichette (labels)
    k = df['strike'].values
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    oi = df['openInterest'].values
    types = df['type'].values
    t = max(t_yrs, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    is_call = (types == 'call')
    theta_part1 = -(s * pdf * v) / (2 * np.sqrt(t))
    theta_part2 = r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))
    theta = (theta_part1 - theta_part2) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    # Creiamo un dataframe nuovo di zecca con indice pulito
    return pd.DataFrame({
        'strike': k,
        'Gamma': gamma * mult,
        'Vanna': vanna * mult,
        'Charm': charm * mult,
        'Vega': vega,
        'Theta': theta
    })

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V36")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
        num_levels = st.sidebar.slider("ZOOM AREA (Punti)", 100, 2500, 1000)
        main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- PULIZIA RADICALE DUPLICATI ---
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            # Uniamo e resettiamo l'indice immediatamente
            df_raw = pd.concat([c, p], axis=0, ignore_index=True)
            
            # Raggruppiamo per strike e tipo, prendendo il primo valore per la vola e sommando l'OI
            # Questo elimina i duplicati alla radice
            df_clean = df_raw.groupby(['strike', 'type'], as_index=False).agg({
                'impliedVolatility': 'first',
                'openInterest': 'sum'
            })
            
            # Calcolo Greche (Lavora su array, quindi non può generare l'errore label)
            df_res = fast_engine(df_clean, spot, t_yrs)
            
            # Consolidamento per strike (somma Call + Put per ogni livello)
            df_total = df_res.groupby('strike', as_index=False).sum().sort_values('strike').reset_index(drop=True)
            
            # Zero Gamma
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']

            # Binning per visualizzazione
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'}).reset_index(drop=True)
            
            df_plot_zoom = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # Dashboard
            st.markdown(f"## 🏛️ {active_t} Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPOT", f"{spot:.2f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.0f}")
            m3.metric("CALL WALL", f"{call_wall:.0f}")
            
            g_sum = df_total['Gamma'].sum()
            bias = "BULLISH" if g_sum > 0 else "BEARISH"
            m4.markdown(f"<div style='text-align:center; padding:10px; border-radius:10px; border:2px solid {'#00ff00' if bias=='BULLISH' else '#ff4444'};'>BIAS: {bias}</div>", unsafe_allow_html=True)

            # Grafico
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_plot_zoom['strike'], x=df_plot_zoom[main_metric], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in df_plot_zoom[main_metric]],
                width=strike_step * 0.8
            ))
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO G")
            fig.update_layout(template="plotly_dark", height=800, yaxis=dict(dtick=strike_step))
            st.plotly_chart(fig, use_container_width=True)

            # Tabella
            st.markdown("### 📊 Livelli Chiave")
            table_data = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:15]].sort_values('strike', ascending=False).reset_index(drop=True)
            st.dataframe(table_data[['strike', 'Gamma', 'Vega', 'Theta', 'Vanna']].style.format(precision=1).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Theta']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
