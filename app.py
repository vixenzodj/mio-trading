import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V29", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE (MOLTO PIÙ VELOCE) ---
def fast_greeks(df, s, t, r=0.045):
    k = df['strike'].values
    v = df['impliedVolatility'].values
    oi = df['openInterest'].values
    types = df['type'].values
    
    # Evitiamo divisioni per zero
    v = np.where(v <= 0, 1e-9, v)
    t = max(t, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Calcoli in blocco
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    # Theta e correzione segno per Put
    is_call = (types == 'call')
    theta_part1 = -(s * pdf * v) / (2 * np.sqrt(t))
    theta_part2 = r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))
    theta = (theta_part1 - theta_part2) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    return pd.DataFrame({
        'strike': k,
        'Gamma': gamma * mult,
        'Vanna': vanna * mult,
        'Vega': vega,
        'Theta': theta
    })

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V29")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA", exps)
    strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
    zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 20, 5)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # Unione rapida
            calls, puts = chain.calls.copy(), chain.puts.copy()
            calls['type'], puts['type'] = 'call', 'put'
            df_raw = pd.concat([calls, puts])
            
            # Calcolo greche istantaneo
            df_greeks = fast_greeks(df_raw, spot, t_yrs)
            
            # Aggregazione per bin (Scrematura)
            df_greeks['bin'] = np.floor(df_greeks['strike'] / strike_step) * strike_step
            df_plot = df_greeks.groupby('bin').sum().reset_index().rename(columns={'bin': 'strike'})
            
            # Zero Gamma Cumulativo (Punto di equilibrio)
            df_greeks = df_greeks.sort_values('strike')
            df_greeks['cum_gamma'] = df_greeks['Gamma'].cumsum()
            z_gamma = df_greeks.loc[df_greeks['cum_gamma'].abs().idxmin(), 'strike']

            # Zoom e Wall
            limit = (spot * zoom_pct) / 100
            df_view = df_plot[(df_plot['strike'] >= spot - limit) & (df_plot['strike'] <= spot + limit)]
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- DASHBOARD ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPOT", f"{spot:.2f}")
            m2.metric("VOL TRIGGER", f"{z_gamma:.0f}")
            m3.metric("CALL WALL", f"{call_wall:.0f}")
            bias = "BULLISH" if df_greeks['Gamma'].sum() > 0 else "BEARISH"
            m4.markdown(f"**BIAS:** <span style='color:{'#00ff00' if bias=='BULLISH' else '#ff4444'}'>{bias}</span>", unsafe_allow_html=True)

            # --- GRAFICO (Leggero per il browser) ---
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h',
                marker_color=['#00ff00' if x > 0 else '#00aaff' for x in df_view[main_metric]],
                width=strike_step * 0.7
            ))

            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO G")

            fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0, r=0, t=30, b=0),
                              yaxis=dict(range=[spot-limit, spot+limit], dtick=strike_step))
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA METRICHE (A colpo d'occhio) ---
            st.markdown("### 📊 Livelli di Difesa")
            targets = [z_gamma, call_wall, put_wall]
            # Mostra solo i 10 strike più importanti intorno allo spot per non bloccare Chrome
            table_df = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:12]].sort_values('strike', ascending=False)
            
            st.table(table_df[['strike', 'Gamma', 'Vanna', 'Vega', 'Theta']].style.format(precision=1).applymap(
                lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4444' if x < 0 else '', subset=['Gamma', 'Vanna', 'Theta']
            ))

    except Exception as e:
        st.error(f"Errore: {e}")
