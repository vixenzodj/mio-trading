import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V31", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def fast_greeks(df, s, t, r=0.045):
    # Usiamo i valori puliti dal dataframe
    k = df['strike'].values
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    oi = df['openInterest'].values
    types = df['type'].values
    t = max(t, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    vega = s * pdf * np.sqrt(t) * oi * 100
    
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
    }, index=df.index) # Manteniamo l'indice per evitare disallineamenti

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V31")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        def_idx = 5 if "NDX" in t_str or "SPX" in t_str else 2
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 2, 5, 10, 25, 50, 100, 250], index=def_idx)
        zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 15, 5)
        main_metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Vega', 'Theta'])

        hist = t_obj.history(period='2d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            calls, puts = chain.calls.copy(), chain.puts.copy()
            calls['type'], puts['type'] = 'call', 'put'
            
            # --- SOLUZIONE AL BUG: Pulizia Duplicati ---
            df_raw = pd.concat([calls, puts]).reset_index(drop=True)
            # Se ci sono due opzioni con stesso strike e tipo, sommiamo l'OI
            df_raw = df_raw.groupby(['strike', 'type'], as_index=False).agg({
                'impliedVolatility': 'mean',
                'openInterest': 'sum',
                'lastPrice': 'mean'
            })
            
            # Calcolo greche su dati puliti
            df_greeks = fast_greeks(df_raw, spot, t_yrs)
            
            # Punto di equilibrio (Zero Gamma)
            df_sorted = df_greeks.groupby('strike').sum().reset_index().sort_values('strike')
            df_sorted['cum_gamma'] = df_sorted['Gamma'].cumsum()
            z_gamma = df_sorted.loc[df_sorted['cum_gamma'].abs().idxmin(), 'strike']

            # Aggregazione per bin (Scrematura visiva)
            df_greeks['bin'] = np.floor(df_greeks['strike'] / strike_step) * strike_step
            df_plot = df_greeks.groupby('bin').sum(numeric_only=True).reset_index().rename(columns={'bin': 'strike'})
            
            # Zoom dinamico
            limit = (spot * zoom_pct) / 100
            df_view = df_plot[(df_plot['strike'] >= spot - limit) & (df_plot['strike'] <= spot + limit)].copy()
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPOT PRICE", f"{spot:.2f}")
            m2.metric("VOL TRIGGER (0G)", f"{z_gamma:.0f}")
            m3.metric("CALL WALL", f"{call_wall:.0f}")
            bias_val = df_sorted['Gamma'].sum()
            m4.markdown(f"<div style='text-align:center; padding:10px; border-radius:5px; background:{'#00ff0022' if bias_val>0 else '#ff444422'}; border:1px solid {'#00ff00' if bias_val>0 else '#ff4444'}'><b>BIAS: {'BULLISH' if bias_val>0 else 'BEARISH'}</b></div>", unsafe_allow_html=True)

            # --- GRAFICO DINAMICO ---
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h',
                marker_color=['#00ff00' if x > 0 else '#00aaff' for x in df_view[main_metric]],
                width=strike_step * 0.8,
                text=[f"{v/1e6:.1f}M" if abs(v)>5e5 else "" for v in df_view[main_metric]],
                textposition='outside'
            ))

            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO G")
            fig.add_hline(y=call_wall, line_color="red", annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="green", annotation_text="PUT WALL")

            fig.update_layout(template="plotly_dark", height=800, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis=dict(range=[spot-limit, spot+limit], dtick=strike_step, title="STRIKE"),
                              xaxis=dict(title=f"Net {main_metric} Exposure ($)"))
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA VALORI ESATTI ---
            st.markdown("### 📊 Struttura Greche (Livelli ATM)")
            table_df = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:15]].sort_values('strike', ascending=False)
            
            def color_gex(val):
                color = '#00ff00' if val > 0 else '#ff4444' if val < 0 else 'white'
                return f'color: {color}'

            st.dataframe(table_df[['strike', 'Gamma', 'Vanna', 'Vega', 'Theta']].style.format(precision=1).map(color_gex, subset=['Gamma', 'Vanna', 'Theta']), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
