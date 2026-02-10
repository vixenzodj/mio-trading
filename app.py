import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V37", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def fast_engine(df, spot, t_yrs, r=0.045):
    s = float(spot)
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
    theta = (-(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    return pd.DataFrame({
        'strike': k, 'Gamma': gamma * mult, 'Vanna': vanna * mult, 
        'Charm': charm * mult, 'Vega': vega, 'Theta': theta
    })

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V37")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
        num_levels = st.sidebar.slider("ZOOM AREA", 100, 2500, 1000)

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # Pulizia e Calcolo
            df_raw = pd.concat([chain.calls.assign(type='call'), chain.puts.assign(type='put')], ignore_index=True)
            df_clean = df_raw.groupby(['strike', 'type'], as_index=False).agg({'impliedVolatility': 'first', 'openInterest': 'sum'})
            df_res = fast_engine(df_clean, spot, t_yrs)
            
            # --- LOGICA DI REGIME (Normalizzazione 0-1) ---
            def get_regime_score(series):
                net = series.sum()
                total_abs = series.abs().sum()
                return net / total_abs if total_abs != 0 else 0

            g_score = get_regime_score(df_res['Gamma'])
            v_score = get_regime_score(df_res['Vanna'])
            c_score = get_regime_score(df_res['Charm'])

            # --- DASHBOARD NUMERICO ---
            st.markdown(f"## 🏛️ {active_t} Professional Regime Monitor")
            
            # Funzione per interpretare il valore
            def interpret(val):
                abs_v = abs(val)
                if abs_v < 0.3: return "DEBOLE", "#777"
                if abs_v < 0.7: return "MODERATO", "#ffaa00"
                return "ESTREMO", "#00ff00" if val > 0 else "#ff4444"

            c1, c2, c3, c4 = st.columns(4)
            
            for col, name, score in zip([c1, c2, c3], ["GAMMA", "VANNA", "CHARM"], [g_score, v_score, c_score]):
                status, color = interpret(score)
                col.markdown(f"""
                <div style="background:#1e1e1e; padding:15px; border-radius:10px; border-left: 5px solid {color};">
                    <p style="margin:0; color:#aaa; font-size:12px;">{name} REGIME SCORE</p>
                    <h2 style="margin:0; color:{color};">{score:+.2f}</h2>
                    <p style="margin:0; font-size:14px; font-weight:bold;">{status}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- CALCOLO LIVELLI ---
            df_total = df_res.groupby('strike', as_index=False).sum().sort_values('strike').reset_index(drop=True)
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']
            
            # Aggregazione per grafico
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'})
            df_view = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()

            # --- GRAFICO ---
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df_view['strike'], x=df_view['Gamma'], orientation='h', 
                                 marker_color=['#00ff00' if x>=0 else '#00aaff' for x in df_view['Gamma']], width=strike_step*0.8))
            
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0,r=0,t=30,b=0), yaxis=dict(dtick=strike_step))
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA PROFESSIONALE ---
            st.markdown("### 📊 Market Maker Inventory (Detailed Levels)")
            table_data = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:15]].sort_values('strike', ascending=False)
            st.dataframe(table_data[['strike', 'Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']].style.format(precision=1).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Charm', 'Theta']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
