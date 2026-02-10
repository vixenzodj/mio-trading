import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V38", initial_sidebar_state="expanded")
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

# --- SIDEBAR & SELEZIONE ---
st.sidebar.header("🕹️ GEX ENGINE V38")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
        num_levels = st.sidebar.slider("ZOOM AREA", 100, 2500, 1000)
        
        # SELETTORE METRICA PER IL GRAFICO
        main_metric = st.sidebar.radio("VISUALIZZA NEL GRAFICO:", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- SOLUZIONE DEFINITIVA AI DUPLICATI ---
            c = chain.calls.copy().assign(type='call')
            p = chain.puts.copy().assign(type='put')
            df_raw = pd.concat([c, p], axis=0, ignore_index=True)
            
            # Fondiamo gli strike identici (stesso strike, stesso tipo) sommando l'Open Interest
            df_clean = df_raw.groupby(['strike', 'type'], as_index=False).agg({
                'impliedVolatility': 'mean',
                'openInterest': 'sum'
            }).reset_index(drop=True)
            
            # Calcolo Greche
            df_res = fast_engine(df_clean, spot, t_yrs)
            
            # Calcolo Regime Scores
            def get_score(series):
                net = series.sum()
                total = series.abs().sum()
                return net / total if total != 0 else 0

            g_score, v_score, c_score = get_score(df_res['Gamma']), get_score(df_res['Vanna']), get_score(df_res['Charm'])

            # --- HEADER INDICATORI (IL CAMPANELLO) ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            i1, i2, i3, i4 = st.columns(4)
            
            metrics_data = [
                ("GAMMA", g_score, "🛡️ STABILE" if g_score > 0 else "⚠️ VOLATILE"),
                ("VANNA", v_score, "🌊 FLUIDO" if v_score > 0 else "🌪️ SKEW"),
                ("CHARM", c_score, "⏳ DECADIMENTO" if c_score > 0 else "⚡ MOMENTUM")
            ]

            for col, (name, score, status) in zip([i1, i2, i3], metrics_data):
                color = "#00ff00" if score > 0 else "#ff4444"
                col.markdown(f"""
                <div style="background:#1e1e1e; padding:10px; border-radius:8px; border-bottom: 3px solid {color}; text-align:center;">
                    <p style="margin:0; color:#aaa; font-size:12px;">{name} REGIME</p>
                    <h3 style="margin:0; color:{color};">{score:+.2f}</h3>
                    <p style="margin:0; font-size:12px;">{status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            bias = "🟢 BULLISH" if g_score > 0 and v_score > 0 else "🔴 BEARISH" if g_score < 0 else "🟡 NEUTRAL"
            i4.markdown(f"<div style='background:#1e1e1e; padding:18px; border-radius:8px; text-align:center; height:100%'><b>BIAS: {bias}</b></div>", unsafe_allow_html=True)

            # --- ELABORAZIONE GRAFICO ---
            df_total = df_res.groupby('strike', as_index=False).sum().sort_values('strike').reset_index(drop=True)
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']
            
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'})
            df_view = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()

            # --- GRAFICO DINAMICO ---
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h', 
                marker_color=['#00ff00' if x>=0 else '#00aaff' for x in df_view[main_metric]],
                width=strike_step*0.8, name=main_metric
            ))
            
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            
            fig.update_layout(template="plotly_dark", height=700, margin=dict(l=0,r=0,t=20,b=0),
                              xaxis=dict(title=f"Esposizione Netta {main_metric}"),
                              yaxis=dict(dtick=strike_step, title="STRIKE"))
            st.plotly_chart(fig, use_container_width=True)

            # --- METRICHE NUMERICHE SOTTO IL GRAFICO ---
            st.markdown("### 📊 Market Inventory Data")
            table_data = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:20]].sort_values('strike', ascending=False).reset_index(drop=True)
            
            st.dataframe(table_data[['strike', 'Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']].style.format(precision=1).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Charm', 'Theta']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
