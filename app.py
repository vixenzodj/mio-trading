import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V14", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE GRECHE PROFESSIONALE (Standard Istituzionale) ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        
        r = 0.045 # Tasso Risk-Free stimato
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        
        # Formule standard Black-Scholes pesate per Open Interest
        # Calcoliamo la DOLLAR GAMMA: (Gamma * Spot^2 * 0.01) * OI * 100
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100 * (s**2) * 0.01
        vanna = ((pdf * d1) / v) * oi * s * 0.01
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi * 100
        vega = (s * pdf * np.sqrt(t_yrs)) * oi * 100
        theta = ((-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2))) * oi * 100
        
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0.0]*5)

def find_localized_levels(df, spot):
    # Filtriamo solo gli strike vicini al prezzo attuale (±10%) per evitare errori statistici
    df_local = df[(df['strike'] >= spot * 0.90) & (df['strike'] <= spot * 1.10)].copy()
    
    if df_local.empty: return spot, spot, spot
    
    # Call Wall: Massimo Gamma Netto Positivo
    call_wall = float(df_local.loc[df_local['Gamma'].idxmax(), 'strike'])
    # Put Wall: Massimo Gamma Netto Negativo (Supporto)
    put_wall = float(df_local.loc[df_local['Gamma'].idxmin(), 'strike'])
    
    # Zero Gamma Flip: Il punto dove il segno cambia vicino allo Spot
    df_local['gamma_sign'] = np.sign(df_local['Gamma'])
    # Cerchiamo il cambio di segno più vicino allo spot
    df_local['sign_change'] = df_local['gamma_sign'].diff().fillna(0) != 0
    changes = df_local[df_local['sign_change']]
    
    if not changes.empty:
        z_gamma = float(changes.iloc[(changes['strike'] - spot).abs().argsort()[:1]]['strike'].values[0])
    else:
        z_gamma = float(df_local.iloc[(df_local['Gamma']).abs().argsort()[:1]]['strike'].values[0])
        
    return call_wall, put_wall, z_gamma

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE PRO V14")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
    strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
    num_levels = st.sidebar.slider("VISIBILITÀ RANGE", 10, 200, 80)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.1) / 365
            ch = t_obj.option_chain(sel_exp)
            
            # --- CALCOLO CORE ---
            c_v = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            p_v = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike'].astype(float)})
            metrics = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
            for i, m in enumerate(metrics):
                if m in ['Vega', 'Theta']:
                    df[m] = c_v[i] + p_v[i] # Vega/Theta sono additivi (costo/rischio totale)
                else:
                    df[m] = c_v[i] - p_v[i] # Gamma/Vanna/Charm sono direzionali (Net)

            # --- LIVELLI REALI (LOCALIZZATI) ---
            c_wall, p_wall, z_gamma = find_localized_levels(df, spot)

            # --- BIAS DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} Analysis | Expiry: {sel_exp}")
            
            # Formula Bias: Somma pesata di tutte le greche (Gamma, Vanna, Charm, Theta)
            bias_val = (df['Gamma'].sum() * 1.0) + (df['Vanna'].sum() * 0.4) + (df['Charm'].sum() * 0.4) + (df['Theta'].sum() * 0.2)
            b_label, b_color = ("STRONG LONG", "#00ff00") if bias_val > 0 else ("STRONG SHORT", "#ff4444")

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div style='text-align:center;border:2px solid {b_color};padding:10px;border-radius:10px;'><h4>INSTITUTIONAL BIAS</h4><h2 style='color:{b_color};'>{b_label}</h2></div>", unsafe_allow_html=True)
            m2.metric("CALL WALL", f"{c_wall:.0f}", f"{c_wall-spot:.1f} pts")
            m3.metric("PUT WALL", f"{p_wall:.0f}", f"{p_wall-spot:.1f} pts")
            m4.metric("ZERO GAMMA", f"{z_gamma:.0f}", f"FLIP POINT", delta_color="off")

            # --- GRAFICO ---
            df['bin'] = (df['strike'] / strike_step).round() * strike_step
            df_p = df.groupby('bin')[metrics].sum().reset_index().rename(columns={'bin': 'strike'})
            df_p = df_p[(df_p['strike'] >= spot - (strike_step * num_levels/2)) & (df_p['strike'] <= spot + (strike_step * num_levels/2))]

            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_p[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_p['strike'], x=df_p[main_metric], orientation='h', 
                marker_color=colors, width=strike_step*0.8,
                text=[f"{v/1e6:.1f}M" if abs(v)>1e6 else "" for v in df_p[main_metric]], textposition='outside'
            ))

            # Linee Chirurgiche
            fig.add_hline(y=c_wall, line_color="red", line_width=3, annotation_text="CALL WALL")
            fig.add_hline(y=p_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=850,
                yaxis=dict(title="STRIKE", tickformat=".0f", gridcolor="#333"),
                xaxis=dict(title=f"Dollar {main_metric} Exposure ($)", zerolinecolor="white"),
                title=f"Distribuzione Esposizione Reale - {main_metric.upper()}"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore: {e}")
