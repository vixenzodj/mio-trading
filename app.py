import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V18", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE DI CALCOLO ISTITUZIONALE ---
def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        
        # DOLLAR GAMMA EXPOSURE (Standard Goldman/Gexbot)
        # Formula: 0.5 * Gamma * Spot^2 * 0.01 * OI
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * (s**2) * 0.01 * oi * 100
        vanna = s * pdf * d1 / v * 0.01 * oi
        charm = (pdf * (r / (v * np.sqrt(t_yrs)) - d1 / (2 * t_yrs))) * oi * 100
        vega = s * pdf * np.sqrt(t_yrs) * oi * 100
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2 if row['type']=='call' else -d2)) * oi * 100
        
        mult = 1 if row['type'] == 'call' else -1
        return pd.Series([gamma * mult, vanna * mult, charm * mult, vega, theta])
    except: return pd.Series([0.0]*5)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V18")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
    num_levels = st.sidebar.slider("ZOOM AREA PREZZO", 10, 100, 40)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- ELABORAZIONE DATI ---
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            c_res = c.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            p_res = p.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            
            cols = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
            df_c = pd.DataFrame(c_res.values, columns=cols); df_c['strike'] = c['strike'].values
            df_p = pd.DataFrame(p_res.values, columns=cols); df_p['strike'] = p['strike'].values
            
            df_total = pd.concat([df_c, df_p]).groupby('strike', as_index=False).sum()
            df_total = df_total.sort_values('strike')

            # --- IL VERO CALCOLO ZERO GAMMA (Gexbot Style) ---
            # 1. Filtriamo solo gli strike attivi (±15% dallo spot) per evitare i 15.000
            df_active = df_total[(df_total['strike'] >= spot * 0.85) & (df_total['strike'] <= spot * 1.15)].copy()
            
            # 2. Calcoliamo il Gamma Cumulativo Netto
            df_active['cum_gamma'] = df_active['Gamma'].cumsum()
            
            # 3. Lo Zero Flip è il punto dove il cumulato passa da negativo a positivo (o viceversa)
            zero_idx = (df_active['cum_gamma']).abs().idxmin()
            z_gamma = df_active.loc[zero_idx, 'strike']
            
            # Muri entro il range operativo
            call_wall = df_active.loc[df_active['Gamma'].idxmax(), 'strike']
            put_wall = df_active.loc[df_active['Gamma'].idxmin(), 'strike']

            # --- DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            
            # Calcolo BIAS REALE: Pressione Gamma Netta
            bias_val = df_active['Gamma'].sum()
            b_label, b_color = ("BULLISH (Long Gamma)", "#00ff00") if bias_val > 0 else ("BEARISH (Short Gamma)", "#ff4444")

            d1, d2, d3, d4 = st.columns(4)
            d1.markdown(f"<div style='border:2px solid {b_color};padding:10px;border-radius:10px;text-align:center;'>INSTITUTIONAL BIAS<br><b style='color:{b_color};'>{b_label}</b></div>", unsafe_allow_html=True)
            d2.metric("CALL WALL", f"{call_wall:.0f}", f"{call_wall-spot:.1f}")
            d3.metric("PUT WALL", f"{put_wall:.0f}", f"{put_wall-spot:.1f}")
            d4.metric("ZERO FLIP", f"{z_gamma:.0f}", "VOL TRIGGER")

            # --- GRAFICO DINAMICO ---
            # Filtriamo solo l'area intorno allo spot per una granularità perfetta
            # Non usiamo più step fissi ma mostriamo gli strike reali con OI significativo
            df_plot = df_active[(df_active['strike'] >= spot * (1 - num_levels/1000)) & (df_active['strike'] <= spot * (1 + num_levels/1000))]

            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h', 
                marker_color=colors, width=1.5,
                text=[f"{v/1e6:.1f}M" if abs(v)>5e5 else "" for v in df_plot[main_metric]], textposition='outside'
            ))

            # Linee Operative
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO FLIP")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=900,
                yaxis=dict(title="STRIKE", autorange=True, tickformat=".0f", gridcolor="#333", nticks=40),
                xaxis=dict(title=f"Dollar {main_metric} Exposure ($)", zerolinecolor="white"),
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore: {e}")
