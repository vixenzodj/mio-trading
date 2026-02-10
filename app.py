import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO V22", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# --- MOTORE DI CALCOLO ISTITUZIONALE ---
def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi < 1: return [0.0]*5
        
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        
        # Dollar Gamma Standard
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * (s**2) * 0.01 * oi * 100
        vanna = s * pdf * d1 / v * 0.01 * oi
        charm = (pdf * (r / (v * np.sqrt(t_yrs)) - d1 / (2 * t_yrs))) * oi * 100
        vega = s * pdf * np.sqrt(t_yrs) * oi * 100
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2 if row['type']=='call' else -d2)) * oi * 100
        
        mult = 1 if row['type'] == 'call' else -1
        return [gamma * mult, vanna * mult, charm * mult, vega, theta]
    except: return [0.0]*5

# --- SIDEBAR: INPUT LIBERO ---
st.sidebar.header("🕹️ GEX TERMINAL V22")
# Qui puoi scrivere qualsiasi ticker (es. MSTR, NVDA, ^NDX)
ticker_input = st.sidebar.text_input("INSERISCI TICKER (es. NDX, TSLA, SPX)", "NDX").upper().strip()
t_str = f"^{ticker_input}" if ticker_input in ["NDX", "SPX", "RUT"] else ticker_input

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE (Granularità)", [1, 5, 10, 25, 50, 100, 250], index=5)
        zoom_range = st.sidebar.slider("RANGE VISIBILE (Punti)", 100, 5000, 1500)
        main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- ELABORAZIONE DATI ---
            cols = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
            c_data = [calc_greeks_pro(row, spot, t_yrs) for _, row in chain.calls.iterrows()]
            p_data = [calc_greeks_pro(row, spot, t_yrs) for _, row in chain.puts.iterrows()]
            
            df_all = pd.concat([
                pd.DataFrame(c_data, columns=cols).assign(strike=chain.calls['strike'].values),
                pd.DataFrame(p_data, columns=cols).assign(strike=chain.puts['strike'].values)
            ]).groupby('strike', as_index=False).sum().sort_values('strike')

            # --- ZERO GAMMA (GEXBOT CALIBRATION) ---
            # Filtro per focalizzarsi sulla liquidità reale (Gexbot style)
            df_near = df_all[(df_all['strike'] >= spot * 0.8) & (df_all['strike'] <= spot * 1.2)].copy()
            df_near['cum_gamma'] = df_near['Gamma'].cumsum()
            z_gamma = df_near.loc[df_near['cum_gamma'].abs().idxmin(), 'strike']

            # --- BINNING PER GRAFICO ---
            df_all['bin'] = np.floor(df_all['strike'] / strike_step) * strike_step
            df_plot = df_all.groupby('bin', as_index=False)[cols].sum().rename(columns={'bin': 'strike'})
            
            # Zoom per evitare lo schiacciamento dello screenshot precedente
            df_plot = df_plot[(df_plot['strike'] >= spot - zoom_range) & (df_plot['strike'] <= spot + zoom_range)]
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- UI ---
            st.markdown(f"## 🏛️ {ticker_input} GEX | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ZERO FLIP", f"{z_gamma:.0f}")
            m2.metric("CALL WALL", f"{call_wall:.0f}")
            m3.metric("PUT WALL", f"{put_wall:.0f}")
            m4.metric("TARGET", "VOL TRIGGER")

            # --- GRAFICO DINAMICO ---
            fig = go.Figure()
            # Colore basato sulla metrica (Call vs Put Gamma)
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h', 
                marker_color=colors, width=strike_step * 0.8,
                name=main_metric
            ))

            # Marcatori a video
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=3, line_dash="dash", annotation_text="ZERO GAMMA")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"SPOT: {spot:.2f}")
            fig.add_hline(y=call_wall, line_color="red", line_width=2, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="green", line_width=2, annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=900,
                yaxis=dict(title="STRIKE", autorange="reversed", gridcolor="#333", nticks=60),
                xaxis=dict(title=f"Net Dollar {main_metric} Exposure ($)", zerolinecolor="white")
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ticker non trovato o errore dati: {e}")
