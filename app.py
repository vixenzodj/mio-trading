import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

st.set_page_config(layout="wide", page_title="GEX PRO V19", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE PROFESSIONALE ---
def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi < 5: return pd.Series([0.0]*5) # Filtro OI minimo
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * (s**2) * 0.01 * oi * 100
        vanna = s * pdf * d1 / v * 0.01 * oi
        charm = (pdf * (r / (v * np.sqrt(t_yrs)) - d1 / (2 * t_yrs))) * oi * 100
        vega = s * pdf * np.sqrt(t_yrs) * oi * 100
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2 if row['type']=='call' else -d2)) * oi * 100
        mult = 1 if row['type'] == 'call' else -1
        return pd.Series([gamma * mult, vanna * mult, charm * mult, vega, theta])
    except: return pd.Series([0.0]*5)

# --- SIDEBAR ---
st.sidebar.header("🕹️ CONTROLLO GRANULARITÀ")
active_t = st.sidebar.selectbox("ASSET", ["NDX", "SPX", "TSLA", "NVDA", "AAPL", "QQQ", "SPY"])
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA", exps)
    
    # RIPRISTINO STEP MANUALE (Per risolvere il problema della distanza visiva)
    strike_step = st.sidebar.selectbox("STEP STRIKE (Granularità)", [1, 5, 10, 20, 25, 50, 100], index=5)
    zoom_range = st.sidebar.slider("RANGE VISIBILE (Punti)", 100, 2000, 800)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            # Calcolo Greche
            df_c = pd.DataFrame(c.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1).values, columns=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
            df_c['strike'] = c['strike'].values
            df_p = pd.DataFrame(p.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1).values, columns=['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
            df_p['strike'] = p['strike'].values
            
            df_all = pd.concat([df_c, df_p]).groupby('strike', as_index=False).sum()
            
            # --- CALCOLO ZERO GAMMA "LOCALIZZATO" ---
            # Filtriamo solo strike vicino allo spot per evitare i 15.000 (Gexbot Logic)
            df_near = df_all[(df_all['strike'] >= spot * 0.92) & (df_all['strike'] <= spot * 1.08)].copy()
            df_near['cum_gamma'] = df_near['Gamma'].cumsum()
            z_gamma = df_near.loc[df_near['cum_gamma'].abs().idxmin(), 'strike']

            # --- AGGREGAZIONE PER VISUALIZZAZIONE PULITA ---
            df_all['bin'] = np.floor(df_all['strike'] / strike_step) * strike_step
            df_plot = df_all.groupby('bin', as_index=False).sum()
            df_plot = df_plot.rename(columns={'bin': 'strike'})
            
            # Taglio del grafico per evitare lo schiacciamento
            df_plot = df_plot[(df_plot['strike'] >= spot - zoom_range) & (df_plot['strike'] <= spot + zoom_range)]
            
            # Muri Operativi
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- UI ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ZERO FLIP", f"{z_gamma:.0f}")
            c2.metric("CALL WALL", f"{call_wall:.0f}")
            c3.metric("PUT WALL", f"{put_wall:.0f}")
            c4.metric("SPOT", f"{spot:.2f}")

            # GRAFICO
            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h', 
                marker_color=colors, width=strike_step * 0.8
            ))

            # Linee Gexbot-Style
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="VOL TRIGGER")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=call_wall, line_color="red", line_width=2, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="green", line_width=2, annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=900,
                yaxis=dict(title="STRIKE", gridcolor="#333", nticks=50, autorange="reversed"),
                xaxis=dict(title=f"Net Dollar {main_metric}", zerolinecolor="white")
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore: {e}")
