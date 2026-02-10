import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V28", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
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
st.sidebar.header("🕹️ GEX ENGINE V28")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
    
    # Adattamento automatico dello step strike in base al prezzo
    default_step = 50 if "NDX" in t_str or "SPX" in t_str else 5
    strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 2, 5, 10, 25, 50, 100, 250], index=TICKER_LIST.index(active_t) if active_t in TICKER_LIST else 4)
    
    # Zoom dinamico basato su percentuale del prezzo per non rompere la visuale
    zoom_pct = st.sidebar.slider("ZOOM AREA PREZZO (% dallo Spot)", 1, 20, 5)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            num_levels = (spot * zoom_pct) / 100 # Calcolo dinamico dello zoom
            
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            # Uniamo e calcoliamo greche
            df_all = pd.concat([c, p])
            res = df_all.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            cols = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
            df_res = pd.DataFrame(res.values, columns=cols)
            df_res['strike'] = df_all['strike'].values
            
            # Raggruppamento
            df_total = df_res.groupby('strike', as_index=False).sum().sort_values('strike')

            # --- LOGICA ZERO GAMMA ---
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']

            # --- SCREMATURA E ZOOM ---
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False)[cols].sum().rename(columns={'bin': 'strike'})
            
            # Filtro chirurgico: solo strike con valore reale e dentro lo zoom
            df_plot_zoom = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- BIAS ---
            gamma_net = df_total['Gamma'].sum()
            bias = "BULLISH" if gamma_net > 0 else "BEARISH"
            b_col = "#00ff00" if bias == "BULLISH" else "#ff4444"

            # --- DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} Terminal | Spot: {spot:.2f}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("SPOT", f"{spot:.2f}")
            d2.metric("VOL TRIGGER", f"{z_gamma:.0f}")
            d3.metric("CALL WALL", f"{call_wall:.0f}")
            d4.markdown(f"<div style='background:{b_col}22; border:1px solid {b_col}; padding:10px; border-radius:5px; text-align:center;'>BIAS: <b>{bias}</b></div>", unsafe_allow_html=True)

            # --- GRAFICO PRO ---
            fig = go.Figure()
            
            # Barre con colori dinamici
            fig.add_trace(go.Bar(
                y=df_plot_zoom['strike'], x=df_plot_zoom[main_metric],
                orientation='h', marker_color=['#00ff00' if x > 0 else '#00aaff' for x in df_plot_zoom[main_metric]],
                width=strike_step * 0.8,
                text=[f"{v/1e6:.1f}M" if abs(v)>1000 else "" for v in df_plot_zoom[main_metric]],
                textposition='outside'
            ))

            # Linee di riferimento
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            fig.add_hline(y=call_wall, line_color="red", annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=900,
                yaxis=dict(title="STRIKE", range=[spot - num_levels, spot + num_levels], gridcolor="#333", dtick=strike_step),
                xaxis=dict(title=f"Net {main_metric} Exposure ($)", zerolinecolor="white"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA DETTAGLIATA (A COLPO D'OCCHIO) ---
            st.markdown("### 🛡️ Livelli di Difesa & Greche Real-Time")
            # Mostriamo i livelli chiave + un buffer intorno allo spot
            key_strikes = [z_gamma, call_wall, put_wall]
            table_data = df_plot[df_plot['strike'].apply(lambda x: any(abs(x - k) < strike_step for k in key_strikes) or abs(x - spot) < strike_step*2)]
            table_data = table_data.sort_values('strike', ascending=False).drop_duplicates()
            
            st.table(table_data[['strike', 'Gamma', 'Vanna', 'Vega', 'Theta']].style.format(precision=2).applymap(
                lambda x: 'color: #00ff00' if x > 0 else 'color: #ff4444' if x < 0 else 'color: white'
            ))

    except Exception as e:
        st.error(f"Errore: {e}")
