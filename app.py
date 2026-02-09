import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Professional GEX Terminal", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# --- LISTA TICKER ---
TICKER_LIST = ["NDX", "SPX", "QQQ", "SPY", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN"]

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE CALCOLO GRECHE ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        r = 0.045
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0.0]*5)

# --- SIDEBAR ---
st.sidebar.header("🕹️ DASHBOARD SETTINGS")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
trade_mode = st.sidebar.selectbox("TIMEFRAME", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])

# NUOVE FUNZIONI DI GRANULARITÀ
st.sidebar.subheader("🎯 VISUALIZZAZIONE")
strike_step = st.sidebar.selectbox("GRANULARITÀ STRIKE (Step)", [1, 5, 10, 25, 50, 100], index=2)
num_strikes = st.sidebar.slider("NUMERO STRIKE VISIBILI", 10, 100, 40)
main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

t_str = fix_ticker(active_t)
try:
    t_obj = yf.Ticker(t_str)
    hist = t_obj.history(period='1d')
    if not hist.empty:
        spot = hist['Close'].iloc[-1]
        exps = t_obj.options
        idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
        sel_exp = exps[min(idx, len(exps)-1)]
        
        # Caricamento Opzioni
        t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel_exp)
        
        # Calcolo Greche
        c_m = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p_m = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        
        df = pd.DataFrame({'strike': ch.calls['strike']})
        df['Gamma'], df['Vanna'], df['Charm'] = c_m[0]-p_m[0], c_m[1]-p_m[1], c_m[2]-p_m[2]
        df['Vega'], df['Theta'] = c_m[3]+p_m[3], c_m[4]+p_m[4]

        # --- LOGICA DI SCREMATURA (IL CUORE DELLA TUA RICHIESTA) ---
        # 1. Filtriamo solo gli strike arrotondati allo step scelto
        df = df[df['strike'] % strike_step == 0]
        
        # 2. Prendiamo i top N strike più vicini al prezzo
        df['dist'] = abs(df['strike'] - spot)
        df_z = df.sort_values('dist').head(num_strikes).sort_values('strike')

        # Identificazione Muri
        call_wall = df.loc[df['Gamma'].idxmax(), 'strike']
        put_wall = df.loc[df['Gamma'].idxmin(), 'strike']

        # --- GRAFICO GEXBOT STYLE ---
        fig = go.Figure()
        
        # Barre Mirror
        colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_z[main_metric]]
        fig.add_trace(go.Bar(y=df_z['strike'].astype(str), x=df_z[main_metric], orientation='h', 
                             marker_color=colors, name=main_metric))

        # Annotazioni Prezzo e Muri
        fig.add_vline(x=0, line_color="white", line_width=1)
        
        # Layout pulito
        fig.update_layout(
            template="plotly_dark", height=800,
            title=f"<b>{active_t} NET {main_metric.upper()}</b> - {sel_exp} (Step: {strike_step})",
            xaxis_title=f"{main_metric} Exposure",
            yaxis=dict(title="STRIKE PRICE", autorange="reversed"),
            bargap=0.1
        )
        
        # --- UI INDICATORI ---
        st.subheader(f"📊 Market Intelligence: {active_t}")
        c1, c2, c3 = st.columns(3)
        c1.metric("PREZZO SPOT", f"{spot:.2f}")
        c2.metric("CALL WALL (Resistenza)", f"{call_wall}")
        c3.metric("PUT WALL (Supporto)", f"{put_wall}")

        st.plotly_chart(fig, use_container_width=True)

        # Metriche Totali
        st.divider()
        m_cols = st.columns(5)
        for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
            val = df[m].sum()
            m_cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")

except Exception as e:
    st.error(f"Errore nel caricamento dei dati: {e}")
