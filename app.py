import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Ultimate GEX Terminal V3", initial_sidebar_state="expanded")

# --- AUTO-REFRESH (Impostato a 60 secondi per Scalping) ---
st_autorefresh(interval=60000, key="global_refresh")

# --- FUNZIONE DI MAPPATURA INDICI ---
def map_ticker(symbol):
    # Gli indici puri spesso non hanno catene opzioni accessibili facilmente. 
    # Usiamo gli ETF corrispondenti che sono il riferimento per i Market Maker.
    mapping = {
        "NDX": "QQQ",
        "SPX": "SPY",
        "RUT": "IWM",
        "DJI": "DIA"
    }
    return mapping.get(symbol.upper(), symbol.upper())

# --- MOTORE DI CALCOLO GRECHE COMPLETO (BSM) ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
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
    except: return pd.Series([0]*5)

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=60)
def load_market_data(symbol, exp_idx, zoom_val):
    target = map_ticker(symbol)
    t_obj = yf.Ticker(target)
    hist = t_obj.history(period='1d')
    if hist.empty: return None
    
    spot = hist['Close'].iloc[-1]
    exps = t_obj.options
    if not exps: return None
    
    sel_exp = exps[min(exp_idx, len(exps)-1)]
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    chain = t_obj.option_chain(sel_exp)
    
    c_res = chain.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    p_res = chain.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': chain.calls['strike']})
    df['Gamma'] = c_res[0] - p_res[0]
    df['Vanna'] = c_res[1] - p_res[1]
    df['Charm'] = c_res[2] - p_res[2]
    df['Vega'] = c_res[3] + p_res[3]
    df['Theta'] = c_res[4] + p_res[4]
    
    l, u = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
    return spot, df[(df['strike']>=l) & (df['strike']<=u)], sel_exp, df

# --- INTERFACCIA ---
st.sidebar.header("🎯 CONTROLLO ASSET")
input_ticker = st.sidebar.text_input("INSERISCI TICKER (es. NDX, NVDA, TSLA)", "QQQ").upper()

trading_mode = st.sidebar.selectbox("MODALITÀ TRADING", ["SCALPING", "INTRADAY", "SWING"])
# Auto-selezione scadenza in base al modo
expiry_default = 0 if trading_mode == "SCALPING" else (2 if trading_mode == "INTRADAY" else 5)

try:
    ticker_mapped = map_ticker(input_ticker)
    t_engine = yf.Ticker(ticker_mapped)
    avail_exps = t_engine.options
    
    selected_exp_idx = st.sidebar.selectbox("SCADENZA", range(len(avail_exps)), index=min(expiry_default, len(avail_exps)-1), format_func=lambda x: avail_exps[x])
    zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 50, 5)
    metric_to_plot = st.sidebar.radio("VISUALIZZA NEL GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    data = load_market_data(input_ticker, selected_exp_idx, zoom_pct)
    
    if data:
        spot, df_zoom, exp_date, df_full = data
        z_flip = df_zoom.loc[df_zoom[metric_to_plot].abs().idxmin(), 'strike']

        # 1. BIAS STATISTICO (LOGICA COMBINATA GAMMA + VANNA)
        g_sum = df_full['Gamma'].sum()
        v_sum = df_full['Vanna'].sum()
        
        if g_sum > 0 and v_sum > 0: bias, b_col = "STRONG LONG", "#00ff00"
        elif g_sum < 0 and v_sum < 0: bias, b_col = "STRONG SHORT", "#ff4444"
        elif g_sum < 0: bias, b_col = "SHORT BIAS", "#ffaa00"
        else: bias, b_col = "LONG BIAS", "#00aaff"

        st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:#1e1e1e; border-left:10px solid {b_col};'>"
                    f"<h1 style='color:{b_col}; margin:0;'>{bias} | {input_ticker}</h1>"
                    f"<p style='color:white;'>Regime di mercato basato su esposizione Gamma e Vanna globale.</p></div>", unsafe_allow_html=True)

        # 2. GRAFICO GEXBOT STYLE
        max_val = df_zoom[metric_to_plot].abs().max()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df_zoom['strike'], x=df_zoom[metric_to_plot], orientation='h', 
                             marker_color=np.where(df_zoom[metric_to_plot]>=0, '#00ff00', '#00aaff')))
        fig.add_hline(y=spot, line_color="cyan", line_width=2, annotation_text=f"SPOT: {spot:.2f}")
        fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text=f"ZERO {metric_to_plot}")
        fig.update_layout(template="plotly_dark", height=700, xaxis=dict(range=[-max_val*1.1, max_val*1.1]))
        st.plotly_chart(fig, use_container_width=True)

        # 3. TUTTE LE METRICHE (REPLACE COMPLETE)
        st.divider()
        st.subheader("📊 Analisi Greche Totali")
        cols = st.columns(5)
        m_list = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
        for i, m in enumerate(m_list):
            total_val = df_full[m].sum()
            cols[i].metric(f"TOTAL {m}", f"{total_val/1e6:.2f}M" if abs(total_val)>1e5 else f"{total_val:.2f}")

except Exception as e:
    st.error(f"Errore nel caricamento del ticker '{input_ticker}'. Assicurati che il simbolo sia corretto (es. NVDA, AAPL, QQQ).")
