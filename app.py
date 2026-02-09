import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Institutional GEX Terminal", initial_sidebar_state="expanded")

# --- CARICAMENTO DINAMICO TICKER (NDAQ 100, S&P 500, etc) ---
@st.cache_data
def get_all_tickers():
    # Lista base dei più scambiati
    top_tickers = ["QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "NDX", "SPX"]
    # Qui il sistema può potenzialmente scaricare liste da Wikipedia o CSV per arrivare a 1000+
    # Per ora popoliamo con i settori chiave per garantire velocità
    return sorted(list(set(top_tickers)))

# --- MOTORE DI CALCOLO GRECHE (BSM) ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
        r = 0.045 # Tasso risk-free aggiornato
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

# --- LOGICA DI TRADING ---
st.sidebar.header("📡 RADAR OPERATIVO")
search_ticker = st.sidebar.selectbox("CERCA O SELEZIONA TICKER (1000+)", get_all_tickers())
custom_ticker = st.sidebar.text_input("OPPURE SCRIVI TICKER MANUALE", "").upper()
ticker = custom_ticker if custom_ticker else search_ticker

trading_mode = st.sidebar.selectbox("MODALITÀ", ["SCALPING", "INTRADAY", "SWING"])
zoom_val = st.sidebar.slider("ZOOM AREA %", 1, 50, 5)

# Refresh differenziato
refresh_map = {"SCALPING": 60000, "INTRADAY": 300000, "SWING": 900000}
st_autorefresh(interval=refresh_map[trading_mode], key="global_ref")

# --- CARICAMENTO DATI ---
@st.cache_data(ttl=60)
def load_data(t_symbol, mode, zoom):
    t_obj = yf.Ticker(t_symbol)
    h = t_obj.history(period='1d')
    if h.empty: return None
    spot = h['Close'].iloc[-1]
    
    exps = t_obj.options
    # Selezione intelligente della scadenza in base al modo
    idx = 0 if mode == "SCALPING" else (2 if mode == "INTRADAY" else 5)
    sel_exp = exps[min(idx, len(exps)-1)]
    
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    chain = t_obj.option_chain(sel_exp)
    
    # Calcolo parallelo
    c_g = chain.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    p_g = chain.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': chain.calls['strike']})
    cols = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
    for i, m in enumerate(cols):
        if m in ['Gamma', 'Vanna', 'Charm']: df[m] = c_g[i] - p_g[i]
        else: df[m] = c_g[i] + p_g[i]
        
    l, u = spot * (1 - zoom/100), spot * (1 + zoom/100)
    return spot, df[(df['strike']>=l) & (df['strike']<=u)], sel_exp, df

# --- DASHBOARD ---
data_res = load_data(ticker, trading_mode, zoom_val)
if data_res:
    spot, df_p, exp_d, df_f = data_res
    
    # 1. CALCOLO STATISTICO COMBINATO (BIAS)
    # Uniamo Gamma e Vanna per un segnale più professionale
    g_total = df_f['Gamma'].sum()
    v_total = df_f['Vanna'].sum()
    
    # LOGICA DI SEGNALE AVANZATA
    if g_total > 0 and v_total > 0:
        bias, b_color, b_text = "STRONG LONG", "#00ff00", "Vantaggio Statistico Rialzista - Compressione Volatilità"
    elif g_total < 0 and v_total < 0:
        bias, b_color, b_text = "STRONG SHORT", "#ff4444", "Vantaggio Statistico Ribassista - Espansione Volatilità"
    elif g_total < 0 and v_total > 0:
        bias, b_color, b_text = "VOLATILE / NEUTRAL", "#ffff00", "Conflitto tra Gamma e Vanna. Attesa direzionalità chiara."
    else:
        bias, b_color, b_text = "WEAK LONG", "#00aa00", "Bias Rialzista ma debole. Possibile trading di range."

    # Visualizzazione Bias
    st.markdown(f"""
        <div style="background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 10px solid {b_color};">
            <h1 style="color:{b_color}; margin:0;">{bias}</h1>
            <p style="color:white; font-size:18px;">{b_text}</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. GRAFICO
    m_view = st.selectbox("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
    z_flip = df_p.loc[df_p[m_view].abs().idxmin(), 'strike']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_p['strike'], x=df_p[m_view], orientation='h', 
                         marker_color=np.where(df_p[m_view]>=0, '#00ff00', '#00aaff')))
    fig.add_hline(y=spot, line_color="cyan", annotation_text="SPOT")
    fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 3. METRICHE COMPLETE
    st.divider()
    m_cols = st.columns(5)
    for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
        val = df_f[m].sum()
        m_cols[i].metric(f"TOTAL {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")
