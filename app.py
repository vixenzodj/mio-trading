import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Real-Index GEX Terminal", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# --- FUNZIONE CORREZIONE SIMBOLI PER INDICI ---
def fix_ticker(symbol):
    s = symbol.upper().strip()
    # YFinance richiede il cappelletto ^ per gli indici diretti
    if s in ["NDX", "SPX", "RUT", "VIX"]:
        return f"^{s}"
    return s

# --- MOTORE GRECHE ---
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
    ticker_str = fix_ticker(symbol)
    t_obj = yf.Ticker(ticker_str)
    
    # Per gli indici usiamo il prezzo 'last' più preciso
    hist = t_obj.history(period='1d')
    if hist.empty: return None
    spot = hist['Close'].iloc[-1]
    
    exps = t_obj.options
    if not exps: return None
    
    sel_exp = exps[min(exp_idx, len(exps)-1)]
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    chain = t_obj.option_chain(sel_exp)
    
    # Processiamo CALL e PUT
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

# --- UI ---
st.sidebar.header("🕹️ CONTROLLO INDICI")
input_ticker = st.sidebar.text_input("TICKER (NDX, SPX, NVDA)", "NDX").upper()

mode = st.sidebar.selectbox("TRADING MODE", ["SCALPING", "INTRADAY", "SWING"])
exp_def = 0 if mode == "SCALPING" else 2

try:
    ticker_ready = fix_ticker(input_ticker)
    t_info = yf.Ticker(ticker_ready)
    avail_exps = t_info.options
    
    sel_idx = st.sidebar.selectbox("SCADENZA", range(len(avail_exps)), index=min(exp_def, len(avail_exps)-1), format_func=lambda x: avail_exps[x])
    zoom = st.sidebar.slider("ZOOM %", 1, 30, 5)
    metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    data = load_market_data(input_ticker, sel_idx, zoom)
    
    if data:
        spot, df_z, exp_d, df_f = data
        z_flip = df_z.loc[df_z[metric].abs().idxmin(), 'strike']

        # LOGICA BIAS STATISTICO
        g_sum, v_sum = df_f['Gamma'].sum(), df_f['Vanna'].sum()
        bias, color = ("STRONG LONG", "#00ff00") if g_sum > 0 and v_sum > 0 else (("STRONG SHORT", "#ff4444") if g_sum < 0 and v_sum < 0 else ("NEUTRAL/VOLATILE", "#ffff00"))

        st.markdown(f"<div style='padding:20px; background:#151515; border-left:10px solid {color};'>"
                    f"<h1 style='color:{color};'>{bias} | {input_ticker} @ {spot:.2f}</h1></div>", unsafe_allow_html=True)

        # GRAFICO
        max_v = df_z[metric].abs().max()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[metric], orientation='h', 
                             marker_color=np.where(df_z[metric]>=0, '#00ff00', '#00aaff')))
        fig.add_hline(y=spot, line_color="cyan", annotation_text=f"PREZZO INDICE: {spot:.2f}")
        fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO GAMMA")
        fig.update_layout(template="plotly_dark", height=700, xaxis=dict(range=[-max_v*1.1, max_v*1.1]))
        st.plotly_chart(fig, use_container_width=True)

        # DASHBOARD METRICHE TOTALI
        st.subheader("📊 Analisi Greche Totali (Catena Completa)")
        cols = st.columns(5)
        for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
            val = df_f[m].sum()
            cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")

except Exception as e:
    st.error(f"Errore: {input_ticker} potrebbe non avere opzioni attive o il simbolo è errato.")
