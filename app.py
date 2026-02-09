import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Ultimate GEX Intelligence Terminal", initial_sidebar_state="expanded")

# --- TRADING MODE SELECTOR ---
st.sidebar.header("🕹️ STRATEGY MODE")
trading_mode = st.sidebar.selectbox(
    "Seleziona Tipo di Trading",
    ["SCALPING (0DTE Focus)", "INTRADAY (Weekly Focus)", "SWING (Monthly/Institutional)"]
)

if trading_mode == "SCALPING (0DTE Focus)":
    refresh_rate, default_zoom, expiry_auto = 60000, 3, 0
    st_autorefresh(interval=refresh_rate, key="scalp_ref")
elif trading_mode == "INTRADAY (Weekly Focus)":
    refresh_rate, default_zoom, expiry_auto = 300000, 8, 1
    st_autorefresh(interval=refresh_rate, key="intra_ref")
else:
    refresh_rate, default_zoom, expiry_auto = 900000, 20, 3
    st_autorefresh(interval=refresh_rate, key="swing_ref")

# --- MOTORE DI CALCOLO COMPLETO ---
def get_all_greeks(row, spot, t_yrs, r=0.04):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
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

@st.cache_data(ttl=60)
def get_data(ticker, exp_idx, zoom_pct):
    t_obj = yf.Ticker(ticker)
    hist = t_obj.history(period='1d')
    if hist.empty: return None
    spot = hist['Close'].iloc[-1]
    sel_exp = t_obj.options[exp_idx]
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    opts = t_obj.option_chain(sel_exp)
    
    c_res = opts.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    p_res = opts.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    
    df = pd.DataFrame({'strike': opts.calls['strike']})
    df['Gamma'] = c_res[0] - p_res[0]
    df['Vanna'] = c_res[1] - p_res[1]
    df['Charm'] = c_res[2] - p_res[2]
    df['Vega'] = c_res[3] + p_res[3]
    df['Theta'] = c_res[4] + p_res[4]
    
    l, u = spot * (1 - zoom_pct/100), spot * (1 + zoom_pct/100)
    return spot, df[(df['strike'] >= l) & (df['strike'] <= u)], sel_exp, df

# --- UI PRINCIPALE ---
ticker = st.sidebar.text_input("SYMBOL", "QQQ").upper()
try:
    exps = yf.Ticker(ticker).options
    exp_idx = st.sidebar.selectbox("EXPIRY", range(len(exps)), index=min(expiry_auto, len(exps)-1), format_func=lambda x: exps[x])
    zoom = st.sidebar.slider("ZOOM AREA %", 1, 50, default_zoom)
    metric_sel = st.sidebar.radio("VISUALIZZA NEL GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])
except: st.stop()

data = get_data(ticker, exp_idx, zoom)
if data:
    spot, df_plot, exp_date, df_full = data
    zero_flip = df_plot.loc[df_plot[metric_sel].abs().idxmin(), 'strike']
    
    # --- GRAFICO GEXBOT STYLE ---
    max_v = df_plot[metric_sel].abs().max()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_plot['strike'], x=df_plot[metric_sel], orientation='h',
                         marker_color=np.where(df_plot[metric_sel]>=0, '#00ff00', '#00aaff')))
    fig.add_hline(y=spot, line_color="cyan", annotation_text="SPOT")
    fig.add_hline(y=zero_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
    fig.update_layout(template="plotly_dark", height=700, xaxis=dict(range=[-max_v*1.1, max_v*1.1]))
    st.plotly_chart(fig, use_container_width=True)

    # --- PANNELLO STATISTICO & SEGNALI ---
    st.markdown("---")
    st.subheader("🧠 Intelligence Report & Signal")
    
    # Logica di Analisi
    total_gamma = df_full['Gamma'].sum()
    total_vanna = df_full['Vanna'].sum()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("🎯 **BIAS OPERATIVO**")
        if total_gamma > 0:
            st.success(f"MODE: **LONG BIAS** (Gamma Positivo). Il mercato tende a comprimere la volatilità. Buy the dips.")
        else:
            st.error(f"MODE: **SHORT/AGGRESSIVE** (Gamma Negativo). Il mercato tende a espandere i movimenti. Sell the rips.")
            
        if abs(spot - zero_flip) / spot < 0.01:
            st.warning("⚠️ **VOLATILITY TRIGGER:** Il prezzo è vicino allo Zero Flip. Attesa esplosione di direzionalità!")

    with c2:
        st.info("📈 **PROIEZIONE STATISTICA**")
        if trading_mode.startswith("SCALPING"):
            st.write("**Breve Termine (0DTE):** Focus sui muri immediati. Elevata probabilità di mean reversion se Gamma è alto.")
        elif trading_mode.startswith("INTRADAY"):
            st.write("**Medio Termine:** Direzionalità guidata dal Vanna. Se Vanna è +, il calo di IV spinge il prezzo su.")
        else:
            st.write("**Lungo Termine (Institutional):** Accumulazione profonda. I muri mensili fungono da calamite strutturali.")

    # --- TUTTE LE METRICHE (RESTORED) ---
    st.markdown("### 📊 Market Snapshot (Tutte le Metriche)")
    m_cols = st.columns(5)
    for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
        val = df_full[m].sum()
        m_cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")
