import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="PRO Scalper Terminal", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="datarefresh") # Refresh ogni minuto per scalping

# --- MOTORE CALCOLO GRECHE ---
def get_greeks(row, spot, t_yrs, r=0.04):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
        
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        cdf = norm.cdf(d1)
        
        gamma = pdf / (s * v * np.sqrt(t_yrs))
        vanna = (pdf * d1) / v
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) ))
        vega = s * pdf * np.sqrt(t_yrs)
        theta = -(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)
        
        return pd.Series([gamma * oi * 100, vanna * oi, charm * oi, vega * oi, theta * oi])
    except:
        return pd.Series([0]*5)

@st.cache_data(ttl=60)
def get_full_market_data(ticker, exp_idx, zoom_pct):
    t_obj = yf.Ticker(ticker)
    hist = t_obj.history(period='1d')
    if hist.empty: return None
    spot = hist['Close'].iloc[-1]
    
    sel_exp = t_obj.options[exp_idx]
    dt_exp = datetime.strptime(sel_exp, '%Y-%m-%d')
    t_yrs = max((dt_exp - datetime.now()).days, 0.5) / 365
    
    opts = t_obj.option_chain(sel_exp)
    calls, puts = opts.calls, opts.puts
    
    # Calcolo Greche
    c_grk = calls.apply(lambda r: get_greeks(r, spot, t_yrs), axis=1)
    p_grk = puts.apply(lambda r: get_greeks(r, spot, t_yrs), axis=1)
    
    # Merge dei dati
    df = pd.DataFrame({'strike': calls['strike']})
    metrics = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
    for i, m in enumerate(metrics):
        # Per Gamma, Vanna, Charm usiamo la differenza (Net)
        # Per Vega, Theta usiamo la somma (Esposizione totale)
        if m in ['Gamma', 'Vanna', 'Charm']:
            df[m] = c_grk[i] - p_grk[i]
        else:
            df[m] = c_grk[i] + p_grk[i]
    
    # Filtro Zoom
    l_bound, u_bound = spot * (1 - zoom_pct/100), spot * (1 + zoom_pct/100)
    df_zoom = df[(df['strike'] >= l_bound) & (df['strike'] <= u_bound)].copy()
    
    return spot, df_zoom, sel_exp, df

# --- UI ---
st.sidebar.title("🚀 SCALPER SETTINGS")
ticker = st.sidebar.text_input("SYMBOL", "QQQ").upper()
try:
    exps = yf.Ticker(ticker).options
    exp_idx = st.sidebar.selectbox("EXPIRY (0DTE = Top)", range(len(exps)), format_func=lambda x: exps[x])
except: st.stop()

zoom = st.sidebar.slider("ZOOM (RANGE %)", 1, 30, 5)
metric_sel = st.sidebar.radio("INDICATORE", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

data = get_full_market_data(ticker, exp_idx, zoom)
if data:
    spot, df_plot, exp_date, df_full = data

    # 1. Calcolo ZERO GAMMA DINAMICO (Local Flip Point)
    # Cerchiamo dove la metrica passa da + a - nell'area visibile
    try:
        # Troviamo lo strike dove il valore è più vicino allo zero nel range zoommato
        zero_flip = df_plot.loc[df_plot[metric_sel].abs().idxmin(), 'strike']
    except: zero_flip = spot

    # 2. Status Metrica Corrente
    current_val = df_full.iloc[(df_full['strike']-spot).abs().idxmin()][metric_sel]
    status = "POSITIVO" if current_val > 0 else "NEGATIVO"
    color = "#00ff00" if status == "POSITIVO" else "#ff4444"

    # --- GRAFICO ---
    max_val = df_plot[metric_sel].abs().max()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_plot['strike'], x=df_plot[metric_sel], orientation='h',
        marker_color=np.where(df_plot[metric_sel] >= 0, '#00ff00', '#00aaff'),
        hovertemplate="Strike: %{y}<br>Value: %{x:.2f}<extra></extra>"
    ))

    # Linee Operative
    fig.add_hline(y=spot, line_color="cyan", line_width=2, annotation_text=f"SPOT: {spot:.2f}")
    fig.add_hline(y=zero_flip, line_dash="dash", line_color="yellow", line_width=2, 
                  annotation_text=f"ZERO {metric_sel.upper()}: {zero_flip}")

    fig.update_layout(
        template="plotly_dark", height=800,
        title=f"PRO {metric_sel.upper()} PROFILE - {ticker} ({exp_date})",
        xaxis=dict(range=[-max_val*1.1, max_val*1.1], title=f"NET {metric_sel.upper()} EXPOSURE"),
        yaxis=dict(title="STRIKE", autorange=False, range=[df_plot['strike'].min(), df_plot['strike'].max()]),
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- FOOTER STATISTICO ---
    st.markdown(f"### 📊 Market Regime: <span style='color:{color}'>{metric_sel} {status}</span>", unsafe_allow_html=True)
    cols = st.columns(5)
    metrics_list = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
    for i, m in enumerate(metrics_list):
        val = df_full[m].sum()
        cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")
