import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V55 - ULTIMATE", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh")

# --- CORE QUANT ENGINE (BLACK-SCHOLES) ---
def calculate_gex_at_price(price, df, r=0.045):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    exposure_size = df['openInterest'].fillna(0).values + (df['volume'].fillna(0).values * 0.5)
    d1 = (np.log(price/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (price * iv * np.sqrt(T))
    side = np.where(df['type'] == 'call', 1, -1)
    return np.sum(gamma * exposure_size * 100 * price * side)

def get_greeks_pro(df, S, r=0.045):
    if df.empty: return df
    df = df[df['impliedVolatility'] > 0.01].copy()
    K, iv, T = df['strike'].values, df['impliedVolatility'].values, np.maximum(df['dte_years'].values, 0.0001)
    oi_vol_weighted = df['openInterest'].fillna(0).values + (df['volume'].fillna(0).values * 0.5)
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    side = np.where(df['type'] == 'call', 1, -1)
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi_vol_weighted * 100 * side
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi_vol_weighted * side
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi_vol_weighted * 100 * side
    df['Vega']  = S * pdf * np.sqrt(T) * 0.01 * oi_vol_weighted * 100
    df['Theta'] = ((-(S * pdf * iv) / (2 * np.sqrt(T))) - side * (r * K * np.exp(-r * T) * norm.cdf(d2 * side))) * (1/365) * oi_vol_weighted * 100
    return df

# --- DATA FETCHING ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker, dates):
    t = yf.Ticker(ticker)
    frames = []
    for d in dates:
        try:
            oc = t.option_chain(d)
            frames.append(pd.concat([oc.calls.assign(type='call', exp=d), oc.puts.assign(type='put', exp=d)]))
        except: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.markdown("## ⚙️ QUANT ENGINE V55")
asset = st.sidebar.selectbox("TICKER", ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA"])
t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
current_ticker = t_map.get(asset, asset)

ticker_obj = yf.Ticker(current_ticker)
h = ticker_obj.history(period='1d')
if h.empty: st.stop()
spot = h['Close'].iloc[-1]

available_dates = ticker_obj.options
today = datetime.now()
selected_dte = st.sidebar.multiselect("SCADENZE 0DTE/1DTE", [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates], default=[f"{(datetime.strptime(available_dates[0], '%Y-%m-%d') - today).days + 1} DTE | {available_dates[0]}"])

metric = st.sidebar.radio("METRICA GRAFICO PRINCIPALE", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
gran = st.sidebar.select_slider("GRANULARITÀ", options=[1, 2, 5, 10, 20, 50, 100], value=10 if "NDX" in asset else 5)
zoom_val = st.sidebar.slider("ZOOM AREA %", 0.5, 15.0, 3.0)

if selected_dte:
    target_dates = [d.split('| ')[1] for d in selected_dte]
    raw_data = fetch_data(current_ticker, target_dates)
    
    if not raw_data.empty:
        raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
        
        # 1. Zero Gamma
        try: z_gamma = brentq(calculate_gex_at_price, spot * 0.90, spot * 1.10, args=(raw_data,))
        except: z_gamma = spot 

        # 2. Greche
        df = get_greeks_pro(raw_data, spot)
        agg = df.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
        
        # 3. Muri intelligenti (Solo nel range visibile per lo scalping)
        lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
        visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
        
        if not visible_agg.empty:
            c_wall = visible_agg.loc[visible_agg['Gamma'].idxmax(), 'strike']
            p_wall = visible_agg.loc[visible_agg['Gamma'].idxmin(), 'strike']
        else:
            c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
            p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']

        # --- DASHBOARD ---
        st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CALL WALL (Res)", f"{c_wall:.0f}")
        m2.metric("ZERO GAMMA (Pivot)", f"{z_gamma:.2f}")
        m3.metric("PUT WALL (Sup)", f"{p_wall:.0f}")
        m4.metric("NET GEX", f"${agg['Gamma'].sum()/1e6:.1f}M")

        # --- FUNZIONE 1: OVERVIEW MULTI-METRICA ---
        st.markdown("---")
        st.markdown("### 📊 Real-Time Multi-Metric Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        for col, m_name, color in zip([ov1, ov2, ov3, ov4], ["Vanna", "Charm", "Vega", "Theta"], ["#00BFFF", "#FF00FF", "#FFFF00", "#FFA500"]):
            mini_fig = go.Figure()
            mini_fig.add_trace(go.Scatter(x=agg['strike'], y=agg[m_name], fill='tozeroy', line_color=color))
            mini_fig.update_layout(title=m_name, height=150, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark", xaxis_visible=False, yaxis_visible=False)
            col.plotly_chart(mini_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")

        # --- FUNZIONE 2: GRAFICO PRINCIPALE CON MURI CHIARI ---
        p_df = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)].copy()
        p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
        p_df = p_df.groupby('bin', as_index=False).sum()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=p_df['bin'], x=p_df[metric], orientation='h',
            marker=dict(color=['#00FF41' if x >= 0 else '#0074D9' for x in p_df[metric]], line_width=0),
            width=gran * 0.85, name=metric
        ))

        # Linee Livelli con stile Istituzionale
        fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT PRICE", annotation_position="top right")
        fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA FLIP")
        
        # Muri ricalcolati per essere sempre visibili se nel range
        fig.add_hline(y=c_wall, line_color="#FF4136", line_width=3, annotation_text=f"CALL WALL @{c_wall:.0f}", annotation_bgcolor="#FF4136")
        fig.add_hline(y=p_wall, line_color="#2ECC40", line_width=3, annotation_text=f"PUT WALL @{p_wall:.0f}", annotation_bgcolor="#2ECC40")

        fig.update_layout(
            template="plotly_dark", height=800, margin=dict(l=0,r=0,t=0,b=0),
            yaxis=dict(range=[lo, hi], dtick=gran, gridcolor="#333", title="STRIKE"),
            xaxis=dict(title=f"Net {metric} Exposure", gridcolor="#333", zerolinecolor="#FFF")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.code(f"Pivots: 0G@{z_gamma:.2f} | CW@{c_wall:.0f} | PW@{p_wall:.0f}")
