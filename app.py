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
st.set_page_config(layout="wide", page_title="SENTINEL GEX V58 - PRO", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh")

# --- CORE QUANT ENGINE ---
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

# --- NAVIGAZIONE ---
st.sidebar.markdown("## 🧭 SISTEMA")
menu = st.sidebar.radio("Seleziona Vista:", ["🏟️ DASHBOARD SINGOLA", "🔥 SCANNER HOT TICKERS"])

# =================================================================
# PAGINA 1: DASHBOARD SINGOLA (IDENTICA AL TUO ORIGINALE)
# =================================================================
if menu == "🏟️ DASHBOARD SINGOLA":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛰️ SENTINEL V58 HUB")
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "MSTR"]

    new_asset = st.sidebar.text_input("➕ CARICA TICKER", "").upper().strip()
    if new_asset and new_asset not in st.session_state.ticker_list:
        st.session_state.ticker_list.insert(0, new_asset)
        st.rerun()

    asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
    t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
    current_ticker = t_map.get(asset, asset)

    ticker_obj = yf.Ticker(current_ticker)
    h = ticker_obj.history(period='1d')
    if h.empty: st.stop()
    spot = h['Close'].iloc[-1]

    available_dates = ticker_obj.options
    today = datetime.now()
    date_options = [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates]
    selected_dte = st.sidebar.multiselect("SCADENZE", date_options, default=[date_options[0]])

    metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    gran = st.sidebar.select_slider("GRANULARITÀ", options=[1, 2, 5, 10, 20, 25, 50, 100, 250], value=5)
    zoom_val = st.sidebar.slider("ZOOM %", 0.5, 15.0, 3.0)

    if selected_dte:
        target_dates = [d.split('| ')[1] for d in selected_dte]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_min = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            sd_move = spot * mean_iv * np.sqrt(max(dte_min, 1)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move
            sd2_up, sd2_down = spot + (sd_move * 2), spot - (sd_move * 2)

            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            agg = df.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            c_wall = visible_agg.loc[visible_agg['Gamma'].idxmax(), 'strike'] if not visible_agg.empty else spot
            p_wall = visible_agg.loc[visible_agg['Gamma'].idxmin(), 'strike'] if not visible_agg.empty else spot

            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.2f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            st.markdown("---")
            st.markdown("### 🛰️ Real-Time Metric Regime & Market Direction")
            
            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            net_vega, net_theta = agg['Vega'].sum(), agg['Theta'].sum()

            r1, r2, r3, r4, r5 = st.columns(5)
            for name, val, col in [("GAMMA", net_gamma, r1), ("VANNA", net_vanna, r2), ("CHARM", net_charm, r3), ("VEGA", net_vega, r4), ("THETA", net_theta, r5)]:
                reg = "POSITIVO" if val > 0 else "NEGATIVO"
                col.markdown(f"**{name}**")
                col.markdown(f"<h3 style='color:{'#00FF41' if val > 0 else '#FF4136'}; margin:0;'>{reg}</h3>", unsafe_allow_html=True)
                col.caption(f"Net: ${val/1e6:.2f}M")

            st.markdown("#### 🧭 MARKET DIRECTION INDICATOR")
            
            direction = "NEUTRALE / ATTESA"; bias_color = "gray"
            if net_gamma < 0 and net_vanna < 0:
                direction = "🔴 PERICOLO ESTREMO: SHORT GAMMA + NEGATIVE VANNA (Crash Risk)"; bias_color = "#8B0000"
            elif net_gamma < 0:
                direction = "🔴 ACCELERAZIONE VOLATILITÀ (Short Gamma Bias)"; bias_color = "#FF4136"
