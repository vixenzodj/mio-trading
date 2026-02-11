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
    # Calcolo esposizione in Dollari
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

# --- SIDEBAR ---
st.sidebar.markdown("## 🛰️ SENTINEL V58 HUB")

if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA", "AAPL", "MSTR", "BTC-USD"]

new_asset = st.sidebar.text_input("➕ AGGIUNGI TICKER", "").upper().strip()
if new_asset and new_asset not in st.session_state.ticker_list:
    st.session_state.ticker_list.insert(0, new_asset)
    st.rerun()

asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
current_ticker = t_map.get(asset, asset)

ticker_obj = yf.Ticker(current_ticker)
h = ticker_obj.history(period='1d')
if h.empty: 
    st.error(f"Ticker {asset} non trovato.")
    st.stop()
spot = h['Close'].iloc[-1]

available_dates = ticker_obj.options
if not available_dates:
    st.warning(f"Nessuna opzione per {asset}")
    st.stop()

today = datetime.now()
date_options = [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates]
selected_dte = st.sidebar.multiselect("SCADENZE", date_options, default=[date_options[0]])

metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
gran = st.sidebar.select_slider("GRANULARITÀ PREZZO", options=[1, 2, 5, 10, 20, 25, 50, 100, 250], value=5)
zoom_val = st.sidebar.slider("ZOOM AREA %", 0.5, 15.0, 3.0)

if selected_dte:
    target_dates = [d.split('| ')[1] for d in selected_dte]
    raw_data = fetch_data(current_ticker, target_dates)
    
    if not raw_data.empty:
        raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
        try: z_gamma = brentq(calculate_gex_at_price, spot * 0.90, spot * 1.10, args=(raw_data,))
        except: z_gamma = spot 

        df = get_greeks_pro(raw_data, spot)
        agg = df.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
        
        # --- PULIZIA NUMERI INFINITESIMALI (Fix per le tue immagini) ---
        # Qualunque valore minore di 0.01 dollari viene forzato a 0
        for m in ["Gamma", "Vanna", "Charm", "Vega", "Theta"]:
            agg.loc[agg[m].abs() < 1e-2, m] = 0

        lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
        visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
        
        c_wall = visible_agg.loc[visible_agg['Gamma'].idxmax(), 'strike'] if not visible_agg.empty else spot
        p_wall = visible_agg.loc[visible_agg['Gamma'].idxmin(), 'strike'] if not visible_agg.empty else spot

        st.subheader(f"🏟️ {asset} | Spot: ${spot:,.2f}")
        
        # --- INDICATORI SUPERIORI ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CALL WALL", f"${c_wall:,.0f}")
        m2.metric("ZERO GAMMA", f"${z_gamma:,.2f}")
        m3.metric("PUT WALL", f"${p_wall:,.0f}")
        m4.metric("SPOT", f"${spot:,.2f}")

        # --- LOGICA BIAS ---
        st.markdown("#### 🧭 MARKET DIRECTION INDICATOR")
        net_gamma = agg['Gamma'].sum()
        net_vanna = agg['Vanna'].sum()
        net_charm = agg['Charm'].sum()
        net_theta = agg['Theta'].sum()
        net_vega = agg['Vega'].sum()

        if net_gamma < 0 and net_vanna < 0:
            direction = "🔴 PERICOLO CRASH: SHORT GAMMA + VANNA NEGATIVA"; bias_color = "#8B0000"
        elif net_gamma < 0:
            direction = "🔴 ACCELERAZIONE VOLATILITÀ (Short Gamma)"; bias_color = "#FF4136"
        elif spot < z_gamma:
            direction = "🟠 PRESSIONE RIBASSISTA (Sotto Zero Gamma)"; bias_color = "#FF851B"
        elif net_gamma > 0 and net_charm < 0:
            direction = "🟢 REVERSIONE POSITIVA (Charm Support)"; bias_color = "#2ECC40"
        elif net_gamma > 0 and abs(net_theta) > abs(net_vega):
            direction = "⚪ CONSOLIDAMENTO / THETA BURN"; bias_color = "#AAAAAA"
        else:
            direction = "🔵 LONG GAMMA / STABILITÀ"; bias_color = "#0074D9"

        st.markdown(f"<div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center;'> <b style='color:black; font-size:20px;'>{direction}</b> </div>", unsafe_allow_html=True)

        # --- GRAFICO CON FORMATTAZIONE DOLLARI ---
        p_df = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)].copy()
        p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
        p_df = p_df.groupby('bin', as_index=False).sum()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=p_df['bin'], 
            x=p_df[metric], 
            orientation='h',
            marker=dict(color=['#00FF41' if x >= 0 else '#0074D9' for x in p_df[metric]]),
            hovertemplate="Prezzo: $%{y:,.2f}<br>Esposizione: $%{x:,.2s}<extra></extra>"
        ))
        
        fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
        fig.add_hline(y=z_gamma, line_color="#FFD700", line_dash="dash", annotation_text="ZERO GAMMA")

        fig.update_layout(
            template="plotly_dark", height=700,
            margin=dict(l=0, r=0, t=30, b=0),
            # ASSE Y: Prezzi puliti con il simbolo $ e virgole
            yaxis=dict(
                title="Prezzo Strike ($)",
                tickformat="$~s", # Formato valuta compatto
                gridcolor="#333",
                range=[lo, hi]
            ),
            # ASSE X: Valori in dollari (k=mila, M=milioni)
            xaxis=dict(
                title=f"Esposizione Netta {metric} ($)",
                tickformat="$.2s", # Forza il formato $100k, $1M ecc.
                zerolinecolor="white"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
