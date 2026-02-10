import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V52 - PIVOT", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh")

# --- ENGINE MATEMATICO PROFESSIONALE ---
def get_greeks_pro(df, S, r=0.045):
    if df.empty: return df
    # Filtraggio rigoroso per scarti di dati Yahoo Finance
    df = df[df['impliedVolatility'] > 0.01].copy()
    
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    side = np.where(df['type'] == 'call', 1, -1)
    
    # Dollar Gamma (Esposizione monetaria istituzionale)
    # Formula: 0.5 * Gamma * S^2 * 0.01
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi * 100 * side
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi * side
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi * 100 * side
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
st.sidebar.markdown("## 🛰️ SENTINEL V52 PIVOT")
asset = st.sidebar.selectbox("TICKER", ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA"])
t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
current_ticker = t_map.get(asset, asset)

ticker_obj = yf.Ticker(current_ticker)
h = ticker_obj.history(period='1d')
if h.empty: st.stop()
spot = h['Close'].iloc[-1]

available_dates = ticker_obj.options
today = datetime.now()
dte_labels = [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates]
selected_dte = st.sidebar.multiselect("SCADENZE", dte_labels, default=dte_labels[:1])

metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm"])
gran = st.sidebar.select_slider("GRANULARITÀ", options=[1, 2, 5, 10, 20, 50, 100], value=10 if "NDX" in asset else 5)
zoom_val = st.sidebar.slider("ZOOM %", 0.5, 10.0, 2.0)

if selected_dte:
    target_dates = [d.split('| ')[1] for d in selected_dte]
    raw_data = fetch_data(current_ticker, target_dates)
    
    if not raw_data.empty:
        raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
        df = get_greeks_pro(raw_data, spot)
        
        # Aggregazione per Strike
        agg = df.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm"]].sum()
        
        # --- CALCOLO ZERO GAMMA PROFESSIONALE (FLIP POINT) ---
        # Filtriamo un range stretto intorno allo spot per trovare il "Flip" reale
        near_spot = agg[(agg['strike'] > spot * 0.95) & (agg['strike'] < spot * 1.05)].sort_values('strike')
        
        try:
            # Troviamo il punto dove il segno cambia (Zero Crossing)
            # Creiamo una funzione di interpolazione basata sul Gamma Netto
            f_flip = interp1d(near_spot['Gamma'], near_spot['strike'], bounds_error=False, fill_value="extrapolate")
            z_gamma = float(f_flip(0))
            
            # Coerenza: Se lo zero gamma calcolato è fuori dal range locale, 
            # prendiamo lo strike con il valore Gamma più vicino allo zero (il "gap")
            if z_gamma < near_spot['strike'].min() or z_gamma > near_spot['strike'].max():
                z_gamma = near_spot.loc[near_spot['Gamma'].abs().idxmin(), 'strike']
        except:
            z_gamma = spot

        c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
        p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']

        # UI DASHBOARD
        st.subheader(f"🏟️ {asset} Scalping Terminal | Spot: {spot:.2f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("CALL WALL", f"{c_wall:.0f}")
        c2.metric("ZERO GAMMA (PIVOT)", f"{z_gamma:.2f}")
        c3.metric("PUT WALL", f"{p_wall:.0f}")

        # --- GRAFICO IDENTICO A GEXBOT ---
        lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
        p_df = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)].copy()
        p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
        p_df = p_df.groupby('bin', as_index=False).sum()

        fig = go.Figure()
        
        # Colori Gexbot: Neon Green (Positive) / Sky Blue (Negative)
        fig.add_trace(go.Bar(
            y=p_df['bin'], x=p_df[metric], orientation='h',
            marker=dict(color=['#00FF41' if x >= 0 else '#0074D9' for x in p_df[metric]], line_width=0),
            width=gran * 0.8
        ))

        # Linee di trading
        fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
        fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="VOL TRIGGER (0-G)")
        
        if lo <= c_wall <= hi: fig.add_hline(y=c_wall, line_color="#FF4136", annotation_text="CALL WALL")
        if lo <= p_wall <= hi: fig.add_hline(y=p_wall, line_color="#2ECC40", annotation_text="PUT WALL")

        fig.update_layout(
            template="plotly_dark", height=800, margin=dict(l=0,r=0,t=30,b=0),
            yaxis=dict(range=[lo, hi], dtick=gran, gridcolor="#222"),
            xaxis=dict(title=f"Net {metric} Exposure", gridcolor="#222")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Output per TradingView
        st.code(f"Pivots: 0G@{z_gamma:.2f} | CW@{c_wall:.0f} | PW@{p_wall:.0f}")
