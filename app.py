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
st.set_page_config(layout="wide", page_title="SENTINEL GEX V50", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh") # Refresh 1 min per scalping

# --- ENGINE MATEMATICO ---
def get_greeks_notional(df, S, r=0.045):
    if df.empty: return df
    K, iv, T = df['strike'].values, np.maximum(df['impliedVolatility'].values, 0.001), np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    side = np.where(df['type'] == 'call', 1, -1)
    
    # Calcolo Notional (Esposizione monetaria reale)
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi * 100 * side
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi * side
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi * 100 * side
    df['Vega']  = S * pdf * np.sqrt(T) * 0.01 * oi * 100
    df['Theta'] = ((-(S * pdf * iv) / (2 * np.sqrt(T))) - side * (r * K * np.exp(-r * T) * norm.cdf(d2 * side))) * (1/365) * oi * 100
    return df

# --- DATA FETCHING ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_deep_chain(ticker, dates):
    t = yf.Ticker(ticker)
    frames = []
    for d in dates:
        try:
            oc = t.option_chain(d)
            frames.append(pd.concat([oc.calls.assign(type='call', exp=d), oc.puts.assign(type='put', exp=d)]))
        except: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --- INTERFACCIA ---
st.sidebar.markdown("## 🛰️ SENTINEL V50")
asset = st.sidebar.selectbox("TICKER", ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA"])
t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
current_ticker = t_map.get(asset, asset)

# Prezzo Spot Realtime
ticker_obj = yf.Ticker(current_ticker)
spot = ticker_obj.history(period='1d')['Close'].iloc[-1]

# Gestione Scadenze
available_dates = ticker_obj.options
today = datetime.now()
dte_labels = [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates]
selected_dte = st.sidebar.multiselect("SCADENZE ATTIVE", dte_labels, default=dte_labels[:1])

# Parametri Grafici
metric = st.sidebar.radio("ANALISI FLUSSI", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
gran = st.sidebar.select_slider("GRANULARITÀ STRIKE", options=[1, 2, 5, 10, 20, 50, 100], value=10 if "NDX" in asset else 5)
zoom_val = st.sidebar.slider("ZOOM AREA %", 0.5, 15.0, 3.0)

if selected_dte:
    target_dates = [d.split('| ')[1] for d in selected_dte]
    data = fetch_deep_chain(current_ticker, target_dates)
    
    if not data.empty:
        data['dte_years'] = data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
        processed = get_greeks_notional(data, spot)
        
        # Consolidamento per Strike
        final_df = processed.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
        
        # Calcolo Livelli Chiave
        c_wall = final_df.loc[final_df['Gamma'].idxmax(), 'strike']
        p_wall = final_df.loc[final_df['Gamma'].idxmin(), 'strike']
        
        # Zero Gamma Preciso (Cross-over)
        sort_df = final_df.sort_values('strike')
        f_z = interp1d(sort_df['Gamma'].cumsum(), sort_df['strike'], fill_value="extrapolate")
        z_gamma = float(f_z(0))

        # DASHBOARD
        st.subheader(f"🏟️ {asset} Terminal | Spot: {spot:.2f}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CALL WALL (Res)", f"{c_wall:.0f}")
        k2.metric("PUT WALL (Sup)", f"{p_wall:.0f}")
        k3.metric("ZERO GAMMA (Pivot)", f"{z_gamma:.2f}")
        k4.metric(f"NET {metric.upper()}", f"${final_df[metric].sum()/1e6:.1f}M")

        # --- GRAFICO GEXBOT STYLE ---
        limit_lo, limit_hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
        p_df = final_df[(final_df['strike'] >= limit_lo) & (final_df['strike'] <= limit_hi)].copy()
        p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
        p_df = p_df.groupby('bin', as_index=False).sum()

        fig = go.Figure()
        
        # Barre con colori neon
        fig.add_trace(go.Bar(
            y=p_df['bin'], x=p_df[metric], orientation='h',
            marker=dict(color=['#00FF00' if x >= 0 else '#00BFFF' for x in p_df[metric]],
                        line=dict(width=0)),
            width=gran * 0.9, name=metric
        ))

        # Linee di trading (identiche a GexBot)
        fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
        fig.add_hline(y=z_gamma, line_color="#FFFF00", line_dash="dash", annotation_text="ZERO G")
        
        if limit_lo <= c_wall <= limit_hi:
            fig.add_hline(y=c_wall, line_color="#FF0000", line_width=2, annotation_text="CALL WALL")
        if limit_lo <= p_wall <= limit_hi:
            fig.add_hline(y=p_wall, line_color="#00FF00", line_width=2, annotation_text="PUT WALL")

        fig.update_layout(
            template="plotly_dark", height=900,
            yaxis=dict(title="STRIKE", range=[limit_lo, limit_hi], dtick=gran, gridcolor="#222"),
            xaxis=dict(title=f"Net {metric} Exposure ($)", gridcolor="#222", zerolinecolor="#666"),
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # LISTA MURO PER TRADINGVIEW
        with st.expander("📝 Livelli per TradingView (Copia/Incolla)"):
            st.code(f"Spot: {spot:.2f}\nZero Gamma: {z_gamma:.2f}\nCall Wall: {c_wall:.2f}\nPut Wall: {p_wall:.2f}")

else:
    st.info("Seleziona almeno una scadenza nella barra laterale per iniziare l'analisi.")
