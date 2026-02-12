import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V58 - LIGHT", initial_sidebar_state="expanded")
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
today = datetime.now()

if menu == "🏟️ DASHBOARD SINGOLA":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛰️ SENTINEL HUB")
    
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "MSTR"]

    new_asset = st.sidebar.text_input("➕ CARICA TICKER", "").upper().strip()
    if new_asset and new_asset not in st.session_state.ticker_list:
        st.session_state.ticker_list.insert(0, new_asset)
        st.rerun()

    asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
    t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
    current_ticker = t_map.get(asset, asset)

    # --- LOGICA INTELLIGENTE PER EVITARE PESANTEZZA ---
    # Se è un indice pesante (NDX > 15000, SPX > 5000), imposta granularità alta di default
    default_gran = 1.0
    if "NDX" in asset or "SPX" in asset:
        default_gran = 25.0 # NDX ha bisogno di step 25 per essere fluido
    elif "NVDA" in asset or "SMCI" in asset or "MSTR" in asset:
        default_gran = 5.0
    
    ticker_obj = yf.Ticker(current_ticker)
    h = ticker_obj.history(period='1d')
    if h.empty: st.stop()
    spot = h['Close'].iloc[-1]

    # Date
    available_dates = ticker_obj.options
    all_dates_info = []
    for d in available_dates:
        try:
            dt_obj = datetime.strptime(d, '%Y-%m-%d')
            dte = (dt_obj - today).days + 1
            if 0 <= dte <= 60: # Limitiamo a 60 giorni per velocità
                all_dates_info.append({"label": f"{dte} DTE | {d}", "date": d, "dte": dte})
        except: continue
    
    all_dates_info = sorted(all_dates_info, key=lambda x: x['dte'])
    date_labels = [x['label'] for x in all_dates_info]
    
    # Seleziona meno date di default per alleggerire il carico iniziale
    default_sel = date_labels[:2] if date_labels else []
    selected_dte_labels = st.sidebar.multiselect("SCADENZE", date_labels, default=default_sel)

    metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    
    # Slider Granularità (con default intelligente calcolato sopra)
    gran = st.sidebar.select_slider("GRANULARITÀ STRIKE (Alto = +Veloce)", options=[0.5, 1, 2.5, 5, 10, 25, 50, 100], value=default_gran)
    zoom_val = st.sidebar.slider("ZOOM %", 1.0, 15.0, 4.0)

    if selected_dte_labels:
        target_dates = [label.split('| ')[1] for label in selected_dte_labels]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            # Calcoli base
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_ref = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            sd_move = spot * mean_iv * np.sqrt(max(dte_ref, 1)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move

            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            
            # --- AGGREGAZIONE (Cruciale per performance) ---
            df['strike_bin'] = (np.round(df['strike'] / gran) * gran)
            agg = df.groupby('strike_bin', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
            agg = agg.rename(columns={'strike_bin': 'strike'})
            
            # Filtro Zoom
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            c_wall = visible_agg.loc[visible_agg['Gamma'].idxmax(), 'strike'] if not visible_agg.empty else spot
            p_wall = visible_agg.loc[visible_agg['Gamma'].idxmin(), 'strike'] if not visible_agg.empty else spot

            # --- HEADER ---
            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.2f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")
            
            st.markdown("---")

            # --- MARKET DIRECTION INDICATOR (RICHIESTO) ---
            net_gamma = agg['Gamma'].sum()
            net_vanna = agg['Vanna'].sum()
            net_charm = agg['Charm'].sum()
            net_vega = agg['Vega'].sum()
            net_theta = agg['Theta'].sum()

            direction = "NEUTRALE / ATTESA"; bias_color = "gray"
            if net_gamma < 0 and net_vanna < 0:
                direction = "🔴 PERICOLO ESTREMO: SHORT GAMMA + NEGATIVE VANNA (Crash Risk)"; bias_color = "#8B0000"
            elif net_gamma < 0:
                direction = "🔴 ACCELERAZIONE VOLATILITÀ (Short Gamma Bias)"; bias_color = "#FF4136"
            elif spot < z_gamma:
                direction = "🟠 PRESSIONE DI VENDITA (Sotto Zero Gamma)"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0:
                direction = "🟢 REVERSIONE VERSO LO SPOT (Charm Support)"; bias_color = "#2ECC40"
            elif net_gamma > 0 and abs(net_theta) > abs(net_vega):
                direction = "⚪ CONSOLIDAMENTO / THETA DECAY (Range Bound)"; bias_color = "#AAAAAA"
            else:
                direction = "🔵 LONG GAMMA / STABILITÀ (Bassa Volatilità)"; bias_color = "#0074D9"

            # Display Indicatore
            st.markdown(f"<div style='background-color:{bias_color}; padding:10px; border-radius:5px; text-align:center; margin-bottom: 20px;'> <b style='color:black; font-size:20px;'>{direction}</b> </div>", unsafe_allow_html=True)
            
            # --- SINGOLO GRAFICO OTTIMIZZATO (RICHIESTO) ---
            fig = go.Figure()
            
            # Barre
            fig.add_trace(go.Bar(
                y=visible_agg['strike'], 
                x=visible_agg[metric], 
                orientation='h',
                marker=dict(color=['#00FF41' if x >= 0 else '#0074D9' for x in visible_agg[metric]], line_width=0),
                width=gran * 0.85 # Larghezza dinamica in base alla granularità
            ))
            
            # Linee
            fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G FLIP")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CW")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PW")
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="longdash", annotation_text="+1SD")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="longdash", annotation_text="-1SD")

            fig.update_layout(
                template="plotly_dark", height=700, 
                margin=dict(l=0,r=0,t=0,b=0),
                yaxis=dict(range=[lo, hi], dtick=gran, gridcolor="#333", title="Strike Price"),
                xaxis=dict(title=f"Net {metric}", tickformat="$.2s")
            )
            
            st.plotly_chart(fig, use_container_width=True)

# --- SCANNER (INVARIATO) ---
elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Professional Market Scanner")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 AGGIORNA", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        expiry_mode = st.selectbox("📅 SCADENZE:", ["0-1 DTE (Scalp)", "Mensile (Swing)"])
    
    tickers_50 = ["^NDX", "^SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "MSTR", "COIN", "PLTR"]
    
    scan_results = []
    progress = st.progress(0)
    
    for i, t_name in enumerate(tickers_50):
        try:
            t_obj = yf.Ticker(t_name)
            hist = t_obj.history(period='5d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            
            opts = t_obj.options
            if not opts: continue
            target_opt = opts[0] if "0-1" in expiry_mode else (opts[2] if len(opts)>2 else opts[0])
            
            oc = t_obj.option_chain(target_opt)
            df = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            
            try: zg = brentq(calculate_gex_at_price, px*0.8, px*1.2, args=(df,))
            except: zg = px
            
            dist = ((px - zg)/px)*100
            status = "✅ SOPRA 0G" if px > zg else "🔻 SOTTO 0G"
            if abs(dist) < 0.3: status = "🔥 FLIP 0G"
            
            scan_results.append({"Ticker": t_name, "Price": px, "0-Gamma": zg, "Dist%": dist, "Status": status})
        except: continue
        progress.progress((i+1)/len(tickers_50))
        
    if scan_results:
        st.dataframe(pd.DataFrame(scan_results).sort_values("Dist%", key=abs), use_container_width=True, height=600)
