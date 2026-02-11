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
st.set_page_config(layout="wide", page_title="SENTINEL GEX V58 - PRO", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh")

# --- CORE QUANT ENGINE (Logica Comune) ---
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

# --- SIDEBAR NAVIGAZIONE ---
st.sidebar.title("🛰️ SENTINEL V58")
menu = st.sidebar.radio("SISTEMA", ["📊 TERMINALE GEX", "🔥 SCANNER HOT TICKERS"])

# Lista estesa dei 50 Ticker (puoi aggiungerne altri qui)
TICKER_50 = [
    "SPX", "NDX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", 
    "META", "GOOGL", "AMD", "MSTR", "COIN", "MARA", "RIOT", "BITO", "SMCI", "AVGO",
    "LLY", "JPM", "GS", "NFLX", "DIS", "BABA", "TSM", "PLTR", "SNOW", "U", "ARM"
]

# --- PAGINA 1: TERMINALE GEX (Il tuo codice originale intatto) ---
if menu == "📊 TERMINALE GEX":
    asset = st.sidebar.selectbox("SELEZIONA ASSET", TICKER_50)
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
    gran = st.sidebar.select_slider("GRANULARITÀ", options=[1, 2, 5, 10, 20, 25, 50, 100], value=5)
    zoom_val = st.sidebar.slider("ZOOM %", 0.5, 15.0, 3.0)

    if selected_dte:
        target_dates = [d.split('| ')[1] for d in selected_dte]
        raw_data = fetch_data(current_ticker, target_dates)
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            # Deviazioni Standard
            mean_iv = raw_data['impliedVolatility'].mean()
            sd_move = spot * mean_iv * np.sqrt(0.5/365)
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

            st.subheader(f"🏟️ {asset} | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.2f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            # Market Direction Indicator
            net_gamma, net_vanna = agg['Gamma'].sum(), agg['Vanna'].sum()
            direction = "NEUTRALE"; bias_col = "gray"
            if net_gamma < 0: direction = "🔴 SHORT GAMMA (Accelerazione)"; bias_col = "#FF4136"
            elif spot < z_gamma: direction = "🟠 SOTTO ZERO GAMMA"; bias_col = "#FF851B"
            else: direction = "🔵 LONG GAMMA (Stabilità)"; bias_col = "#0074D9"
            st.markdown(f"<div style='background-color:{bias_col}; padding:10px; border-radius:5px; text-align:center;'><b>{direction}</b></div>", unsafe_allow_html=True)

            # Grafico
            p_df = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)].copy()
            p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
            p_df = p_df.groupby('bin', as_index=False).sum()
            p_df[metric] = p_df[metric].apply(lambda x: x if abs(x) > 1e-8 else 0)

            fig = go.Figure()
            fig.add_trace(go.Bar(y=p_df['bin'], x=p_df[metric], orientation='h', marker_color='#00FF41'))
            fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, annotation_text="0-G")
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="dash", annotation_text="1SD")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="dash")
            fig.update_layout(template="plotly_dark", height=700, xaxis=dict(tickformat="$.3s"))
            st.plotly_chart(fig, use_container_width=True)

# --- PAGINA 2: SCANNER HOT TICKERS (Tabella dei 50) ---
elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Market Scanner - Analisi Quantitativa 50 Asset")
    st.markdown("Monitoraggio istantaneo della distanza dallo Zero Gamma e Greche Nette.")
    
    scan_results = []
    progress_bar = st.progress(0)
    
    for i, t_code in enumerate(TICKER_50):
        try:
            t_obj = yf.Ticker(t_code)
            hist = t_obj.history(period='1d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            
            # Prendi dati opzioni (prima scadenza)
            opt_date = t_obj.options[0]
            oc = t_obj.option_chain(opt_date)
            df_opt = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            df_opt['dte_years'] = 0.5 / 365
            
            # Calcolo 0G e Greche
            g, v, th = 0, 0, 0
            try:
                g_df = get_greeks_pro(df_opt, px)
                g, v, th = g_df['Gamma'].sum(), g_df['Vanna'].sum(), g_df['Theta'].sum()
                zg = brentq(calculate_gex_at_price, px*0.8, px*1.2, args=(df_opt,))
            except: zg = px
            
            dist_zg = ((px - zg) / px) * 100
            
            scan_results.append({
                "Ticker": t_code,
                "Prezzo": round(px, 2),
                "Gamma ($M)": round(g/1e6, 2),
                "Vanna ($M)": round(v/1e6, 2),
                "Theta ($M)": round(th/1e6, 2),
                "Dist. 0G %": round(dist_zg, 2),
                "Alert": "🔥 HOT" if abs(dist_zg) < 0.6 else "OK"
            })
        except: continue
        progress_bar.progress((i + 1) / len(TICKER_50))

    final_df = pd.DataFrame(scan_results).sort_values(by="Dist. 0G %")
    
    # Styling Tabella
    def color_status(val):
        color = '#FF4136' if abs(val) < 0.6 else 'white'
        return f'color: {color}; font-weight: bold' if abs(val) < 0.6 else ''

    st.dataframe(final_df.style.applymap(color_status, subset=['Dist. 0G %']), use_container_width=True, height=800)
    st.success("Scansione completata. I ticker in rosso sono vicini al punto di rottura (Zero Gamma).")
