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
st.set_page_config(layout="wide", page_title="SENTINEL GEX V63 - FULL PRO", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="sentinel_refresh")

# --- CORE QUANT ENGINE ---
def calculate_gex_at_price(price, df, mode='hybrid', r=0.045):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    
    if mode == 'oi':
        exposure_size = df['openInterest'].fillna(0).values
    elif mode == 'vol':
        exposure_size = df['volume'].fillna(0).values
    else: # hybrid
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
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "MSTR"]
    
    new_asset = st.sidebar.text_input("➕ CARICA TICKER", "").upper().strip()
    if new_asset and new_asset not in st.session_state.ticker_list:
        st.session_state.ticker_list.insert(0, new_asset)
        st.rerun()

    asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
    t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
    current_ticker = t_map.get(asset, asset)

    default_gran = 1.0
    if "NDX" in asset: default_gran = 25.0
    elif "SPX" in asset: default_gran = 10.0
    elif any(x in asset for x in ["NVDA", "MSTR", "SMCI"]): default_gran = 5.0
    
    ticker_obj = yf.Ticker(current_ticker)
    h = ticker_obj.history(period='1d')
    if h.empty: st.stop()
    spot = h['Close'].iloc[-1]

    try:
        available_dates = ticker_obj.options
    except:
        st.error("⚠️ Yahoo Finance Rate Limit. Attendi un minuto.")
        st.stop()

    all_dates_info = []
    for d in available_dates:
        try:
            dt_obj = datetime.strptime(d, '%Y-%m-%d')
            dte = (dt_obj - today).days + 1
            if dte >= 0:
                all_dates_info.append({"label": f"{dte} DTE | {d}", "date": d, "dte": dte})
        except: continue
    
    date_labels = [x['label'] for x in all_dates_info]
    selected_dte_labels = st.sidebar.multiselect("SCADENZE", date_labels, default=date_labels[:1])
    metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    gran = st.sidebar.select_slider("GRANULARITÀ", options=[0.5, 1, 2.5, 5, 10, 25, 50, 100], value=default_gran)
    zoom_val = st.sidebar.slider("ZOOM %", 1.0, 20.0, 5.0)

    if selected_dte_labels:
        target_dates = [label.split('| ')[1] for label in selected_dte_labels]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: max((datetime.strptime(x, '%Y-%m-%d') - today).days, 0.5)) / 365
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_ref = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            
            if 'prev_iv' not in st.session_state: st.session_state.prev_iv = mean_iv
            iv_change = mean_iv - st.session_state.prev_iv
            st.session_state.prev_iv = mean_iv

            sd_move = spot * mean_iv * np.sqrt(max(dte_ref, 0.5)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move

            try: z_gamma_oi = brentq(calculate_gex_at_price, spot * 0.80, spot * 1.20, args=(raw_data, 'oi'))
            except: z_gamma_oi = spot
            try: z_gamma_vol = brentq(calculate_gex_at_price, spot * 0.80, spot * 1.20, args=(raw_data, 'vol'))
            except: z_gamma_vol = spot
            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data, 'hybrid'))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            df['strike_bin'] = (np.round(df['strike'] / gran) * gran)
            agg = df.groupby('strike_bin', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum().rename(columns={'strike_bin': 'strike'})
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
            p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']
            v_trigger = agg.loc[agg['Vanna'].abs().idxmax(), 'strike']

            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")

            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            direction = "NEUTRALE"; bias_color = "gray"
            
            if net_gamma < 0 and net_vanna < 0:
                direction = "☢️ PERICOLO ESTREMO"; bias_color = "#8B0000"
            elif net_gamma < 0:
                direction = "🔴 SHORT GAMMA BIAS"; bias_color = "#FF4136"
            elif spot < z_gamma:
                direction = "🟠 PRESSIONE SOTTO ZERO GAMMA"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0:
                direction = "🚀 BULLISH FLOW"; bias_color = "#2ECC40"
            else:
                direction = "🔵 LONG GAMMA STABILITÀ"; bias_color = "#0074D9"
            
            st.markdown(f"### 📊 Real-Time Metric Regime")
            c_reg1, c_reg2, c_reg3, c_reg4 = st.columns(4)
            c_reg1.metric("Net Gamma", f"{net_gamma:,.0f}", delta="LONG" if net_gamma > 0 else "SHORT")
            c_reg2.metric("Net Vanna", f"{net_vanna:,.0f}", delta="STABLE" if net_vanna > 0 else "UNSTABLE")
            c_reg3.metric("Net Charm", f"{net_charm:,.0f}", delta="SUPPORT" if net_charm < 0 else "DECAY")
            c_reg4.metric("Market Regime", "VOL DRIVEN" if net_gamma < 0 else "SPOT DRIVEN")

            st.markdown(f"<div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center; margin-bottom: 25px;'><b style='color:white; font-size:24px;'>BIAS: {direction}</b></div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("0-G (OI / VOL)", f"{z_gamma_oi:.0f} | {z_gamma_vol:.0f}", delta=f"{z_gamma_vol - z_gamma_oi:+.2f} DIFF", delta_color="inverse")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            def get_dist(target, spot):
                d = ((target - spot) / spot) * 100
                color = "#00FF41" if d > 0 else "#FF4136"
                return f"<span style='color:{color};'>{d:+.2f}%</span>"

            st.markdown(f"""<div style='background-color:rgba(30, 30, 30, 0.8); padding:10px; border-radius:5px; border: 1px solid #444; margin-bottom: 20px; display: flex; justify-content: space-around;'>
                <div><b>📍 DIST. CW:</b> {get_dist(c_wall, spot)}</div>
                <div><b>📍 0G-STATIC:</b> {get_dist(z_gamma_oi, spot)}</div>
                <div><b>📍 0G-DYNAMIC:</b> {get_dist(z_gamma_vol, spot)}</div>
                <div><b>📍 DIST. VT:</b> {get_dist(v_trigger, spot)}</div>
                <div><b>📍 DIST. PW:</b> {get_dist(p_wall, spot)}</div>
            </div>""", unsafe_allow_html=True)

            col_view, col_vol = st.columns([2, 1])
            with col_view: view_mode = st.radio("👁️ VISTA GRAFICO:", ["📊 Vista Standard", "🌪️ Vanna View"], horizontal=True)
            with col_vol: st.metric("📈 VOLATILITÀ IV", f"{mean_iv*100:.2f}%", delta=f"{iv_change*100:.2f}%", delta_color="inverse")

            fig = go.Figure()
            if "Standard" in view_mode:
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg[metric], orientation='h', marker=dict(color=['#00FF41' if x >= 0 else '#FF4136' for x in visible_agg[metric]]), name=metric))
            else:
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg['Gamma'], orientation='h', marker=dict(color='rgba(100, 100, 100, 0.3)'), name="Gamma", xaxis="x1"))
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg['Vanna'], orientation='h', marker=dict(color=['#00FFFF' if x >= 0 else '#FF00FF' for x in visible_agg['Vanna']]), width=gran * 0.4, name="Vanna", xaxis="x2"))
                fig.update_layout(xaxis=dict(title="Gamma"), xaxis2=dict(title="Vanna", overlaying="x", side="top"), barmode='overlay')

            fig.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
            fig.add_hline(y=z_gamma_oi, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G STATIC")
            fig.add_hline(y=z_gamma_vol, line_color="#00BFFF", line_width=2, line_dash="dot", annotation_text="0-G DYNAMIC")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CW")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PW")
            fig.update_layout(template="plotly_dark", height=850, yaxis=dict(range=[lo, hi], dtick=gran))
            st.plotly_chart(fig, use_container_width=True)

elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Professional Market Scanner")
    expiry_mode = st.selectbox("📅 SCADENZE:", ["0-1 DTE (Scalping/Intraday)", "Prossima Scadenza Mensile (Swing)"])
    tickers_50 = ["^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA"]
    
    scan_results = []
    progress_bar = st.progress(0)
    for i, t_name in enumerate(tickers_50):
        try:
            t_obj = yf.Ticker(t_name)
            hist = t_obj.history(period='1d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            opts = t_obj.options
            target_opt = opts[0] if "0-1 DTE" in expiry_mode else (opts[2] if len(opts) > 2 else opts[0])
            oc = t_obj.option_chain(target_opt)
            df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            df_scan['dte_years'] = max((datetime.strptime(target_opt, '%Y-%m-%d') - today).days + 1, 0.5) / 365
            
            try: zg_oi = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan, 'oi'))
            except: zg_oi = px
            try: zg_vol = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan, 'vol'))
            except: zg_vol = px
            
            divergenza = ((zg_vol - zg_oi) / zg_oi) * 100
            pressione = "🟢 BULLISH" if zg_vol < zg_oi else "🔴 BEARISH"
            if abs(divergenza) < 0.1: pressione = "⚪ NEUTRALE"

            avg_iv = df_scan['impliedVolatility'].mean()
            sd_move = px * avg_iv * np.sqrt(df_scan['dte_years'].iloc[0])
            sd1_up, sd1_down = px + sd_move, px - sd_move # <--- RIGA CORRETTA QUI
            
            dist_zg_pct = ((px - zg_vol) / px) * 100
            is_above_0g = px > zg_vol
            
            if not is_above_0g: status_label = "🔻 SOTTO 0G"
            else: status_label = "✅ SOPRA 0G"
            if abs(dist_zg_pct) < 0.3: status_label = "🔥 FLIP"
            
            scan_results.append({
                "Ticker": t_name.replace("^", ""), "Prezzo": round(px, 2), 
                "0-G Static": round(zg_oi, 2), "0-G Dynamic": round(zg_vol, 2),
                "Div %": f"{divergenza:+.2f}%", "Pressione": pressione, "Analisi": status_label, "_sort": abs(dist_zg_pct)
            })
        except: continue
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values("_sort").drop(columns=["_sort"])
        st.dataframe(final_df.style.applymap(lambda x: 'color: #FF4136' if '🔴' in str(x) or '🔻' in str(x) else 'color: #2ECC40' if '🟢' in str(x) or '✅' in str(x) else ''), use_container_width=True, height=800)
