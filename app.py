import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import time

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V63 - FULL PRO", initial_sidebar_state="expanded")

# AGGIORNAMENTO AUTOMATICO: Impostato a 120.000 ms (2 minuti)
st_autorefresh(interval=120000, key="sentinel_refresh")

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

def calculate_0g_dynamic(price, df, r=0.045):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    exposure_size = df['volume'].fillna(0).values 
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

# --- FUNZIONE SCANNER (Cache sincronizzata a 2 min) ---
@st.cache_data(ttl=115, show_spinner=False) 
def fetch_scanner_ticker(t_name, expiry_mode_str, today_str):
    try:
        t_obj = yf.Ticker(t_name)
        hist = t_obj.history(period='2d') # Ridotto a 2gg per velocità
        if hist.empty: return None
        px = hist['Close'].iloc[-1]
        opts = t_obj.options
        if not opts: return None
        target_opt = opts[0] if "0-1 DTE" in expiry_mode_str else (opts[2] if len(opts) > 2 else opts[0])
        oc = t_obj.option_chain(target_opt)
        df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
        
        today_obj = datetime.strptime(today_str, '%Y-%m-%d')
        dte_years = max((datetime.strptime(target_opt, '%Y-%m-%d') - today_obj).days + 1, 0.5) / 365
        df_scan['dte_years'] = dte_years
        df_scan = df_scan[(df_scan['strike'] > px*0.8) & (df_scan['strike'] < px*1.2)]
        
        return px, df_scan, dte_years
    except:
        return None

# --- NAVIGAZIONE ---
st.sidebar.markdown("## 🧭 SISTEMA")
st.sidebar.info("Aggiornamento Automatico: Ogni 2 minuti")
menu = st.sidebar.radio("Seleziona Vista:", ["🏟️ DASHBOARD SINGOLA", "🔥 SCANNER HOT TICKERS"])
today = datetime.now()
today_str_format = today.strftime('%Y-%m-%d')

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

    available_dates = ticker_obj.options
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
            
            sd_move = spot * mean_iv * np.sqrt(max(dte_ref, 0.5)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move

            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 
            try: z_gamma_dyn = brentq(calculate_0g_dynamic, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma_dyn = spot

            df = get_greeks_pro(raw_data, spot)
            df['strike_bin'] = (np.round(df['strike'] / gran) * gran)
            agg = df.groupby('strike_bin', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum().rename(columns={'strike_bin': 'strike'})
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
            p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']
            v_trigger = agg.loc[agg['Vanna'].abs().idxmax(), 'strike']

            st.subheader(f"🏟️ {asset} | Spot: {spot:.2f}")

            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            direction = "NEUTRALE"; bias_color = "gray"
            if net_gamma < 0 and net_vanna < 0: direction = "☢️ PERICOLO ESTREMO"; bias_color = "#8B0000"
            elif net_gamma < 0: direction = "🔴 SHORT GAMMA BIAS"; bias_color = "#FF4136"
            elif spot < z_gamma: direction = "🟠 PRESSIONE SOTTO ZERO GAMMA"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0: direction = "🚀 BULLISH FLOW"; bias_color = "#2ECC40"
            else: direction = "🔵 LONG GAMMA / STABILITÀ"; bias_color = "#0074D9"
            
            st.markdown(f"<div style='background-color:{bias_color}; padding:10px; border-radius:10px; text-align:center; color:white; font-size:20px;'><b>BIAS: {direction}</b></div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.0f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg[metric], orientation='h', marker=dict(color=['#00FF41' if x >= 0 else '#FF4136' for x in visible_agg[metric]])))
            fig.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_dash="dash", annotation_text="0-G")
            fig.update_layout(template="plotly_dark", height=800, yaxis=dict(range=[lo, hi], dtick=gran))
            st.plotly_chart(fig, use_container_width=True)

elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Market Scanner (50 Tickers - Auto Refresh 2m)")
    
    expiry_mode = st.selectbox("📅 SCADENZE:", ["0-1 DTE (Scalping/Intraday)", "Prossima Scadenza Mensile"])
    
    # TUTTI I 50 TICKER RICHIESTI
    tickers_50 = [
        "^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", 
        "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", 
        "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", 
        "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", 
        "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA", "IBM"
    ]
    
    scan_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t_name in enumerate(tickers_50):
        status_text.text(f"Scansione: {t_name} ({i+1}/{len(tickers_50)})")
        
        data_pack = fetch_scanner_ticker(t_name, expiry_mode, today_str_format)
        time.sleep(0.1) # Pausa minima per evitare blocchi Yahoo
        
        if data_pack:
            px, df_scan, dte_years = data_pack
            try:
                try: zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
                except: zg_val = px
                try: zg_dyn = brentq(calculate_0g_dynamic, px*0.75, px*1.25, args=(df_scan,))
                except: zg_dyn = px

                df_scan_greeks = get_greeks_pro(df_scan, px)
                net_vanna_scan = df_scan_greeks['Vanna'].sum() if not df_scan_greeks.empty else 0
                net_charm_scan = df_scan_greeks['Charm'].sum() if not df_scan_greeks.empty else 0
                
                # Scoring
                p_score = 4 if (px > zg_val and px > zg_dyn) else (-4 if (px < zg_val and px < zg_dyn) else 0)
                v_score = 3 if net_vanna_scan > 0 else -3
                c_score = 3 if net_charm_scan < 0 else -3
                ss = p_score + v_score + c_score

                v_icon = "🟢" if net_vanna_scan > 0 else "🔴"
                c_icon = "🔵" if net_charm_scan < 0 else "🔴"
                
                if ss >= 8: verdict = "🚀 FULL LONG"
                elif ss <= -8: verdict = "☢️ CRASH RISK"
                elif px > zg_val and px < zg_dyn: verdict = "⚠️ DISTRIBUZIONE"
                elif px < zg_val and px > zg_dyn: verdict = "🔥 SHORT SQUEEZE"
                else: verdict = "⚖️ NEUTRO"

                scan_results.append({
                    "Ticker": t_name.replace("^", ""), 
                    "Score": int(ss), 
                    "Verdict": verdict,
                    "Vanna/Charm": f"{v_icon} | {c_icon}",
                    "Prezzo": round(px, 2), 
                    "0-G Static": round(zg_val, 2),
                    "Dist. 0G %": round(((px - zg_val) / px) * 100, 2),
                    "Analisi": "✅ SOPRA 0G" if px > zg_val else "🔻 SOTTO 0G"
                })
            except: pass
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values(by="Score", ascending=False)
        
        def color_logic(row):
            styles = [''] * len(row)
            score_val = row['Score']
            if score_val >= 8: styles[1] = 'background-color: #2ECC40; color: white'
            elif score_val <= -8: styles[1] = 'background-color: #8B0000; color: white'
            return styles

        st.dataframe(final_df.style.apply(color_logic, axis=1), use_container_width=True, height=800)
        st.success(f"Scansione completata alle {datetime.now().strftime('%H:%M:%S')}. Prossimo aggiornamento tra 2 minuti.")
