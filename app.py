import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time  # <-- Manteniamo l'import per il delay anti-ban
import requests

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V63 - FULL PRO", initial_sidebar_state="expanded")

# --- CORE QUANT ENGINE ---
def calculate_gex_at_price(price, df, r=0.045):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    # Logica originale: OI + 50% Volume
    exposure_size = df['openInterest'].fillna(0).values + (df['volume'].fillna(0).values * 0.5)
    d1 = (np.log(price/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (price * iv * np.sqrt(T))
    side = np.where(df['type'] == 'call', 1, -1)
    return np.sum(gamma * exposure_size * 100 * price * side)

def calculate_0g_dynamic(price, df, r=0.045):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    # NUOVA LOGICA: Solo Volumi di giornata
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

# --- FUNZIONE PROTETTIVA PER LO SCANNER (Evita Ban IP da Yahoo) ---
@st.cache_data(ttl=300, show_spinner=False) # Aggiorna max ogni 5 minuti per ticker
def fetch_scanner_ticker(t_name, expiry_mode_str, today_str):
    try:
        t_obj = yf.Ticker(t_name)
        hist = t_obj.history(period='5d')
        if hist.empty: return None
        px = hist['Close'].iloc[-1]
        opts = t_obj.options
        if not opts: return None
        target_opt = opts[0] if "0-1 DTE" in expiry_mode_str else (opts[2] if len(opts) > 2 else opts[0])
        oc = t_obj.option_chain(target_opt)
        df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
        
        # Conversione stringa a datetime
        today_obj = datetime.strptime(today_str, '%Y-%m-%d')
        dte_years = max((datetime.strptime(target_opt, '%Y-%m-%d') - today_obj).days + 1, 0.5) / 365
        df_scan['dte_years'] = dte_years
        df_scan = df_scan[(df_scan['strike'] > px*0.7) & (df_scan['strike'] < px*1.3)]
        
        return px, df_scan, dte_years
    except:
        return None

def fetch_yahoo_history(symbol, timeframe, start_str, end_str):
    # Mapping Timeframe Yahoo
    tf_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1H": "1h", "1D": "1d"}
    tf = tf_map.get(timeframe, "1d")
    
    # Controllo Limiti Yahoo Intraday
    start_dt = datetime.strptime(start_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_str, '%Y-%m-%d')
    delta_days = (end_dt - start_dt).days
    
    if delta_days > 7 and tf == "1m":
        st.warning("⚠️ Yahoo limita i dati 1Min a 7 giorni. Passaggio automatico a 1H.")
        tf = "1h"
    elif delta_days > 60 and tf in ["5m", "15m"]:
        st.warning("⚠️ Yahoo limita i dati 5Min/15Min a 60 giorni. Passaggio automatico a 1D.")
        tf = "1d"
    elif delta_days > 730 and tf == "1h":
        st.warning("⚠️ Yahoo limita i dati 1H a 730 giorni. Passaggio automatico a 1D.")
        tf = "1d"

    try:
        df = yf.download(symbol, start=start_str, end=end_str, interval=tf, progress=False)
        
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        col_map = {'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
        df.rename(columns=col_map, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Errore Yahoo Finance: {e}")
        return pd.DataFrame()

def fetch_alpaca_history(symbol, timeframe, start_str, end_str):
    if symbol.startswith("^"):
        return fetch_yahoo_history(symbol, timeframe, start_str, end_str)

    symbol_map = {
        "SPX": "SPY", "NDX": "QQQ", "RUT": "IWM", "DJI": "DIA", "VIX": "VIXY"
    } 
    alpaca_sym = symbol_map.get(symbol.upper(), symbol.upper().replace("^", ""))
    
    headers = {
        "APCA-API-KEY-ID": "PKQVMHYR25JUXQVLTEEBEKVIMV",
        "APCA-API-SECRET-KEY": "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7"
    }
    
    tf_map = {"1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1H": "1Hour", "1D": "1Day"}
    tf = tf_map.get(timeframe, "1Day")
    
    now_utc = datetime.utcnow()
    safe_end_dt = now_utc - timedelta(minutes=20)
    
    try:
        req_end_obj = datetime.strptime(end_str, '%Y-%m-%d')
        if req_end_obj.date() >= now_utc.date():
            final_end = safe_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            final_end = end_str + "T23:59:59Z"
    except:
        final_end = end_str + "T23:59:59Z"

    url = f"https://data.alpaca.markets/v2/stocks/{alpaca_sym}/bars"
    
    all_bars = []
    next_token = None
    
    # Progress Bar per download lunghi
    p_bar = st.progress(0, text="Scaricamento dati storici (Pagination)...")
    
    while True:
        params = {
            "start": start_str + "T00:00:00Z",
            "end": final_end,
            "timeframe": tf,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "iex",
            "page_token": next_token
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and data["bars"]:
                    all_bars.extend(data["bars"])
                    next_token = data.get("next_page_token")
                    
                    # Update progress (fake visual feedback)
                    current_len = len(all_bars)
                    p_bar.progress(min(current_len % 100, 90), text=f"Scaricati {current_len} records...")
                    
                    if not next_token:
                        break
                else:
                    break
            else:
                st.error(f"Errore Alpaca API ({response.status_code}): {response.text}")
                break
        except Exception as e:
            st.error(f"Errore Connessione Alpaca: {e}")
            break
            
    p_bar.empty()
    
    if all_bars:
        df = pd.DataFrame(all_bars)
        df['t'] = pd.to_datetime(df['t'])
        df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
        
    return pd.DataFrame()

# --- NAVIGAZIONE ---
st.sidebar.markdown("## 🧭 SISTEMA")
menu = st.sidebar.radio("Seleziona Vista:", ["🏟️ DASHBOARD SINGOLA", "🔥 SCANNER HOT TICKERS", "🔙 BACKTESTING STRATEGIA"])

# --- REFRESH CONFIG ---
# Dashboard: refresh ogni 1 minuto (60000 ms)
# Scanner: refresh ogni 5 minuti (300000 ms) per evitare Rate Limit
if menu == "🏟️ DASHBOARD SINGOLA":
    st_autorefresh(interval=60000, key="sentinel_dash_refresh")
elif menu == "🔥 SCANNER HOT TICKERS":
    st_autorefresh(interval=300000, key="sentinel_scan_refresh")
# --------------------------

today = datetime.now()
today_str_format = today.strftime('%Y-%m-%d') # Per la cache

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
    except Exception as e:
        st.error("⚠️ Yahoo Finance ti ha temporaneamente bloccato per troppe richieste (Rate Limit). Cambia rete/IP o attendi 10 minuti prima di riprovare.")
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
            
            if 'prev_iv' not in st.session_state:
                st.session_state.prev_iv = mean_iv
            iv_change = mean_iv - st.session_state.prev_iv
            st.session_state.prev_iv = mean_iv

            # --- MODIFICA ASIMMETRICA DS (SKEW DRIVEN) ---
            # 1. Estrazione IV Specifica (Skew) su 0DTE/1DTE
            try:
                skew_date = available_dates[0]
                skew_data = fetch_data(current_ticker, [skew_date])
                
                if not skew_data.empty:
                    # Target: 2% OTM (~25 Delta proxy)
                    c_target = spot * 1.02
                    p_target = spot * 0.98
                    
                    c_skew = skew_data[skew_data['type'] == 'call']
                    p_skew = skew_data[skew_data['type'] == 'put']
                    
                    # Trova lo strike più vicino al target
                    c_iv = c_skew.iloc[(c_skew['strike'] - c_target).abs().argmin()]['impliedVolatility'] if not c_skew.empty else mean_iv
                    p_iv = p_skew.iloc[(p_skew['strike'] - p_target).abs().argmin()]['impliedVolatility'] if not p_skew.empty else mean_iv
                else:
                    c_iv = p_iv = mean_iv
            except:
                c_iv = p_iv = mean_iv

            # 2. Calcolo Fixed 1-Day Move (1/252)
            one_day_factor = np.sqrt(1/252)
            
            # 3. Creazione delle 4 Linee Asimmetriche
            sd1_up = spot * (1 + (c_iv * one_day_factor))
            sd2_up = spot * (1 + (c_iv * 2 * one_day_factor))
            sd1_down = spot * (1 - (p_iv * one_day_factor))
            sd2_down = spot * (1 - (p_iv * 2 * one_day_factor))
            
            skew_factor = p_iv / c_iv if c_iv > 0 else 1.0
            # ---------------------------------------------

            # CALCOLO 0-GAMMA ORIGINALE
            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 
            
            # CALCOLO 0-GAMMA DINAMICO (SOLO VOLUMI)
            try: z_gamma_dyn = brentq(calculate_0g_dynamic, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma_dyn = spot

            df = get_greeks_pro(raw_data, spot)
            
            # --- LOGICA DI AGGREGAZIONE MATEMATICA (Binning Dinamico) ---
            # Usiamo floor division per forzare ogni contratto nel proprio bin matematico
            pivot_series = (df['strike'] // gran) * gran
            
            # Aggregazione Totale su Pivot
            agg = df.groupby(pivot_series).agg({
                'Gamma': 'sum', 
                'Vanna': 'sum', 
                'Charm': 'sum', 
                'Vega': 'sum', 
                'Theta': 'sum'
            }).reset_index()
            
            # Rinomina la colonna pivot
            if 'strike' not in agg.columns:
                agg.rename(columns={'index': 'strike', agg.columns[0]: 'strike'}, inplace=True)
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            # Calcolo Muri basato sui dati aggregati
            if not agg.empty:
                c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
                p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']
                v_trigger = agg.loc[agg['Vanna'].abs().idxmax(), 'strike']
            else:
                c_wall = p_wall = v_trigger = spot

            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")

            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            direction = "NEUTRALE"; bias_color = "gray"
            
            if net_gamma < 0 and net_vanna < 0:
                direction = "☢️ PERICOLO ESTREMO (Crash Risk / Short Gamma & Vanna)"; bias_color = "#8B0000"
            elif net_gamma < 0:
                direction = "🔴 SHORT GAMMA BIAS (Espansione Volatilità)"; bias_color = "#FF4136"
            elif spot < z_gamma:
                direction = "🟠 PRESSIONE SOTTO ZERO GAMMA (Vulnerabilità)"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0:
                direction = "🚀 BULLISH FLOW (Charm Support / Long Gamma)"; bias_color = "#2ECC40"
            else:
                direction = "🔵 LONG GAMMA / STABILITÀ (Contrazione Volatilità)"; bias_color = "#0074D9"
            
            st.markdown(f"### 📊 Real-Time Metric Regime")
            c_reg1, c_reg2, c_reg3, c_reg4 = st.columns(4)
            c_reg1.metric("Net Gamma", f"{net_gamma:,.0f}", delta=f"{'LONG' if net_gamma > 0 else 'SHORT'}")
            c_reg2.metric("Net Vanna", f"{net_vanna:,.0f}", delta=f"{'STABLE' if net_vanna > 0 else 'UNSTABLE'}")
            c_reg3.metric("Net Charm", f"{net_charm:,.0f}", delta=f"{'SUPPORT' if net_charm < 0 else 'DECAY'}")
            c_reg4.metric("SKEW FACTOR (P/C)", f"{skew_factor:.2f}x")

            st.markdown(f"""
                <div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center; margin-top: 10px; margin-bottom: 25px;'>
                    <b style='color:white; font-size:24px;'>MARKET BIAS: {direction}</b>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA (STA/DYN)", f"{z_gamma:.0f} / {z_gamma_dyn:.0f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{spot*one_day_factor*mean_iv:.2f}")

            st.markdown("---")
            
            def get_dist(target, spot):
                d = ((target - spot) / spot) * 100
                color = "#00FF41" if d > 0 else "#FF4136"
                return f"<span style='color:{color};'>{d:+.2f}%</span>"

            sd_up_pct = ((sd1_up - spot) / spot) * 100
            sd_dn_pct = ((sd1_down - spot) / spot) * 100

            st.markdown(f"""
                <div style='background-color:rgba(30, 30, 30, 0.8); padding:10px; border-radius:5px; border: 1px solid #444; margin-bottom: 20px; display: flex; justify-content: space-around;'>
                    <div><b>📍 DIST. CW:</b> {get_dist(c_wall, spot)}</div>
                    <div><b>📍 DIST. 0G-DYN:</b> {get_dist(z_gamma_dyn, spot)}</div>
                    <div><b>📍 DIST. VT:</b> {get_dist(v_trigger, spot)}</div>
                    <div><b>📍 DIST. PW:</b> {get_dist(p_wall, spot)}</div>
                    <div><b>📍 1SD UP/DN:</b> <span style='color:#FFA500;'>{sd_up_pct:+.2f}% / {sd_dn_pct:+.2f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

            # --- INIZIO NUOVO HUD QUANTISTICO ON-DEMAND ---
            with st.expander("🔍 🧠 HUD QUANTISTICO: SENTIMENT & CONFLUENZA GREEKS (Clicca per espandere)"):
                pos_score = 4 if (spot > z_gamma and spot > z_gamma_dyn) else (-4 if (spot < z_gamma and spot < z_gamma_dyn) else 0)
                vanna_score = 3 if net_vanna > 0 else -3
                charm_score = 3 if net_charm < 0 else -3
                total_ss = pos_score + vanna_score + charm_score
                
                hud_color = "#2ECC40" if total_ss >= 5 else ("#FF4136" if total_ss <= -5 else "#FFDC00")
                
                pos_text = "🟢 SOPRA entrambi 0-G (Pieno controllo acquirenti)" if pos_score == 4 else ("🔴 SOTTO entrambi 0-G (Pieno controllo venditori)" if pos_score == -4 else "🟡 Divergenza OI vs Volumi (Fase incerta)")
                vanna_text = "🟢 Stabile (Nessuno Squeeze Imminente)" if vanna_score == 3 else "🔴 Pericolo Squeeze (Dealer costretti a comprare/vendere in corsa)"
                charm_text = "🔵 Supporto Passivo (Il tempo aiuta i Long)" if charm_score == 3 else "🔴 Flusso in Uscita (Il tempo pesa sul prezzo)"

                st.markdown(f"""
                <div style='background-color:rgba(15,15,15,0.9); padding:20px; border: 2px solid {hud_color}; border-radius:10px;'>
                    <h2 style='text-align:center; color:{hud_color}; margin-top:0;'>SENTIMENT SCORE: {total_ss} / 10</h2>
                    <hr style='border-color:#333;'>
                    <div style='display:flex; justify-content:space-between; text-align:center;'>
                        <div style='width:30%;'>
                            <h4 style='color:white;'>⚡ Forza Prezzo (40%)</h4>
                            <p style='color:lightgray;'><i>Confluenza 0G Statico / Dinamico</i></p>
                            <b>{pos_text}</b>
                        </div>
                        <div style='width:30%;'>
                            <h4 style='color:white;'>🌪️ Forza Vanna (30%)</h4>
                            <p style='color:lightgray;'><i>Rischio accelerazione Volatilità</i></p>
                            <b>{vanna_text}</b>
                        </div>
                        <div style='width:30%;'>
                            <h4 style='color:white;'>⏳ Forza Charm (30%)</h4>
                            <p style='color:lightgray;'><i>Supporto/Pressione legati al Tempo</i></p>
                            <b>{charm_text}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # --- FINE NUOVO HUD ---

            col_view, col_vol = st.columns([2, 1])
            with col_view:
                view_mode = st.radio("👁️ VISTA GRAFICO:", ["📊 Vista Standard (Metrica Singola)", "🌪️ Vanna View (Overlay Gamma + Vanna)"], horizontal=True)
            with col_vol:
                st.metric("📈 VOLATILITÀ CHAIN IV (Dinamica)", f"{mean_iv*100:.2f}%", delta=f"{iv_change*100:.2f}%", delta_color="inverse")

            fig = go.Figure()

            if view_mode == "📊 Vista Standard (Metrica Singola)":
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg[metric], 
                    orientation='h', 
                    marker=dict(color=['#00FF41' if x >= 0 else '#FF4136' for x in visible_agg[metric]]),
                    name=metric
                ))
                xaxis_title = f"Net {metric} Exposure"
            else:
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg['Gamma'], 
                    orientation='h', 
                    marker=dict(color='rgba(100, 100, 100, 0.3)', line=dict(width=0)), 
                    name="Gamma (Background)",
                    xaxis="x1"
                ))
                
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg['Vanna'], 
                    orientation='h', 
                    marker=dict(
                        color=['#00FFFF' if x >= 0 else '#FF00FF' for x in visible_agg['Vanna']], 
                        line=dict(color='white', width=1)
                    ),
                    width=gran * 0.4, 
                    name="Vanna (Focus)",
                    xaxis="x2"
                ))

                fig.update_layout(
                    xaxis=dict(title="Gamma Exposure", side="bottom", showgrid=False),
                    xaxis2=dict(title="Vanna Exposure (Scaled)", side="top", overlaying="x", showgrid=False, zerolinecolor="white"),
                    barmode='overlay'
                )
                xaxis_title = "Vanna vs Gamma Overlay (Dual Axis)"

            for strike in visible_agg['strike']:
                fig.add_hline(y=strike, line_width=0.3, line_dash="dot", line_color="rgba(255,255,255,0.2)")

            fig.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G STATIC")
            fig.add_hline(y=z_gamma_dyn, line_color="#00BFFF", line_width=2, line_dash="dot", annotation_text="0-G DYNAMIC (VOL)")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CW")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PW")
            fig.add_hline(y=v_trigger, line_color="#FF00FF", line_width=2, line_dash="longdash", annotation_text="VANNA TRIGGER")
            
            # --- VISUALIZZAZIONE LINEE ASIMMETRICHE ---
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="dash", annotation_text=f"+1SD (Call IV) {sd1_up:.2f}")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="dash", annotation_text=f"-1SD (Put IV) {sd1_down:.2f}")
            fig.add_hline(y=sd2_up, line_color="#FF0000", line_dash="solid", annotation_text=f"+2SD {sd2_up:.2f}")
            fig.add_hline(y=sd2_down, line_color="#FF0000", line_dash="solid", annotation_text=f"-2SD {sd2_down:.2f}")

            # Nota Skew Factor nella legenda (dummy trace)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'), 
                                     name=f"Skew Factor: {skew_factor:.2f}x (Put/Call IV)", showlegend=True))

            fig.update_layout(template="plotly_dark", height=850, margin=dict(l=0,r=0,t=40,b=0), yaxis=dict(range=[lo, hi], dtick=gran))
            st.plotly_chart(fig, use_container_width=True)

elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Professional Market Scanner (50 Tickers)")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 AGGIORNA SCANNER", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        expiry_mode = st.selectbox("📅 SELEZIONE SCADENZE:", ["0-1 DTE (Scalping/Intraday)", "Prossima Scadenza Mensile (Swing)"])
    
    tickers_50 = ["^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA"]
    
    scan_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t_name in enumerate(tickers_50):
        status_text.text(f"Scansione in profondità: {t_name} ({i+1}/{len(tickers_50)})")
        
        # --- UTILIZZO FUNZIONE PROTETTA ---
        data_pack = fetch_scanner_ticker(t_name, expiry_mode, today_str_format)
        
        # Micro-pausa fondamentale impostata a 0.5s per evitare il ban IP da Yahoo
        time.sleep(0.5) 
        
        if data_pack is None:
            progress_bar.progress((i + 1) / len(tickers_50))
            continue
            
        px, df_scan, dte_years = data_pack
        
        try:
            # Calcolo 0-G Statico e Dinamico
            try: zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
            except: zg_val = px
            try: zg_dyn = brentq(calculate_0g_dynamic, px*0.75, px*1.25, args=(df_scan,))
            except: zg_dyn = px

            # Calcolo Greche Scanner
            df_scan_greeks = get_greeks_pro(df_scan, px)
            net_vanna_scan = df_scan_greeks['Vanna'].sum() if not df_scan_greeks.empty else 0
            net_charm_scan = df_scan_greeks['Charm'].sum() if not df_scan_greeks.empty else 0
            
            # Motore di Scoring Confluenza (Max +10, Min -10)
            p_score = 4 if (px > zg_val and px > zg_dyn) else (-4 if (px < zg_val and px < zg_dyn) else 0)
            v_score = 3 if net_vanna_scan > 0 else -3
            c_score = 3 if net_charm_scan < 0 else -3
            ss = p_score + v_score + c_score

            v_icon = "🟢" if net_vanna_scan > 0 else "🔴"
            c_icon = "🔵" if net_charm_scan < 0 else "🔴"
            
            # Cluster/Market Regime
            if ss >= 8: verdict = "🚀 CONFLUENZA FULL LONG"
            elif ss <= -8: verdict = "☢️ CRASH RISK / FULL SHORT"
            elif px > zg_val and px < zg_dyn: verdict = "⚠️ DISTRIBUZIONE (Volumi in uscita)"
            elif px < zg_val and px > zg_dyn: verdict = "🔥 SHORT SQUEEZE IN ATTO"
            elif net_vanna_scan < 0 and px > zg_dyn: verdict = "🌪️ GAMMA SQUEEZE (Alta Volatilità)"
            else: verdict = "⚖️ NEUTRO / RANGE BOUND"

            avg_iv = df_scan['impliedVolatility'].mean()
            sd_move = px * avg_iv * np.sqrt(dte_years)
            sd1_up, sd1_down = px + sd_move, px - sd_move
            dist_zg_pct = ((px - zg_val) / px) * 100
            is_above_0g = px > zg_val
            near_sd_up = abs(px - sd1_up) / px < 0.005
            near_sd_down = abs(px - sd1_down) / px < 0.005
            
            if not is_above_0g: 
                if near_sd_down: status_label = "🔴 < 0G | TEST -1SD (Bounce?)"
                elif px < sd1_down: status_label = "⚫ < 0G | SOTTO -1SD (Short Ext)"
                else: status_label = "🔻 SOTTO 0G (Short Bias)"
            else: 
                if near_sd_up: status_label = "🟡 > 0G | TEST +1SD (Breakout?)"
                elif px > sd1_up: status_label = "🟢 > 0G | SOPRA +1SD (Long Ext)"
                elif near_sd_down: status_label = "🟢 > 0G | DIP BUY (Test -1SD)"
                else: status_label = "✅ SOPRA 0G (Long Bias)"
            
            if abs(dist_zg_pct) < 0.3: status_label = "🔥 FLIP IMMINENTE (0G)"
            
            scan_results.append({
                "Ticker": t_name.replace("^", ""), 
                "Score": int(ss),                 
                "Verdict (Regime)": verdict,      
                "Greche V|C": f"V:{v_icon} C:{c_icon}", 
                "Prezzo": round(px, 2), 
                "0-G Static": round(zg_val, 2), 
                "0-G Dynamic": round(zg_dyn, 2),
                "1SD Range": f"{sd1_down:.0f}-{sd1_up:.0f}", 
                "Dist. 0G %": round(dist_zg_pct, 2), 
                "Analisi": status_label, 
                "_sort_score": -ss,                
                "_sort_dist": abs(dist_zg_pct)
            })
        except: pass
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values(by=["_sort_score", "_sort_dist"]).drop(columns=["_sort_score", "_sort_dist"])
        
        def color_logic_pro(row):
            styles = [''] * len(row)
            # Colore per Score
            score_idx = row.index.get_loc('Score')
            val_score = row['Score']
            if val_score >= 8: styles[score_idx] = 'background-color: #2ECC40; color: white; font-weight: bold'
            elif val_score <= -8: styles[score_idx] = 'background-color: #8B0000; color: white; font-weight: bold'
            elif val_score > 0: styles[score_idx] = 'color: #2ECC40; font-weight: bold'
            elif val_score < 0: styles[score_idx] = 'color: #FF4136; font-weight: bold'
            
            # Colore per Analisi (Originale)
            analisi_idx = row.index.get_loc('Analisi')
            val_ana = row['Analisi']
            if "🔥" in val_ana: styles[analisi_idx] = 'background-color: #8B0000; color: white'
            elif "🔴" in val_ana: styles[analisi_idx] = 'color: #FF4136; font-weight: bold'
            elif "🟢" in val_ana: styles[analisi_idx] = 'color: #2ECC40; font-weight: bold'
            elif "🟡" in val_ana: styles[analisi_idx] = 'color: #FFDC00'
            elif "✅" in val_ana: styles[analisi_idx] = 'color: #0074D9'

            return styles

        st.dataframe(final_df.style.apply(color_logic_pro, axis=1), use_container_width=True, height=800)

elif menu == "🔙 BACKTESTING STRATEGIA":
    st.title("🔙 Backtesting Strategia GEX (Advanced Builder)")
    
    st.markdown("""
    <div style='background-color:rgba(0, 100, 255, 0.1); padding:15px; border-radius:5px; border: 1px solid #0074D9; margin-bottom: 20px;'>
    <b>🛠️ COSTRUTTORE STRATEGIE:</b> Qui puoi simulare migliaia di combinazioni.
    Scegli le date, definisci i <b>Trigger di Ingresso</b> (es. Rimbalzo sui Muri, Breakout 0-Gamma) e decidi se attendere la <b>Chiusura Candela</b> per confermare il segnale.
    </div>
    """, unsafe_allow_html=True)

    # --- 1. CONFIGURAZIONE DATI & PERIODO ---
    st.subheader("1️⃣ Dati & Periodo")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bt_ticker = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "AMD", "COIN", "^SPX", "^NDX", "^RUT"])
    with c2:
        bt_tf = st.selectbox("Timeframe", ["5Min", "15Min", "1H", "1D"], index=2)
    with c3:
        # Date Range Picker
        today = datetime.now()
        default_start = today - timedelta(days=365)
        date_range = st.date_input("Periodo di Test", [default_start, today], max_value=today)
    with c4:
        initial_capital = st.number_input("Capitale Iniziale ($)", value=10000)

    # --- 1.1 ORARI DI TRADING (SESSIONI) ---
    st.subheader("⏰ Orari Operativi (Fuso Orario Dati)")
    t1, t2, t3 = st.columns(3)
    with t1:
        use_time_filter = st.checkbox("Filtra per Orario", value=False)
    with t2:
        start_time = st.time_input("Ora Inizio", value=datetime.strptime("09:30", "%H:%M").time())
    with t3:
        end_time = st.time_input("Ora Fine", value=datetime.strptime("16:00", "%H:%M").time())

    # --- 2. COSTRUTTORE STRATEGIA ---
    st.markdown("---")
    st.subheader("2️⃣ Costruttore Regole di Ingresso")
    
    col_long, col_short, col_conf = st.columns(3)
    
    with col_long:
        st.markdown("### 🟢 LONG SETUP")
        long_trigger = st.selectbox("Entra LONG quando il Prezzo:", [
            "Nessun Long",
            "Rimbalza su Put Wall (Supporto)",
            "Rompe a Rialzo 0-Gamma (Trend)",
            "Rompe a Rialzo Call Wall (Squeeze)",
            "Rompe a Rialzo +1SD (Momentum)"
        ])
        
    with col_short:
        st.markdown("### 🔴 SHORT SETUP")
        short_trigger = st.selectbox("Entra SHORT quando il Prezzo:", [
            "Nessun Short",
            "Rimbalza su Call Wall (Resistenza)",
            "Rompe a Ribasso 0-Gamma (Trend)",
            "Rompe a Ribasso Put Wall (Crash)",
            "Rompe a Ribasso -1SD (Momentum)"
        ])

    with col_conf:
        st.markdown("### ⚙️ FILTRI & CONFERME")
        entry_mode = st.radio("Modalità di Ingresso:", [
            "⚡ Instant Touch (Appena tocca il livello)",
            "🕯️ Candle Close (Attendi chiusura candela)"
        ], help="Instant: entra subito durante la candela. Candle Close: entra all'apertura della candela successiva se la condizione è confermata.")
        
        use_trend_filter = st.checkbox("Filtro Trend (SMA 200)", value=False, help="Long solo se Prezzo > SMA200, Short solo se Prezzo < SMA200")
        
        # SENSITIVITY SLIDER
        st.markdown("### 🎚️ SENSIBILITÀ LIVELLI")
        level_sensitivity = st.slider("Moltiplicatore Distanza Muri (Basso = Più Trade)", 0.5, 4.0, 1.5, 0.1, help="Valori più bassi avvicinano i muri al prezzo (più segnali, più rumore). Valori alti li allontanano (meno segnali, più affidabili).")

    # --- 3. GESTIONE RISCHIO ---
    st.markdown("---")
    st.subheader("3️⃣ Gestione Rischio & Uscita")
    r1, r2, r3 = st.columns(3)
    with r1:
        rr_ratio = st.selectbox("Rischio : Rendimento", ["1:1", "1:1.5", "1:2", "1:3", "1:5", "Dynamic (Opposite Wall)"])
    with r2:
        risk_per_trade = st.slider("Rischio per Trade (%)", 0.1, 5.0, 1.0)
    with r3:
        sl_type = st.selectbox("Stop Loss Mode", ["Fixed ATR (Volatility Based)", "Fixed % (Static)"])

    # Parsing R:R
    rr_map = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0, "1:3": 3.0, "1:5": 5.0, "Dynamic (Opposite Wall)": "DYNAMIC"}
    target_mult = rr_map[rr_ratio]

    if st.button("🚀 AVVIA SIMULAZIONE STRATEGIA", type="primary"):
        if len(date_range) != 2:
            st.error("Seleziona una data di inizio e fine valide.")
            st.stop()
            
        start_date_str = date_range[0].strftime('%Y-%m-%d')
        end_date_str = date_range[1].strftime('%Y-%m-%d')
        
        with st.spinner(f"Elaborazione Strategia su {bt_ticker} dal {start_date_str} al {end_date_str}..."):
            # 1. Fetch Price History
            df_hist = fetch_alpaca_history(bt_ticker, bt_tf, start_date_str, end_date_str)
            
            if df_hist.empty:
                st.error("❌ Nessun dato storico trovato. Prova a cambiare date o Ticker.")
                st.stop()
            
            # 2. CALCOLO LIVELLI GEX SINTETICI & INDICATORI
            df_hist['Returns'] = df_hist['Close'].pct_change()
            df_hist['Roll_Vol'] = df_hist['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Proxy Levels
            df_hist['ZeroGamma_Sim'] = df_hist['Close'].rolling(window=20).mean() # Proxy dinamico
            
            # ATR Calculation
            high_low = df_hist['High'] - df_hist['Low']
            high_close = np.abs(df_hist['High'] - df_hist['Close'].shift())
            low_close = np.abs(df_hist['Low'] - df_hist['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df_hist['ATR'] = true_range.rolling(14).mean()
            
            # Dynamic Walls (Volume Weighted Logic Attempt)
            # Se abbiamo Volume, usiamolo per pesare la larghezza dei muri?
            # Idea: High Volume = Stronger Walls (More OI usually). 
            # Ma qui stiamo simulando la distanza.
            # Usiamo Volatility Multiplier standard per ora, ma assicuriamo che i dati siano reali.
            
            # Utilizziamo la Sensibilità scelta dall'utente per modulare la distanza
            # NUOVA FORMULA: level_sensitivity è ora un moltiplicatore DIRETTO dell'ampiezza.
            # (1 + Roll_Vol) aggiunge un componente dinamico ma controllato.
            df_hist['Vol_Mult'] = level_sensitivity * (1 + df_hist['Roll_Vol'])
            
            df_hist['CallWall_Sim'] = df_hist['ZeroGamma_Sim'] + (df_hist['ATR'] * df_hist['Vol_Mult'])
            df_hist['PutWall_Sim'] = df_hist['ZeroGamma_Sim'] - (df_hist['ATR'] * df_hist['Vol_Mult'])
            
            # SD Lines (Bollinger-like for Momentum)
            # Anche le SD devono scalare con la sensibilità per coerenza
            df_hist['SD1_Up'] = df_hist['ZeroGamma_Sim'] + (df_hist['ATR'] * level_sensitivity)
            df_hist['SD1_Down'] = df_hist['ZeroGamma_Sim'] - (df_hist['ATR'] * level_sensitivity)
            
            # Trend Filter
            df_hist['SMA200'] = df_hist['Close'].rolling(window=200).mean()
            
            df_hist.dropna(inplace=True)
            
            # 3. ENGINE DI TRADING AVANZATO
            balance = initial_capital
            equity_curve = [initial_capital]
            trades = []
            position = None 
            
            wait_for_close = "Candle Close" in entry_mode
            
            for i in range(len(df_hist)):
                if i < 1: continue
                
                curr_bar = df_hist.iloc[i]
                prev_bar = df_hist.iloc[i-1]
                
                price = curr_bar['Close']
                ts = curr_bar['datetime']
                
                # --- TIME FILTER CHECK ---
                if use_time_filter:
                    # Assumiamo che ts sia datetime
                    # Se i dati sono daily, il time filter non ha senso (ts.time() è 00:00)
                    if bt_tf == "1D":
                        pass # Ignora filtro su daily
                    else:
                        bar_time = ts.time()
                        if not (start_time <= bar_time <= end_time):
                            # Se siamo fuori orario, chiudiamo posizioni intraday? 
                            # O semplicemente non entriamo?
                            # Per ora: NON ENTRIAMO. Le posizioni aperte restano overnight (swing).
                            # Se si vuole chiudere a fine sessione, serve logica extra.
                            
                            # Gestione Posizione (Swing): continuiamo a monitorare SL/TP anche fuori orario?
                            # Nei dati storici intraday, le barre fuori orario potrebbero non esserci se il feed è solo RTH.
                            pass

                # Levels
                zg = prev_bar['ZeroGamma_Sim']
                cw = prev_bar['CallWall_Sim']
                pw = prev_bar['PutWall_Sim']
                sd_up = prev_bar['SD1_Up']
                sd_dn = prev_bar['SD1_Down']
                atr = prev_bar['ATR']
                sma200 = prev_bar['SMA200']
                
                # --- GESTIONE POSIZIONE ---
                if position:
                    # Check Exit
                    exit_res = None
                    exit_pnl = 0
                    
                    if position['type'] == 'long':
                        if curr_bar['Low'] <= position['sl']:
                            exit_price = position['sl'] # Slippage sim
                            exit_res = 'LOSS'
                        elif curr_bar['High'] >= position['tp']:
                            exit_price = position['tp']
                            exit_res = 'WIN'
                            
                    elif position['type'] == 'short':
                        if curr_bar['High'] >= position['sl']:
                            exit_price = position['sl']
                            exit_res = 'LOSS'
                        elif curr_bar['Low'] <= position['tp']:
                            exit_price = position['tp']
                            exit_res = 'WIN'
                            
                    if exit_res:
                        if position['type'] == 'long':
                            pnl = (exit_price - position['entry']) * position['size']
                        else:
                            pnl = (position['entry'] - exit_price) * position['size']
                            
                        balance += pnl
                        trades.append({'time': ts, 'type': f'EXIT {exit_res}', 'price': exit_price, 'pnl': pnl, 'res': exit_res})
                        position = None
                    
                    equity_curve.append(balance)
                    continue 

                # --- VALUTAZIONE SEGNALI ---
                # Verifica Orario per NUOVI ingressi
                can_enter = True
                if use_time_filter and bt_tf != "1D":
                     bar_time = ts.time()
                     if not (start_time <= bar_time <= end_time):
                         can_enter = False
                
                signal = None
                
                if can_enter:
                    # Trend Filter Check
                    trend_ok_long = (curr_bar['Close'] > sma200) if use_trend_filter else True
                    trend_ok_short = (curr_bar['Close'] < sma200) if use_trend_filter else True
                    
                    # --- LONG LOGIC ---
                    if trend_ok_long and "Nessun" not in long_trigger:
                        trigger_met = False
                        
                        if "Rimbalza su Put Wall" in long_trigger:
                            if wait_for_close:
                                if curr_bar['Low'] <= pw and curr_bar['Close'] > pw: trigger_met = True
                            else:
                                if curr_bar['Low'] <= pw: trigger_met = True
                                
                        elif "Rompe a Rialzo 0-Gamma" in long_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] < zg and curr_bar['Close'] > zg: trigger_met = True
                            else:
                                if prev_bar['Close'] < zg and curr_bar['High'] > zg: trigger_met = True
                                
                        elif "Rompe a Rialzo Call Wall" in long_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] < cw and curr_bar['Close'] > cw: trigger_met = True
                            else:
                                if prev_bar['Close'] < cw and curr_bar['High'] > cw: trigger_met = True

                        elif "Rompe a Rialzo +1SD" in long_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] < sd_up and curr_bar['Close'] > sd_up: trigger_met = True
                            else:
                                if prev_bar['Close'] < sd_up and curr_bar['High'] > sd_up: trigger_met = True
                                
                        if trigger_met: signal = 'long'

                    # --- SHORT LOGIC ---
                    if trend_ok_short and "Nessun" not in short_trigger and signal is None:
                        trigger_met = False
                        
                        if "Rimbalza su Call Wall" in short_trigger:
                            if wait_for_close:
                                if curr_bar['High'] >= cw and curr_bar['Close'] < cw: trigger_met = True
                            else:
                                if curr_bar['High'] >= cw: trigger_met = True
                                
                        elif "Rompe a Ribasso 0-Gamma" in short_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] > zg and curr_bar['Close'] < zg: trigger_met = True
                            else:
                                if prev_bar['Close'] > zg and curr_bar['Low'] < zg: trigger_met = True
                                
                        elif "Rompe a Ribasso Put Wall" in short_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] > pw and curr_bar['Close'] < pw: trigger_met = True
                            else:
                                if prev_bar['Close'] > pw and curr_bar['Low'] < pw: trigger_met = True

                        elif "Rompe a Ribasso -1SD" in short_trigger:
                            if wait_for_close:
                                if prev_bar['Close'] > sd_dn and curr_bar['Close'] < sd_dn: trigger_met = True
                            else:
                                if prev_bar['Close'] > sd_dn and curr_bar['Low'] < sd_dn: trigger_met = True
                                
                        if trigger_met: signal = 'short'
                
                # --- ENTRY EXECUTION ---
                if signal:
                    # Calculate SL/TP
                    risk_amt = balance * (risk_per_trade / 100)
                    
                    if sl_type == "Fixed ATR (Volatility Based)":
                        sl_dist = atr * 2.0
                    else:
                        sl_dist = price * 0.01 # 1% fixed
                        
                    if signal == 'long':
                        sl_price = price - sl_dist
                        risk_per_share = price - sl_price
                        
                        if target_mult == "DYNAMIC":
                            tp_price = cw # Target Call Wall
                            if tp_price <= price: tp_price = price + (risk_per_share * 2) # Fallback
                        else:
                            tp_price = price + (risk_per_share * target_mult)
                            
                        if risk_per_share > 0:
                            size = risk_amt / risk_per_share
                            position = {'type': 'long', 'entry': price, 'sl': sl_price, 'tp': tp_price, 'size': size}
                            trades.append({'time': ts, 'type': 'ENTRY LONG', 'price': price, 'pnl': 0, 'res': 'OPEN'})
                            
                    elif signal == 'short':
                        sl_price = price + sl_dist
                        risk_per_share = sl_price - price
                        
                        if target_mult == "DYNAMIC":
                            tp_price = pw # Target Put Wall
                            if tp_price >= price: tp_price = price - (risk_per_share * 2)
                        else:
                            tp_price = price - (risk_per_share * target_mult)
                            
                        if risk_per_share > 0:
                            size = risk_amt / risk_per_share
                            position = {'type': 'short', 'entry': price, 'sl': sl_price, 'tp': tp_price, 'size': size}
                            trades.append({'time': ts, 'type': 'ENTRY SHORT', 'price': price, 'pnl': 0, 'res': 'OPEN'})

            # --- RISULTATI ---
            st.success("✅ Simulazione Completata!")
            
            closed_trades = [t for t in trades if t['res'] in ['WIN', 'LOSS']]
            wins = [t for t in closed_trades if t['res'] == 'WIN']
            losses = [t for t in closed_trades if t['res'] == 'LOSS']
            
            total_trades = len(closed_trades)
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            total_pnl = sum([t['pnl'] for t in closed_trades])
            profit_factor = (sum([t['pnl'] for t in wins]) / abs(sum([t['pnl'] for t in losses]))) if losses else 99.9
            
            # Coverage Stats
            trading_days = df_hist['datetime'].dt.date.nunique()
            days_with_trades = pd.to_datetime([t['time'] for t in trades]).date
            active_days = len(set(days_with_trades))
            coverage_pct = (active_days / trading_days * 100) if trading_days > 0 else 0
            
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Net Profit", f"${total_pnl:,.2f}", delta=f"{(total_pnl/initial_capital)*100:.2f}%")
            k2.metric("Total Trades", f"{total_trades}", help="Numero totale di operazioni chiuse (Win + Loss)")
            k3.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
            k4.metric("Profit Factor", f"{profit_factor:.2f}")
            k5.metric("Active Days", f"{active_days}/{trading_days}", f"{coverage_pct:.1f}% Coverage")
            
            st.area_chart(equity_curve)
            
            # Grafico Tecnico
            fig = go.Figure()
            display_limit = 1000 # Mostra ultimi 1000 periodi per velocità
            df_chart = df_hist.tail(display_limit)
            
            fig.add_trace(go.Candlestick(x=df_chart['datetime'], open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Prezzo'))
            fig.add_trace(go.Scatter(x=df_chart['datetime'], y=df_chart['CallWall_Sim'], line=dict(color='green', width=1, dash='dash'), name='Call Wall (Sim)'))
            fig.add_trace(go.Scatter(x=df_chart['datetime'], y=df_chart['PutWall_Sim'], line=dict(color='red', width=1, dash='dash'), name='Put Wall (Sim)'))
            fig.add_trace(go.Scatter(x=df_chart['datetime'], y=df_chart['ZeroGamma_Sim'], line=dict(color='yellow', width=2), name='0-Gamma (Sim)'))
            
            trade_df = pd.DataFrame(trades)
            if not trade_df.empty:
                trade_df = trade_df[trade_df['time'] >= df_chart['datetime'].iloc[0]]
                entries = trade_df[trade_df['type'].str.contains('ENTRY')]
                exits_win = trade_df[trade_df['res'] == 'WIN']
                exits_loss = trade_df[trade_df['res'] == 'LOSS']
                
                fig.add_trace(go.Scatter(x=entries['time'], y=entries['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='blue'), name='Entry'))
                fig.add_trace(go.Scatter(x=exits_win['time'], y=exits_win['price'], mode='markers', marker=dict(symbol='circle', size=8, color='green'), name='Take Profit'))
                fig.add_trace(go.Scatter(x=exits_loss['time'], y=exits_loss['price'], mode='markers', marker=dict(symbol='x', size=8, color='red'), name='Stop Loss'))

            fig.update_layout(title=f"Analisi Tecnica (Ultimi {display_limit} bars)", template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📜 Storico Operazioni"):
                st.dataframe(pd.DataFrame(trades))
