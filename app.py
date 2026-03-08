import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time as dt_time
import time  # <-- Manteniamo l'import per il delay anti-ban
import requests

from botocore.exceptions import ClientError
import os
import io

s3_client = boto3.client('s3',
    endpoint_url='https://files.massive.com',
    aws_access_key_id='fc19982d-d244-499b-823a-710891d5757e',
    aws_secret_access_key='XE4AM3OmmVZpjqXhfOXzxmREDpvYbuo1'
)
MASSIVE_BUCKET = 'flatfiles'
LOCAL_DB_DIR = 'local_database'
os.makedirs(LOCAL_DB_DIR, exist_ok=True)

# --- STRATEGY PARAMETER GRID ---
STRATEGY_PARAM_GRID = {
    "RSI Mean Reversion": {'period': range(10, 22, 2), 'ob': range(65, 85, 5), 'os': range(20, 40, 5)},
    "MACD Crossover": {'fast': range(8, 16, 2), 'slow': range(20, 32, 2), 'signal': range(7, 12, 1)},
    "Bollinger Breakout": {'period': range(15, 30, 5), 'std_dev': [1.5, 2.0, 2.5]},
    "Golden/Death Cross": {'fast': range(40, 60, 10), 'slow': range(150, 250, 50)},
    "Stochastic Oscillator": {'k_period': range(10, 20, 2), 'ob': range(75, 95, 5), 'os': range(5, 25, 5)},
    "CCI Momentum": {'period': range(10, 30, 5)},
    "Williams %R Reversal": {'period': range(10, 30, 5)},
    "HMA Trend": {'period': range(10, 40, 5)},
    "TEMA Crossover": {'period': range(10, 40, 5)},
    "KAMA Trend": {'period': range(10, 40, 5)},
    "Aroon Oscillator": {'period': range(15, 35, 5)},
    "SuperTrend Reversal": {'period': range(7, 15, 2)},
    "Parabolic SAR": {},
    "TSI Crossover": {},
    "UO Overbought/Oversold": {},
    "Keltner Channel Breakout": {},
    "Donchian Channel Breakout": {},
    "Chaikin Volatility": {},
    "CMF Trend": {},
    "VWAP Crossover": {},
    "AD Line Trend": {},
    "Vortex Crossover": {},
    "Choppiness Index Breakout": {},
    "KST Crossover": {},
    "Coppock Curve": {},
    "Ichimoku Cloud Breakout": {},
    "Awesome Oscillator": {},
    "PPO Crossover": {},
    "Mass Index Reversal": {},
    "Ulcer Index Safety": {},
    "WMA Trend": {'period': range(10, 40, 5)},
    "TRIMA Crossover": {'period': range(10, 40, 5)},
    "CMO Reversal": {},
    "Momentum Breakout": {'period': range(5, 20, 5)},
    "BOP Trend": {},
    "TRIX Crossover": {},
    "StochRSI Reversal": {},
    "TSF Trend": {}
}

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
        df.ffill().bfill(inplace=True)
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
        "APCA-API-KEY-ID": st.session_state.get("alpaca_api_key", "PKQVMHYR25JUXQVLTEEBEKVIMV"),
        "APCA-API-SECRET-KEY": st.session_state.get("alpaca_secret_key", "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7")
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
st.sidebar.markdown("## 🔑 API KEYS")
st.session_state.alpaca_api_key = st.sidebar.text_input("Alpaca API Key ID", value=st.session_state.get("alpaca_api_key", "PKQVMHYR25JUXQVLTEEBEKVIMV"), type="password")
st.session_state.alpaca_secret_key = st.sidebar.text_input("Alpaca Secret Key", value=st.session_state.get("alpaca_secret_key", "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7"), type="password")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📁 DATABASE LOCALE")
uploaded_file = st.sidebar.file_uploader("Carica file CSV (Database Locale)", type=['csv'])
if uploaded_file is not None:
    file_path = os.path.join(LOCAL_DB_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File {uploaded_file.name} salvato permanentemente nel Database Locale.")
st.sidebar.markdown("---")

st.sidebar.markdown("## 🧭 SISTEMA")
menu = st.sidebar.radio("Seleziona Vista:", ["🏟️ DASHBOARD SINGOLA", "🔥 SCANNER HOT TICKERS", "🔙 BACKTESTING STRATEGIA", "🛠️ STRATEGY BUILDER"])

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

            # --- INIZIO HUD QUANTISTICO (VERSIONE BILANCIATA PRO) ---
            with st.expander("🔍 🧠 HUD QUANTISTICO: SENTIMENT & CONFLUENZA GREEKS (Clicca per espandere)"):
                import math 
                
                # 1. LOGICA MATEMATICA ORIGINALE (4, 3, 3) - INVARIATA
                pos_score = 4 if (spot > z_gamma and spot > z_gamma_dyn) else (-4 if (spot < z_gamma and spot < z_gamma_dyn) else 0)
                vanna_score = 3 if net_vanna > 0 else -3
                charm_score = 3 if net_charm < 0 else -3
                total_ss = pos_score + vanna_score + charm_score
                
                hud_color = "#2ECC40" if total_ss >= 5 else ("#FF4136" if total_ss <= -5 else "#FFDC00")
                
                # 2. NUOVO BILANCIAMENTO "ANTI-OSCURAMENTO"
                # Usiamo un moltiplicatore più dolce (800 invece di 5000) e mettiamo un tetto (CAP)
                p_dist_raw = (abs(spot - z_gamma_dyn) / spot) * 800
                p_intensity = min(15, p_dist_raw) # Il prezzo non può pesare più di "15 punti" nella torta totale
                
                # Le Greche rimangono logaritmiche per gestire i milioni/miliardi senza esplodere
                v_intensity = math.log10(abs(net_vanna) + 1)
                c_intensity = math.log10(abs(net_charm) + 1)
                
                total_intensity = p_intensity + v_intensity + c_intensity
                
                if total_intensity > 0:
                    p_w = int((p_intensity / total_intensity) * 100)
                    v_w = int((v_intensity / total_intensity) * 100)
                    c_w = 100 - p_w - v_w 
                else:
                    p_w, v_w, c_w = 40, 30, 30

                # 3. TESTI COMPLETI ORIGINALI (Invariati)
                pos_text = "🟢 SOPRA entrambi 0-G (Pieno controllo acquirenti)" if pos_score == 4 else ("🔴 SOTTO entrambi 0-G (Pieno controllo venditori)" if pos_score == -4 else "🟡 Divergenza OI vs Volumi (Fase incerta)")
                vanna_text = "🟢 Stabile (Nessuno Squeeze Imminente)" if vanna_score == 3 else "🔴 Pericolo Squeeze (Dealer costretti a comprare/vendere in corsa)"
                charm_text = "🔵 Supporto Passivo (Il tempo aiuta i Long)" if charm_score == 3 else "🔴 Flusso in Uscita (Il tempo pesa sul prezzo)"

                # 4. LOGICA SEGNALI E RISCHIO (Invariata)
                abs_ss = abs(total_ss)
                if total_ss >= 8:
                    res_sig, res_strat, res_target = "🚀 STRONG BUY", "Long Call / Bull Call Spread", "Call Wall"
                elif total_ss <= -8:
                    res_sig, res_strat, res_target = "☢️ STRONG SELL", "Long Put / Bear Put Spread", "Put Wall"
                elif total_ss >= 4:
                    res_sig, res_strat, res_target = "🟢 BUY ON DIP", "Bull Put Spread (Credit)", "+1 SD Line"
                elif total_ss <= -4:
                    res_sig, res_strat, res_target = "🔴 SELL ON RALLY", "Bear Call Spread (Credit)", "-1 SD Line"
                else:
                    res_sig, res_strat, res_target = "⚖️ NEUTRAL", "Wait / Iron Condor", "Gamma Flip Zone"

                res_risk = "2.0% (ALTO)" if abs_ss >= 8 else ("1.0% (MEDIO)" if abs_ss >= 4 else "0.0% (NO TRADE)")
                res_rr = "1:3+" if abs_ss >= 8 else ("1:2" if abs_ss >= 4 else "N/A")

                # 5. INTERFACCIA (Testi lunghi + Percentuali dinamiche)
                st.markdown(f"""
<div style='background-color:rgba(15,15,15,0.9); padding:20px; border: 2px solid {hud_color}; border-radius:10px;'>
<h2 style='text-align:center; color:{hud_color}; margin-top:0;'>SENTIMENT SCORE: {total_ss} / 10</h2>
<h3 style='text-align:center; color:white; margin-bottom:15px;'>AZIONE: <span style='color:{hud_color};'>{res_sig}</span></h3>
<hr style='border-color:#333;'>
<div style='display:flex; justify-content:space-between; text-align:center;'>
<div style='width:30%;'>
<h4 style='color:white;'>⚡ Forza Prezzo ({p_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Confluenza 0G Statico / Dinamico</i></p>
<b style='font-size:13px; color:white;'>{pos_text}</b>
</div>
<div style='width:30%;'>
<h4 style='color:white;'>🌪️ Forza Vanna ({v_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Rischio accelerazione Volatilità</i></p>
<b style='font-size:13px; color:white;'>{vanna_text}</b>
</div>
<div style='width:30%;'>
<h4 style='color:white;'>⏳ Forza Charm ({c_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Supporto/Pressione legati al Tempo</i></p>
<b style='font-size:13px; color:white;'>{charm_text}</b>
</div>
</div>
<hr style='border-color:#333; margin-top:20px;'>
<div style='display:flex; justify-content:space-between; text-align:center; background:rgba(255,255,255,0.05); padding:15px; border-radius:8px;'>
<div style='width:33%;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>STRATEGIA</p>
<b style='color:white;'>{res_strat}</b>
</div>
<div style='width:33%; border-left:1px solid #444; border-right:1px solid #444;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>RISCHIO CONSIGLIATO</p>
<b style='color:white;'>{res_risk}</b>
</div>
<div style='width:33%;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>TARGET / R:R</p>
<b style='color:white;'>{res_target} ({res_rr})</b>
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

            # --- FIX FINALE: LEGENDA SOPRA IL DOPPIO ASSE ---
            fig.update_layout(
                template="plotly_dark", 
                height=850, 
                margin=dict(l=0, r=0, t=100, b=0), # Aumentato 't' a 100 per far stare asse + legenda
                yaxis=dict(range=[lo, hi], dtick=gran),
                legend=dict(
                    orientation="h",        # Legenda orizzontale
                    yanchor="bottom",
                    y=1.12,                 # Alzata a 1.12 per non toccare l'asse Vanna
                    xanchor="left",
                    x=0.01,                 
                    bgcolor="rgba(0,0,0,0)" 
                )
            )
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
            
            # Motore di Scoring Confluenza
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

            # --- MODIFICA ASIMMETRICA DS (IDENTICA ALLA DASHBOARD) ---
            try:
                mean_iv = df_scan['impliedVolatility'].mean()
                
                if not df_scan.empty:
                    c_target = px * 1.02
                    p_target = px * 0.98
                    c_skew = df_scan[df_scan['type'] == 'call']
                    p_skew = df_scan[df_scan['type'] == 'put']
                    
                    def clean_iv(val):
                        if val is None: return mean_iv / 100 if mean_iv > 1 else mean_iv
                        return val / 100 if val > 1.5 else val

                    raw_c_iv = c_skew.iloc[(c_skew['strike'] - c_target).abs().argmin()]['impliedVolatility'] if not c_skew.empty else mean_iv
                    raw_p_iv = p_skew.iloc[(p_skew['strike'] - p_target).abs().argmin()]['impliedVolatility'] if not p_skew.empty else mean_iv
                    
                    c_iv = clean_iv(raw_c_iv)
                    p_iv = clean_iv(raw_p_iv)
                else:
                    c_iv = p_iv = clean_iv(mean_iv)
            except:
                c_iv = p_iv = 0.15 # Fallback prudenziale

            # 2. Calcolo Fixed 1-Day Move (1/252)
            one_day_factor = np.sqrt(1/252)
            
            # 3. Creazione delle 4 Linee Asimmetriche per lo Scanner
            sd1_up = px * (1 + (c_iv * one_day_factor))
            sd2_up = px * (1 + (c_iv * 2 * one_day_factor))
            sd1_down = px * (1 - (p_iv * one_day_factor))
            sd2_down = px * (1 - (p_iv * 2 * one_day_factor))
            
            # 4. Skew Factor
            skew_factor = p_iv / c_iv if c_iv > 0 else 1.0

            # --- MOTORE OPPORTUNITÀ MEAN REVERSION ---
            if px <= sd2_down:
                reversion_signal = "💎 BUY REVERSION (2DS)"
                rev_score = 2
            elif px <= sd1_down:
                reversion_signal = "🟢 BUY REVERSION (1DS)"
                rev_score = 1
            elif px >= sd2_up:
                reversion_signal = "💀 SELL REVERSION (2DS)"
                rev_score = -2
            elif px >= sd1_up:
                reversion_signal = "🟠 SELL REVERSION (1DS)"
                rev_score = -1
            else:
                reversion_signal = "---"
                rev_score = 0
            # ---------------------------------------------

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
                "0-G Dynamic": round(zg_dyn, 2), # --- NUOVA COLONNA DINAMICA ---
                "1SD Range": f"{sd1_down:.0f} - {sd1_up:.0f}", 
                "2SD Range": f"{sd2_down:.0f} - {sd2_up:.0f}", # --- NUOVA COLONNA 2SD ---
                "Dist. 0G %": round(dist_zg_pct, 2), 
                "OPPORTUNITÀ": reversion_signal,  
                "Analisi": status_label, 
                "_rev_score": rev_score,          
                "_sort_score": -ss,                 
                "_sort_dist": abs(dist_zg_pct)
            })
        except: pass
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values(by=["_sort_score", "_sort_dist"]).drop(columns=["_sort_score", "_sort_dist", "_rev_score"])
        
        def color_logic_pro(row):
            styles = [''] * len(row)
            
            # --- Colore per Score ---
            if 'Score' in row.index:
                score_idx = row.index.get_loc('Score')
                val_score = row['Score']
                if val_score >= 8: styles[score_idx] = 'background-color: #2ECC40; color: white; font-weight: bold'
                elif val_score <= -8: styles[score_idx] = 'background-color: #8B0000; color: white; font-weight: bold'
                elif val_score > 0: styles[score_idx] = 'color: #2ECC40; font-weight: bold'
                elif val_score < 0: styles[score_idx] = 'color: #FF4136; font-weight: bold'
            
            # --- Colore per OPPORTUNITÀ ---
            if 'OPPORTUNITÀ' in row.index:
                opp_idx = row.index.get_loc('OPPORTUNITÀ')
                val_opp = row['OPPORTUNITÀ']
                if "💎 BUY" in val_opp:
                    styles[opp_idx] = 'background-color: #00FF00; color: black; font-weight: bold; border: 1px solid white'
                elif "🟢 BUY" in val_opp:
                    styles[opp_idx] = 'color: #00FF00; font-weight: bold'
                elif "💀 SELL" in val_opp:
                    styles[opp_idx] = 'background-color: #FF0000; color: white; font-weight: bold; border: 1px solid white'
                elif "🟠 SELL" in val_opp:
                    styles[opp_idx] = 'color: #FF0000; font-weight: bold'

            # --- Colore per Analisi ---
            if 'Analisi' in row.index:
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
    st.title("🛠️ Professional Backtesting Suite")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Risk & Robustness")
    friction_pct = st.sidebar.slider("Execution Friction (%)", 0.00, 0.50, 0.00, 0.01)

    def normalize_key(d, possible_keys):
        for k in d.keys():
            if k.lower() in [pk.lower() for pk in possible_keys]:
                return d[k]
        return None

    def apply_friction_post_process(trades_list, initial_capital, friction_pct):
        if not trades_list:
            return trades_list, [initial_capital]
            
        new_trades = []
        balance = initial_capital
        equity_curve = [balance]
        
        for t in trades_list:
            t_copy = dict(t)
            t_type = str(normalize_key(t_copy, ['type', 'Type']) or '').upper()
            price = normalize_key(t_copy, ['price', 'Price', 'Entry Price', 'Exit Price']) or 0
            pnl = normalize_key(t_copy, ['pnl', 'PnL']) or 0
            
            friction_multiplier = 1 - (friction_pct / 100)
            new_price = price * friction_multiplier
            pnl = pnl * friction_multiplier
            t_copy['price'] = new_price
            t_copy['pnl'] = pnl
            balance += pnl
            t_copy['balance'] = balance
            equity_curve.append(balance)
            
            new_trades.append(t_copy)
                
        return new_trades, equity_curve

    def calculate_advanced_metrics(trades_list):
        fallback = {'expectancy': 0, 'profit_factor': 0, 'max_drawdown': 0, 'win_rate': 0, 'total_profit_abs': 0, 'max_dd_abs': 0}
        if not trades_list:
            return fallback
            
        df = pd.DataFrame(trades_list)
        df.columns = [str(c).lower() for c in df.columns]
        
        if 'pnl' not in df.columns:
            return fallback
            
        exits = df[df['pnl'].notna()]
        if exits.empty:
            return fallback
            
        wins = exits[exits['pnl'] > 0]['pnl']
        losses = exits[exits['pnl'] < 0]['pnl']
        
        win_rate = len(wins) / len(exits)
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
        
        total_profit_abs = exits['pnl'].sum()
        
        bal_col = 'balance' if 'balance' in df.columns else None
        max_dd = 0
        max_dd_abs = 0
        if bal_col:
            curve = df[bal_col].tolist()
            peak = curve[0]
            for val in curve:
                if val > peak: peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                dd_abs = peak - val
                if dd > max_dd: max_dd = dd
                if dd_abs > max_dd_abs: max_dd_abs = dd_abs
                
        return {
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd * 100,
            'win_rate': win_rate * 100,
            'total_profit_abs': total_profit_abs,
            'max_dd_abs': max_dd_abs
        }

    def run_monte_carlo(trades_list, initial_capital, simulations=1000):
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        if not trades_list:
            return None
            
        df_res = pd.DataFrame(trades_list)
        if 'pnl' in df_res.columns:
            pnls = df_res[df_res['pnl'].notna()]['pnl'].values
        else:
            return None
            
        n_trades = len(pnls)
        if n_trades == 0:
            return None
            
        # Fixed forward-horizon
        sim_length = min(50, n_trades)
        
        # Vectorized Monte Carlo: Sample with replacement
        random_indices = np.random.randint(0, n_trades, size=(simulations, sim_length))
        simulated_pnls = pnls[random_indices]
        
        # Calculate equity curves
        equity_curves = np.cumsum(simulated_pnls, axis=1) + initial_capital
        
        # Prepend initial capital to the beginning of each curve
        starting_capital = np.full((simulations, 1), initial_capital)
        equity_curves = np.hstack((starting_capital, equity_curves))
        
        # Calculate median curve
        median_curve = np.median(equity_curves, axis=0)
        
        # Calculate quantitative analytics
        final_balances = equity_curves[:, -1]
        prob_profit = (np.sum(final_balances > initial_capital) / simulations) * 100
        
        # Risk of Ruin: equity drops below initial_capital * 0.80 at any point
        ruin_threshold = initial_capital * 0.80
        ruined_simulations = np.any(equity_curves < ruin_threshold, axis=1)
        risk_of_ruin = (np.sum(ruined_simulations) / simulations) * 100
        
        median_final_balance = np.median(final_balances)
        
        # Visualization with Plotly
        fig = go.Figure()
        
        # Performance optimization: Plot all 1000 lines as a single trace separated by NaNs
        # This prevents Plotly from crashing the browser when rendering 1000 individual traces
        x_base = np.arange(sim_length + 1)
        x_all = np.tile(np.append(x_base, np.nan), simulations)
        y_all = np.hstack((equity_curves, np.full((simulations, 1), np.nan))).flatten()
        
        # Add all simulated paths (Gray, low opacity)
        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add Median Curve (Gold, bold)
        fig.add_trace(go.Scatter(
            x=x_base,
            y=median_curve,
            mode='lines',
            line=dict(color='gold', width=3),
            name='Median (50th Percentile)'
        ))
        
        fig.update_layout(
            title='🔬 Monte Carlo Robustness Analysis (Forward 50 Trades)',
            xaxis_title='Trade Number',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig, prob_profit, risk_of_ruin, median_final_balance
    
    # Engine Selection
    engine_choice = st.radio("Seleziona Motore di Backtesting:", 
                             ["🧬 MOTORE A: GEX & Options Hybrid Simulator", 
                              "📈 MOTORE B: Technical Strategy Hub (Pure Trading)"], 
                             horizontal=True)
    
    # Common Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        # Ticker Selection with Predefined List + Custom
        PREDEFINED_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META", "NFLX"]
        ticker_select = st.selectbox("Seleziona Ticker", ["Seleziona..."] + PREDEFINED_TICKERS + ["Inserisci Manualmente"])
        
        if ticker_select == "Inserisci Manualmente":
            ticker = st.text_input("Inserisci Simbolo Ticker", value="SPY").upper()
        elif ticker_select != "Seleziona...":
            ticker = ticker_select
        else:
            ticker = "SPY" # Default

    with c2: timeframe = st.selectbox("Timeframe", ["1D", "1H", "15Min", "5Min"], index=0)
    with c3: 
        start_date = st.date_input("Data Inizio", value=datetime.now() - timedelta(days=365*2))
    with c4: 
        end_date = st.date_input("Data Fine", value=datetime.now())
        initial_capital = st.number_input("Capitale Iniziale ($)", value=10000)

    # Session State for Data Verification
    if 'backtest_data' not in st.session_state:
        st.session_state.backtest_data = None
    if 'backtest_ticker' not in st.session_state:
        st.session_state.backtest_ticker = None

    # --- DATA FETCHING ENHANCED ---
    def fetch_data_smart(ticker, timeframe, start_date, end_date):
        import io
        import requests
        
        df = pd.DataFrame()
        
        # Determine asset type
        is_forex = "=X" in ticker
        is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
        is_crypto = "-USD" in ticker
        is_stock = not (is_forex or is_index or is_crypto)
        
        days_requested = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        # ENGINE 1: Alpaca (Primary for US Stocks/ETFs)
        if is_stock and not is_crypto:
            try:
                tf_alpaca = timeframe
                if timeframe == "1D": tf_alpaca = "1Day"
                elif timeframe == "1H": tf_alpaca = "1Hour"
                elif timeframe == "15Min": tf_alpaca = "15Min"
                elif timeframe == "5Min": tf_alpaca = "5Min"
                
                df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            except Exception as e:
                print(f"Alpaca fetch failed: {e}")
        
        clean_ticker = ticker.replace('=X', '').replace('^', '')

        # ENGINE 1: Massive Cloud
        if df.empty:
            try:
                obj = s3_client.get_object(Bucket=MASSIVE_BUCKET, Key=f"{clean_ticker}.csv")
                df_massive = pd.read_csv(io.BytesIO(obj['Body'].read()))
                
                # Standardize columns
                rename_map = {c: c.capitalize() for c in df_massive.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
                if 'timestamp' in df_massive.columns: rename_map['timestamp'] = 'datetime'
                elif 'time' in df_massive.columns: rename_map['time'] = 'datetime'
                elif 'date' in df_massive.columns: rename_map['date'] = 'datetime'
                
                df_massive.rename(columns=rename_map, inplace=True)
                
                if 'datetime' in df_massive.columns:
                    df_massive['datetime'] = pd.to_datetime(df_massive['datetime'])
                    df_massive = df_massive[(df_massive['datetime'] >= pd.to_datetime(start_date)) & (df_massive['datetime'] <= pd.to_datetime(end_date))]
                    
                    if not df_massive.empty:
                        st.success("✅ Dati recuperati dai server cloud Massive.")
                        df = df_massive
            except Exception as e:
                pass

        # ENGINE 2: Local Database
        if df.empty:
            try:
                local_path = os.path.join(LOCAL_DB_DIR, f"{clean_ticker}.csv")
                if os.path.exists(local_path):
                    df_local = pd.read_csv(local_path)
                    
                    # Standardize columns
                    rename_map = {c: c.capitalize() for c in df_local.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
                    if 'timestamp' in df_local.columns: rename_map['timestamp'] = 'datetime'
                    elif 'time' in df_local.columns: rename_map['time'] = 'datetime'
                    elif 'date' in df_local.columns: rename_map['date'] = 'datetime'
                    
                    df_local.rename(columns=rename_map, inplace=True)
                    
                    if 'datetime' in df_local.columns:
                        df_local['datetime'] = pd.to_datetime(df_local['datetime'])
                        df_local = df_local[(df_local['datetime'] >= pd.to_datetime(start_date)) & (df_local['datetime'] <= pd.to_datetime(end_date))]
                        
                        if not df_local.empty:
                            st.success("📂 Dati recuperati dal Database Locale.")
                            df = df_local
            except Exception as e:
                pass

        # ENGINE 3: yfinance (Fallback)
        if df.empty:
            try:
                tf_yf = "1d"
                if timeframe == "1D": tf_yf = "1d"
                elif timeframe == "1H": tf_yf = "1h"
                elif timeframe == "15Min": tf_yf = "15m"
                elif timeframe == "5Min": tf_yf = "5m"
                
                actual_start = start_date
                if tf_yf in ["5m", "15m", "1h"] and days_requested > 60:
                    actual_start = end_date - timedelta(days=60)
                    st.warning(f"⚠️ yfinance supporta solo 60 giorni per il timeframe {tf_yf}. Date troncate.")
                
                df_yf = yf.download(ticker, start=actual_start, end=end_date, interval=tf_yf, progress=False)
                if not df_yf.empty:
                    df_yf.reset_index(inplace=True)
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)
                    rename_map = {
                        'Date': 'datetime', 'Datetime': 'datetime',
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    }
                    df_yf.rename(columns=rename_map, inplace=True)
                    if 'datetime' not in df_yf.columns and df_yf.index.name in ['Date', 'Datetime']:
                        df_yf.reset_index(inplace=True)
                        df_yf.rename(columns={df_yf.index.name: 'datetime'}, inplace=True)
                    df_yf['datetime'] = pd.to_datetime(df_yf['datetime'])
                    
                    if not df_yf.empty:
                        st.warning("⚠️ Dati presi da Yahoo Finance (Limiti applicati).")
                        df = df_yf
            except Exception as e:
                pass
                
        # ENGINE 4: Fatal Error
        if df.empty:
            st.error("❌ ERRORE CRITICO: Dati non trovati in nessun motore (Alpaca, Massive, Locale, Yahoo). Per favore, carica un file CSV manualmente usando l'apposito uploader per testare questo asset.")
            st.stop()
                
        if not df.empty:
            cols = df.select_dtypes(include=['float64']).columns
            if not cols.empty:
                df[cols] = df[cols].astype('float32')
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            df.sort_values('datetime', inplace=True)
            df.ffill().bfill(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    # Data Verification Step
    st.markdown("---")
    if st.button("🔍 Verifica Disponibilità Dati Storici"):
        with st.spinner(f"Ricerca dati storici per {ticker} dal {start_date} al {end_date}..."):
            df_check = fetch_data_smart(ticker, timeframe, start_date, end_date)
            
            if not df_check.empty:
                # Check actual date range
                min_date = df_check['datetime'].min().date()
                max_date = df_check['datetime'].max().date()
                count = len(df_check)
                
                st.success(f"✅ Dati Trovati! {count} candele disponibili.")
                st.info(f"📅 Range Disponibile: {min_date} -> {max_date}")
                
                if min_date > start_date:
                    st.warning(f"⚠️ Attenzione: I dati iniziano dal {min_date}, successivi alla data richiesta {start_date}.")
                
                st.session_state.backtest_data = df_check
                st.session_state.backtest_ticker = ticker
            else:
                st.error(f"❌ Nessun dato trovato per {ticker} nel range selezionato. Prova a cambiare date o ticker.")
                st.session_state.backtest_data = None

    # --- BACKTESTING ENGINE & VISUALIZER ---
    class TechnicalIndicators:
        # --- TREND ---
        @staticmethod
        def sma(series, period): return series.rolling(period).mean()
        @staticmethod
        def ema(series, period): return series.ewm(span=period, adjust=False).mean()
        @staticmethod
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        @staticmethod
        def hma(series, period):
            half_length = int(period / 2)
            sqrt_length = int(np.sqrt(period))
            wmaf = TechnicalIndicators.wma(series, half_length)
            wmas = TechnicalIndicators.wma(series, period)
            return TechnicalIndicators.wma(2 * wmaf - wmas, sqrt_length)
        @staticmethod
        def tema(series, period):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            ema3 = TechnicalIndicators.ema(ema2, period)
            return 3 * ema1 - 3 * ema2 + ema3
        @staticmethod
        def dema(series, period):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            return 2 * ema1 - ema2
        @staticmethod
        def kama(series, period=10, pow1=2, pow2=30):
            change = abs(series - series.shift(period))
            volatility = series.diff().abs().rolling(window=period).sum()
            er = change / volatility
            sc = (er * (2.0 / (pow1 + 1) - 2.0 / (pow2 + 1)) + 2.0 / (pow2 + 1)) ** 2
            kama = [series.values[period-1]]
            for i in range(period, len(series)):
                kama.append(kama[-1] + sc.values[i] * (series.values[i] - kama[-1]))
            return pd.Series(kama, index=series.index[period-1:])

        @staticmethod
        def macd(series, fast=12, slow=26, signal=9):
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        
        @staticmethod
        def adx(df, period=14):
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = TechnicalIndicators.atr(df, period)
            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            return dx.ewm(alpha=1/period).mean()

        @staticmethod
        def aroon(df, period=25):
            aroon_up = 100 * df['High'].rolling(period + 1).apply(lambda x: x.argmax()) / period
            aroon_down = 100 * df['Low'].rolling(period + 1).apply(lambda x: x.argmin()) / period
            return aroon_up, aroon_down

        @staticmethod
        def cci(df, period=20):
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (tp - sma) / (0.015 * mad)

        @staticmethod
        def supertrend(df, period=10, multiplier=3):
            hl2 = (df['High'] + df['Low']) / 2
            atr = TechnicalIndicators.atr(df, period)
            upper = hl2 + (multiplier * atr)
            lower = hl2 - (multiplier * atr)
            return upper, lower

        @staticmethod
        def parabolic_sar(df, af=0.02, max_af=0.2):
            high, low = df['High'], df['Low']
            sar = [low[0]]
            ep = high[0]
            acc = af
            trend = 1 # 1 for long, -1 for short
            
            for i in range(1, len(df)):
                prev_sar = sar[-1]
                if trend == 1:
                    curr_sar = prev_sar + acc * (ep - prev_sar)
                    curr_sar = min(curr_sar, low[i-1])
                    if i > 1: curr_sar = min(curr_sar, low[i-2])
                    
                    if low[i] < curr_sar:
                        trend = -1
                        curr_sar = ep
                        ep = low[i]
                        acc = af
                    else:
                        if high[i] > ep:
                            ep = high[i]
                            acc = min(acc + af, max_af)
                else:
                    curr_sar = prev_sar + acc * (ep - prev_sar)
                    curr_sar = max(curr_sar, high[i-1])
                    if i > 1: curr_sar = max(curr_sar, high[i-2])
                    
                    if high[i] > curr_sar:
                        trend = 1
                        curr_sar = ep
                        ep = high[i]
                        acc = af
                    else:
                        if low[i] < ep:
                            ep = low[i]
                            acc = min(acc + af, max_af)
                sar.append(curr_sar)
            return pd.Series(sar, index=df.index)

        # --- MOMENTUM ---
        @staticmethod
        def rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        @staticmethod
        def stochastic(df, k_period=14, d_period=3):
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            return k, d

        @staticmethod
        def williams_r(df, period=14):
            highest_high = df['High'].rolling(period).max()
            lowest_low = df['Low'].rolling(period).min()
            return ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100

        @staticmethod
        def roc(series, period=12):
            return ((series - series.shift(period)) / series.shift(period)) * 100
        
        @staticmethod
        def tsi(series, r=25, s=13):
            m = series.diff()
            m1 = m.ewm(span=r).mean().ewm(span=s).mean()
            m2 = abs(m).ewm(span=r).mean().ewm(span=s).mean()
            return 100 * (m1 / m2)

        @staticmethod
        def uo(df, p1=7, p2=14, p3=28):
            bp = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
            tr = TechnicalIndicators.atr(df, 1) # True Range 1-period approximation
            avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
            avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
            avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
            return 100 * (4*avg1 + 2*avg2 + avg3) / 7

        # --- VOLATILITY ---
        @staticmethod
        def bollinger_bands(series, period=20, std_dev=2):
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            return sma + (std * std_dev), sma - (std * std_dev)

        @staticmethod
        def atr(df, period=14):
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            return true_range.rolling(period).mean()

        @staticmethod
        def keltner_channels(df, period=20, mult=2):
            ema = TechnicalIndicators.ema(df['Close'], period)
            atr = TechnicalIndicators.atr(df, 10)
            return ema + (mult * atr), ema - (mult * atr)

        @staticmethod
        def donchian_channels(df, period=20):
            return df['High'].rolling(period).max(), df['Low'].rolling(period).min()

        @staticmethod
        def chaikin_volatility(df, period=10, roc_period=10):
            hl = df['High'] - df['Low']
            ema_hl = TechnicalIndicators.ema(hl, period)
            return TechnicalIndicators.roc(ema_hl, roc_period)

        # --- VOLUME ---
        @staticmethod
        def obv(df):
            return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        @staticmethod
        def mfi(df, period=14):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
            return 100 - (100 / (1 + positive_flow / negative_flow))

        @staticmethod
        def cmf(df, period=20):
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_volume = mf_multiplier * df['Volume']
            return mf_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()

        @staticmethod
        def vwap(df):
            v = df['Volume'].values
            tp = (df['High'] + df['Low'] + df['Close']).values / 3
            return df.assign(vwap=(tp * v).cumsum() / v.cumsum())['vwap']
        
        @staticmethod
        def ad_line(df):
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            return (clv * df['Volume']).cumsum()

        # --- NEW INDICATORS (BATCH 2) ---
        @staticmethod
        def vortex(df, period=14):
            tr = TechnicalIndicators.atr(df, 1).rolling(period).sum()
            vm_plus = abs(df['High'] - df['Low'].shift(1)).rolling(period).sum()
            vm_minus = abs(df['Low'] - df['High'].shift(1)).rolling(period).sum()
            return vm_plus / tr, vm_minus / tr

        @staticmethod
        def chop(df, period=14):
            tr = TechnicalIndicators.atr(df, 1).rolling(period).sum()
            r = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
            return 100 * np.log10(tr / r) / np.log10(period)

        @staticmethod
        def kst(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15):
            roc1 = TechnicalIndicators.roc(df['Close'], r1).rolling(n1).mean()
            roc2 = TechnicalIndicators.roc(df['Close'], r2).rolling(n2).mean()
            roc3 = TechnicalIndicators.roc(df['Close'], r3).rolling(n3).mean()
            roc4 = TechnicalIndicators.roc(df['Close'], r4).rolling(n4).mean()
            return (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)

        @staticmethod
        def coppock(df, wma_period=10, roc1=14, roc2=11):
            roc_sum = TechnicalIndicators.roc(df['Close'], roc1) + TechnicalIndicators.roc(df['Close'], roc2)
            return roc_sum.ewm(span=wma_period).mean()

        @staticmethod
        def ichimoku(df):
            nine_period_high = df['High'].rolling(window=9).max()
            nine_period_low = df['Low'].rolling(window=9).min()
            tenkan_sen = (nine_period_high + nine_period_low) / 2

            period26_high = df['High'].rolling(window=26).max()
            period26_low = df['Low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2

            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            period52_high = df['High'].rolling(window=52).max()
            period52_low = df['Low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

            chikou_span = df['Close'].shift(-26)

            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        
        @staticmethod
        def ao(df):
            mp = (df['High'] + df['Low']) / 2
            return mp.rolling(5).mean() - mp.rolling(34).mean()

        @staticmethod
        def ppo(df, fast=12, slow=26):
            fast_ema = TechnicalIndicators.ema(df['Close'], fast)
            slow_ema = TechnicalIndicators.ema(df['Close'], slow)
            return ((fast_ema - slow_ema) / slow_ema) * 100

        @staticmethod
        def mass_index(df, period=25, ema_period=9):
            high_low = df['High'] - df['Low']
            ema1 = high_low.ewm(span=ema_period).mean()
            ema2 = ema1.ewm(span=ema_period).mean()
            ratio = ema1 / ema2
            return ratio.rolling(period).sum()

        @staticmethod
        def ulcer_index(df, period=14):
            close = df['Close']
            max_close = close.rolling(period).max()
            drawdown = 100 * ((close - max_close) / max_close)
            sq_drawdown = drawdown ** 2
            return np.sqrt(sq_drawdown.rolling(period).mean())

        # --- NEW INDICATORS (BATCH 3) ---
        @staticmethod
        def wma(series, period=9):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        @staticmethod
        def trima(series, period=18):
            return TechnicalIndicators.sma(TechnicalIndicators.sma(series, period), period) # Approx

        @staticmethod
        def cmo(series, period=14):
            # Chande Momentum Oscillator
            diff = series.diff()
            pos = diff.where(diff > 0, 0)
            neg = abs(diff.where(diff < 0, 0))
            sum_pos = pos.rolling(period).sum()
            sum_neg = neg.rolling(period).sum()
            return 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)

        @staticmethod
        def mom(series, period=10):
            return series.diff(period)

        @staticmethod
        def bop(df):
            # Balance of Power
            return (df['Close'] - df['Open']) / (df['High'] - df['Low'])

        @staticmethod
        def trix(series, period=15):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            ema3 = TechnicalIndicators.ema(ema2, period)
            return ema3.pct_change() * 100

        @staticmethod
        def stochrsi(series, period=14):
            rsi = TechnicalIndicators.rsi(series, period)
            min_rsi = rsi.rolling(period).min()
            max_rsi = rsi.rolling(period).max()
            return (rsi - min_rsi) / (max_rsi - min_rsi)

        @staticmethod
        def stddev(series, period=20):
            return series.rolling(period).std()

        @staticmethod
        def tsf(series, period=14):
            # Time Series Forecast (Linear Regression Forecast)
            # Simplified: Linear Reg at end point
            x = np.arange(period)
            def linreg_pred(y):
                if len(y) < period: return np.nan
                slope, intercept, _, _, _ = stats.linregress(x, y)
                return slope * (period - 1) + intercept
            return series.rolling(period).apply(linreg_pred, raw=True)

        @staticmethod
        def process_indicator(data):
            """Standardize input into clean float32 array"""
            if isinstance(data, pd.Series):
                return data.astype('float32').fillna(0).values
            elif isinstance(data, pd.DataFrame):
                return data.astype('float32').fillna(0).values
            elif isinstance(data, np.ndarray):
                return np.nan_to_num(data.astype('float32'))
            return np.array(data, dtype='float32')

        @classmethod
        def get_registry(cls):
            return {
                "RSI": {"func": cls.rsi, "params": {"period": 14}, "input": "Close", "outputs": ["RSI"]},
                "MACD": {"func": cls.macd, "params": {"fast": 12, "slow": 26, "signal": 9}, "input": "Close", "outputs": ["MACD", "MACD_Signal"]},
                "Bollinger": {"func": cls.bollinger_bands, "params": {"period": 20, "std_dev": 2}, "input": "Close", "outputs": ["BB_Upper", "BB_Lower"]},
                "ATR": {"func": cls.atr, "params": {"period": 14}, "input": "df", "outputs": ["ATR"]},
                "SMA200": {"func": cls.sma, "params": {"period": 200}, "input": "Close", "outputs": ["SMA200"]},
                "SMA50": {"func": cls.sma, "params": {"period": 50}, "input": "Close", "outputs": ["SMA50"]},
                "EMA20": {"func": cls.ema, "params": {"period": 20}, "input": "Close", "outputs": ["EMA20"]},
                "Stoch_K": {"func": cls.stochastic, "params": {"k_period": 14, "d_period": 3}, "input": "df", "outputs": ["Stoch_K", "Stoch_D"]},
                "ADX": {"func": cls.adx, "params": {"period": 14}, "input": "df", "outputs": ["ADX"]},
                "CCI": {"func": cls.cci, "params": {"period": 20}, "input": "df", "outputs": ["CCI"]},
                "WilliamsR": {"func": cls.williams_r, "params": {"period": 14}, "input": "df", "outputs": ["WilliamsR"]},
                "ROC": {"func": cls.roc, "params": {"period": 12}, "input": "Close", "outputs": ["ROC"]},
                "OBV": {"func": cls.obv, "params": {}, "input": "df", "outputs": ["OBV"]},
                "MFI": {"func": cls.mfi, "params": {"period": 14}, "input": "df", "outputs": ["MFI"]},
                "HMA20": {"func": cls.hma, "params": {"period": 20}, "input": "Close", "outputs": ["HMA20"]},
                "TEMA20": {"func": cls.tema, "params": {"period": 20}, "input": "Close", "outputs": ["TEMA20"]},
                "DEMA20": {"func": cls.dema, "params": {"period": 20}, "input": "Close", "outputs": ["DEMA20"]},
                "KAMA20": {"func": cls.kama, "params": {"period": 20}, "input": "Close", "outputs": ["KAMA20"]},
                "Aroon": {"func": cls.aroon, "params": {"period": 25}, "input": "df", "outputs": ["Aroon_Up", "Aroon_Down"]},
                "SuperTrend": {"func": cls.supertrend, "params": {"period": 10, "multiplier": 3}, "input": "df", "outputs": ["SuperTrend_Upper", "SuperTrend_Lower"]},
                "Parabolic_SAR": {"func": cls.parabolic_sar, "params": {}, "input": "df", "outputs": ["Parabolic_SAR"]},
                "TSI": {"func": cls.tsi, "params": {}, "input": "Close", "outputs": ["TSI"]},
                "UO": {"func": cls.uo, "params": {}, "input": "df", "outputs": ["UO"]},
                "KC": {"func": cls.keltner_channels, "params": {}, "input": "df", "outputs": ["KC_Upper", "KC_Lower"]},
                "DC": {"func": cls.donchian_channels, "params": {}, "input": "df", "outputs": ["DC_Upper", "DC_Lower"]},
                "Chaikin_Vol": {"func": cls.chaikin_volatility, "params": {}, "input": "df", "outputs": ["Chaikin_Vol"]},
                "CMF": {"func": cls.cmf, "params": {}, "input": "df", "outputs": ["CMF"]},
                "VWAP": {"func": cls.vwap, "params": {}, "input": "df", "outputs": ["VWAP"]},
                "AD_Line": {"func": cls.ad_line, "params": {}, "input": "df", "outputs": ["AD_Line"]},
                "Vortex": {"func": cls.vortex, "params": {}, "input": "df", "outputs": ["Vortex_Plus", "Vortex_Minus"]},
                "Chop": {"func": cls.chop, "params": {}, "input": "df", "outputs": ["Chop_Index"]},
                "KST": {"func": cls.kst, "params": {}, "input": "df", "outputs": ["KST"]},
                "Coppock": {"func": cls.coppock, "params": {}, "input": "df", "outputs": ["Coppock"]},
                "Ichimoku": {"func": cls.ichimoku, "params": {}, "input": "df", "outputs": ["Tenkan", "Kijun", "SpanA", "SpanB", "Chikou"]},
                "AO": {"func": cls.ao, "params": {}, "input": "df", "outputs": ["AO"]},
                "PPO": {"func": cls.ppo, "params": {}, "input": "df", "outputs": ["PPO"]},
                "Mass_Index": {"func": cls.mass_index, "params": {}, "input": "df", "outputs": ["Mass_Index"]},
                "Ulcer_Index": {"func": cls.ulcer_index, "params": {}, "input": "df", "outputs": ["Ulcer_Index"]},
                "WMA20": {"func": cls.wma, "params": {"period": 20}, "input": "Close", "outputs": ["WMA20"]},
                "TRIMA20": {"func": cls.trima, "params": {"period": 20}, "input": "Close", "outputs": ["TRIMA20"]},
                "CMO": {"func": cls.cmo, "params": {}, "input": "Close", "outputs": ["CMO"]},
                "MOM10": {"func": cls.mom, "params": {"period": 10}, "input": "Close", "outputs": ["MOM10"]},
                "BOP": {"func": cls.bop, "params": {}, "input": "df", "outputs": ["BOP"]},
                "TRIX": {"func": cls.trix, "params": {}, "input": "Close", "outputs": ["TRIX"]},
                "StochRSI": {"func": cls.stochrsi, "params": {}, "input": "Close", "outputs": ["StochRSI"]},
                "STDDEV": {"func": cls.stddev, "params": {}, "input": "Close", "outputs": ["STDDEV"]},
                "TSF": {"func": cls.tsf, "params": {}, "input": "Close", "outputs": ["TSF"]},
            }

    class StrategyLib:
        @staticmethod
        def get_signal_func(strategy_name):
            strategies = {
                "RSI Mean Reversion": StrategyLib.rsi_mean_reversion,
                "MACD Crossover": StrategyLib.macd_crossover,
                "Bollinger Breakout": StrategyLib.bollinger_breakout,
                "Golden/Death Cross": StrategyLib.golden_death_cross,
                "Stochastic Oscillator": StrategyLib.stochastic_oscillator,
                "CCI Momentum": StrategyLib.cci_momentum,
                "Williams %R Reversal": StrategyLib.williams_r_reversal,
                "HMA Trend": StrategyLib.hma_trend,
                "TEMA Crossover": StrategyLib.tema_crossover,
                "KAMA Trend": StrategyLib.kama_trend,
                "Aroon Oscillator": StrategyLib.aroon_oscillator,
                "SuperTrend Reversal": StrategyLib.supertrend_reversal,
                "Parabolic SAR": StrategyLib.parabolic_sar_strategy,
                "TSI Crossover": StrategyLib.tsi_crossover,
                "UO Overbought/Oversold": StrategyLib.uo_strategy,
                "Keltner Channel Breakout": StrategyLib.keltner_channel_breakout,
                "Donchian Channel Breakout": StrategyLib.donchian_channel_breakout,
                "Chaikin Volatility": StrategyLib.chaikin_volatility_strategy,
                "CMF Trend": StrategyLib.cmf_trend,
                "VWAP Crossover": StrategyLib.vwap_crossover,
                "AD Line Trend": StrategyLib.ad_line_trend,
                "Vortex Crossover": StrategyLib.vortex_crossover,
                "Choppiness Index Breakout": StrategyLib.choppiness_index_breakout,
                "KST Crossover": StrategyLib.kst_crossover,
                "Coppock Curve": StrategyLib.coppock_curve,
                "Ichimoku Cloud Breakout": StrategyLib.ichimoku_cloud_breakout,
                "Awesome Oscillator": StrategyLib.awesome_oscillator,
                "PPO Crossover": StrategyLib.ppo_crossover,
                "Mass Index Reversal": StrategyLib.mass_index_reversal,
                "Ulcer Index Safety": StrategyLib.ulcer_index_safety,
                "WMA Trend": StrategyLib.wma_trend,
                "TRIMA Crossover": StrategyLib.trima_crossover,
                "CMO Reversal": StrategyLib.cmo_reversal,
                "Momentum Breakout": StrategyLib.momentum_breakout,
                "BOP Trend": StrategyLib.bop_trend,
                "TRIX Crossover": StrategyLib.trix_crossover,
                "StochRSI Reversal": StrategyLib.stochrsi_reversal,
                "TSF Trend": StrategyLib.tsf_trend,
            }
            return strategies.get(strategy_name)

        @staticmethod
        def rsi_mean_reversion(df, params, cache=None):
            p = params.get('period', 14)
            if cache is not None and ('RSI', p) in cache:
                rsi = cache[('RSI', p)]
            else:
                rsi = TechnicalIndicators.rsi(df['Close'], p)
                if cache is not None: cache[('RSI', p)] = rsi
            
            prev = rsi.shift(1)
            curr = rsi
            os, ob = params.get('os', 30), params.get('ob', 70)
            long_sig = (prev < os) & (curr > os)
            short_sig = (prev > ob) & (curr < ob)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def macd_crossover(df, params, cache=None):
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            sig = params.get('signal', 9)
            
            key = ('MACD', fast, slow, sig)
            if cache is not None and key in cache:
                macd, signal_line = cache[key]
            else:
                macd, signal_line = TechnicalIndicators.macd(df['Close'], fast, slow, sig)
                if cache is not None: cache[key] = (macd, signal_line)

            long_sig = (macd.shift(1) < signal_line.shift(1)) & (macd > signal_line)
            short_sig = (macd.shift(1) > signal_line.shift(1)) & (macd < signal_line)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def bollinger_breakout(df, params, cache=None):
            p = params.get('period', 20)
            std = params.get('std_dev', 2)
            
            key = ('BB', p, std)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.bollinger_bands(df['Close'], p, std)
                if cache is not None: cache[key] = (upper, lower)

            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_lower, prev_upper = lower.shift(1), upper.shift(1)
            long_sig = (prev_close < prev_lower) & (curr_close > prev_lower)
            short_sig = (prev_close > prev_upper) & (curr_close < prev_upper)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def golden_death_cross(df, params, cache=None):
            fast_p = params.get('fast', 50)
            slow_p = params.get('slow', 200)
            
            key_fast = ('SMA', fast_p)
            if cache is not None and key_fast in cache:
                fast_ma = cache[key_fast]
            else:
                fast_ma = TechnicalIndicators.sma(df['Close'], fast_p)
                if cache is not None: cache[key_fast] = fast_ma
                
            key_slow = ('SMA', slow_p)
            if cache is not None and key_slow in cache:
                slow_ma = cache[key_slow]
            else:
                slow_ma = TechnicalIndicators.sma(df['Close'], slow_p)
                if cache is not None: cache[key_slow] = slow_ma

            long_sig = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)
            short_sig = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def stochastic_oscillator(df, params, cache=None):
            k_p = params.get('k_period', 14)
            d_p = params.get('d_period', 3)
            
            key = ('Stoch', k_p, d_p)
            if cache is not None and key in cache:
                k, d = cache[key]
            else:
                k, d = TechnicalIndicators.stochastic(df, k_p, d_p)
                if cache is not None: cache[key] = (k, d)

            prev_k, curr_k = k.shift(1), k
            os, ob = params.get('os', 20), params.get('ob', 80)
            long_sig = (prev_k < os) & (curr_k > os)
            short_sig = (prev_k > ob) & (curr_k < ob)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cci_momentum(df, params, cache=None):
            p = params.get('period', 20)
            key = ('CCI', p)
            if cache is not None and key in cache:
                cci = cache[key]
            else:
                cci = TechnicalIndicators.cci(df, p)
                if cache is not None: cache[key] = cci
                
            prev_cci, curr_cci = cci.shift(1), cci
            long_sig = (prev_cci < -100) & (curr_cci > -100)
            short_sig = (prev_cci > 100) & (curr_cci < 100)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def williams_r_reversal(df, params, cache=None):
            p = params.get('period', 14)
            key = ('WilliamsR', p)
            if cache is not None and key in cache:
                wr = cache[key]
            else:
                wr = TechnicalIndicators.williams_r(df, p)
                if cache is not None: cache[key] = wr
                
            prev_wr, curr_wr = wr.shift(1), wr
            long_sig = (prev_wr < -80) & (curr_wr > -80)
            short_sig = (prev_wr > -20) & (curr_wr < -20)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def hma_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('HMA', p)
            if cache is not None and key in cache:
                hma = cache[key]
            else:
                hma = TechnicalIndicators.hma(df['Close'], p)
                if cache is not None: cache[key] = hma
                
            long_sig = (hma.shift(1) > hma.shift(2)) & (df['Close'].shift(1) < hma.shift(1)) & (df['Close'] > hma)
            short_sig = (hma.shift(1) < hma.shift(2)) & (df['Close'].shift(1) > hma.shift(1)) & (df['Close'] < hma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tema_crossover(df, params, cache=None):
            p = params.get('period', 20)
            key = ('TEMA', p)
            if cache is not None and key in cache:
                tema = cache[key]
            else:
                tema = TechnicalIndicators.tema(df['Close'], p)
                if cache is not None: cache[key] = tema
                
            long_sig = (df['Close'].shift(1) < tema.shift(1)) & (df['Close'] > tema)
            short_sig = (df['Close'].shift(1) > tema.shift(1)) & (df['Close'] < tema)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def kama_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('KAMA', p)
            if cache is not None and key in cache:
                kama = cache[key]
            else:
                kama = TechnicalIndicators.kama(df['Close'], p)
                if cache is not None: cache[key] = kama
                
            prev_kama, curr_kama = kama.shift(1), kama
            long_sig = prev_kama < curr_kama
            short_sig = prev_kama > curr_kama
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def aroon_oscillator(df, params, cache=None):
            p = params.get('period', 25)
            key = ('Aroon', p)
            if cache is not None and key in cache:
                aroon_up, aroon_down = cache[key]
            else:
                aroon_up, aroon_down = TechnicalIndicators.aroon(df, p)
                if cache is not None: cache[key] = (aroon_up, aroon_down)
                
            prev_up, curr_up = aroon_up.shift(1), aroon_up
            prev_down, curr_down = aroon_down.shift(1), aroon_down
            long_sig = (prev_up < prev_down) & (curr_up > curr_down)
            short_sig = (prev_up > prev_down) & (curr_up < curr_down)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def supertrend_reversal(df, params, cache=None):
            p = params.get('period', 10)
            m = params.get('multiplier', 3)
            key = ('SuperTrend', p, m)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.supertrend(df, p, m)
                if cache is not None: cache[key] = (upper, lower)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_upper, curr_upper = upper.shift(1), upper
            prev_lower, curr_lower = lower.shift(1), lower
            long_sig = (prev_close < prev_upper) & (curr_close > curr_lower)
            short_sig = (prev_close > prev_lower) & (curr_close < curr_upper)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def parabolic_sar_strategy(df, params, cache=None):
            key = ('Parabolic_SAR',)
            if cache is not None and key in cache:
                sar = cache[key]
            else:
                sar = TechnicalIndicators.parabolic_sar(df)
                if cache is not None: cache[key] = sar
                
            prev_sar, curr_sar = sar.shift(1), sar
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_sar > prev_close) & (curr_sar < curr_close)
            short_sig = (prev_sar < prev_close) & (curr_sar > curr_close)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tsi_crossover(df, params, cache=None):
            key = ('TSI',)
            if cache is not None and key in cache:
                tsi = cache[key]
            else:
                tsi = TechnicalIndicators.tsi(df['Close'])
                if cache is not None: cache[key] = tsi
                
            prev_tsi, curr_tsi = tsi.shift(1), tsi
            long_sig = (prev_tsi < 0) & (curr_tsi > 0)
            short_sig = (prev_tsi > 0) & (curr_tsi < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def uo_strategy(df, params, cache=None):
            key = ('UO',)
            if cache is not None and key in cache:
                uo = cache[key]
            else:
                uo = TechnicalIndicators.uo(df)
                if cache is not None: cache[key] = uo
                
            prev_uo, curr_uo = uo.shift(1), uo
            long_sig = (prev_uo < 30) & (curr_uo > 30)
            short_sig = (prev_uo > 70) & (curr_uo < 70)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def keltner_channel_breakout(df, params, cache=None):
            key = ('KC',)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.keltner_channels(df)
                if cache is not None: cache[key] = (upper, lower)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_upper, curr_upper = upper.shift(1), upper
            prev_lower, curr_lower = lower.shift(1), lower
            long_sig = (prev_close < prev_upper) & (curr_close > curr_upper)
            short_sig = (prev_close > prev_lower) & (curr_close < curr_lower)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def donchian_channel_breakout(df, params, cache=None):
            key = ('DC',)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.donchian_channels(df)
                if cache is not None: cache[key] = (upper, lower)
                
            curr_close = df['Close']
            prev_upper = upper.shift(1)
            prev_lower = lower.shift(1)
            long_sig = curr_close >= prev_upper
            short_sig = curr_close <= prev_lower
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def chaikin_volatility_strategy(df, params, cache=None):
            key = ('Chaikin_Vol',)
            if cache is not None and key in cache:
                cv = cache[key]
            else:
                cv = TechnicalIndicators.chaikin_volatility(df)
                if cache is not None: cache[key] = cv
                
            prev_cv, curr_cv = cv.shift(1), cv
            long_sig = (prev_cv < 0) & (curr_cv > 0)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cmf_trend(df, params, cache=None):
            key = ('CMF',)
            if cache is not None and key in cache:
                cmf = cache[key]
            else:
                cmf = TechnicalIndicators.cmf(df)
                if cache is not None: cache[key] = cmf
                
            prev_cmf, curr_cmf = cmf.shift(1), cmf
            long_sig = (prev_cmf < 0) & (curr_cmf > 0)
            short_sig = (prev_cmf > 0) & (curr_cmf < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def vwap_crossover(df, params, cache=None):
            key = ('VWAP',)
            if cache is not None and key in cache:
                vwap = cache[key]
            else:
                vwap = TechnicalIndicators.vwap(df)
                if cache is not None: cache[key] = vwap
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_vwap, curr_vwap = vwap.shift(1), vwap
            long_sig = (prev_close < prev_vwap) & (curr_close > curr_vwap)
            short_sig = (prev_close > prev_vwap) & (curr_close < curr_vwap)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ad_line_trend(df, params, cache=None):
            key = ('AD_Line',)
            if cache is not None and key in cache:
                ad = cache[key]
            else:
                ad = TechnicalIndicators.ad_line(df)
                if cache is not None: cache[key] = ad
                
            prev_ad, curr_ad = ad.shift(1), ad
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_ad < curr_ad) & (prev_close > curr_close)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def vortex_crossover(df, params, cache=None):
            key = ('Vortex',)
            if cache is not None and key in cache:
                vp, vm = cache[key]
            else:
                vp, vm = TechnicalIndicators.vortex(df)
                if cache is not None: cache[key] = (vp, vm)
                
            prev_vp, curr_vp = vp.shift(1), vp
            prev_vm, curr_vm = vm.shift(1), vm
            long_sig = (prev_vp < prev_vm) & (curr_vp > curr_vm)
            short_sig = (prev_vp > prev_vm) & (curr_vp < curr_vm)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def choppiness_index_breakout(df, params, cache=None):
            key = ('Chop',)
            if cache is not None and key in cache:
                chop = cache[key]
            else:
                chop = TechnicalIndicators.chop(df)
                if cache is not None: cache[key] = chop
                
            prev_chop, curr_chop = chop.shift(1), chop
            long_sig = (prev_chop > 61.8) & (curr_chop < 61.8)
            short_sig = (prev_chop < 38.2) & (curr_chop > 38.2)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def kst_crossover(df, params, cache=None):
            key = ('KST',)
            if cache is not None and key in cache:
                kst = cache[key]
            else:
                kst = TechnicalIndicators.kst(df)
                if cache is not None: cache[key] = kst
                
            prev_kst, curr_kst = kst.shift(1), kst
            long_sig = (prev_kst < 0) & (curr_kst > 0)
            short_sig = (prev_kst > 0) & (curr_kst < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def coppock_curve(df, params, cache=None):
            key = ('Coppock',)
            if cache is not None and key in cache:
                cop = cache[key]
            else:
                cop = TechnicalIndicators.coppock(df)
                if cache is not None: cache[key] = cop
                
            prev_cop, curr_cop = cop.shift(1), cop
            long_sig = (prev_cop < 0) & (curr_cop > 0)
            short_sig = (prev_cop > 0) & (curr_cop < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ichimoku_cloud_breakout(df, params, cache=None):
            key = ('Ichimoku',)
            if cache is not None and key in cache:
                tenkan, kijun, span_a, span_b, chikou = cache[key]
            else:
                tenkan, kijun, span_a, span_b, chikou = TechnicalIndicators.ichimoku(df)
                if cache is not None: cache[key] = (tenkan, kijun, span_a, span_b, chikou)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_span_a, curr_span_a = span_a.shift(1), span_a
            curr_span_b = span_b
            long_sig = (prev_close < prev_span_a) & (curr_close > curr_span_a) & (curr_close > curr_span_b)
            short_sig = (prev_close > prev_span_a) & (curr_close < curr_span_a) & (curr_close < curr_span_b)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def awesome_oscillator(df, params, cache=None):
            key = ('AO',)
            if cache is not None and key in cache:
                ao = cache[key]
            else:
                ao = TechnicalIndicators.ao(df)
                if cache is not None: cache[key] = ao
                
            prev_ao, curr_ao = ao.shift(1), ao
            long_sig = (prev_ao < 0) & (curr_ao > 0)
            short_sig = (prev_ao > 0) & (curr_ao < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ppo_crossover(df, params, cache=None):
            key = ('PPO',)
            if cache is not None and key in cache:
                ppo = cache[key]
            else:
                ppo = TechnicalIndicators.ppo(df)
                if cache is not None: cache[key] = ppo
                
            prev_ppo, curr_ppo = ppo.shift(1), ppo
            long_sig = (prev_ppo < 0) & (curr_ppo > 0)
            short_sig = (prev_ppo > 0) & (curr_ppo < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def mass_index_reversal(df, params, cache=None):
            key = ('Mass_Index',)
            if cache is not None and key in cache:
                mi = cache[key]
            else:
                mi = TechnicalIndicators.mass_index(df)
                if cache is not None: cache[key] = mi
                
            prev_mi, curr_mi = mi.shift(1), mi
            long_sig = (prev_mi > 27) & (curr_mi < 27)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ulcer_index_safety(df, params, cache=None):
            key = ('Ulcer_Index',)
            if cache is not None and key in cache:
                ui = cache[key]
            else:
                ui = TechnicalIndicators.ulcer_index(df)
                if cache is not None: cache[key] = ui
                
            prev_ui, curr_ui = ui.shift(1), ui
            long_sig = (prev_ui > 5) & (curr_ui < 5)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def wma_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('WMA', p)
            if cache is not None and key in cache:
                wma = cache[key]
            else:
                wma = TechnicalIndicators.wma(df['Close'], p)
                if cache is not None: cache[key] = wma
                
            prev_wma, curr_wma = wma.shift(1), wma
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_wma < curr_wma) & (prev_close > curr_wma)
            short_sig = (prev_wma > curr_wma) & (prev_close < curr_wma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def trima_crossover(df, params, cache=None):
            p = params.get('period', 20)
            key = ('TRIMA', p)
            if cache is not None and key in cache:
                trima = cache[key]
            else:
                trima = TechnicalIndicators.trima(df['Close'], p)
                if cache is not None: cache[key] = trima
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_trima, curr_trima = trima.shift(1), trima
            long_sig = (prev_close < prev_trima) & (curr_close > curr_trima)
            short_sig = (prev_close > prev_trima) & (curr_close < curr_trima)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cmo_reversal(df, params, cache=None):
            key = ('CMO',)
            if cache is not None and key in cache:
                cmo = cache[key]
            else:
                cmo = TechnicalIndicators.cmo(df['Close'])
                if cache is not None: cache[key] = cmo
                
            prev_cmo, curr_cmo = cmo.shift(1), cmo
            long_sig = (prev_cmo < -50) & (curr_cmo > -50)
            short_sig = (prev_cmo > 50) & (curr_cmo < 50)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def momentum_breakout(df, params, cache=None):
            p = params.get('period', 10)
            key = ('MOM', p)
            if cache is not None and key in cache:
                mom = cache[key]
            else:
                mom = TechnicalIndicators.mom(df['Close'], p)
                if cache is not None: cache[key] = mom
                
            prev_mom, curr_mom = mom.shift(1), mom
            long_sig = (prev_mom < 0) & (curr_mom > 0)
            short_sig = (prev_mom > 0) & (curr_mom < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def bop_trend(df, params, cache=None):
            key = ('BOP',)
            if cache is not None and key in cache:
                bop = cache[key]
            else:
                bop = TechnicalIndicators.bop(df)
                if cache is not None: cache[key] = bop
                
            prev_bop, curr_bop = bop.shift(1), bop
            long_sig = (prev_bop < 0) & (curr_bop > 0)
            short_sig = (prev_bop > 0) & (curr_bop < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def trix_crossover(df, params, cache=None):
            key = ('TRIX',)
            if cache is not None and key in cache:
                trix = cache[key]
            else:
                trix = TechnicalIndicators.trix(df['Close'])
                if cache is not None: cache[key] = trix
                
            prev_trix, curr_trix = trix.shift(1), trix
            long_sig = (prev_trix < 0) & (curr_trix > 0)
            short_sig = (prev_trix > 0) & (curr_trix < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def stochrsi_reversal(df, params, cache=None):
            key = ('StochRSI',)
            if cache is not None and key in cache:
                srsi = cache[key]
            else:
                srsi = TechnicalIndicators.stochrsi(df['Close'])
                if cache is not None: cache[key] = srsi
                
            prev_srsi, curr_srsi = srsi.shift(1), srsi
            long_sig = (prev_srsi < 0.2) & (curr_srsi > 0.2)
            short_sig = (prev_srsi > 0.8) & (curr_srsi < 0.8)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tsf_trend(df, params, cache=None):
            key = ('TSF',)
            if cache is not None and key in cache:
                tsf = cache[key]
            else:
                tsf = TechnicalIndicators.tsf(df['Close'])
                if cache is not None: cache[key] = tsf
                
            prev_tsf, curr_tsf = tsf.shift(1), tsf
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_tsf < curr_tsf) & (prev_close > curr_tsf)
            short_sig = (prev_tsf > curr_tsf) & (prev_close < curr_tsf)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

    class BacktestEngine:
        def __init__(self, ticker, start_date, end_date, timeframe, initial_capital=10000):
            self.ticker = ticker
            self.start_date = start_date
            self.end_date = end_date
            self.timeframe = timeframe
            self.initial_capital = initial_capital
            self.df = pd.DataFrame()

        def fetch_data(self):
            self.df = fetch_data_smart(self.ticker, self.timeframe, self.start_date, self.end_date)
            return not self.df.empty

        def add_technical_indicators(self):
            if self.df.empty: return
            
            registry = TechnicalIndicators.get_registry()
            
            for name, config in registry.items():
                try:
                    # Prepare Input
                    if config['input'] == 'df':
                        data = self.df
                    else:
                        data = self.df[config['input']]
                    
                    # Execute Calculation
                    result = config['func'](data, **config['params'])
                    
                    # Handle Outputs
                    if isinstance(result, tuple):
                        for i, out_col in enumerate(config['outputs']):
                            if i < len(result):
                                self.df[out_col] = result[i]
                    else:
                        if config['outputs']:
                            self.df[config['outputs'][0]] = result
                            
                except Exception as e:
                    pass

            # Memory Optimization: Convert all float64 to float32
            # This is crucial for 50k+ candles on Streamlit Cloud
            cols = self.df.select_dtypes(include=['float64']).columns
            if not cols.empty:
                self.df[cols] = self.df[cols].astype('float32')
            
            # Data Integrity: Use ffill().bfill() instead of dropna to keep full dataframe length
            self.df.ffill().bfill(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        def add_gex_levels(self, sensitivity=1.5):
            if self.df.empty: return
            # Synthetic GEX Logic
            self.df['Returns'] = self.df['Close'].pct_change()
            self.df['Roll_Vol'] = self.df['Returns'].rolling(window=20).std() * np.sqrt(252)
            self.df['ZeroGamma'] = self.df['Close'].rolling(window=20).mean()
            
            # Dynamic Walls based on Volatility and Sensitivity
            vol_mult = sensitivity * (1 + self.df['Roll_Vol'])
            self.df['CallWall'] = self.df['ZeroGamma'] + (self.df['ATR'] * vol_mult)
            self.df['PutWall'] = self.df['ZeroGamma'] - (self.df['ATR'] * vol_mult)

        def optimize_strategy(self, strategy_type, param_ranges, risk_reward, risk_per_trade, time_ranges=None):
            import itertools
            import time

            # 1. Data Preparation (Vectorized & Float32)
            # Ensure base indicators are present (ATR is critical for SL/TP)
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()

            # Work with a copy to avoid side effects, convert to float32 for memory/speed
            # We use numpy arrays directly for the optimization loop
            opens = self.df['Open'].values.astype(np.float32)
            highs = self.df['High'].values.astype(np.float32)
            lows = self.df['Low'].values.astype(np.float32)
            closes = self.df['Close'].values.astype(np.float32)
            atrs = self.df['ATR'].values.astype(np.float32)
            dates = self.df['datetime'].values
            n_candles = len(closes)

            # 2. Parameter & Time Setup
            keys = list(param_ranges.keys()) if param_ranges else []
            values = [param_ranges[k] for k in keys] if keys else []
            param_combinations = list(itertools.product(*values)) if values else [()]

            time_combinations = [(None, None)]
            if time_ranges:
                start_times = time_ranges.get('start_times', [])
                end_times = time_ranges.get('end_times', [])
                if start_times and end_times:
                    time_combinations = list(itertools.product(start_times, end_times))

            total_iterations = len(param_combinations) * len(time_combinations)
            
            # Results containers
            best_win_rate = 0.0
            best_pnl = -float('inf')
            best_params = {}
            best_time_config = {'start': None, 'end': None}
            results = []

            # UI Feedback
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time_perf = time.time()
            counter = 0

            # Memoization for expensive indicator calculations
            # Key: (IndicatorName, ParamValue), Value: NumpyArray
            indicator_cache = {}

            # 3. Optimization Loop
            for t_start, t_end in time_combinations:
                # Pre-calculate Time Mask
                time_mask = np.ones(n_candles, dtype=bool)
                if t_start and t_end:
                    # Vectorized Time Filtering
                    # Convert datetime objects to minutes from midnight for fast comparison
                    # We assume dates are numpy datetime64[ns]
                    dt_index = pd.to_datetime(dates)
                    minutes = dt_index.hour * 60 + dt_index.minute
                    start_min = t_start.hour * 60 + t_start.minute
                    end_min = t_end.hour * 60 + t_end.minute
                    time_mask = (minutes >= start_min) & (minutes <= end_min)

                for params_tuple in param_combinations:
                    counter += 1
                    params = dict(zip(keys, params_tuple)) if keys else {}

                    # Update UI periodically
                    if counter % 10 == 0 or counter == total_iterations:
                        progress = counter / total_iterations
                        elapsed = time.time() - start_time_perf
                        rem = (elapsed / progress) - elapsed if progress > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"AI Optimization: {counter}/{total_iterations} | Best WR: {best_win_rate:.1f}%")

                    # --- A. Signal Generation (Vectorized) ---
                    signals = np.zeros(n_candles, dtype=np.int8) # 0: None, 1: Long, -1: Short

                    try:
                        # Use StrategyLib for signal generation
                        signal_func = StrategyLib.get_signal_func(strategy_type)
                        if signal_func:
                            import inspect
                            sig_params = inspect.signature(signal_func).parameters
                            if 'cache' in sig_params:
                                long_sig, short_sig = signal_func(self.df, params, cache=indicator_cache)
                            else:
                                long_sig, short_sig = signal_func(self.df, params)
                            
                            # Handle potential NaNs in signals
                            if isinstance(long_sig, pd.Series):
                                long_sig = long_sig.fillna(False).values
                            if isinstance(short_sig, pd.Series):
                                short_sig = short_sig.fillna(False).values
                                
                            print(f'Testing {strategy_type} with {params} -> Raw signals: {np.sum(long_sig | short_sig)}')
                                
                            signals[long_sig] = 1
                            signals[short_sig] = -1
                        else:
                            # Fallback or continue
                            continue

                    except Exception as e:
                        print(f"Strategy Error in {strategy_type} with params {params}: {e}")
                        continue

                    raw_signal_count = np.sum(signals != 0)

                    # Apply Time Mask
                    signals[~time_mask] = 0
                    final_signal_count = np.sum(signals != 0)

                    print(f"DEBUG: Params {params} generated {final_signal_count} signals")
                    if final_signal_count == 0:
                        if raw_signal_count == 0:
                            print(f"  -> Reason: Indicator produced 0 signals (possibly all NaNs or no crossovers).")
                        else:
                            print(f"  -> Reason: Time filter blocked all {raw_signal_count} signals.")
                        continue

                    # --- B. Vectorized Trade Simulation (Isolation & Next-Bar) ---
                    # Identify potential entry points (Signal at T -> Entry at T+1)
                    sig_indices = np.where(signals != 0)[0]
                    if len(sig_indices) == 0: continue

                    trade_pnl = []
                    last_exit_idx = -1

                    # Fast Loop over Signals (NOT Candles)
                    for idx in sig_indices:
                        entry_idx = idx + 1
                        if entry_idx >= n_candles: break
                        
                        # Trade Isolation
                        if entry_idx <= last_exit_idx: continue

                        # Setup
                        direction = signals[idx]
                        entry_price = opens[entry_idx] # Entry at Open T+1
                        atr_val = atrs[idx] # ATR at Signal Candle
                        
                        if np.isnan(atr_val) or atr_val == 0: continue

                        sl_dist = atr_val * 1.5 # Fixed as per original
                        
                        if direction == 1: # Long
                            sl = entry_price - sl_dist
                            tp = entry_price + (sl_dist * risk_reward)
                            
                            # Vectorized Exit Search
                            future_lows = lows[entry_idx:]
                            future_highs = highs[entry_idx:]
                            
                            sl_hit = future_lows <= sl
                            tp_hit = future_highs >= tp
                            
                            # Find first occurrence
                            first_sl = np.argmax(sl_hit) if sl_hit.any() else n_candles
                            first_tp = np.argmax(tp_hit) if tp_hit.any() else n_candles
                            
                            # Determine Outcome
                            if first_sl == n_candles and first_tp == n_candles:
                                last_exit_idx = n_candles # Held till end
                            elif first_sl <= first_tp: # SL hit first or same candle (Conservative)
                                trade_pnl.append(-sl_dist)
                                last_exit_idx = entry_idx + first_sl
                            else: # TP hit first
                                trade_pnl.append(sl_dist * risk_reward)
                                last_exit_idx = entry_idx + first_tp
                                
                        else: # Short
                            sl = entry_price + sl_dist
                            tp = entry_price - (sl_dist * risk_reward)
                            
                            future_lows = lows[entry_idx:]
                            future_highs = highs[entry_idx:]
                            
                            sl_hit = future_highs >= sl
                            tp_hit = future_lows <= tp
                            
                            first_sl = np.argmax(sl_hit) if sl_hit.any() else n_candles
                            first_tp = np.argmax(tp_hit) if tp_hit.any() else n_candles
                            
                            if first_sl == n_candles and first_tp == n_candles:
                                last_exit_idx = n_candles
                            elif first_sl <= first_tp:
                                trade_pnl.append(-sl_dist)
                                last_exit_idx = entry_idx + first_sl
                            else:
                                trade_pnl.append(sl_dist * risk_reward)
                                last_exit_idx = entry_idx + first_tp

                    # --- C. Metrics & Best Selection ---
                    if trade_pnl:
                        pnl_arr = np.array(trade_pnl)
                        wins = np.sum(pnl_arr > 0)
                        count = len(pnl_arr)
                        wr = (wins / count) * 100
                        tot_pnl = np.sum(pnl_arr)
                        
                        if wr > best_win_rate:
                            best_win_rate = wr
                            best_pnl = tot_pnl
                            best_params = params
                            best_time_config = {'start': t_start, 'end': t_end}
                        elif wr == best_win_rate and tot_pnl > best_pnl:
                            best_pnl = tot_pnl
                            best_params = params
                            best_time_config = {'start': t_start, 'end': t_end}
                            
                        results.append({
                            'params': params,
                            'time_config': {'start': t_start, 'end': t_end},
                            'win_rate': wr,
                            'trades': count,
                            'pnl': tot_pnl
                        })

            progress_bar.empty()
            status_text.empty()
            return best_params, best_time_config, best_win_rate, results

        def optimize_hybrid_strategy(self, param_ranges, time_ranges=None):
            best_win_rate = 0
            best_params = {}
            best_time_config = {'start': None, 'end': None}
            results = []
            
            keys = list(param_ranges.keys())
            import itertools
            values = [param_ranges[k] for k in keys]
            param_combinations = list(itertools.product(*values))
            
            time_combinations = [(None, None)]
            if time_ranges:
                start_times = time_ranges.get('start_times', [])
                end_times = time_ranges.get('end_times', [])
                if start_times and end_times:
                    time_combinations = list(itertools.product(start_times, end_times))

            total_iterations = len(param_combinations) * len(time_combinations)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            counter = 0
            start_time_perf = time.time()

            for t_start, t_end in time_combinations:
                if t_start and t_end and t_start >= t_end: continue

                for combo in param_combinations:
                    counter += 1
                    # combo: (long_trigger, short_trigger, sensitivity, rr, risk_pct)
                    # Map keys to args
                    p = dict(zip(keys, combo))
                    
                    if counter % 5 == 0:
                        progress = counter / total_iterations
                        elapsed = time.time() - start_time_perf
                        rem = (elapsed / progress) - elapsed if progress > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Ottimizzazione GEX... {counter}/{total_iterations} (Rimanente: {rem:.1f}s)")

                    # Recalculate GEX levels if sensitivity changes? 
                    # Optimization: GEX levels depend on sensitivity. 
                    # Ideally we pre-calculate, but here we might need to re-run add_gex_levels if sensitivity changes.
                    # However, add_gex_levels modifies self.df in place. This is tricky for optimization loop.
                    # Solution: Calculate GEX walls dynamically inside run_hybrid or reset DF.
                    # Better: Pre-calculate GEX for all sensitivity levels in ranges? No, memory heavy.
                    # Acceptable: Re-calculate GEX inside loop (slower) or clone DF.
                    # Let's clone DF for safety or modify run_hybrid to accept walls directly?
                    # For now, let's re-run add_gex_levels. It's fast enough for vector ops.
                    
                    self.add_gex_levels(p['sensitivity'])
                    
                    trades, _ = self.run_hybrid_strategy(
                        p['long_trigger'], p['short_trigger'], p['rr'], p['risk_pct'], 
                        start_time=t_start, end_time=t_end, entry_mode=p.get('entry_mode', 'Standard')
                    )
                    
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_params = p
                            best_time_config = {'start': t_start, 'end': t_end}
                            
                        results.append({
                            'params': p,
                            'time_config': {'start': t_start, 'end': t_end},
                            'win_rate': win_rate,
                            'trades': len(trades),
                            'pnl': df_res['pnl'].sum()
                        })

            progress_bar.empty()
            status_text.empty()
            return best_params, best_time_config, best_win_rate, results

        def run_hybrid_strategy(self, long_trigger, short_trigger, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard"):
            trades = []
            balance = self.initial_capital
            equity_curve = [balance]
            position = None

            for i in range(len(self.df)):
                if i < 200: continue # Warmup
                curr = self.df.iloc[i]
                prev = self.df.iloc[i-1]
                
                # Schedule Check
                if start_time and end_time:
                    curr_time = curr['datetime'].time()
                    if not (start_time <= curr_time <= end_time):
                        equity_curve.append(balance)
                        continue

                # Exit Logic
                if position:
                    exit_res = None
                    if position['type'] == 'long':
                        if curr['Low'] <= position['sl']: exit_res, exit_price = 'LOSS', position['sl']
                        elif curr['High'] >= position['tp']: exit_res, exit_price = 'WIN', position['tp']
                    else:
                        if curr['High'] >= position['sl']: exit_res, exit_price = 'LOSS', position['sl']
                        elif curr['Low'] <= position['tp']: exit_res, exit_price = 'WIN', position['tp']
                    
                    if exit_res:
                        pnl = (exit_price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - exit_price) * position['size']
                        balance += pnl
                        trades.append({'time': curr['datetime'], 'type': f'EXIT {exit_res}', 'price': exit_price, 'pnl': pnl, 'balance': balance, 'Logica': f"Exit: {exit_res}"})
                        position = None
                    equity_curve.append(balance)
                    continue

                # Entry Logic (Hybrid GEX)
                signal = None
                # Long
                if long_trigger == "Bounce Put Wall" and prev['Low'] <= prev['PutWall'] and curr['Close'] > prev['PutWall']: signal = 'long'
                elif long_trigger == "Breakout 0-Gamma" and prev['Close'] < prev['ZeroGamma'] and curr['Close'] > prev['ZeroGamma']: signal = 'long'
                elif long_trigger == "Breakout Call Wall" and prev['Close'] < prev['CallWall'] and curr['Close'] > prev['CallWall']: signal = 'long'
                
                # Short
                if short_trigger == "Bounce Call Wall" and prev['High'] >= prev['CallWall'] and curr['Close'] < prev['CallWall']: signal = 'short'
                elif short_trigger == "Breakdown 0-Gamma" and prev['Close'] > prev['ZeroGamma'] and curr['Close'] < prev['ZeroGamma']: signal = 'short'
                elif short_trigger == "Breakdown Put Wall" and prev['Close'] > prev['PutWall'] and curr['Close'] < prev['PutWall']: signal = 'short'

                if signal:
                    entry_price = curr['Close']
                    sl_dist = curr['ATR'] * 1.5
                    
                    if entry_mode == "Retest":
                        entry_price = (curr['High'] + curr['Low']) / 2
                    
                    risk_amt = balance * (risk_per_trade / 100)
                    logic_str = f"Trigger: {long_trigger if signal == 'long' else short_trigger} | RR: {risk_reward} -> Signal: {signal.upper()}"
                    
                    if signal == 'long':
                        sl = entry_price - sl_dist
                        tp = entry_price + (sl_dist * risk_reward)
                        size = risk_amt / (entry_price - sl) if (entry_price - sl) > 0 else 0
                        if size > 0:
                            position = {'type': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': entry_price, 'pnl': 0, 'balance': balance, 'Logica': logic_str})
                    else:
                        sl = entry_price + sl_dist
                        tp = entry_price - (sl_dist * risk_reward)
                        size = risk_amt / (sl - entry_price) if (sl - entry_price) > 0 else 0
                        if size > 0:
                            position = {'type': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': entry_price, 'pnl': 0, 'balance': balance, 'Logica': logic_str})
                
                equity_curve.append(balance)
                
            return trades, equity_curve

        def execute_trades_agnostic(self, signals, risk_reward, risk_per_trade, sl_atr_mult=1.5, start_time=None, end_time=None, strategy_name="", params=None):
            # Prepare Data Arrays
            opens = self.df['Open'].values
            highs = self.df['High'].values
            lows = self.df['Low'].values
            closes = self.df['Close'].values
            datetimes = self.df['datetime'].values
            atrs = self.df['ATR'].values if 'ATR' in self.df.columns else np.zeros_like(closes)
            
            n_candles = len(closes)
            trades = []
            
            # Equity Curve Initialization
            equity_curve = np.full(n_candles, self.initial_capital, dtype=np.float32)
            balance = self.initial_capital
            
            last_exit_idx = -1
            
            # Diagnostics
            diag = {
                'total_signals': 0,
                'skipped_isolation': 0,
                'skipped_time': 0,
                'skipped_invalid_atr': 0,
                'skipped_size': 0,
                'executed': 0
            }
            
            # Get indices where signal is not 0 (up to n-2)
            sig_indices = np.where(signals[:-1] != 0)[0]
            diag['total_signals'] = len(sig_indices)
            
            for idx in sig_indices:
                entry_idx = idx + 1
                
                # Trade Isolation
                if entry_idx <= last_exit_idx:
                    diag['skipped_isolation'] += 1
                    continue
                
                # Time Check
                if start_time and end_time:
                    dt = pd.Timestamp(datetimes[idx])
                    t = dt.time()
                    if not (start_time <= t <= end_time):
                        diag['skipped_time'] += 1
                        continue

                direction = signals[idx] # 1 or -1
                entry_price = opens[entry_idx]
                entry_time = datetimes[entry_idx]
                atr_val = atrs[idx]
                
                if np.isnan(atr_val) or atr_val == 0: 
                    diag['skipped_invalid_atr'] += 1
                    continue
                
                sl_dist = atr_val * sl_atr_mult
                
                # Position Sizing
                risk_amount = balance * (risk_per_trade / 100.0)
                size = risk_amount / sl_dist if sl_dist > 0 else 0
                
                if size <= 0: 
                    diag['skipped_size'] += 1
                    continue

                if direction == 1: # Long
                    sl = entry_price - sl_dist
                    tp = entry_price + (sl_dist * risk_reward)
                    
                    future_lows = lows[entry_idx:]
                    future_highs = highs[entry_idx:]
                    
                    sl_hit_mask = future_lows <= sl
                    tp_hit_mask = future_highs >= tp
                    
                    first_sl = np.argmax(sl_hit_mask) if sl_hit_mask.any() else n_candles
                    first_tp = np.argmax(tp_hit_mask) if tp_hit_mask.any() else n_candles
                    
                    if first_sl == n_candles and first_tp == n_candles:
                        last_exit_idx = n_candles
                        continue
                    elif first_sl <= first_tp:
                        exit_idx = entry_idx + first_sl
                        exit_price = sl
                        exit_type = "SL"
                    else:
                        exit_idx = entry_idx + first_tp
                        exit_price = tp
                        exit_type = "TP"
                        
                    pnl = (exit_price - entry_price) * size
                        
                else: # Short
                    sl = entry_price + sl_dist
                    tp = entry_price - (sl_dist * risk_reward)
                    
                    future_lows = lows[entry_idx:]
                    future_highs = highs[entry_idx:]
                    
                    sl_hit_mask = future_highs >= sl
                    tp_hit_mask = future_lows <= tp
                    
                    first_sl = np.argmax(sl_hit_mask) if sl_hit_mask.any() else n_candles
                    first_tp = np.argmax(tp_hit_mask) if tp_hit_mask.any() else n_candles
                    
                    if first_sl == n_candles and first_tp == n_candles:
                        last_exit_idx = n_candles
                        continue
                    elif first_sl <= first_tp:
                        exit_idx = entry_idx + first_sl
                        exit_price = sl
                        exit_type = "SL"
                    else:
                        exit_idx = entry_idx + first_tp
                        exit_price = tp
                        exit_type = "TP"
                        
                    pnl = (entry_price - exit_price) * size

                # Record Trade
                balance += pnl
                logic_str = f"Trigger: {strategy_name} | Params: {params} -> Signal: {'LONG' if direction == 1 else 'SHORT'}"
                trades.append({
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': datetimes[exit_idx],
                    'Exit Price': exit_price,
                    'pnl': pnl,
                    'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                    'Type': 'long' if direction == 1 else 'short',
                    'Status': exit_type,
                    'type': 'ENTRY LONG' if direction == 1 else 'ENTRY SHORT',
                    'time': entry_time,
                    'price': entry_price,
                    'Logica': logic_str
                })
                
                if exit_idx < n_candles:
                    equity_curve[exit_idx:] = balance
                
                last_exit_idx = exit_idx
                diag['executed'] += 1

            return trades, equity_curve, diag

        def run_technical_strategy(self, strategy_type, params, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard", sl_atr_mult=1.5):
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()
            
            n_candles = len(self.df)
            signal_series = np.zeros(n_candles, dtype=np.int8)
            
            try:
                signal_func = StrategyLib.get_signal_func(strategy_type)
                if signal_func:
                    import inspect
                    sig_params = inspect.signature(signal_func).parameters
                    if 'cache' in sig_params:
                        long_sig, short_sig = signal_func(self.df, params, cache={})
                    else:
                        long_sig, short_sig = signal_func(self.df, params)
                    
                    signal_series[long_sig] = 1
                    signal_series[short_sig] = -1
            except Exception as e:
                st.error(f"Signal Gen Error: {e}")
                return [], []

            trades, equity, diag = self.execute_trades_agnostic(signal_series, risk_reward, risk_per_trade, sl_atr_mult, start_time, end_time, strategy_name=strategy_type, params=params)
            
            if len(trades) == 0:
                st.warning(f"⚠️ Nessuna operazione eseguita. Diagnostica: {diag}")
                
            return trades, equity

        def run_technical_strategy_old(self, strategy_type, params, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard", sl_atr_mult=1.5):
            trades = []
            balance = self.initial_capital
            equity_curve = [balance] * len(self.df)
            
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()
            
            # Convert to records for speed
            records = self.df.to_dict('records')
            n_candles = len(records)
            
            position = None # {type, entry_price, sl, tp, size, entry_time}
            
            # --- Pre-calculate Signals ---
            signal_series = np.zeros(n_candles, dtype=np.int8)
            try:
                signal_func = StrategyLib.get_signal_func(strategy_type)
                if signal_func:
                    # Check if accepts cache
                    import inspect
                    sig_params = inspect.signature(signal_func).parameters
                    if 'cache' in sig_params:
                        long_sig, short_sig = signal_func(self.df, params, cache={})
                    else:
                        long_sig, short_sig = signal_func(self.df, params)
                    
                    signal_series[long_sig] = 1
                    signal_series[short_sig] = -1
            except Exception as e:
                print(f"Signal Gen Error: {e}")

            # Loop from 1 to n (need prev candle for signal)
            for i in range(1, n_candles):
                prev = records[i-1]
                curr = records[i]
                
                # Update Equity (Carry forward)
                equity_curve[i] = balance 
                
                # 1. Manage Open Position
                if position:
                    exit_type = None
                    exit_price = 0.0
                    
                    # Check SL/TP against Current High/Low
                    if position['type'] == 'long':
                        if curr['Low'] <= position['sl']:
                            exit_type = 'SL'
                            exit_price = position['sl']
                        elif curr['High'] >= position['tp']:
                            exit_type = 'TP'
                            exit_price = position['tp']
                    else: # Short
                        if curr['High'] >= position['sl']:
                            exit_type = 'SL'
                            exit_price = position['sl']
                        elif curr['Low'] <= position['tp']:
                            exit_type = 'TP'
                            exit_price = position['tp']
                            
                    if exit_type:
                        # Execute Exit
                        if position['type'] == 'long':
                            pnl = (exit_price - position['entry']) * position['size']
                        else:
                            pnl = (position['entry'] - exit_price) * position['size']
                            
                        balance += pnl
                        equity_curve[i] = balance
                        
                        trades.append({
                            'Entry Time': position['time'],
                            'Entry Price': position['entry'],
                            'Exit Time': curr['datetime'],
                            'Exit Price': exit_price,
                            'pnl': pnl,
                            'Return %': (pnl / (position['entry'] * position['size'])) * 100 if position['size'] > 0 else 0,
                            'Type': position['type'],
                            'Status': exit_type,
                            # Compatibility fields for Visualizer
                            'type': f"ENTRY {position['type'].upper()}", 
                            'time': position['time'], 
                            'price': position['entry']
                        })
                        position = None
                        continue # Isolation: No new entry on same bar as exit
                
                # 2. Check for New Entry (if no position)
                if position is None:
                    # Time Schedule Check (based on Signal Time)
                    if start_time and end_time:
                        t_obj = prev['datetime'].time()
                        if not (start_time <= t_obj <= end_time):
                            continue
                    
                    # Check Pre-calculated Signal (Signal at T (prev) -> Entry at T+1 (curr))
                    # signal_series[i-1] corresponds to signal generated at prev candle
                    sig_val = signal_series[i-1]
                    signal = 'long' if sig_val == 1 else 'short' if sig_val == -1 else None
                        
                    if signal:
                        # Entry Setup
                        entry_price = curr['Open'] # Entry at Open T+1
                        atr = prev['ATR'] # ATR at Signal Candle
                        
                        if atr > 0:
                            sl_dist = atr * sl_atr_mult
                            risk_amt = balance * (risk_per_trade / 100)
                            
                            if signal == 'long':
                                sl = entry_price - sl_dist
                                tp = entry_price + (sl_dist * risk_reward)
                                size = risk_amt / sl_dist
                            else:
                                sl = entry_price + sl_dist
                                tp = entry_price - (sl_dist * risk_reward)
                                size = risk_amt / sl_dist
                                
                            position = {
                                'type': signal,
                                'entry': entry_price,
                                'sl': sl,
                                'tp': tp,
                                'size': size,
                                'time': curr['datetime']
                            }
                            
                            # Check for Immediate Exit (Same Candle)
                            imm_exit = None
                            imm_price = 0.0
                            
                            if signal == 'long':
                                if curr['Low'] <= sl:
                                    imm_exit = 'SL'; imm_price = sl
                                elif curr['High'] >= tp:
                                    imm_exit = 'TP'; imm_price = tp
                            else:
                                if curr['High'] >= sl:
                                    imm_exit = 'SL'; imm_price = sl
                                elif curr['Low'] <= tp:
                                    imm_exit = 'TP'; imm_price = tp
                                    
                            if imm_exit:
                                if signal == 'long': pnl = (imm_price - entry_price) * size
                                else: pnl = (entry_price - imm_price) * size
                                    
                                balance += pnl
                                equity_curve[i] = balance
                                
                                trades.append({
                                    'Entry Time': curr['datetime'],
                                    'Entry Price': entry_price,
                                    'Exit Time': curr['datetime'],
                                    'Exit Price': imm_price,
                                    'pnl': pnl,
                                    'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                                    'Type': signal,
                                    'Status': imm_exit,
                                    'type': f"ENTRY {signal.upper()}",
                                    'time': curr['datetime'],
                                    'price': entry_price
                                })
                                position = None
                            
            return trades, equity_curve

    class Visualizer:
        @staticmethod
        def plot_tradingview_clone(df, trades, engine_type="Hybrid", strategy_name=""):
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df['datetime'],
                            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            name='Price'), row=1, col=1)
            
            # Indicators based on Engine
            if engine_type == "Hybrid":
                if 'ZeroGamma' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['ZeroGamma'], name='Zero Gamma', line=dict(color='orange', width=1)), row=1, col=1)
                if 'CallWall' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['CallWall'], name='Call Wall', line=dict(color='green', dash='dash')), row=1, col=1)
                if 'PutWall' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['PutWall'], name='Put Wall', line=dict(color='red', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['datetime'], y=[0]*len(df), showlegend=False, opacity=0), row=2, col=1)
            else:
                # Technical Indicators Visualization
                if 'SMA200' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=2)), row=1, col=1)
                if 'SMA50' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA50'], name='SMA 50', line=dict(color='cyan', width=1)), row=1, col=1)
                
                if "Bollinger" in strategy_name and 'BB_Upper' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

                # Row 2 Oscillators
                if "RSI" in strategy_name and 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                elif "MACD" in strategy_name and 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=2, col=1)
                    fig.add_bar(x=df['datetime'], y=df['MACD'] - df['MACD_Signal'], name='Histogram', row=2, col=1)
                elif "Stochastic" in strategy_name and 'Stoch_K' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Stoch_K'], name='Stoch K', line=dict(color='blue', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Stoch_D'], name='Stoch D', line=dict(color='orange', width=1)), row=2, col=1)
                    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
                elif "CCI" in strategy_name and 'CCI' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['CCI'], name='CCI', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
                elif "Williams" in strategy_name and 'WilliamsR' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['WilliamsR'], name='Williams %R', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-80, line_dash="dash", line_color="green", row=2, col=1)
                elif "Aroon" in strategy_name and 'Aroon_Up' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Aroon_Up'], name='Aroon Up', line=dict(color='green', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Aroon_Down'], name='Aroon Down', line=dict(color='red', width=1)), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=[0]*len(df), showlegend=False, opacity=0), row=2, col=1)

            # Signals
            buy_signals = [t for t in trades if 'ENTRY LONG' in t['type']]
            sell_signals = [t for t in trades if 'ENTRY SHORT' in t['type']]
            
            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in buy_signals], 
                    y=[t['price'] for t in buy_signals],
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy Signal'
                ), row=1, col=1)
            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in sell_signals], 
                    y=[t['price'] for t in sell_signals],
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell Signal'
                ), row=1, col=1)

            fig.update_layout(
                title=f"TradingView Clone - {engine_type} Strategy ({strategy_name})",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
                height=800,
                dragmode='pan',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Crosshairs
            fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", row=1, col=1)
            fig.update_yaxes(showspikes=True, spikecolor="gray", spikemode="across", row=1, col=1)
            fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", row=2, col=1)
            fig.update_yaxes(showspikes=True, spikecolor="gray", spikemode="across", row=2, col=1)
            
            return fig

    engine = BacktestEngine(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe, initial_capital)

    # Use verified data if available
    if st.session_state.backtest_data is not None and st.session_state.backtest_ticker == ticker:
        engine.df = st.session_state.backtest_data.copy()
        data_ready = True
    else:
        data_ready = False

    if "MOTORE A" in engine_choice:
        st.subheader("🧬 Configurazione Strategia Ibrida GEX")
        
        # Schedule & Entry Mode
        with st.expander("⚙️ Opzioni Avanzate (Orari & Ingresso)"):
            c1, c2, c3 = st.columns(3)
            with c1: use_schedule = st.checkbox("Abilita Orari di Trading")
            with c2: start_t = st.time_input("Inizio Sessione", dt_time(9, 30))
            with c3: end_t = st.time_input("Fine Sessione", dt_time(16, 0))
            
            entry_mode = st.selectbox("Modalità di Ingresso", ["Standard", "Breakout (Close)", "Retest"])
            
        col1, col2 = st.columns(2)
        with col1:
            long_trigger = st.selectbox("Trigger Long", ["Rimbalzo Put Wall", "Breakout 0-Gamma", "Breakout Call Wall", "Nessuno"])
            # Map Italian to English for logic
            long_trigger_map = {"Rimbalzo Put Wall": "Bounce Put Wall", "Breakout 0-Gamma": "Breakout 0-Gamma", "Breakout Call Wall": "Breakout Call Wall", "Nessuno": "None"}
            long_trigger_en = long_trigger_map[long_trigger]

        with col2:
            short_trigger = st.selectbox("Trigger Short", ["Rimbalzo Call Wall", "Breakdown 0-Gamma", "Breakdown Put Wall", "Nessuno"])
            short_trigger_map = {"Rimbalzo Call Wall": "Bounce Call Wall", "Breakdown 0-Gamma": "Breakdown 0-Gamma", "Breakdown Put Wall": "Breakdown Put Wall", "Nessuno": "None"}
            short_trigger_en = short_trigger_map[short_trigger]
        
        sensitivity = st.slider("Sensibilità GEX", 0.5, 3.0, 1.5, 0.1)
        rr = st.slider("Rischio:Rendimento", 1.0, 5.0, 2.0, 0.5)
        risk_pct = st.slider("Rischio per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        
        if data_ready:
            if st.button("🚀 Avvia Simulazione GEX"):
                with st.spinner("Calcolo Livelli GEX e Simulazione..."):
                    # Data already in engine.df
                    engine.add_technical_indicators() # Need ATR
                    engine.add_gex_levels(sensitivity)
                    
                    s_time = start_t if use_schedule else None
                    e_time = end_t if use_schedule else None
                    
                    trades, equity = engine.run_hybrid_strategy(long_trigger_en, short_trigger_en, rr, risk_pct, s_time, e_time, entry_mode)
                    st.session_state['trades'] = trades
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    st.subheader("Professional Risk Dashboard")
                    
                    current_trades = st.session_state.get('trades', [])
                    if not current_trades:
                        st.warning("Esegui il backtest per vedere le analisi avanzate")
                    else:
                        # Apply Friction
                        adjusted_trades, adjusted_equity = apply_friction_post_process(current_trades, initial_capital, friction_pct)
                        
                        # Calculate Metrics
                        metrics = calculate_advanced_metrics(adjusted_trades)
                        
                        # Metric Cards
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)
                        
                        with col1:
                            expectancy = metrics['expectancy']
                            st.metric(
                                label="Mathematical Expectancy",
                                value=f"${expectancy:.2f}",
                                delta="Positive" if expectancy > 0 else "Negative",
                                delta_color="normal" if expectancy > 0 else "inverse"
                            )
                            
                        with col2:
                            profit_factor = metrics['profit_factor']
                            st.metric(
                                label="Profit Factor",
                                value=f"{profit_factor:.2f}",
                                delta="Good" if profit_factor > 1.5 else "Needs Improvement",
                                delta_color="normal" if profit_factor > 1.5 else "inverse"
                            )
                            
                        with col3:
                            win_rate = metrics['win_rate']
                            st.metric(
                                label="Win Rate (%)",
                                value=f"{win_rate:.2f}%",
                                delta="Profitable" if win_rate > 50 else "Unprofitable",
                                delta_color="normal" if win_rate > 50 else "inverse"
                            )
                            
                        with col4:
                            max_dd = metrics['max_drawdown']
                            st.metric(
                                label="Max Drawdown (%)",
                                value=f"{max_dd:.2f}%",
                                delta="High Risk" if max_dd < -20 else "Acceptable",
                                delta_color="inverse" if max_dd < -20 else "normal"
                            )
                            
                        with col5:
                            total_profit_abs = metrics.get('total_profit_abs', 0)
                            st.metric(
                                label="Total Net Profit ($)",
                                value=f"${total_profit_abs:.2f}",
                                delta="Positive" if total_profit_abs > 0 else "Negative",
                                delta_color="normal" if total_profit_abs > 0 else "inverse"
                            )
                            
                        with col6:
                            max_dd_abs = metrics.get('max_dd_abs', 0)
                            st.metric(
                                label="Max Drawdown ($)",
                                value=f"${max_dd_abs:.2f}",
                                delta="High Loss" if max_dd_abs > initial_capital * 0.2 else "Acceptable",
                                delta_color="inverse" if max_dd_abs > initial_capital * 0.2 else "normal"
                            )
                            
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, adjusted_trades, "Hybrid"), use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['drawline']})
                        st.line_chart(adjusted_equity)
                        
                        # Monte Carlo Expander
                        with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                            mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                            if mc_res:
                                mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                                st.plotly_chart(mc_fig, use_container_width=True)
                                
                                st.subheader('🔬 Validazione Statistica Long-Term')
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                                with c2:
                                    st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                                with c3:
                                    st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                                
                                if prob_profit > 75:
                                    st.success('✅ Strategia Robusta')
                                elif prob_profit < 60:
                                    st.warning('⚠️ Strategia Fragile (Flop)')
                                    
                                if risk_of_ruin > 10:
                                    st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                    
                                if len(adjusted_trades) < 30:
                                    st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                            else:
                                st.warning("Not enough data for Monte Carlo simulation.")
                        
                        st.subheader("📝 Dettaglio Operazioni (Explainability)")
                        df_res = pd.DataFrame(adjusted_trades)
                        st.data_editor(df_res, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ Esegui prima la 'Verifica Disponibilità Dati Storici' per abilitare la simulazione.")

    else: # MOTORE B
        st.subheader("📈 Configurazione Hub Strategie Tecniche")
        
        # Schedule & Entry Mode
        with st.expander("⚙️ Opzioni Avanzate (Orari & Ingresso)"):
            c1, c2, c3 = st.columns(3)
            with c1: use_schedule = st.checkbox("Abilita Orari di Trading", value=st.session_state.get('use_schedule', False))
            with c2: start_t = st.time_input("Inizio Sessione", value=st.session_state.get('start_time', dt_time(9, 30)))
            with c3: end_t = st.time_input("Fine Sessione", value=st.session_state.get('end_time', dt_time(16, 0)))
            
            entry_mode = st.selectbox("Modalità di Ingresso", ["Standard", "Breakout (Close)", "Retest"], index=["Standard", "Breakout (Close)", "Retest"].index(st.session_state.get('entry_mode', "Standard")))
        
        # Strategy Selection
        strategies_list = [
            "RSI Mean Reversion", "MACD Crossover", "Bollinger Breakout", "Golden/Death Cross", 
            "Stochastic Oscillator", "CCI Momentum", "Williams %R Reversal",
            "HMA Trend", "TEMA Crossover", "KAMA Trend", "Aroon Oscillator",
            "SuperTrend Reversal", "Parabolic SAR", "TSI Crossover", "UO Overbought/Oversold",
            "Keltner Channel Breakout", "Donchian Channel Breakout", "Chaikin Volatility",
            "CMF Trend", "VWAP Crossover", "AD Line Trend",
            "Vortex Crossover", "Choppiness Index Breakout", "KST Crossover", "Coppock Curve",
            "Ichimoku Cloud Breakout", "Awesome Oscillator", "PPO Crossover", "Mass Index Reversal",
            "Ulcer Index Safety",
            "WMA Trend", "TRIMA Crossover", "CMO Reversal", "Momentum Breakout", "BOP Trend",
            "TRIX Crossover", "StochRSI Reversal", "TSF Trend"
        ]
        # Restore strategy selection if exists
        saved_strat_idx = 0
        if st.session_state.get('strategy_type') in strategies_list:
            saved_strat_idx = strategies_list.index(st.session_state.get('strategy_type'))
            
        strategy_type = st.selectbox("Seleziona Tipo Strategia", strategies_list, index=saved_strat_idx)
        
        # Dynamic Parameters with Session State Persistence
        params = {}
        col1, col2, col3 = st.columns(3)
        
        if strategy_type == "RSI Mean Reversion":
            with col1: params['period'] = st.number_input("Periodo RSI", value=int(st.session_state.get('period_rsi', 14)))
            with col2: params['ob'] = st.number_input("Ipercomprato", value=int(st.session_state.get('ob_rsi', 70)))
            with col3: params['os'] = st.number_input("Ipervenduto", value=int(st.session_state.get('os_rsi', 30)))
        elif strategy_type == "Stochastic Oscillator":
            with col1: params['k_period'] = st.number_input("Periodo K", value=int(st.session_state.get('k_stoch', 14)))
            with col2: params['ob'] = st.number_input("Ipercomprato", value=int(st.session_state.get('ob_stoch', 80)))
            with col3: params['os'] = st.number_input("Ipervenduto", value=int(st.session_state.get('os_stoch', 20)))
        elif strategy_type == "CCI Momentum":
             with col1: params['period'] = st.number_input("Periodo CCI", value=int(st.session_state.get('period_cci', 20)))
        elif strategy_type == "Williams %R Reversal":
             with col1: params['period'] = st.number_input("Periodo Williams %R", value=int(st.session_state.get('period_williams', 14)))
        elif strategy_type == "Bollinger Breakout":
            with col1: params['period'] = st.number_input("Periodo BB", value=int(st.session_state.get('period_bb', 20)))
            with col2: params['std_dev'] = st.number_input("Dev. Std", value=float(st.session_state.get('std_bb', 2.0)))
        elif strategy_type == "MACD Crossover":
            with col1: params['fast'] = st.number_input("Fast Period", value=int(st.session_state.get('fast_macd', 12)))
            with col2: params['slow'] = st.number_input("Slow Period", value=int(st.session_state.get('slow_macd', 26)))
            with col3: params['signal'] = st.number_input("Signal Period", value=int(st.session_state.get('signal_macd', 9)))
        elif strategy_type == "Golden/Death Cross":
            with col1: params['fast'] = st.number_input("Fast MA", value=int(st.session_state.get('fast_gd', 50)))
            with col2: params['slow'] = st.number_input("Slow MA", value=int(st.session_state.get('slow_gd', 200)))
        else:
            # Generic fallback for others to avoid errors if params needed
            with col1: params['period'] = st.number_input("Periodo", value=int(st.session_state.get('period_generic', 14)))
        
        rr = st.slider("Rischio:Rendimento", 1.0, 5.0, float(st.session_state.get('rr', 2.0)), 0.5)
        risk_pct = st.slider("Rischio per Trade (%)", 0.5, 5.0, float(st.session_state.get('risk_pct', 1.0)), 0.1)
        
        if data_ready:
            c_run, c_opt = st.columns([2, 1])
            
            # Check Auto-Run Flag
            auto_run = st.session_state.get('run_backtest_auto', False)
            
            if c_run.button("🚀 Avvia Backtest Tecnico") or auto_run:
                if auto_run: st.session_state['run_backtest_auto'] = False # Reset flag
                
                with st.spinner("Calcolo Indicatori e Simulazione..."):
                    # Data already in engine.df
                    engine.add_technical_indicators()
                    
                    s_time = start_t if use_schedule else None
                    e_time = end_t if use_schedule else None
                    
                    trades, equity = engine.run_technical_strategy(strategy_type, params, rr, risk_pct, s_time, e_time, entry_mode)
                    st.session_state['trades'] = trades
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    # Optimization Feedback: Actual Processed Range
                    if not engine.df.empty:
                        min_date = engine.df['datetime'].min()
                        max_date = engine.df['datetime'].max()
                        st.caption(f"📅 Range Effettivo Processato: {min_date} -> {max_date} ({len(engine.df)} candele)")
                    
                    st.subheader("Professional Risk Dashboard")
                    
                    current_trades = st.session_state.get('trades', [])
                    if not current_trades:
                        st.warning("Esegui il backtest per vedere le analisi avanzate")
                    else:
                        # Apply Friction
                        adjusted_trades, adjusted_equity = apply_friction_post_process(current_trades, initial_capital, friction_pct)
                        
                        # Calculate Metrics
                        metrics = calculate_advanced_metrics(adjusted_trades)
                        
                        # Metric Cards
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)
                        
                        with col1:
                            expectancy = metrics['expectancy']
                            st.metric(
                                label="Mathematical Expectancy",
                                value=f"${expectancy:.2f}",
                                delta="Positive" if expectancy > 0 else "Negative",
                                delta_color="normal" if expectancy > 0 else "inverse"
                            )
                            
                        with col2:
                            profit_factor = metrics['profit_factor']
                            st.metric(
                                label="Profit Factor",
                                value=f"{profit_factor:.2f}",
                                delta="Good" if profit_factor > 1.5 else "Needs Improvement",
                                delta_color="normal" if profit_factor > 1.5 else "inverse"
                            )
                            
                        with col3:
                            win_rate = metrics['win_rate']
                            st.metric(
                                label="Win Rate (%)",
                                value=f"{win_rate:.2f}%",
                                delta="Profitable" if win_rate > 50 else "Unprofitable",
                                delta_color="normal" if win_rate > 50 else "inverse"
                            )
                            
                        with col4:
                            max_dd = metrics['max_drawdown']
                            st.metric(
                                label="Max Drawdown (%)",
                                value=f"{max_dd:.2f}%",
                                delta="High Risk" if max_dd < -20 else "Acceptable",
                                delta_color="inverse" if max_dd < -20 else "normal"
                            )
                            
                        with col5:
                            total_profit_abs = metrics.get('total_profit_abs', 0)
                            st.metric(
                                label="Total Net Profit ($)",
                                value=f"${total_profit_abs:.2f}",
                                delta="Positive" if total_profit_abs > 0 else "Negative",
                                delta_color="normal" if total_profit_abs > 0 else "inverse"
                            )
                            
                        with col6:
                            max_dd_abs = metrics.get('max_dd_abs', 0)
                            st.metric(
                                label="Max Drawdown ($)",
                                value=f"${max_dd_abs:.2f}",
                                delta="High Loss" if max_dd_abs > initial_capital * 0.2 else "Acceptable",
                                delta_color="inverse" if max_dd_abs > initial_capital * 0.2 else "normal"
                            )
                            
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, adjusted_trades, "Technical", strategy_type), use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['drawline']})
                        st.line_chart(adjusted_equity)
                        
                        # Monte Carlo Expander
                        with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                            mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                            if mc_res:
                                mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                                st.plotly_chart(mc_fig, use_container_width=True)
                                
                                st.subheader('🔬 Validazione Statistica Long-Term')
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                                with c2:
                                    st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                                with c3:
                                    st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                                
                                if prob_profit > 75:
                                    st.success('✅ Strategia Robusta')
                                elif prob_profit < 60:
                                    st.warning('⚠️ Strategia Fragile (Flop)')
                                    
                                if risk_of_ruin > 10:
                                    st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                    
                                if len(adjusted_trades) < 30:
                                    st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                            else:
                                st.warning("Not enough data for Monte Carlo simulation.")
                        
                        st.subheader("📝 Dettaglio Operazioni (Explainability)")
                        df_res = pd.DataFrame(adjusted_trades)
                        st.data_editor(df_res, use_container_width=True, hide_index=True)
            
            if c_opt.button("🧠 Ottimizza Strategia (AI)"):
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # --- FAST OPTIMIZATION LOGIC ---
                # 1. Prepare Data (Downsample if needed)
                engine.add_technical_indicators() # Ensure ATR and other base indicators are present
                opt_df = engine.df.copy()
                if len(opt_df) > 10000:
                    status_text.text(f"⚠️ Dati estesi ({len(opt_df)} candele). Ottimizzazione in corso su tutto il dataset...")
                    # No truncation as per user request

                
                # 2. Define Ranges
                opt_config = STRATEGY_PARAM_GRID.get(strategy_type, {})
                
                rr_ranges = [1.5, 2.0, 2.5, 3.0]
                
                # 3. Generate Combinations
                import itertools
                keys = list(opt_config.keys())
                values = [opt_config[k] for k in keys]
                param_combos = list(itertools.product(*values)) if values else [()]
                
                total_steps = len(param_combos) * len(rr_ranges)
                step = 0
                best_res = {'wr': 0, 'pnl': -float('inf'), 'params': {}, 'rr': 0}
                
                # 4. Fast Loop
                opt_cache = {}
                for p_vals in param_combos:
                    curr_p = dict(zip(keys, p_vals))
                    
                    # Generate Signals (Vectorized)
                    sigs = pd.Series(0, index=opt_df.index)
                    
                    try:
                        signal_func = StrategyLib.get_signal_func(strategy_type)
                        if signal_func:
                            long_sig, short_sig = signal_func(opt_df, curr_p, cache=opt_cache)
                            long_sig = long_sig.fillna(False)
                            short_sig = short_sig.fillna(False)
                            sigs = np.where(long_sig, 1, np.where(short_sig, -1, 0))
                    except Exception as e:
                        continue

                    # Get Entry Indices
                    entry_idxs = np.where(sigs != 0)[0]
                    if len(entry_idxs) == 0: continue
                    
                    # 5. Fast Trade Outcome Loop
                    # We assume fixed RR and ATR based SL
                    atr_arr = opt_df['ATR'].values
                    close_arr = opt_df['Close'].values
                    high_arr = opt_df['High'].values
                    low_arr = opt_df['Low'].values
                    times = opt_df['datetime'].values
                    
                    for rr_val in rr_ranges:
                        step += 1
                        if step % 20 == 0:
                            progress_bar.progress(min(step/total_steps, 1.0))
                            status_text.text(f"Scansione AI... WR: {best_res['wr']:.1f}%")
                            
                        wins = 0
                        losses = 0
                        pnl = 0
                        
                        for idx in entry_idxs:
                            if idx >= len(opt_df) - 1: continue
                            
                            entry_price = close_arr[idx]
                            direction = sigs[idx] # 1 or -1
                            atr = atr_arr[idx]
                            if np.isnan(atr): continue
                            
                            sl_dist = atr * 1.5
                            
                            if direction == 1: # Long
                                sl = entry_price - sl_dist
                                tp = entry_price + (sl_dist * rr_val)
                                # Look forward max 50 bars
                                for fwd in range(idx+1, min(idx+51, len(opt_df))):
                                    if low_arr[fwd] <= sl:
                                        losses += 1
                                        pnl -= sl_dist
                                        break
                                    elif high_arr[fwd] >= tp:
                                        wins += 1
                                        pnl += (sl_dist * rr_val)
                                        break
                            else: # Short
                                sl = entry_price + sl_dist
                                tp = entry_price - (sl_dist * rr_val)
                                for fwd in range(idx+1, min(idx+51, len(opt_df))):
                                    if high_arr[fwd] >= sl:
                                        losses += 1
                                        pnl -= sl_dist
                                        break
                                    elif low_arr[fwd] <= tp:
                                        wins += 1
                                        pnl += (sl_dist * rr_val)
                                        break
                        
                        total_trades = wins + losses
                        if total_trades > 0:
                            wr = (wins / total_trades) * 100
                            
                            # Selection Logic: Max WR, then Max PnL
                            if wr > best_res['wr']:
                                best_res = {'wr': wr, 'pnl': pnl, 'params': curr_p, 'rr': rr_val}
                            elif wr == best_res['wr'] and pnl > best_res['pnl']:
                                best_res = {'wr': wr, 'pnl': pnl, 'params': curr_p, 'rr': rr_val}

                progress_bar.empty()
                status_text.empty()
                
                if best_res['wr'] > 0:
                    st.success(f"🏆 Ottimizzazione Completata! WR: {best_res['wr']:.1f}%")
                    
                    # Save to Session State
                    for k, v in best_res['params'].items():
                        if strategy_type == "RSI Mean Reversion":
                            if k == 'period': st.session_state['period_rsi'] = v
                            elif k == 'ob': st.session_state['ob_rsi'] = v
                            elif k == 'os': st.session_state['os_rsi'] = v
                        elif strategy_type == "Stochastic Oscillator":
                            if k == 'k_period': st.session_state['k_stoch'] = v
                            elif k == 'ob': st.session_state['ob_stoch'] = v
                            elif k == 'os': st.session_state['os_stoch'] = v
                        elif strategy_type == "CCI Momentum":
                            if k == 'period': st.session_state['period_cci'] = v
                        elif strategy_type == "Williams %R Reversal":
                            if k == 'period': st.session_state['period_williams'] = v
                        elif strategy_type == "Bollinger Breakout":
                            if k == 'period': st.session_state['period_bb'] = v
                            elif k == 'std_dev': st.session_state['std_bb'] = v
                        elif strategy_type == "MACD Crossover":
                            if k == 'fast': st.session_state['fast_macd'] = v
                            elif k == 'slow': st.session_state['slow_macd'] = v
                            elif k == 'signal': st.session_state['signal_macd'] = v
                        elif strategy_type == "Golden/Death Cross":
                            if k == 'fast': st.session_state['fast_gd'] = v
                            elif k == 'slow': st.session_state['slow_gd'] = v
                        else:
                            if k == 'period': st.session_state['period_generic'] = v
                            
                    st.session_state['rr'] = best_res['rr']
                    st.session_state['run_backtest_auto'] = True
                    
                    st.rerun()
                else:
                    st.error("Nessun risultato valido trovato.")

        else:
            st.info("⚠️ Esegui prima la 'Verifica Disponibilità Dati Storici' per abilitare la simulazione.")

elif menu == "🛠️ STRATEGY BUILDER":
    st.title("🛠️ Strategy Builder (No-Code)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Impostazioni Base")
    
    # Time Filters UI
    start_time = st.sidebar.time_input("Start Trading Time", value=datetime.strptime("09:30", "%H:%M").time())
    end_time = st.sidebar.time_input("End Trading Time", value=datetime.strptime("16:00", "%H:%M").time())
    eod_close = st.sidebar.checkbox("Close all at EOD", value=True)
    
    # ORB UI
    orb_enabled = st.sidebar.checkbox("Enable Opening Range Breakout (ORB)", value=True)
    orb_duration = 15
    if orb_enabled:
        orb_duration = st.sidebar.selectbox("ORB Candle Duration (min)", [5, 15, 30])
        
    # Ticker and Date Range
    st.sidebar.markdown("### 📈 Selezione Asset")
    ticker_choices = [
        "EURUSD=X (Forex)", "GBPUSD=X (Forex)", "USDJPY=X (Forex)", "EURGBP=X (Forex)",
        "^GSPC (S&P500)", "^IXIC (Nasdaq)", "^GDAXI (DAX)", "FTSEMIB.MI (FTSE MIB)",
        "BTC-USD (Crypto)", "ETH-USD (Crypto)", "AAPL (Stock)", "TSLA (Stock)", "NVDA (Stock)",
        "--- INSERIMENTO MANUALE ---"
    ]
    selected_ticker = st.sidebar.selectbox("Ticker", ticker_choices, index=4)
    if selected_ticker == "--- INSERIMENTO MANUALE ---":
        ticker = st.sidebar.text_input("Inserisci Ticker Custom", value="SPY").upper()
    else:
        ticker = selected_ticker.split(" ")[0]
        
    st.sidebar.markdown("### 🛡️ Risk Management")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    rr_ratio = st.sidebar.slider("Target Risk/Reward (R:R)", 1.0, 5.0, 2.0, 0.1)
    sl_mode = st.sidebar.selectbox("Stop Loss Mode", ["Fixed %", "Candle Low/High"])
    fixed_sl_pct = 1.0
    if sl_mode == "Fixed %":
        fixed_sl_pct = st.sidebar.slider("Fixed Stop Loss (%)", 0.1, 5.0, 1.0, 0.1)
    
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with c2: end_date = st.date_input("End Date", value=datetime.now())
    
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
    
    # Duplicate necessary functions
    def normalize_key(d, possible_keys):
        for k in d.keys():
            if k.lower() in [pk.lower() for pk in possible_keys]:
                return d[k]
        return None

    def apply_friction_post_process(trades_list, initial_capital, friction_pct):
        if not trades_list:
            return trades_list, [initial_capital]
            
        new_trades = []
        balance = initial_capital
        equity_curve = [balance]
        
        for t in trades_list:
            t_copy = dict(t)
            t_type = str(normalize_key(t_copy, ['type', 'Type']) or '').upper()
            price = normalize_key(t_copy, ['price', 'Price', 'Entry Price', 'Exit Price']) or 0
            pnl = normalize_key(t_copy, ['pnl', 'PnL']) or 0
            
            friction_multiplier = 1 - (friction_pct / 100)
            new_price = price * friction_multiplier
            pnl = pnl * friction_multiplier
            t_copy['price'] = new_price
            t_copy['pnl'] = pnl
            balance += pnl
            t_copy['balance'] = balance
            equity_curve.append(balance)
            
            new_trades.append(t_copy)
                
        return new_trades, equity_curve

    def calculate_advanced_metrics(trades_list):
        fallback = {'expectancy': 0, 'profit_factor': 0, 'max_drawdown': 0, 'win_rate': 0, 'total_profit_abs': 0, 'max_dd_abs': 0}
        if not trades_list:
            return fallback
            
        df = pd.DataFrame(trades_list)
        df.columns = [str(c).lower() for c in df.columns]
        
        if 'pnl' not in df.columns:
            return fallback
            
        exits = df[df['pnl'].notna()]
        if exits.empty:
            return fallback
            
        wins = exits[exits['pnl'] > 0]['pnl']
        losses = exits[exits['pnl'] < 0]['pnl']
        
        win_rate = len(wins) / len(exits)
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
        
        total_profit_abs = exits['pnl'].sum()
        
        bal_col = 'balance' if 'balance' in df.columns else None
        max_dd = 0
        max_dd_abs = 0
        if bal_col:
            curve = df[bal_col].tolist()
            peak = curve[0]
            for val in curve:
                if val > peak: peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                dd_abs = peak - val
                if dd > max_dd: max_dd = dd
                if dd_abs > max_dd_abs: max_dd_abs = dd_abs
                
        return {
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd * 100,
            'win_rate': win_rate * 100,
            'total_profit_abs': total_profit_abs,
            'max_dd_abs': max_dd_abs
        }

    def run_monte_carlo(trades_list, initial_capital, simulations=1000):
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        if not trades_list:
            return None
            
        df_res = pd.DataFrame(trades_list)
        if 'pnl' in df_res.columns:
            pnls = df_res[df_res['pnl'].notna()]['pnl'].values
        else:
            return None
            
        n_trades = len(pnls)
        if n_trades == 0:
            return None
            
        sim_length = min(50, n_trades)
        
        random_indices = np.random.randint(0, n_trades, size=(simulations, sim_length))
        simulated_pnls = pnls[random_indices]
        
        equity_curves = np.cumsum(simulated_pnls, axis=1) + initial_capital
        
        starting_capital = np.full((simulations, 1), initial_capital)
        equity_curves = np.hstack((starting_capital, equity_curves))
        
        median_curve = np.median(equity_curves, axis=0)
        
        final_balances = equity_curves[:, -1]
        prob_profit = (np.sum(final_balances > initial_capital) / simulations) * 100
        
        ruin_threshold = initial_capital * 0.80
        ruined_simulations = np.any(equity_curves < ruin_threshold, axis=1)
        risk_of_ruin = (np.sum(ruined_simulations) / simulations) * 100
        
        median_final_balance = np.median(final_balances)
        
        fig = go.Figure()
        
        x_base = np.arange(sim_length + 1)
        x_all = np.tile(np.append(x_base, np.nan), simulations)
        y_all = np.hstack((equity_curves, np.full((simulations, 1), np.nan))).flatten()
        
        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_base,
            y=median_curve,
            mode='lines',
            line=dict(color='gold', width=3),
            name='Median (50th Percentile)'
        ))
        
        fig.update_layout(
            title='🔬 Monte Carlo Robustness Analysis (Forward 50 Trades)',
            xaxis_title='Trade Number',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig, prob_profit, risk_of_ruin, median_final_balance

    def fetch_data_smart(ticker, timeframe, start_date, end_date):
        import io
        import requests
        
        df = pd.DataFrame()
        
        # Determine asset type
        is_forex = "=X" in ticker
        is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
        is_crypto = "-USD" in ticker
        is_stock = not (is_forex or is_index or is_crypto)
        
        days_requested = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        # ENGINE 1: Alpaca (Primary for US Stocks/ETFs)
        if is_stock and not is_crypto:
            try:
                tf_alpaca = timeframe
                if timeframe == "1d": tf_alpaca = "1Day"
                elif timeframe == "1h": tf_alpaca = "1Hour"
                elif timeframe == "15m": tf_alpaca = "15Min"
                elif timeframe == "5m": tf_alpaca = "5Min"
                elif timeframe == "1m": tf_alpaca = "1Min"
                
                df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            except Exception as e:
                print(f"Alpaca fetch failed: {e}")
        
        clean_ticker = ticker.replace('=X', '').replace('^', '')

        # ENGINE 1: Massive Cloud
        if df.empty:
            try:
                obj = s3_client.get_object(Bucket=MASSIVE_BUCKET, Key=f"{clean_ticker}.csv")
                df_massive = pd.read_csv(io.BytesIO(obj['Body'].read()))
                
                # Standardize columns
                rename_map = {c: c.capitalize() for c in df_massive.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
                if 'timestamp' in df_massive.columns: rename_map['timestamp'] = 'datetime'
                elif 'time' in df_massive.columns: rename_map['time'] = 'datetime'
                elif 'date' in df_massive.columns: rename_map['date'] = 'datetime'
                
                df_massive.rename(columns=rename_map, inplace=True)
                
                if 'datetime' in df_massive.columns:
                    df_massive['datetime'] = pd.to_datetime(df_massive['datetime'])
                    df_massive = df_massive[(df_massive['datetime'] >= pd.to_datetime(start_date)) & (df_massive['datetime'] <= pd.to_datetime(end_date))]
                    
                    if not df_massive.empty:
                        st.success("✅ Dati recuperati dai server cloud Massive.")
                        df = df_massive
            except Exception as e:
                pass

        # ENGINE 2: Local Database
        if df.empty:
            try:
                local_path = os.path.join(LOCAL_DB_DIR, f"{clean_ticker}.csv")
                if os.path.exists(local_path):
                    df_local = pd.read_csv(local_path)
                    
                    # Standardize columns
                    rename_map = {c: c.capitalize() for c in df_local.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
                    if 'timestamp' in df_local.columns: rename_map['timestamp'] = 'datetime'
                    elif 'time' in df_local.columns: rename_map['time'] = 'datetime'
                    elif 'date' in df_local.columns: rename_map['date'] = 'datetime'
                    
                    df_local.rename(columns=rename_map, inplace=True)
                    
                    if 'datetime' in df_local.columns:
                        df_local['datetime'] = pd.to_datetime(df_local['datetime'])
                        df_local = df_local[(df_local['datetime'] >= pd.to_datetime(start_date)) & (df_local['datetime'] <= pd.to_datetime(end_date))]
                        
                        if not df_local.empty:
                            st.success("📂 Dati recuperati dal Database Locale.")
                            df = df_local
            except Exception as e:
                pass

        # ENGINE 3: yfinance (Fallback)
        if df.empty:
            try:
                tf_yf = "1d"
                if timeframe == "1m": tf_yf = "1m"
                elif timeframe == "5m": tf_yf = "5m"
                elif timeframe == "15m": tf_yf = "15m"
                elif timeframe == "1h": tf_yf = "1h"
                elif timeframe == "1d": tf_yf = "1d"
                
                actual_start = start_date
                if tf_yf in ["1m"] and days_requested > 7:
                    actual_start = end_date - timedelta(days=7)
                    st.warning("⚠️ yfinance supporta solo 7 giorni per il timeframe 1m. Date troncate.")
                elif tf_yf in ["5m", "15m", "1h"] and days_requested > 60:
                    actual_start = end_date - timedelta(days=60)
                    st.warning(f"⚠️ yfinance supporta solo 60 giorni per il timeframe {tf_yf}. Date troncate.")
                
                df_yf = yf.download(ticker, start=actual_start, end=end_date, interval=tf_yf, progress=False)
                if not df_yf.empty:
                    df_yf.reset_index(inplace=True)
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)
                    rename_map = {
                        'Date': 'datetime', 'Datetime': 'datetime',
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    }
                    df_yf.rename(columns=rename_map, inplace=True)
                    if 'datetime' not in df_yf.columns and df_yf.index.name in ['Date', 'Datetime']:
                        df_yf.reset_index(inplace=True)
                        df_yf.rename(columns={df_yf.index.name: 'datetime'}, inplace=True)
                    df_yf['datetime'] = pd.to_datetime(df_yf['datetime'])
                    
                    if not df_yf.empty:
                        st.warning("⚠️ Dati presi da Yahoo Finance (Limiti applicati).")
                        df = df_yf
            except Exception as e:
                pass
                
        # ENGINE 4: Fatal Error
        if df.empty:
            st.error("❌ ERRORE CRITICO: Dati non trovati in nessun motore (Alpaca, Massive, Locale, Yahoo). Per favore, carica un file CSV manualmente usando l'apposito uploader per testare questo asset.")
            st.stop()
                
        if not df.empty:
            cols = df.select_dtypes(include=['float64']).columns
            if not cols.empty:
                df[cols] = df[cols].astype('float32')
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            df.sort_values('datetime', inplace=True)
            df.ffill().bfill(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def run_custom_strategy(df, start_time, end_time, eod_close, orb_enabled, orb_duration, initial_capital, risk_per_trade, rr_ratio, sl_mode, fixed_sl_pct):
        trades = []
        if df.empty:
            return trades
            
        df = df.copy()
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        in_position = False
        entry_price = 0
        entry_time = None
        position_type = None
        size = 0
        sl_price = 0
        tp_price = 0
        
        grouped = df.groupby('date')
        
        for date, group in grouped:
            group = group.sort_values('datetime').reset_index(drop=True)
            
            orb_high = None
            orb_low = None
            orb_end_time = None
            
            if orb_enabled:
                if not group.empty:
                    first_candle_time = group.iloc[0]['datetime']
                    orb_end_time = first_candle_time + pd.Timedelta(minutes=orb_duration)
                    
                    orb_data = group[group['datetime'] < orb_end_time]
                    if not orb_data.empty:
                        orb_high = orb_data['High'].max()
                        orb_low = orb_data['Low'].min()
            
            for idx, row in group.iterrows():
                current_time = row['time']
                current_datetime = row['datetime']
                
                is_trading_time = start_time <= current_time <= end_time
                
                if in_position:
                    exit_triggered = False
                    exit_type = ""
                    exit_price = 0
                    
                    if position_type == 'LONG':
                        if row['Low'] <= sl_price:
                            exit_triggered = True
                            exit_type = "SL"
                            exit_price = sl_price
                        elif row['High'] >= tp_price:
                            exit_triggered = True
                            exit_type = "TP"
                            exit_price = tp_price
                    elif position_type == 'SHORT':
                        if row['High'] >= sl_price:
                            exit_triggered = True
                            exit_type = "SL"
                            exit_price = sl_price
                        elif row['Low'] <= tp_price:
                            exit_triggered = True
                            exit_type = "TP"
                            exit_price = tp_price
                            
                    if not exit_triggered and eod_close and current_time >= end_time:
                        exit_triggered = True
                        exit_type = "EOD"
                        exit_price = row['Close']
                    elif not exit_triggered and not is_trading_time:
                        exit_triggered = True
                        exit_type = "Out of Time"
                        exit_price = row['Close']
                    
                    if exit_triggered:
                        pnl = (exit_price - entry_price) if position_type == 'LONG' else (entry_price - exit_price)
                        pnl *= size
                        
                        trades.append({
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': current_datetime,
                            'Exit Price': exit_price,
                            'pnl': pnl,
                            'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                            'Type': 'long' if position_type == 'LONG' else 'short',
                            'Status': exit_type,
                            'type': f'EXIT {exit_type}',
                            'time': current_datetime,
                            'price': exit_price,
                            'Logica': f"Exit: {exit_type}"
                        })
                        in_position = False
                
                if not in_position and is_trading_time:
                    if orb_enabled and orb_high is not None and orb_low is not None:
                        if current_datetime >= orb_end_time:
                            if row['High'] > orb_high:
                                in_position = True
                                position_type = 'LONG'
                                entry_price = max(row['Open'], orb_high)
                                entry_time = current_datetime
                                
                                if sl_mode == "Fixed %":
                                    sl_price = entry_price * (1 - fixed_sl_pct / 100)
                                else:
                                    sl_price = row['Low']
                                    if sl_price >= entry_price:
                                        sl_price = entry_price * 0.999
                                        
                                tp_price = entry_price + ((entry_price - sl_price) * rr_ratio)
                                
                                risk_amount = initial_capital * (risk_per_trade / 100)
                                risk_per_unit = entry_price - sl_price
                                size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                                
                            elif row['Low'] < orb_low:
                                in_position = True
                                position_type = 'SHORT'
                                entry_price = min(row['Open'], orb_low)
                                entry_time = current_datetime
                                
                                if sl_mode == "Fixed %":
                                    sl_price = entry_price * (1 + fixed_sl_pct / 100)
                                else:
                                    sl_price = row['High']
                                    if sl_price <= entry_price:
                                        sl_price = entry_price * 1.001
                                        
                                tp_price = entry_price - ((sl_price - entry_price) * rr_ratio)
                                
                                risk_amount = initial_capital * (risk_per_trade / 100)
                                risk_per_unit = sl_price - entry_price
                                size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                                
                    elif not orb_enabled:
                        if current_time >= start_time:
                            in_position = True
                            position_type = 'LONG'
                            entry_price = row['Close']
                            entry_time = current_datetime
                            
                            if sl_mode == "Fixed %":
                                sl_price = entry_price * (1 - fixed_sl_pct / 100)
                            else:
                                sl_price = row['Low']
                                if sl_price >= entry_price:
                                    sl_price = entry_price * 0.999
                                    
                            tp_price = entry_price + ((entry_price - sl_price) * rr_ratio)
                            
                            risk_amount = initial_capital * (risk_per_trade / 100)
                            risk_per_unit = entry_price - sl_price
                            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                            
        return trades

    if st.button("🚀 Esegui Strategia Custom"):
        with st.spinner("Fetching data and running strategy..."):
            df = fetch_data_smart(ticker, timeframe, start_date, end_date)
            if not df.empty:
                trades = run_custom_strategy(df, start_time, end_time, eod_close, orb_enabled, orb_duration, initial_capital, risk_per_trade, rr_ratio, sl_mode, fixed_sl_pct)
                
                if trades:
                    friction_pct = 0.0
                    adjusted_trades, adjusted_equity = apply_friction_post_process(trades, initial_capital, friction_pct)
                    
                    st.subheader("📊 Risultati Strategia")
                    metrics = calculate_advanced_metrics(adjusted_trades)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    col2.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                    col4.metric("Total Profit", f"${metrics['total_profit_abs']:.2f}")
                    
                    st.line_chart(adjusted_equity)
                    
                    with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                        mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                        if mc_res:
                            mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                            st.plotly_chart(mc_fig, use_container_width=True)
                            
                            st.subheader('🔬 Validazione Statistica Long-Term')
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                            with c2:
                                st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                            with c3:
                                st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                            
                            if prob_profit > 75:
                                st.success('✅ Strategia Robusta')
                            elif prob_profit < 60:
                                st.warning('⚠️ Strategia Fragile (Flop)')
                                
                            if risk_of_ruin > 10:
                                st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                
                            if len(adjusted_trades) < 30:
                                st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                        else:
                            st.warning("Not enough data for Monte Carlo simulation.")
                            
                    st.subheader("📝 Dettaglio Operazioni")
                    st.dataframe(pd.DataFrame(adjusted_trades))
                else:
                    st.warning("Nessun trade generato con questi parametri.")
            else:
                st.error("Errore nel recupero dei dati storici.")
