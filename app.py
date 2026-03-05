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
        """
        Tries to fetch data from Alpaca first, falls back to yfinance if empty or error.
        """
        df = pd.DataFrame()
        
        # 1. Try Alpaca
        try:
            # Convert timeframe to Alpaca format if needed
            tf_alpaca = timeframe
            if timeframe == "1D": tf_alpaca = "1Day"
            elif timeframe == "1H": tf_alpaca = "1Hour"
            elif timeframe == "15Min": tf_alpaca = "15Min"
            elif timeframe == "5Min": tf_alpaca = "5Min"
            
            df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Alpaca fetch failed: {e}")

        # 2. Fallback to yfinance if Alpaca failed or returned empty
        if df.empty:
            st.warning(f"⚠️ Dati Alpaca non disponibili per {ticker}. Tentativo con Yahoo Finance (storico più profondo).")
            try:
                # Convert timeframe to yfinance format
                tf_yf = "1d"
                if timeframe == "1D": tf_yf = "1d"
                elif timeframe == "1H": tf_yf = "1h"
                elif timeframe == "15Min": tf_yf = "15m"
                elif timeframe == "5Min": tf_yf = "5m"
                
                # yfinance download
                df = yf.download(ticker, start=start_date, end=end_date, interval=tf_yf, progress=False)
                
                if not df.empty:
                    # Standardize columns
                    df.reset_index(inplace=True)
                    # Handle MultiIndex columns if present (yfinance update)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    # Rename to match system standard
                    rename_map = {
                        'Date': 'datetime', 'Datetime': 'datetime',
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    }
                    df.rename(columns=rename_map, inplace=True)
                    
                    # Ensure datetime column exists
                    if 'datetime' not in df.columns and df.index.name in ['Date', 'Datetime']:
                        df.reset_index(inplace=True)
                        df.rename(columns={df.index.name: 'datetime'}, inplace=True)
                    
                    # Filter by date just in case
                    df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception as e:
                st.error(f"Errore Yahoo Finance: {e}")
                
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
            # Core Indicators
            self.df['RSI'] = TechnicalIndicators.rsi(self.df['Close'])
            self.df['MACD'], self.df['MACD_Signal'] = TechnicalIndicators.macd(self.df['Close'])
            self.df['BB_Upper'], self.df['BB_Lower'] = TechnicalIndicators.bollinger_bands(self.df['Close'])
            self.df['ATR'] = TechnicalIndicators.atr(self.df)
            self.df['SMA200'] = TechnicalIndicators.sma(self.df['Close'], 200)
            self.df['SMA50'] = TechnicalIndicators.sma(self.df['Close'], 50)
            self.df['EMA20'] = TechnicalIndicators.ema(self.df['Close'], 20)
            self.df['Stoch_K'], self.df['Stoch_D'] = TechnicalIndicators.stochastic(self.df)
            self.df['ADX'] = TechnicalIndicators.adx(self.df)
            self.df['CCI'] = TechnicalIndicators.cci(self.df)
            self.df['WilliamsR'] = TechnicalIndicators.williams_r(self.df)
            self.df['ROC'] = TechnicalIndicators.roc(self.df['Close'])
            self.df['OBV'] = TechnicalIndicators.obv(self.df)
            self.df['MFI'] = TechnicalIndicators.mfi(self.df)
            
            # New Indicators
            self.df['HMA20'] = TechnicalIndicators.hma(self.df['Close'], 20)
            self.df['TEMA20'] = TechnicalIndicators.tema(self.df['Close'], 20)
            self.df['DEMA20'] = TechnicalIndicators.dema(self.df['Close'], 20)
            self.df['KAMA20'] = TechnicalIndicators.kama(self.df['Close'], 20)
            self.df['Aroon_Up'], self.df['Aroon_Down'] = TechnicalIndicators.aroon(self.df)
            self.df['SuperTrend_Upper'], self.df['SuperTrend_Lower'] = TechnicalIndicators.supertrend(self.df)
            self.df['Parabolic_SAR'] = TechnicalIndicators.parabolic_sar(self.df)
            self.df['TSI'] = TechnicalIndicators.tsi(self.df['Close'])
            self.df['UO'] = TechnicalIndicators.uo(self.df)
            self.df['KC_Upper'], self.df['KC_Lower'] = TechnicalIndicators.keltner_channels(self.df)
            self.df['DC_Upper'], self.df['DC_Lower'] = TechnicalIndicators.donchian_channels(self.df)
            self.df['Chaikin_Vol'] = TechnicalIndicators.chaikin_volatility(self.df)
            self.df['CMF'] = TechnicalIndicators.cmf(self.df)
            self.df['VWAP'] = TechnicalIndicators.vwap(self.df)
            self.df['AD_Line'] = TechnicalIndicators.ad_line(self.df)

            # New Indicators (Batch 2)
            self.df['Vortex_Plus'], self.df['Vortex_Minus'] = TechnicalIndicators.vortex(self.df)
            self.df['Chop_Index'] = TechnicalIndicators.chop(self.df)
            self.df['KST'] = TechnicalIndicators.kst(self.df)
            self.df['Coppock'] = TechnicalIndicators.coppock(self.df)
            self.df['Tenkan'], self.df['Kijun'], self.df['SpanA'], self.df['SpanB'], self.df['Chikou'] = TechnicalIndicators.ichimoku(self.df)
            self.df['AO'] = TechnicalIndicators.ao(self.df)
            self.df['PPO'] = TechnicalIndicators.ppo(self.df)
            self.df['Mass_Index'] = TechnicalIndicators.mass_index(self.df)
            self.df['Ulcer_Index'] = TechnicalIndicators.ulcer_index(self.df)

            # New Indicators (Batch 3)
            self.df['WMA20'] = TechnicalIndicators.wma(self.df['Close'], 20)
            self.df['TRIMA20'] = TechnicalIndicators.trima(self.df['Close'], 20)
            self.df['CMO'] = TechnicalIndicators.cmo(self.df['Close'])
            self.df['MOM10'] = TechnicalIndicators.mom(self.df['Close'], 10)
            self.df['BOP'] = TechnicalIndicators.bop(self.df)
            self.df['TRIX'] = TechnicalIndicators.trix(self.df['Close'])
            self.df['StochRSI'] = TechnicalIndicators.stochrsi(self.df['Close'])
            self.df['STDDEV'] = TechnicalIndicators.stddev(self.df['Close'])
            self.df['TSF'] = TechnicalIndicators.tsf(self.df['Close'])

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
            best_win_rate = 0
            best_params = {}
            best_time_config = {'start': None, 'end': None}
            results = []
            
            # Parameter Combinations
            keys = list(param_ranges.keys())
            import itertools
            values = [param_ranges[k] for k in keys]
            param_combinations = list(itertools.product(*values))
            
            # Time Combinations
            time_combinations = [(None, None)] # Default: No schedule
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
                # Validate time range
                if t_start and t_end and t_start >= t_end: continue

                for combo in param_combinations:
                    counter += 1
                    current_params = dict(zip(keys, combo))
                    
                    # Update Progress
                    if counter % 10 == 0 or counter == total_iterations:
                        progress = counter / total_iterations
                        elapsed = time.time() - start_time_perf
                        estimated_total = elapsed / progress if progress > 0 else 0
                        remaining = estimated_total - elapsed
                        progress_bar.progress(progress)
                        status_text.text(f"Ottimizzazione in corso... {counter}/{total_iterations} (Stimato rimanente: {remaining:.1f}s)")

                    trades, _ = self.run_technical_strategy(strategy_type, current_params, risk_reward, risk_per_trade, start_time=t_start, end_time=t_end)
                    
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        
                        # Score: Win Rate weighted by number of trades (to avoid 100% WR with 1 trade)
                        # Simple logic: Prefer higher WR, but if WR is equal, prefer more trades.
                        # For now, just maximize WR, but require min trades?
                        
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_params = current_params
                            best_time_config = {'start': t_start, 'end': t_end}
                        
                        results.append({
                            'params': current_params, 
                            'time_config': {'start': t_start, 'end': t_end},
                            'win_rate': win_rate, 
                            'trades': len(trades),
                            'pnl': df_res['pnl'].sum()
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
                        trades.append({'time': curr['datetime'], 'type': f'EXIT {exit_res}', 'price': exit_price, 'pnl': pnl, 'balance': balance})
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
                    
                    if signal == 'long':
                        sl = entry_price - sl_dist
                        tp = entry_price + (sl_dist * risk_reward)
                        size = risk_amt / (entry_price - sl) if (entry_price - sl) > 0 else 0
                        if size > 0:
                            position = {'type': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': entry_price, 'pnl': 0, 'balance': balance})
                    else:
                        sl = entry_price + sl_dist
                        tp = entry_price - (sl_dist * risk_reward)
                        size = risk_amt / (sl - entry_price) if (sl - entry_price) > 0 else 0
                        if size > 0:
                            position = {'type': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': entry_price, 'pnl': 0, 'balance': balance})
                
                equity_curve.append(balance)
                
            return trades, equity_curve

        def run_technical_strategy(self, strategy_type, params, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard", sl_atr_mult=1.5):
            trades = []
            balance = self.initial_capital
            equity_curve = [balance]
            position = None
            
            for i in range(len(self.df)):
                if i < 200: continue
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
                        trades.append({'time': curr['datetime'], 'type': f'EXIT {exit_res}', 'price': exit_price, 'pnl': pnl, 'balance': balance})
                        position = None
                    equity_curve.append(balance)
                    continue

                # Entry Logic (Dynamic based on Strategy Type)
                signal = None
                
                if strategy_type == "RSI Mean Reversion":
                    # RSI Oversold/Overbought
                    if prev['RSI'] < params['os'] and curr['RSI'] > params['os']: signal = 'long'
                    elif prev['RSI'] > params['ob'] and curr['RSI'] < params['ob']: signal = 'short'
                    
                elif strategy_type == "MACD Crossover":
                    # MACD Line crosses Signal Line
                    if prev['MACD'] < prev['MACD_Signal'] and curr['MACD'] > curr['MACD_Signal']: signal = 'long'
                    elif prev['MACD'] > prev['MACD_Signal'] and curr['MACD'] < curr['MACD_Signal']: signal = 'short'
                    
                elif strategy_type == "Bollinger Breakout":
                    # Close breaks Upper/Lower Band
                    if prev['Close'] < prev['BB_Lower'] and curr['Close'] > prev['BB_Lower']: signal = 'long' # Reversal from bottom
                    elif prev['Close'] > prev['BB_Upper'] and curr['Close'] < prev['BB_Upper']: signal = 'short' # Reversal from top
                    
                elif strategy_type == "Golden/Death Cross":
                    # SMA 50 crosses SMA 200
                    if prev['SMA50'] < prev['SMA200'] and curr['SMA50'] > curr['SMA200']: signal = 'long'
                    elif prev['SMA50'] > prev['SMA200'] and curr['SMA50'] < curr['SMA200']: signal = 'short'
                    
                elif strategy_type == "Stochastic Oscillator":
                    # K crosses D in OS/OB zones
                    if prev['Stoch_K'] < params['os'] and curr['Stoch_K'] > params['os']: signal = 'long'
                    elif prev['Stoch_K'] > params['ob'] and curr['Stoch_K'] < params['ob']: signal = 'short'

                elif strategy_type == "CCI Momentum":
                    if prev['CCI'] < -100 and curr['CCI'] > -100: signal = 'long'
                    elif prev['CCI'] > 100 and curr['CCI'] < 100: signal = 'short'

                elif strategy_type == "Williams %R Reversal":
                    if prev['WilliamsR'] < -80 and curr['WilliamsR'] > -80: signal = 'long'
                    elif prev['WilliamsR'] > -20 and curr['WilliamsR'] < -20: signal = 'short'

                # --- NEW STRATEGIES ---
                elif strategy_type == "HMA Trend":
                    # HMA Slope
                    if prev['HMA20'] < curr['HMA20'] and prev['Close'] > curr['HMA20']: signal = 'long'
                    elif prev['HMA20'] > curr['HMA20'] and prev['Close'] < curr['HMA20']: signal = 'short'

                elif strategy_type == "TEMA Crossover":
                    if prev['Close'] < prev['TEMA20'] and curr['Close'] > curr['TEMA20']: signal = 'long'
                    elif prev['Close'] > prev['TEMA20'] and curr['Close'] < curr['TEMA20']: signal = 'short'

                elif strategy_type == "KAMA Trend":
                    if prev['KAMA20'] < curr['KAMA20']: signal = 'long'
                    elif prev['KAMA20'] > curr['KAMA20']: signal = 'short'

                elif strategy_type == "Aroon Oscillator":
                    if prev['Aroon_Up'] < prev['Aroon_Down'] and curr['Aroon_Up'] > curr['Aroon_Down']: signal = 'long'
                    elif prev['Aroon_Up'] > prev['Aroon_Down'] and curr['Aroon_Up'] < curr['Aroon_Down']: signal = 'short'

                elif strategy_type == "SuperTrend Reversal":
                    if prev['Close'] < prev['SuperTrend_Upper'] and curr['Close'] > curr['SuperTrend_Lower']: signal = 'long' # Trend change
                    elif prev['Close'] > prev['SuperTrend_Lower'] and curr['Close'] < curr['SuperTrend_Upper']: signal = 'short'

                elif strategy_type == "Parabolic SAR":
                    if prev['Parabolic_SAR'] > prev['Close'] and curr['Parabolic_SAR'] < curr['Close']: signal = 'long'
                    elif prev['Parabolic_SAR'] < prev['Close'] and curr['Parabolic_SAR'] > curr['Close']: signal = 'short'

                elif strategy_type == "TSI Crossover":
                    if prev['TSI'] < 0 and curr['TSI'] > 0: signal = 'long'
                    elif prev['TSI'] > 0 and curr['TSI'] < 0: signal = 'short'

                elif strategy_type == "UO Overbought/Oversold":
                    if prev['UO'] < 30 and curr['UO'] > 30: signal = 'long'
                    elif prev['UO'] > 70 and curr['UO'] < 70: signal = 'short'

                elif strategy_type == "Keltner Channel Breakout":
                    if prev['Close'] < prev['KC_Upper'] and curr['Close'] > curr['KC_Upper']: signal = 'long'
                    elif prev['Close'] > prev['KC_Lower'] and curr['Close'] < curr['KC_Lower']: signal = 'short'

                elif strategy_type == "Donchian Channel Breakout":
                    if curr['Close'] >= prev['DC_Upper']: signal = 'long'
                    elif curr['Close'] <= prev['DC_Lower']: signal = 'short'

                elif strategy_type == "Chaikin Volatility":
                    # Volatility expansion
                    if prev['Chaikin_Vol'] < 0 and curr['Chaikin_Vol'] > 0: signal = 'long' # Just a trigger example

                elif strategy_type == "CMF Trend":
                    if prev['CMF'] < 0 and curr['CMF'] > 0: signal = 'long'
                    elif prev['CMF'] > 0 and curr['CMF'] < 0: signal = 'short'

                elif strategy_type == "VWAP Crossover":
                    if prev['Close'] < prev['VWAP'] and curr['Close'] > curr['VWAP']: signal = 'long'
                    elif prev['Close'] > prev['VWAP'] and curr['Close'] < curr['VWAP']: signal = 'short'

                elif strategy_type == "AD Line Trend":
                    if prev['AD_Line'] < curr['AD_Line'] and prev['Close'] > curr['Close']: signal = 'long' # Divergence-ish

                # --- NEW STRATEGIES (BATCH 2) ---
                elif strategy_type == "Vortex Crossover":
                    if prev['Vortex_Plus'] < prev['Vortex_Minus'] and curr['Vortex_Plus'] > curr['Vortex_Minus']: signal = 'long'
                    elif prev['Vortex_Plus'] > prev['Vortex_Minus'] and curr['Vortex_Plus'] < curr['Vortex_Minus']: signal = 'short'

                elif strategy_type == "Choppiness Index Breakout":
                    if prev['Chop_Index'] > 61.8 and curr['Chop_Index'] < 61.8: signal = 'long' # Breakout from chop
                    elif prev['Chop_Index'] < 38.2 and curr['Chop_Index'] > 38.2: signal = 'short' # Entering chop? (Simplified logic)

                elif strategy_type == "KST Crossover":
                    if prev['KST'] < 0 and curr['KST'] > 0: signal = 'long'
                    elif prev['KST'] > 0 and curr['KST'] < 0: signal = 'short'

                elif strategy_type == "Coppock Curve":
                    if prev['Coppock'] < 0 and curr['Coppock'] > 0: signal = 'long'
                    elif prev['Coppock'] > 0 and curr['Coppock'] < 0: signal = 'short'

                elif strategy_type == "Ichimoku Cloud Breakout":
                    # Simple Kumo Breakout
                    if prev['Close'] < prev['SpanA'] and curr['Close'] > curr['SpanA'] and curr['Close'] > curr['SpanB']: signal = 'long'
                    elif prev['Close'] > prev['SpanA'] and curr['Close'] < curr['SpanA'] and curr['Close'] < curr['SpanB']: signal = 'short'

                elif strategy_type == "Awesome Oscillator":
                    if prev['AO'] < 0 and curr['AO'] > 0: signal = 'long'
                    elif prev['AO'] > 0 and curr['AO'] < 0: signal = 'short'

                elif strategy_type == "PPO Crossover":
                    if prev['PPO'] < 0 and curr['PPO'] > 0: signal = 'long'
                    elif prev['PPO'] > 0 and curr['PPO'] < 0: signal = 'short'

                elif strategy_type == "Mass Index Reversal":
                    if prev['Mass_Index'] > 27 and curr['Mass_Index'] < 27: signal = 'long' # Reversal bulge

                elif strategy_type == "Ulcer Index Safety":
                    if prev['Ulcer_Index'] > 5 and curr['Ulcer_Index'] < 5: signal = 'long' # Volatility calming down

                # --- NEW STRATEGIES (BATCH 3) ---
                elif strategy_type == "WMA Trend":
                    if prev['WMA20'] < curr['WMA20'] and prev['Close'] > curr['WMA20']: signal = 'long'
                    elif prev['WMA20'] > curr['WMA20'] and prev['Close'] < curr['WMA20']: signal = 'short'

                elif strategy_type == "TRIMA Crossover":
                    if prev['Close'] < prev['TRIMA20'] and curr['Close'] > curr['TRIMA20']: signal = 'long'
                    elif prev['Close'] > prev['TRIMA20'] and curr['Close'] < curr['TRIMA20']: signal = 'short'

                elif strategy_type == "CMO Reversal":
                    if prev['CMO'] < -50 and curr['CMO'] > -50: signal = 'long'
                    elif prev['CMO'] > 50 and curr['CMO'] < 50: signal = 'short'

                elif strategy_type == "Momentum Breakout":
                    if prev['MOM10'] < 0 and curr['MOM10'] > 0: signal = 'long'
                    elif prev['MOM10'] > 0 and curr['MOM10'] < 0: signal = 'short'

                elif strategy_type == "BOP Trend":
                    if prev['BOP'] < 0 and curr['BOP'] > 0: signal = 'long'
                    elif prev['BOP'] > 0 and curr['BOP'] < 0: signal = 'short'

                elif strategy_type == "TRIX Crossover":
                    if prev['TRIX'] < 0 and curr['TRIX'] > 0: signal = 'long'
                    elif prev['TRIX'] > 0 and curr['TRIX'] < 0: signal = 'short'

                elif strategy_type == "StochRSI Reversal":
                    if prev['StochRSI'] < 0.2 and curr['StochRSI'] > 0.2: signal = 'long'
                    elif prev['StochRSI'] > 0.8 and curr['StochRSI'] < 0.8: signal = 'short'

                elif strategy_type == "TSF Trend":
                    if prev['TSF'] < curr['TSF'] and prev['Close'] > curr['TSF']: signal = 'long'
                    elif prev['TSF'] > curr['TSF'] and prev['Close'] < curr['TSF']: signal = 'short'


                if signal:
                    entry_price = curr['Close']
                    sl_dist = curr['ATR'] * sl_atr_mult
                    
                    if entry_mode == "Retest":
                        entry_price = (curr['High'] + curr['Low']) / 2
                    
                    risk_amt = balance * (risk_per_trade / 100)
                    
                    if signal == 'long':
                        sl = entry_price - sl_dist
                        tp = entry_price + (sl_dist * risk_reward)
                        size = risk_amt / (entry_price - sl) if (entry_price - sl) > 0 else 0
                        if size > 0:
                            position = {'type': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': entry_price, 'pnl': 0, 'balance': balance})
                    else:
                        sl = entry_price + sl_dist
                        tp = entry_price - (sl_dist * risk_reward)
                        size = risk_amt / (sl - entry_price) if (sl - entry_price) > 0 else 0
                        if size > 0:
                            position = {'type': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': entry_price, 'pnl': 0, 'balance': balance})
                
                equity_curve.append(balance)
                
            return trades, equity_curve

    class Visualizer:
        @staticmethod
        def plot_tradingview_clone(df, trades, engine_type="Hybrid", strategy_name=""):
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df['datetime'],
                            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            name='Price'))
            
            # Indicators based on Engine
            if engine_type == "Hybrid":
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['ZeroGamma'], name='Zero Gamma', line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['CallWall'], name='Call Wall', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['PutWall'], name='Put Wall', line=dict(color='red', dash='dash')))
            else:
                # Technical Indicators Visualization
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA50'], name='SMA 50', line=dict(color='cyan', width=1)))
                
                if "Bollinger" in strategy_name:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')))
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot')))

            # Signals
            buy_signals = [t for t in trades if 'ENTRY LONG' in t['type']]
            sell_signals = [t for t in trades if 'ENTRY SHORT' in t['type']]
            
            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in buy_signals], 
                    y=[t['price'] for t in buy_signals],
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy Signal'
                ))
            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in sell_signals], 
                    y=[t['price'] for t in sell_signals],
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell Signal'
                ))

            fig.update_layout(
                title=f"TradingView Clone - {engine_type} Strategy ({strategy_name})",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600,
                yaxis_title="Price",
                xaxis_title="Date"
            )
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
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    # Metrics
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        total_pnl = df_res['pnl'].sum()
                        pf = df_res[df_res['pnl'] > 0]['pnl'].sum() / abs(df_res[df_res['pnl'] < 0]['pnl'].sum()) if len(df_res[df_res['pnl'] < 0]) > 0 else float('inf')
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Profitto Netto", f"${total_pnl:.2f}")
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Profit Factor", f"{pf:.2f}")
                        m4.metric("Saldo Finale", f"${equity[-1]:.2f}")
                        
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, trades, "Hybrid"), use_container_width=True)
                        st.line_chart(equity)
                        st.dataframe(df_res)
                    else:
                        st.warning("Nessuna operazione generata con le impostazioni correnti.")
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
        
        # Generic fallback for others to avoid errors if params needed
        if not params:
             # Add generic period if strategy might need it (heuristic)
             if "period" not in params: params['period'] = 14
        
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
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        total_pnl = df_res['pnl'].sum()
                        pf = df_res[df_res['pnl'] > 0]['pnl'].sum() / abs(df_res[df_res['pnl'] < 0]['pnl'].sum()) if len(df_res[df_res['pnl'] < 0]) > 0 else float('inf')
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Profitto Netto", f"${total_pnl:.2f}")
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Profit Factor", f"{pf:.2f}")
                        m4.metric("Saldo Finale", f"${equity[-1]:.2f}")
                        
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, trades, "Technical", strategy_type), use_container_width=True)
                        st.line_chart(equity)
                        st.dataframe(df_res)
                    else:
                        st.warning("Nessuna operazione generata con le impostazioni correnti.")
            
            if c_opt.button("🧠 Ottimizza Strategia (AI)"):
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # --- FAST OPTIMIZATION LOGIC ---
                # 1. Prepare Data (Downsample if needed)
                engine.add_technical_indicators() # Ensure ATR and other base indicators are present
                opt_df = engine.df.copy()
                if len(opt_df) > 10000:
                    status_text.text(f"⚠️ Dati troppo estesi ({len(opt_df)} candele). Ottimizzazione sugli ultimi 5000 periodi per velocità.")
                    opt_df = opt_df.iloc[-5000:].reset_index(drop=True)
                
                # 2. Define Ranges
                opt_config = {}
                recalc_func = None
                
                if strategy_type == "RSI Mean Reversion":
                    opt_config = {'period': range(10, 22, 2), 'ob': range(65, 85, 5), 'os': range(20, 40, 5)}
                    def r_func(d, p): d['RSI'] = TechnicalIndicators.rsi(d['Close'], p['period'])
                    recalc_func = r_func
                elif strategy_type == "Stochastic Oscillator":
                    opt_config = {'k_period': range(10, 20, 2), 'ob': range(75, 95, 5), 'os': range(5, 25, 5)}
                    def r_func(d, p): d['Stoch_K'], d['Stoch_D'] = TechnicalIndicators.stochastic(d, p['k_period'])
                    recalc_func = r_func
                elif strategy_type == "Bollinger Breakout":
                    opt_config = {'period': [20, 30], 'std_dev': [2.0, 2.5]}
                    def r_func(d, p): d['BB_Upper'], d['BB_Lower'] = TechnicalIndicators.bollinger_bands(d['Close'], p['period'], p['std_dev'])
                    recalc_func = r_func
                else:
                    # Generic fallback
                    opt_config = {'dummy': [1]}
                
                rr_ranges = [1.5, 2.0, 2.5, 3.0]
                
                # 3. Generate Combinations
                import itertools
                keys = list(opt_config.keys())
                values = [opt_config[k] for k in keys]
                param_combos = list(itertools.product(*values))
                
                total_steps = len(param_combos) * len(rr_ranges)
                step = 0
                best_res = {'wr': 0, 'pnl': -float('inf'), 'params': {}, 'rr': 0}
                
                # 4. Fast Loop
                for p_vals in param_combos:
                    curr_p = dict(zip(keys, p_vals))
                    
                    # Recalculate Indicators (Vectorized)
                    if recalc_func:
                        try: recalc_func(opt_df, curr_p)
                        except: continue
                    else:
                        engine.add_technical_indicators() # Ensure base exists
                        
                    # Generate Signals (Vectorized)
                    # This is a simplified signal generation for common strategies to speed up scanning
                    # For complex ones, we might fallback to iteration, but here we try vector
                    sigs = pd.Series(0, index=opt_df.index)
                    
                    try:
                        if strategy_type == "RSI Mean Reversion":
                            rsi_arr = opt_df['RSI'].values
                            # Long: prev < os and curr > os
                            sigs = np.where((rsi_arr[:-1] < curr_p['os']) & (rsi_arr[1:] > curr_p['os']), 1, 
                                   np.where((rsi_arr[:-1] > curr_p['ob']) & (rsi_arr[1:] < curr_p['ob']), -1, 0))
                            # Pad the first element lost by slicing
                            sigs = np.insert(sigs, 0, 0)
                            
                        elif strategy_type == "Stochastic Oscillator":
                            k_arr = opt_df['Stoch_K'].values
                            sigs = np.where((k_arr[:-1] < curr_p['os']) & (k_arr[1:] > curr_p['os']), 1,
                                   np.where((k_arr[:-1] > curr_p['ob']) & (k_arr[1:] < curr_p['ob']), -1, 0))
                            sigs = np.insert(sigs, 0, 0)
                            
                        elif strategy_type == "Bollinger Breakout":
                            c = opt_df['Close'].values
                            l = opt_df['BB_Lower'].values
                            u = opt_df['BB_Upper'].values
                            sigs = np.where((c[:-1] < l[:-1]) & (c[1:] > l[1:]), 1,
                                   np.where((c[:-1] > u[:-1]) & (c[1:] < u[1:]), -1, 0))
                            sigs = np.insert(sigs, 0, 0)
                            
                        else:
                            # Fallback: Use the engine's logic but on the smaller DF (slower but compatible)
                            # We skip this for now to ensure speed for the main requested strategies
                            pass
                            
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
                        if k == 'period': st.session_state['period_rsi'] = v; st.session_state['period_cci'] = v; st.session_state['period_williams'] = v; st.session_state['period_bb'] = v
                        if k == 'ob': st.session_state['ob_rsi'] = v; st.session_state['ob_stoch'] = v
                        if k == 'os': st.session_state['os_rsi'] = v; st.session_state['os_stoch'] = v
                        if k == 'k_period': st.session_state['k_stoch'] = v
                        if k == 'std_dev': st.session_state['std_bb'] = v
                        
                    st.session_state['rr'] = best_res['rr']
                    st.session_state['run_backtest_auto'] = True
                    
                    st.rerun()
                else:
                    st.error("Nessun risultato valido trovato.")

        else:
            st.info("⚠️ Esegui prima la 'Verifica Disponibilità Dati Storici' per abilitare la simulazione.")



