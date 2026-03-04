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
    st.title("🔙 Backtesting Strategia GEX (Advanced Builder)")
    
    st.markdown("""
    <div style='background-color:rgba(0, 100, 255, 0.1); padding:15px; border-radius:5px; border: 1px solid #0074D9; margin-bottom: 20px;'>
    <b>🛠️ COSTRUTTORE STRATEGIE:</b> Qui puoi simulare migliaia di combinazioni.
    Scegli le date, definisci i <b>Trigger di Ingresso</b> (es. Rimbalzo sui Muri, Breakout 0-Gamma) e decidi se attendere la <b>Chiusura Candela</b> per confermare il segnale.
    </div>
    """, unsafe_allow_html=True)

# --- BACKTESTING ENGINE & VISUALIZER ---

class TechnicalIndicators:
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        # Basic implementation of SuperTrend
        hl2 = (df['High'] + df['Low']) / 2
        atr = TechnicalIndicators.atr(df, period)
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        return upper, lower # Simplified for brevity, full logic requires trend state

    @staticmethod
    def atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

class BacktestEngine:
    def __init__(self, ticker, start_date, end_date, timeframe, initial_capital=10000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.df = pd.DataFrame()

    def fetch_data(self):
        # Uses existing fetch_alpaca_history logic but ensures max depth
        self.df = fetch_alpaca_history(self.ticker, self.timeframe, self.start_date, self.end_date)
        return not self.df.empty

    def add_technical_indicators(self):
        if self.df.empty: return
        self.df['RSI'] = TechnicalIndicators.rsi(self.df['Close'])
        self.df['MACD'], self.df['MACD_Signal'] = TechnicalIndicators.macd(self.df['Close'])
        self.df['BB_Upper'], self.df['BB_Lower'] = TechnicalIndicators.bollinger_bands(self.df['Close'])
        self.df['ATR'] = TechnicalIndicators.atr(self.df)
        self.df['SMA200'] = self.df['Close'].rolling(window=200).mean()

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

    def run_hybrid_strategy(self, long_trigger, short_trigger, risk_reward, risk_per_trade):
        trades = []
        balance = self.initial_capital
        equity_curve = [balance]
        position = None

        for i in range(len(self.df)):
            if i < 200: continue # Warmup
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
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
                risk_amt = balance * (risk_per_trade / 100)
                sl_dist = curr['ATR'] * 2
                
                if signal == 'long':
                    sl = curr['Close'] - sl_dist
                    tp = curr['Close'] + (sl_dist * risk_reward)
                    size = risk_amt / (curr['Close'] - sl) if (curr['Close'] - sl) > 0 else 0
                    if size > 0:
                        position = {'type': 'long', 'entry': curr['Close'], 'sl': sl, 'tp': tp, 'size': size}
                        trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': curr['Close'], 'pnl': 0, 'balance': balance})
                else:
                    sl = curr['Close'] + sl_dist
                    tp = curr['Close'] - (sl_dist * risk_reward)
                    size = risk_amt / (sl - curr['Close']) if (sl - curr['Close']) > 0 else 0
                    if size > 0:
                        position = {'type': 'short', 'entry': curr['Close'], 'sl': sl, 'tp': tp, 'size': size}
                        trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': curr['Close'], 'pnl': 0, 'balance': balance})
            
            equity_curve.append(balance)
            
        return trades, equity_curve

    def run_technical_strategy(self, rsi_period, rsi_overbought, rsi_oversold, risk_reward, risk_per_trade):
        trades = []
        balance = self.initial_capital
        equity_curve = [balance]
        position = None
        
        # Recalculate RSI with custom period if needed
        self.df['RSI_Strat'] = TechnicalIndicators.rsi(self.df['Close'], rsi_period)

        for i in range(len(self.df)):
            if i < 200: continue
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Exit Logic (Same as Hybrid)
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

            # Entry Logic (Technical)
            signal = None
            if prev['RSI_Strat'] < rsi_oversold and curr['RSI_Strat'] > rsi_oversold: signal = 'long'
            elif prev['RSI_Strat'] > rsi_overbought and curr['RSI_Strat'] < rsi_overbought: signal = 'short'

            if signal:
                risk_amt = balance * (risk_per_trade / 100)
                sl_dist = curr['ATR'] * 2
                
                if signal == 'long':
                    sl = curr['Close'] - sl_dist
                    tp = curr['Close'] + (sl_dist * risk_reward)
                    size = risk_amt / (curr['Close'] - sl) if (curr['Close'] - sl) > 0 else 0
                    if size > 0:
                        position = {'type': 'long', 'entry': curr['Close'], 'sl': sl, 'tp': tp, 'size': size}
                        trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': curr['Close'], 'pnl': 0, 'balance': balance})
                else:
                    sl = curr['Close'] + sl_dist
                    tp = curr['Close'] - (sl_dist * risk_reward)
                    size = risk_amt / (sl - curr['Close']) if (sl - curr['Close']) > 0 else 0
                    if size > 0:
                        position = {'type': 'short', 'entry': curr['Close'], 'sl': sl, 'tp': tp, 'size': size}
                        trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': curr['Close'], 'pnl': 0, 'balance': balance})
            
            equity_curve.append(balance)
            
        return trades, equity_curve

class Visualizer:
    @staticmethod
    def plot_tradingview_clone(df, trades, engine_type="Hybrid"):
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
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=2)))

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
            title=f"TradingView Clone - {engine_type} Strategy",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=600,
            yaxis_title="Price",
            xaxis_title="Date"
        )
        return fig

# --- MAIN UI REPLACEMENT ---
if menu == "🛠️ BACKTESTING STRATEGIA":
    st.title("🛠️ Professional Backtesting Suite")
    
    # Engine Selection
    engine_choice = st.radio("Seleziona Motore di Backtesting:", 
                             ["🧬 MOTORE A: GEX & Options Hybrid Simulator", 
                              "📈 MOTORE B: Technical Strategy Hub (Pure Trading)"], 
                             horizontal=True)
    
    # Common Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1: ticker = st.text_input("Ticker", value="SPY")
    with c2: timeframe = st.selectbox("Timeframe", ["1D", "1H", "15Min", "5Min"], index=0)
    with c3: 
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
    with c4: 
        end_date = st.date_input("End Date", value=datetime.now())
        initial_capital = st.number_input("Capital", value=10000)

    engine = BacktestEngine(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe, initial_capital)

    if "MOTORE A" in engine_choice:
        st.subheader("🧬 GEX Hybrid Strategy Configuration")
        col1, col2 = st.columns(2)
        with col1:
            long_trigger = st.selectbox("Long Trigger", ["Bounce Put Wall", "Breakout 0-Gamma", "Breakout Call Wall", "None"])
        with col2:
            short_trigger = st.selectbox("Short Trigger", ["Bounce Call Wall", "Breakdown 0-Gamma", "Breakdown Put Wall", "None"])
        
        sensitivity = st.slider("GEX Sensitivity", 0.5, 3.0, 1.5, 0.1)
        rr = st.slider("Risk:Reward", 1.0, 5.0, 2.0, 0.5)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        
        if st.button("🚀 Run GEX Simulation"):
            with st.spinner("Downloading Data & Calculating GEX Walls..."):
                if engine.fetch_data():
                    engine.add_technical_indicators() # Need ATR
                    engine.add_gex_levels(sensitivity)
                    trades, equity = engine.run_hybrid_strategy(long_trigger, short_trigger, rr, risk_pct)
                    
                    # Results
                    st.success(f"Simulation Complete. Total Trades: {len(trades)}")
                    
                    # Metrics
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        total_pnl = df_res['pnl'].sum()
                        pf = df_res[df_res['pnl'] > 0]['pnl'].sum() / abs(df_res[df_res['pnl'] < 0]['pnl'].sum()) if len(df_res[df_res['pnl'] < 0]) > 0 else float('inf')
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Net Profit", f"${total_pnl:.2f}")
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Profit Factor", f"{pf:.2f}")
                        m4.metric("Final Balance", f"${equity[-1]:.2f}")
                        
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, trades, "Hybrid"), use_container_width=True)
                        st.line_chart(equity)
                        st.dataframe(df_res)
                    else:
                        st.warning("No trades generated with current settings.")
                else:
                    st.error("Failed to fetch data.")

    else: # MOTORE B
        st.subheader("📈 Technical Strategy Hub Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.number_input("RSI Period", 14)
        with col2:
            rsi_ob = st.number_input("RSI Overbought", 70)
        with col3:
            rsi_os = st.number_input("RSI Oversold", 30)
            
        rr = st.slider("Risk:Reward", 1.0, 5.0, 2.0, 0.5)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        
        if st.button("🚀 Run Technical Backtest"):
            with st.spinner("Downloading Deep History & Calculating Indicators..."):
                if engine.fetch_data():
                    engine.add_technical_indicators()
                    trades, equity = engine.run_technical_strategy(rsi_period, rsi_ob, rsi_os, rr, risk_pct)
                    
                    # Results
                    st.success(f"Simulation Complete. Total Trades: {len(trades)}")
                    
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        total_pnl = df_res['pnl'].sum()
                        pf = df_res[df_res['pnl'] > 0]['pnl'].sum() / abs(df_res[df_res['pnl'] < 0]['pnl'].sum()) if len(df_res[df_res['pnl'] < 0]) > 0 else float('inf')
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Net Profit", f"${total_pnl:.2f}")
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Profit Factor", f"{pf:.2f}")
                        m4.metric("Final Balance", f"${equity[-1]:.2f}")
                        
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, trades, "Technical"), use_container_width=True)
                        st.line_chart(equity)
                        st.dataframe(df_res)
                    else:
                        st.warning("No trades generated.")
                else:
                    st.error("Failed to fetch data.")


