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

# --- CORE QUANT ENGINE ---
def calculate_gex_at_price(price, df, r=0.045):
    """Calcola l'esposizione Gamma netta dei dealer a un dato prezzo."""
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.0001)
    # OI + 50% Volume come proxy dell'esposizione
    exposure_size = df['openInterest'].fillna(0).values + (df['volume'].fillna(0).values * 0.5)
    
    d1 = (np.log(price/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (price * iv * np.sqrt(T))
    side = np.where(df['type'] == 'call', 1, -1)
    
    return np.sum(gamma * exposure_size * 100 * price * side)

def get_greeks_pro(df, S, r=0.045):
    """Calcola tutte le Greche (Gamma, Vanna, Charm, Vega, Theta)."""
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

# =================================================================
# PAGINA 1: DASHBOARD SINGOLA (INTEGRATA)
# =================================================================
if menu == "🏟️ DASHBOARD SINGOLA":
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛰️ SENTINEL V58 HUB")
    
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "MSTR"]

    new_asset = st.sidebar.text_input("➕ CARICA TICKER", "").upper().strip()
    if new_asset and new_asset not in st.session_state.ticker_list:
        st.session_state.ticker_list.insert(0, new_asset)
        st.rerun()

    asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
    t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
    current_ticker = t_map.get(asset, asset)

    ticker_obj = yf.Ticker(current_ticker)
    h = ticker_obj.history(period='1d')
    if h.empty: 
        st.error("Errore nel caricamento dei dati del Ticker. Riprova.")
        st.stop()
    spot = h['Close'].iloc[-1]

    # --- LOGICA RECUPERO DATE AVANZATA (Dalla tua Versione 2) ---
    available_dates = ticker_obj.options
    all_dates_info = []
    for d in available_dates:
        try:
            dt_obj = datetime.strptime(d, '%Y-%m-%d')
            dte = (dt_obj - today).days + 1
            # Includiamo tutto ciò che è disponibile fino a 90 giorni
            if 0 <= dte <= 90:
                all_dates_info.append({"label": f"{dte} DTE | {d}", "date": d, "dte": dte})
        except: continue
    
    # Ordine cronologico esatto
    all_dates_info = sorted(all_dates_info, key=lambda x: x['dte'])
    date_labels = [x['label'] for x in all_dates_info]

    # Multiselect pre-popolato
    default_sel = date_labels[:3] if date_labels else []
    selected_dte_labels = st.sidebar.multiselect("SCADENZE (TUTTE)", date_labels, default=default_sel)

    metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    
    # --- GRANULARITÀ E ZOOM (Dalla tua Versione 2) ---
    gran = st.sidebar.select_slider("GRANULARITÀ STRIKE", options=[0.5, 1, 2, 5, 10, 20, 25, 50, 100], value=1.0)
    zoom_val = st.sidebar.slider("ZOOM %", 0.5, 20.0, 5.0)

    if selected_dte_labels:
        target_dates = [label.split('| ')[1] for label in selected_dte_labels]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            # Parametri Statistici
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_ref = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            sd_move = spot * mean_iv * np.sqrt(max(dte_ref, 1)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move
            sd2_up, sd2_down = spot + (sd_move * 2), spot - (sd_move * 2)

            # Calcolo Zero Gamma
            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.80, spot * 1.20, args=(raw_data,))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            
            # --- AGGREGAZIONE CON GRANULARITÀ (Dalla Versione 2) ---
            df['strike_bin'] = (np.round(df['strike'] / gran) * gran)
            agg = df.groupby('strike_bin', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
            agg = agg.rename(columns={'strike_bin': 'strike'})
            
            # Filtro Zoom
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            # Muri (Calcolati sulla vista visibile)
            c_wall = visible_agg.loc[visible_agg['Gamma'].idxmax(), 'strike'] if not visible_agg.empty else spot
            p_wall = visible_agg.loc[visible_agg['Gamma'].idxmin(), 'strike'] if not visible_agg.empty else spot

            # --- DISPLAY HEADER ---
            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.2f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            st.markdown("---")
            
            # --- ANALISI DIREZIONALE (Dalla Versione 1 - Molto utile) ---
            st.markdown("### 🛰️ Real-Time Metric Regime")
            net_gamma = agg['Gamma'].sum()
            net_vanna = agg['Vanna'].sum()
            net_charm = agg['Charm'].sum()
            net_vega = agg['Vega'].sum()
            net_theta = agg['Theta'].sum()

            r1, r2, r3, r4, r5 = st.columns(5)
            # Lista metriche
            ms = [("GAMMA", net_gamma, r1), ("VANNA", net_vanna, r2), ("CHARM", net_charm, r3), ("VEGA", net_vega, r4), ("THETA", net_theta, r5)]
            
            for name, val, col in ms:
                col.markdown(f"**{name}**")
                color = '#00FF41' if val > 0 else '#FF4136'
                txt_reg = "POSITIVO" if val > 0 else "NEGATIVO"
                col.markdown(f"<h3 style='color:{color}; margin:0;'>{txt_reg}</h3>", unsafe_allow_html=True)
                col.caption(f"Net: {val/1e6:.2f}M")

            # --- MARKET DIRECTION INDICATOR (Dalla Versione 1) ---
            st.markdown("#### 🧭 MARKET DIRECTION INDICATOR")
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

            st.markdown(f"<div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center;'> <b style='color:black; font-size:20px;'>{direction}</b> </div>", unsafe_allow_html=True)
            st.markdown("---")

            # --- GRAFICO (Engine della Versione 2) ---
            fig = go.Figure()
            # Barre con larghezza basata sulla granularità
            fig.add_trace(go.Bar(
                y=visible_agg['strike'], 
                x=visible_agg[metric], 
                orientation='h',
                marker=dict(color=['#00FF41' if x >= 0 else '#0074D9' for x in visible_agg[metric]], line_width=0),
                width=gran * 0.8 # Larghezza dinamica
            ))
            
            # Linee chiave
            fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G FLIP")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CALL WALL")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PUT WALL")
            
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="longdash", annotation_text="+1SD")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="longdash", annotation_text="-1SD")
            fig.add_hline(y=sd2_up, line_color="#E066FF", line_dash="dashdot", annotation_text="+2SD")
            fig.add_hline(y=sd2_down, line_color="#E066FF", line_dash="dashdot", annotation_text="-2SD")

            fig.update_layout(
                template="plotly_dark", height=800, 
                margin=dict(l=0,r=0,t=0,b=0),
                yaxis=dict(range=[lo, hi], dtick=gran, gridcolor="#333", title="Strike Price"),
                xaxis=dict(title=f"Net {metric}", tickformat="$.3s")
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGINA 2: SCANNER COMPLETO (Dalla Versione 1)
# =================================================================
elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Professional Market Scanner (50 Tickers)")
    st.markdown("Analisi incrociata: **Zero Gamma Flip (0G)** e **Standard Deviation (1SD)**.")
    
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 AGGIORNA SCANNER", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        expiry_mode = st.selectbox("📅 SELEZIONE SCADENZE:", ["0-1 DTE (Scalping/Intraday)", "Prossima Scadenza Mensile (Swing)"])
    
    # LISTA COMPLETA 50 TICKERS
    tickers_50 = [
        "^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", 
        "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", 
        "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", 
        "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", 
        "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA"
    ]
    
    scan_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Ciclo di scansione
    for i, t_name in enumerate(tickers_50):
        status_text.text(f"Scansione: {t_name} ({i+1}/{len(tickers_50)})")
        try:
            # 1. Dati Spot
            t_obj = yf.Ticker(t_name)
            hist = t_obj.history(period='5d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            
            # 2. Gestione Scadenze
            opts = t_obj.options
            if not opts: continue
            
            if "0-1 DTE" in expiry_mode:
                target_opt = opts[0]
            else:
                target_opt = opts[2] if len(opts) > 2 else opts[0]

            oc = t_obj.option_chain(target_opt)
            df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            
            # Calcolo DTE
            try:
                exp_date = datetime.strptime(target_opt, '%Y-%m-%d')
                dte_days = (exp_date - datetime.now()).days + 1
                dte_years = max(dte_days, 0.5) / 365
            except:
                dte_years = 0.5/365
            
            df_scan['dte_years'] = dte_years
            
            # 3. Metriche
            df_scan = df_scan[(df_scan['strike'] > px*0.7) & (df_scan['strike'] < px*1.3)]
            
            try: zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
            except: zg_val = px
            
            avg_iv = df_scan['impliedVolatility'].mean()
            sd_move = px * avg_iv * np.sqrt(dte_years)
            sd1_up, sd1_down = px + sd_move, px - sd_move
            
            # 4. Logica Segnale
            dist_zg_pct = ((px - zg_val) / px) * 100
            is_above_0g = px > zg_val
            near_sd_up = abs(px - sd1_up) / px < 0.005
            near_sd_down = abs(px - sd1_down) / px < 0.005
            
            if not is_above_0g: 
                if near_sd_down: status_label = "🔴 < 0G | TEST -1SD (Bounce?)"
                elif px < sd1_down: status_label = "⚫ < 0G | SOTTO -1SD (Short Ext)"
                elif near_sd_up: status_label = "🟠 < 0G | TEST RESISTENZA"
                else: status_label = "🔻 SOTTO 0G (Short Bias)"
            else: 
                if near_sd_up: status_label = "🟡 > 0G | TEST +1SD (Breakout?)"
                elif px > sd1_up: status_label = "🟢 > 0G | SOPRA +1SD (Long Ext)"
                elif near_sd_down: status_label = "🟢 > 0G | DIP BUY (Test -1SD)"
                else: status_label = "✅ SOPRA 0G (Long Bias)"
                    
            if abs(dist_zg_pct) < 0.3: status_label = "🔥 FLIP IMMINENTE (0G)"

            scan_results.append({
                "Ticker": t_name.replace("^", ""),
                "Prezzo": round(px, 2),
                "0-Gamma": round(zg_val, 2),
                "Dist. 0G %": round(dist_zg_pct, 2),
                "Analisi": status_label,
                "_sort_key": abs(dist_zg_pct)
            })
        except: continue
        progress_bar.progress((i + 1) / len(tickers_50))

    status_text.empty()
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values(by="_sort_key")
        final_df = final_df.drop(columns=["_sort_key"])
        
        def color_logic(val):
            if "🔥" in val: return 'background-color: #8B0000; color: white; font-weight: bold'
            if "🔴" in val: return 'color: #FF4136; font-weight: bold'
            if "⚫" in val: return 'background-color: black; color: #FF4136'
            if "🟢" in val: return 'color: #2ECC40; font-weight: bold'
            if "🟡" in val: return 'color: #FFDC00; font-weight: bold'
            if "✅" in val: return 'color: #0074D9'
            if "🔻" in val: return 'color: #FF851B'
            return ''

        st.dataframe(
            final_df.style.applymap(color_logic, subset=['Analisi']),
            use_container_width=True, 
            height=800,
            column_config={
                "Dist. 0G %": st.column_config.NumberColumn(format="%.2f %%"),
                "Prezzo": st.column_config.NumberColumn(format="$ %.2f"),
            }
        )
    else:
        st.error("Nessun dato trovato. Riprova più tardi.")
