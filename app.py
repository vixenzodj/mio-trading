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

    # --- PROTEZIONE RATE LIMIT YAHOO FINANCE ---
    try:
        available_dates = ticker_obj.options
    except Exception as e:
        st.error("⚠️ Blocco temporaneo di Yahoo Finance (Rate Limit). Attendi un minuto prima del prossimo aggiornamento.")
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
            
            # --- SALVATAGGIO VOLATILITA' PER CALCOLO DELTA ---
            if 'prev_iv' not in st.session_state:
                st.session_state.prev_iv = mean_iv
            iv_change = mean_iv - st.session_state.prev_iv
            st.session_state.prev_iv = mean_iv

            sd_move = spot * mean_iv * np.sqrt(max(dte_ref, 0.5)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move
            sd2_up, sd2_down = spot + (sd_move * 2), spot - (sd_move * 2)

            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            df['strike_bin'] = (np.round(df['strike'] / gran) * gran)
            agg = df.groupby('strike_bin', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum().rename(columns={'strike_bin': 'strike'})
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
            p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']

            # --- HEADER ---
            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")

            # --- RIPRISTINO INTEGRALE MARKET DIRECTION LOGIC ---
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
            
            # --- REAL-TIME METRIC REGIME ---
            st.markdown(f"### 📊 Real-Time Metric Regime")
            c_reg1, c_reg2, c_reg3, c_reg4 = st.columns(4)
            c_reg1.metric("Net Gamma", f"{net_gamma:,.0f}", delta=f"{'LONG' if net_gamma > 0 else 'SHORT'}")
            c_reg2.metric("Net Vanna", f"{net_vanna:,.0f}", delta=f"{'STABLE' if net_vanna > 0 else 'UNSTABLE'}")
            c_reg3.metric("Net Charm", f"{net_charm:,.0f}", delta=f"{'SUPPORT' if net_charm < 0 else 'DECAY'}")
            c_reg4.metric("Market Regime", "VOL DRIVEN" if net_gamma < 0 else "SPOT DRIVEN")

            # Box Direzione
            st.markdown(f"""
                <div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center; margin-top: 10px; margin-bottom: 25px;'>
                    <b style='color:white; font-size:24px;'>MARKET BIAS: {direction}</b>
                </div>
                """, unsafe_allow_html=True)

            # --- METRICHE MURI ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}"); m2.metric("ZERO GAMMA", f"{z_gamma:.2f}"); m3.metric("PUT WALL", f"{p_wall:.0f}"); m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            # --- CONTROLLO TATTICO E INDICATORE VOLATILITA' ---
            st.markdown("---")
            col_view, col_vol = st.columns([2, 1])
            with col_view:
                view_mode = st.radio("👁️ VISTA GRAFICO:", ["📊 Vista Standard (Metrica Singola)", "🌪️ Vanna View (Overlay Gamma + Vanna)"], horizontal=True)
            with col_vol:
                # Contatore Volatilità IV Media: Il delta inverse significa che se IV sale diventa rosso (negativo per lo scalper)
                st.metric("📈 VOLATILITÀ CHAIN IV (Dinamica)", f"{mean_iv*100:.2f}%", delta=f"{iv_change*100:.2f}%", delta_color="inverse")

            # --- GRAFICO ---
            fig = go.Figure()

            if view_mode == "📊 Vista Standard (Metrica Singola)":
                # COMPORTAMENTO ORIGINALE INVARIATO AL 100%
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg[metric], orientation='h', 
                                     marker=dict(color=['#00FF41' if x >= 0 else '#FF4136' for x in visible_agg[metric]], line_width=0), width=gran * 0.8))
                xaxis_title = f"Net {metric}"
            else:
                # OVERLAY GAMMA/VANNA (Barra nella Barra)
                # Calcolo Vanna Max Negativa (Innesco Volatilità)
                try:
                    max_neg_vanna_idx = visible_agg['Vanna'].idxmin()
                    vanna_trigger_strike = visible_agg.loc[max_neg_vanna_idx, 'strike']
                    vanna_trigger_val = visible_agg.loc[max_neg_vanna_idx, 'Vanna']
                    dist_vanna = spot - vanna_trigger_strike
                    dist_vanna_pct = (dist_vanna / spot) * 100
                except:
                    vanna_trigger_strike = spot
                    dist_vanna = 0; dist_vanna_pct = 0
                    vanna_trigger_val = 0

                alert_color = "#FF4136" if vanna_trigger_val < 0 else "#2ECC40"
                
                # BOX HUD informativo per la Vanna View
                st.markdown(f"""
                <div style='background-color:rgba(30,30,30,1); border: 1px solid {alert_color}; border-radius:5px; padding:10px; display:flex; justify-content:space-around; margin-bottom: 10px;'>
                    <div><b>🌪️ VOL TRIGGER STRIKE:</b> {vanna_trigger_strike:.0f}</div>
                    <div style='color:{alert_color}'><b>DISTANZA SPOT:</b> {dist_vanna:.2f} pts ({dist_vanna_pct:.2f}%)</div>
                    <div><b>STATUS:</b> {'⚠️ CRITICAL FLIP' if vanna_trigger_val < 0 and abs(dist_vanna_pct) < 0.5 else 'MONITOR'}</div>
                </div>
                """, unsafe_allow_html=True)

                # 1. Grafico GAMMA in secondo piano (Semi-Trasparente)
                gamma_colors_overlay = ['rgba(0, 255, 65, 0.25)' if x >= 0 else 'rgba(255, 65, 54, 0.25)' for x in visible_agg['Gamma']]
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg['Gamma'], orientation='h', 
                                     marker=dict(color=gamma_colors_overlay, line_width=0), width=gran * 0.8, name="Gamma (Sfondo)"))
                
                # 2. Grafico VANNA in primo piano (Barra più sottile Ciano/Magenta)
                vanna_colors_overlay = ['#FF00FF' if x < 0 else '#00BFFF' for x in visible_agg['Vanna']]
                fig.add_trace(go.Bar(y=visible_agg['strike'], x=visible_agg['Vanna'], orientation='h', 
                                     marker=dict(color=vanna_colors_overlay, line_width=1, line_color='rgba(255,255,255,0.4)'), width=gran * 0.35, name="Vanna (Primo Piano)"))

                # Aggiunta linea Trigger Max Vanna
                fig.add_hline(y=vanna_trigger_strike, line_color="#FF00FF", line_width=2, line_dash="dashdot", annotation_text="MAX VOL TRIGGER")
                
                fig.update_layout(barmode='overlay')
                xaxis_title = "Gamma vs Vanna Exposure Overlay"

            # --- LINEE ORIGINALI INTATTE PER ENTRAMBE LE VISUALIZZAZIONI ---
            for strike in visible_agg['strike']:
                fig.add_hline(y=strike, line_width=0.3, line_dash="dot", line_color="rgba(255,255,255,0.2)")

            fig.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CW")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PW")
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="dash", annotation_text="+1SD")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="dash", annotation_text="-1SD")
            fig.add_hline(y=sd2_up, line_color="#FF0000", line_dash="dot", annotation_text="+2SD")
            fig.add_hline(y=sd2_down, line_color="#FF0000", line_dash="dot", annotation_text="-2SD")

            fig.update_layout(template="plotly_dark", height=850, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(range=[lo, hi], dtick=gran), xaxis=dict(title=xaxis_title, tickformat="$.2s"))
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
        status_text.text(f"Scansione: {t_name} ({i+1}/{len(tickers_50)})")
        try:
            t_obj = yf.Ticker(t_name)
            hist = t_obj.history(period='5d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            opts = t_obj.options
            if not opts: continue
            target_opt = opts[0] if "0-1 DTE" in expiry_mode else (opts[2] if len(opts) > 2 else opts[0])
            oc = t_obj.option_chain(target_opt)
            df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            dte_years = max((datetime.strptime(target_opt, '%Y-%m-%d') - today).days + 1, 0.5) / 365
            df_scan['dte_years'] = dte_years
            df_scan = df_scan[(df_scan['strike'] > px*0.7) & (df_scan['strike'] < px*1.3)]
            try: zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
            except: zg_val = px
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
            scan_results.append({"Ticker": t_name.replace("^", ""), "Prezzo": round(px, 2), "0-Gamma": round(zg_val, 2), "1SD Range": f"{sd1_down:.0f}-{sd1_up:.0f}", "Dist. 0G %": round(dist_zg_pct, 2), "Analisi": status_label, "_sort": abs(dist_zg_pct)})
        except: continue
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values("_sort").drop(columns=["_sort"])
        def color_logic(val):
            if "🔥" in val: return 'background-color: #8B0000; color: white'
            if "🔴" in val: return 'color: #FF4136; font-weight: bold'
            if "🟢" in val: return 'color: #2ECC40; font-weight: bold'
            if "🟡" in val: return 'color: #FFDC00'
            if "✅" in val: return 'color: #0074D9'
            return ''
        st.dataframe(final_df.style.applymap(color_logic, subset=['Analisi']), use_container_width=True, height=800)
