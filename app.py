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
    
    # Calcolo metriche per dashboard e indicatori
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi_vol_weighted * 100 * side
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi_vol_weighted * side
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi_vol_weighted * 100 * side
    df['Vega']  = S * pdf * np.sqrt(T) * 0.01 * oi_vol_weighted * 100
    df['Theta'] = ((-(S * pdf * iv) / (2 * np.sqrt(T))) - side * (r * K * np.exp(-r * T) * norm.cdf(d2 * side))) * (1/365) * oi_vol_weighted * 100
    
    # Calcolo valore assoluto per la visualizzazione bidirezionale dei muri
    df['Gamma_Abs'] = abs(df['Gamma'])
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

# =================================================================
# PAGINA 1: DASHBOARD SINGOLA
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
    if h.empty: st.stop()
    spot = h['Close'].iloc[-1]

    available_dates = ticker_obj.options
    today = datetime.now()
    date_options = [f"{(datetime.strptime(d, '%Y-%m-%d') - today).days + 1} DTE | {d}" for d in available_dates]
    selected_dte = st.sidebar.multiselect("SCADENZE", date_options, default=[date_options[0]])

    metric = st.sidebar.radio("METRICA GRAFICO", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    gran = st.sidebar.select_slider("GRANULARITÀ", options=[1, 2, 5, 10, 20, 25, 50, 100, 250], value=5)
    zoom_val = st.sidebar.slider("ZOOM %", 0.5, 15.0, 3.0)

    if selected_dte:
        target_dates = [d.split('| ')[1] for d in selected_dte]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data['dte_years'] = raw_data['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            avg_dte = raw_data['dte_years'].mean() * 365
            
            # --- CALCOLI VOLATILITÀ E LIVELLI ---
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_min = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            sd_move = spot * mean_iv * np.sqrt(max(dte_min, 1)/365)
            sd1_up, sd1_down = spot + sd_move, spot - sd_move
            sd2_up, sd2_down = spot + (sd_move * 2), spot - (sd_move * 2)

            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data,))
            except: z_gamma = spot 

            df = get_greeks_pro(raw_data, spot)
            
            # Calcolo Muri (per intestazione)
            agg_for_walls = df.groupby(['strike', 'type'], as_index=False)['Gamma_Abs'].sum()
            c_wall = agg_for_walls[agg_for_walls['type']=='call'].sort_values('Gamma_Abs', ascending=False).iloc[0]['strike'] if not agg_for_walls[agg_for_walls['type']=='call'].empty else spot
            p_wall = agg_for_walls[agg_for_walls['type']=='put'].sort_values('Gamma_Abs', ascending=False).iloc[0]['strike'] if not agg_for_walls[agg_for_walls['type']=='put'].empty else spot

            # --- DISPLAY METRICHE ---
            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA", f"{z_gamma:.2f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{sd_move:.2f}")

            st.markdown("---")
            st.markdown("### 🛰️ Real-Time Metric Regime & Market Direction")
            
            agg = df.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            net_vega, net_theta = agg['Vega'].sum(), agg['Theta'].sum()

            r1, r2, r3, r4, r5 = st.columns(5)
            for name, val, col in [("GAMMA", net_gamma, r1), ("VANNA", net_vanna, r2), ("CHARM", net_charm, r3), ("VEGA", net_vega, r4), ("THETA", net_theta, r5)]:
                reg = "POSITIVO" if val > 0 else "NEGATIVO"
                col.markdown(f"**{name}**")
                col.markdown(f"<h3 style='color:{'#00FF41' if val > 0 else '#FF4136'}; margin:0;'>{reg}</h3>", unsafe_allow_html=True)
                col.caption(f"Net: ${val/1e6:.2f}M")

            # --- MARKET DIRECTION INDICATOR (Logica Originale) ---
            st.markdown("#### 🧭 MARKET DIRECTION INDICATOR")
            direction = "NEUTRALE / ATTESA"; bias_color = "gray"
            if net_gamma < 0 and net_vanna < 0: direction = "🔴 PERICOLO ESTREMO: SHORT GAMMA + NEGATIVE VANNA (Crash Risk)"; bias_color = "#8B0000"
            elif net_gamma < 0: direction = "🔴 ACCELERAZIONE VOLATILITÀ (Short Gamma Bias)"; bias_color = "#FF4136"
            elif spot < z_gamma: direction = "🟠 PRESSIONE DI VENDITA (Sotto Zero Gamma)"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0: direction = "🟢 REVERSIONE VERSO LO SPOT (Charm Support)"; bias_color = "#2ECC40"
            elif net_gamma > 0 and abs(net_theta) > abs(net_vega): direction = "⚪ CONSOLIDAMENTO / THETA DECAY (Range Bound)"; bias_color = "#AAAAAA"
            else: direction = "🔵 LONG GAMMA / STABILITÀ (Bassa Volatilità)"; bias_color = "#0074D9"

            st.markdown(f"<div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center;'> <b style='color:black; font-size:20px;'>{direction}</b> </div>", unsafe_allow_html=True)
            st.markdown("---")

            # --- GRAFICO BIDIREZIONALE (Stile GexBot) ---
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            
            fig = go.Figure()

            # Processamento dati per il grafico (separazione Call e Put)
            for t_type, color, name in [('call', '#00FF41', 'CALLS'), ('put', '#FF4136', 'PUTS')]:
                p_df = df[df['type'] == t_type].copy()
                p_df['bin'] = (np.round(p_df['strike'] / gran) * gran)
                binned = p_df.groupby('bin', as_index=False)[metric].sum()
                binned = binned[(binned['bin'] >= lo) & (binned['bin'] <= hi)]
                
                # Le Put vengono visualizzate a sinistra (valori negativi sull'asse X)
                x_vals = binned[metric] if t_type == 'call' else -abs(binned[metric])
                
                fig.add_trace(go.Bar(
                    y=binned['bin'], x=x_vals, orientation='h',
                    name=name, marker_color=color,
                    width=gran * 0.8, hoverinfo='y+x'
                ))

            # --- LINEE CHIAVE ---
            fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G FLIP")
            fig.add_hline(y=c_wall, line_color="#00FF41", line_width=2, annotation_text="CALL WALL")
            fig.add_hline(y=p_wall, line_color="#FF4136", line_width=2, annotation_text="PUT WALL")
            
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="longdash", annotation_text="1SD")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="longdash")
            fig.add_hline(y=sd2_up, line_color="#E066FF", line_dash="dashdot", annotation_text="2SD EXTREME")
            fig.add_hline(y=sd2_down, line_color="#E066FF", line_dash="dashdot")

            fig.update_layout(
                template="plotly_dark", height=850, barmode='relative',
                margin=dict(l=0,r=0,t=30,b=0),
                yaxis=dict(range=[lo, hi], dtick=gran, gridcolor="#333", title="STRIKE PRICE"),
                xaxis=dict(title=f"PRO {metric.upper()} EXPOSURE (PUTS <--- | ---> CALLS)", showgrid=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGINA 2: SCANNER 50 TICKER (Codice Originale Intatto)
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
    
    tickers_50 = ["^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA"]
    
    scan_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t_name in enumerate(tickers_50):
        status_text.text(f"Scansione: {t_name} ({i+1}/{len(tickers_50)})")
        try:
            t_obj = yf.Ticker(t_name)
            hist = t_obj.history(period='1d')
            if hist.empty: continue
            px = hist['Close'].iloc[-1]
            opts = t_obj.options
            target_opt = opts[0] if "0-1 DTE" in expiry_mode else (opts[2] if len(opts)>2 else opts[0])
            oc = t_obj.option_chain(target_opt)
            df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
            dte_days = (datetime.strptime(target_opt, '%Y-%m-%d') - datetime.now()).days + 1
            df_scan['dte_years'] = max(dte_days, 0.5) / 365
            
            zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
            avg_iv = df_scan['impliedVolatility'].mean()
            sd_move = px * avg_iv * np.sqrt(df_scan['dte_years'].iloc[0])
            sd1_up, sd1_down = px + sd_move, px - sd_move
            
            dist_zg_pct = ((px - zg_val) / px) * 100
            is_above_0g = px > zg_val
            
            # Logica Analisi
            if not is_above_0g: status_label = "🔴 < 0G | SOTTO PRESSIONE"
            else: status_label = "✅ > 0G | ZONA STABILE"
            if abs(dist_zg_pct) < 0.4: status_label = "🔥 FLIP IMMINENTE (0G)"

            scan_results.append({
                "Ticker": t_name.replace("^", ""), "Prezzo": round(px, 2),
                "0-Gamma": round(zg_val, 2), "1SD Range": f"{sd1_down:.0f}-{sd1_up:.0f}",
                "Dist. 0G %": round(dist_zg_pct, 2), "Analisi": status_label, "_sort": abs(dist_zg_pct)
            })
        except: continue
        progress_bar.progress((i + 1) / len(tickers_50))

    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values("_sort").drop(columns=["_sort"])
        st.dataframe(final_df.style.applymap(lambda v: 'color: #FF4136' if '🔴' in v else ('color: #2ECC40' if '✅' in v else 'background-color: #8B0000' if '🔥' in v else ''), subset=['Analisi']), use_container_width=True, height=800)
