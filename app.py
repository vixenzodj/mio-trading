import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V42", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE DI CALCOLO (Senza conflitti di nomi) ---
def engine_v42(df_input, spot_price, r_rate=0.045):
    if df_input.empty: return df_input
    
    # Lavoriamo su una copia pulita per evitare "SettingWithCopyWarning"
    df = df_input.copy()
    s = float(spot_price)
    
    # Estrazione array numpy per evitare conflitti di etichette/nomi
    k = df['strike'].to_numpy()
    v = np.where(df['impliedVolatility'].to_numpy() <= 0, 1e-9, df['impliedVolatility'].to_numpy())
    t = np.where(df['dte_years'].to_numpy() <= 0, 1e-9, df['dte_years'].to_numpy())
    oi = df['openInterest'].to_numpy()
    
    d1 = (np.log(s/k) + (r_rate + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Calcolo Dollar Gamma (Standard Istituzionale)
    gamma_val = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna_val = s * pdf * d1 / v * 0.01 * oi
    charm_val = (pdf * (r_rate / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    is_call = (df['type'].values == 'call')
    direction = np.where(is_call, 1, -1)
    
    # Assegnazione pulita
    df['Gamma'] = gamma_val * direction
    df['Vanna'] = vanna_val * direction
    df['Charm'] = charm_val * direction
    
    return df

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEXBOT ENGINE V42")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
ticker_str = fix_ticker(active_t)

if ticker_str:
    t_obj = yf.Ticker(ticker_str)
    try:
        all_exps = t_obj.options
        today_dt = datetime.now()
        
        # Mappatura DTE come richiesto (Gexbot style)
        dte_mapping = []
        for ex in all_exps:
            dt_obj = datetime.strptime(ex, '%Y-%m-%d')
            days_diff = (dt_obj - today_dt).days + 1
            dte_mapping.append({'date': ex, 'dte': days_diff})
        
        df_map = pd.DataFrame(dte_mapping)
        dte_options = [f"{r['dte']} DTE ({r['date']})" for _, r in df_map.iterrows()]
        
        selected_dte_labels = st.sidebar.multiselect("SCADENZE (DTE)", options=dte_options, default=dte_options[:2])
        target_dates = [sel.split('(')[1].replace(')', '') for sel in selected_dte_labels]
        
        strike_granularity = st.sidebar.selectbox("GRANULARITÀ STRIKE", [1, 5, 10, 25, 50, 100], index=3)
        zoom_pct = st.sidebar.slider("ZOOM AREA (+/- %)", 1, 15, 5)
        metric_choice = st.sidebar.radio("METRICA VISIVA", ['Gamma', 'Vanna', 'Charm'])

        hist_data = t_obj.history(period='1d')
        if not hist_data.empty and target_dates:
            spot = float(hist_data['Close'].iloc[-1])
            
            # Caricamento e Unione SENZA duplicare colonne 'strike'
            payload = []
            for d in target_dates:
                oc = t_obj.option_chain(d)
                # Selezioniamo solo le colonne necessarie PRIMA del calcolo
                c_df = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
                p_df = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
                payload.append(pd.concat([c_df, p_df], ignore_index=True))
            
            # Unione finale con azzeramento indici
            full_df = pd.concat(payload, ignore_index=True)
            full_df['dte_years'] = full_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today_dt).days + 0.5) / 365
            
            # Calcolo Greche
            processed_df = engine_v42(full_df, spot)
            
            # --- CONSOLIDAMENTO AGGREGATO (Risolve l'errore 'strike' 2 times) ---
            # Raggruppiamo solo per strike e sommiamo le greche
            final_summary = processed_df.groupby('strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
            
            # Calcolo Zero Gamma (Zero Flip)
            final_summary['cum_sum_gamma'] = final_summary['Gamma'].cumsum()
            zero_gamma_level = final_summary.loc[final_summary['cum_sum_gamma'].abs().idxmin(), 'strike']

            # --- INTERFACCIA ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            
            # Indicatori di Regime (I "Campanelli")
            m1, m2, m3, m4 = st.columns(4)
            g_net = final_summary['Gamma'].sum()
            v_net = final_summary['Vanna'].sum()
            
            def regime_box(label, val, total_abs):
                score = val / total_abs if total_abs > 0 else 0
                color = "#00ff00" if score > 0.2 else "#ff4444" if score < -0.2 else "#ffff00"
                text = "POSITIVO" if score > 0.2 else "NEGATIVO" if score < -0.2 else "NEUTRALE"
                return f"<div style='border:2px solid {color}; padding:10px; border-radius:10px; text-align:center;'><b>{label}</b><br><span style='color:{color}; font-size:18px;'>{text} ({score:+.2f})</span></div>"

            m1.markdown(regime_box("GAMMA REGIME", g_net, final_summary['Gamma'].abs().sum()), unsafe_allow_html=True)
            m2.markdown(regime_box("VANNA REGIME", v_net, final_summary['Vanna'].abs().sum()), unsafe_allow_html=True)
            m3.metric("ZERO GAMMA", f"{zero_gamma_level:.0f}")
            m4.metric("NET GEX", f"${g_net/1e6:.1f}M")

            # --- FILTRO ZOOM & GRAFICO ---
            lb, ub = spot * (1 - zoom_pct/100), spot * (1 + zoom_pct/100)
            plot_data = final_summary[(final_summary['strike'] >= lb) & (final_summary['strike'] <= ub)].copy()
            
            # Arrotondamento per granularità scelta
            plot_data['strike_bin'] = np.round(plot_data['strike'] / strike_granularity) * strike_granularity
            plot_data = plot_data.groupby('strike_bin', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()

            # Plotly (Gexbot Style)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=plot_data['strike_bin'], x=plot_data[metric_choice], orientation='h',
