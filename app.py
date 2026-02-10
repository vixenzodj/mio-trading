import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V41", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE BLINDATO ---
def fast_engine_v41(df, spot, r=0.045):
    if df.empty: return df
    s = float(spot)
    k = df['strike'].values
    # Pulizia IV e DTE per evitare divisioni per zero
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    t = np.where(df['dte_years'].values <= 0, 1e-9, df['dte_years'].values)
    oi = df['openInterest'].values
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Dollar Gamma Formula: 0.5 * Gamma * S^2 * 0.01
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    is_call = (df['type'].values == 'call')
    mult = np.where(is_call, 1, -1)
    
    # Creazione nuove colonne in modo sicuro
    df['Gamma'] = gamma * mult
    df['Vanna'] = vanna * mult
    df['Charm'] = charm * mult
    return df

# --- SIDEBAR: LOGICA GEXBOT ---
st.sidebar.header("🕹️ GEXBOT ENGINE V41")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        all_exps = t_obj.options
        today = datetime.now()
        
        # Mappatura DTE
        exp_list = []
        for e in all_exps:
            d = datetime.strptime(e, '%Y-%m-%d')
            dte = (d - today).days + 1
            exp_list.append({'date': e, 'dte': dte})
        
        df_exp_map = pd.DataFrame(exp_list)
        
        # Selezione multipla DTE (Gexbot Style)
        dte_labels = [f"{row['dte']} DTE ({row['date']})" for _, row in df_exp_map.iterrows()]
        selected_labels = st.sidebar.multiselect("SCADENZE (DTE)", options=dte_labels, default=dte_labels[:2])
        
        # Estrazione date selezionate
        selected_dates = [label.split('(')[1].replace(')', '') for label in selected_labels]
        
        strike_step = st.sidebar.selectbox("GRANULARITÀ STRIKE", [1, 5, 10, 25, 50, 100], index=3)
        zoom_range = st.sidebar.slider("ZOOM AREA (+/- % dallo Spot)", 1, 20, 5)
        main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'])

        hist = t_obj.history(period='1d')
        if not hist.empty and selected_dates:
            spot = float(hist['Close'].iloc[-1])
            
            # Caricamento massivo
            all_data = []
            for d_str in selected_dates:
                chain = t_obj.option_chain(d_str)
                c = chain.calls[['strike', 'impliedVolatility', 'openInterest']].copy().assign(type='call', exp=d_str)
                p = chain.puts[['strike', 'impliedVolatility', 'openInterest']].copy().assign(type='put', exp=d_str)
                all_data.extend([c, p])
            
            df_raw = pd.concat(all_data, ignore_index=True)
            
            # Calcolo DTE temporale
            df_raw['dte_years'] = df_raw['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            # --- FIX DUPLICATE COLUMNS: Grouping radicale ---
            df_clean = df_raw.groupby(['strike', 'type', 'dte_years'], as_index=False).agg({
                'impliedVolatility': 'mean', 'openInterest': 'sum'
            })
            
            # Esecuzione calcoli
            df_res = fast_engine_v41(df_clean, spot)
            
            # Consolidamento finale per Strike (unendo le scadenze)
            df_final = df_res.groupby('strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()

            # --- LOGICA ZERO GAMMA ---
            df_final['cum_gamma'] = df_final['Gamma'].cumsum()
            z_gamma = df_final.loc[df_final['cum_gamma'].abs().idxmin(), 'strike']

            # --- CAMPANELLI DI REGIME ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            
            def get_regime(val):
                if val > 0.5: return "🛡️ POSITIVO", "#00ff00"
                if val < -0.5: return "⚠️ NEGATIVO", "#ff4444"
                return "⚖️ NEUTRALE", "#ffff00"

            net_g = df_final['Gamma'].sum()
            net_v = df_final['Vanna'].sum()
            
            m1, m2, m3, m4 = st.columns(4)
            
            # Indicatori dinamici
            g_status, g_color = get_regime(net_g / df_final['Gamma'].abs().sum())
            m1.markdown(f"<div style='border:1px solid {g_color}; padding:10px; border-radius:10px; text-align:center;'><b>GAMMA REGIME</b><br><span style='color:{g_color}; font-size:20px;'>{g_status}</span></div>", unsafe_allow_html=True)
            
            v_status, v_color = get_regime(net_v / df_final['Vanna'].abs().sum())
            m2.markdown(f"<div style='border:1px solid {v_color}; padding:10px; border-radius:10px; text-align:center;'><b>VANNA REGIME</b><br><span style='color:{v_color}; font-size:20px;'>{v_status}</span></div>", unsafe_allow_html=True)
            
            m3.metric("ZERO GAMMA", f"{z_gamma:.0f}")
            m4.metric("NET EXPOSURE", f"${net_g/1e6:.1f}M")

            # --- FILTRO ZOOM PERCENTUALE ---
            lower_b = spot * (1 - zoom_range/100)
            upper_b = spot * (1 + zoom_range/100)
            df_plot = df_final[(df_final['strike'] >= lower_b) & (df_final['strike'] <= upper_b)].copy()
            
            # Binning per il grafico
            df_plot['bin'] = np.floor(df_plot['strike'] / strike_step) * strike_step
            df_plot = df_plot.groupby('bin', as_index=False).sum().rename(columns={'bin': 'strike'})

            # --- GRAFICO GEXBOT ---
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]],
                width=strike_step * 0.8
            ))
            
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="0-GAMMA")
            
            fig.update_layout(template="plotly_dark", height=800, 
                              yaxis=dict(dtick=strike_step, title="STRIKE"),
                              xaxis=dict(title=f"Net {main_metric} Exposure"))
            
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA GRANULARE ---
            st.markdown("### 📊 Market Maker Inventory")
            st.dataframe(df_plot.sort_values('strike', ascending=False).style.format(precision=1).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Charm']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
