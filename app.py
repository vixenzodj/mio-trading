import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE & CACHE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V43", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh") # Refresh ogni 5 minuti per non saturare la cache

# Funzione per scaricare i dati con cache (velocizza lo zoom)
@st.cache_data(ttl=300)
def fetch_option_data(ticker_str, target_dates):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    for d in target_dates:
        oc = t_obj.option_chain(d)
        c_df = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
        p_df = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
        payload.append(pd.concat([c_df, p_df], ignore_index=True))
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

@st.cache_data(ttl=60)
def get_spot_price(ticker_str):
    t_obj = yf.Ticker(ticker_str)
    return t_obj.history(period='1d')['Close'].iloc[-1]

# --- MOTORE DI CALCOLO ---
def engine_v43(df_input, spot_price, r_rate=0.045):
    if df_input.empty: return df_input
    df = df_input.copy()
    s = float(spot_price)
    k = df['strike'].to_numpy()
    v = np.where(df['impliedVolatility'].to_numpy() <= 0, 1e-9, df['impliedVolatility'].to_numpy())
    t = np.where(df['dte_years'].to_numpy() <= 0, 1e-9, df['dte_years'].to_numpy())
    oi = df['openInterest'].to_numpy()
    
    d1 = (np.log(s/k) + (r_rate + 0.5 * v**2) * t) / (v * np.sqrt(t))
    pdf = norm.pdf(d1)
    
    gamma_val = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna_val = s * pdf * d1 / v * 0.01 * oi
    charm_val = (pdf * (r_rate / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    direction = np.where(df['type'].values == 'call', 1, -1)
    df['Gamma'] = gamma_val * direction
    df['Vanna'] = vanna_val * direction
    df['Charm'] = charm_val * direction
    return df

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEXBOT V43 - FAST")
active_t = st.sidebar.selectbox("ASSET", ["SPX", "NDX", "SPY", "QQQ", "TSLA", "NVDA", "AAPL", "MSFT", "IBIT"])

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

ticker_str = fix_ticker(active_t)
t_obj_init = yf.Ticker(ticker_str)
all_exps = t_obj_init.options

# Mappatura DTE
today_dt = datetime.now()
dte_options = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today_dt).days + 1} DTE ({ex})" for ex in all_exps]
selected_dte = st.sidebar.multiselect("SCADENZE", options=dte_options, default=dte_options[:1])

# Parametri Visuali
strike_granularity = st.sidebar.selectbox("GRANULARITÀ STRIKE", [1, 5, 10, 25, 50], index=2)
zoom_pts = st.sidebar.slider("ZOOM (Range Punti dallo Spot)", 50, 1000, 250)
metric_choice = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'])

if ticker_str and selected_dte:
    try:
        spot = get_spot_price(ticker_str)
        target_dates = [sel.split('(')[1].replace(')', '') for sel in selected_dte]
        
        # Caricamento (da Cache o API)
        full_df = fetch_option_data(ticker_str, target_dates)
        full_df['dte_years'] = full_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today_dt).days + 0.5) / 365
        
        # Calcolo Greche
        processed_df = engine_v43(full_df, spot)
        final_summary = processed_df.groupby('strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
        
        # Zero Gamma
        final_summary['cum_sum_gamma'] = final_summary['Gamma'].cumsum()
        zero_gamma_level = final_summary.loc[final_summary['cum_sum_gamma'].abs().idxmin(), 'strike']

        # --- DASHBOARD ---
        st.markdown(f"## 🏛️ {active_t} Terminal | Spot: {spot:.2f}")
        m1, m2, m3, m4 = st.columns(4)
        g_net = final_summary['Gamma'].sum()
        v_net = final_summary['Vanna'].sum()
        
        m1.metric("NET GEX", f"${g_net/1e6:.1f}M")
        m2.metric("NET VANNA", f"${v_net/1e6:.1f}M")
        m3.metric("ZERO GAMMA", f"{zero_gamma_level:.0f}")
        m4.metric("DTE ATTIVI", len(selected_dte))

        # --- FILTRO ZOOM FISSO ---
        # Filtriamo prima del binning per non perdere precisione
        plot_data = final_summary[(final_summary['strike'] >= spot - zoom_pts) & (final_summary['strike'] <= spot + zoom_pts)].copy()
        plot_data['strike_bin'] = (np.round(plot_data['strike'] / strike_granularity) * strike_granularity)
        plot_data = plot_data.groupby('strike_bin', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()

        # --- GRAFICO PRO ---
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_data['strike_bin'], x=plot_data[metric_choice], orientation='h',
            marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in plot_data[metric_choice]],
            width=strike_granularity * 0.7, # Spazio tra le barre per chiarezza
            hovertemplate="Strike: %{y}<br>Value: %{x:,.0f}<extra></extra>"
        ))
        
        # Linee Target
        fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
        fig.add_hline(y=zero_gamma_level, line_color="yellow", line_dash="dash", annotation_text="0-G")
        
        fig.update_layout(
            template="plotly_dark", height=850,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="STRIKE", dtick=strike_granularity, gridcolor="#333", range=[spot-zoom_pts, spot+zoom_pts]),
            xaxis=dict(title=f"Net {metric_choice} Exposure ($)", zerolinecolor="white")
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.error(f"Errore: {e}")
