import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO V49 - DEEP ENGINE", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# --- MOTORE DI CALCOLO GRECHE ---
def calculate_greeks_pro(df, S, r=0.045):
    if df.empty: return df
    K = df['strike'].values
    iv = np.maximum(df['impliedVolatility'].values, 0.001)
    T = np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    side = np.where(df['type'] == 'call', 1, -1)
    
    # Dollar Exposure Formulas (Institutional Standard)
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi * 100 * side
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi * side
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi * 100 * side
    df['Vega']  = S * pdf * np.sqrt(T) * 0.01 * oi * 100
    df['Theta'] = ((-(S * pdf * iv) / (2 * np.sqrt(T))) - side * (r * K * np.exp(-r * T) * norm.cdf(d2 * side))) * (1/365) * oi * 100
    return df

# --- RECUPERO DATI OTTIMIZZATO ---
@st.cache_data(ttl=300, show_spinner=False)
def get_full_chain(ticker_str, target_dates):
    t_obj = yf.Ticker(ticker_str)
    all_data = []
    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            c = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
            p = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
            all_data.append(pd.concat([c, p]))
        except: continue
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("🧬 GEX V49 DEEP")
asset = st.sidebar.selectbox("ASSET", ["NDX", "SPX", "QQQ", "SPY", "NVDA", "TSLA", "MSTR"])
ticker_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
ticker_str = ticker_map.get(asset, asset)

# Prezzo Spot
t_obj = yf.Ticker(ticker_str)
h = t_obj.history(period='1d')
if h.empty:
    st.error("Errore dati Spot. Riprova più tardi.")
    st.stop()
spot = h['Close'].iloc[-1]

# Scadenze
all_exps = t_obj.options
today = datetime.now()
dte_opts = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today).days + 1} DTE ({ex})" for ex in all_exps]
selected_exps = st.sidebar.multiselect("SCADENZE ATTIVE", dte_opts, default=dte_opts[:2])

# Parametri Grafici
metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"], horizontal=True)
granularity = st.sidebar.selectbox("GRANULARITÀ STRIKE", [1, 5, 10, 25, 50, 100], index=2)
zoom = st.sidebar.slider("ZOOM AREA %", 1, 40, 10)

if selected_exps:
    clean_dates = [x.split('(')[1].replace(')', '') for x in selected_exps]
    raw_df = get_full_chain(ticker_str, clean_dates)
    
    if not raw_df.empty:
        raw_df['dte_years'] = raw_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
        df_calc = calculate_greeks_pro(raw_df, spot)
        
        # Aggregazione Totale
        total = df_calc.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
        
        # Calcolo Livelli Chiave (Muri Globali)
        call_wall = total.loc[total['Gamma'].idxmax(), 'strike']
        put_wall = total.loc[total['Gamma'].idxmin(), 'strike']
        
        # Zero Gamma Interpolato
        try:
            sort_g = total.sort_values('strike')
            f_interp = interp1d(sort_g['Gamma'].cumsum(), sort_g['strike'], fill_value="extrapolate")
            zero_gamma = float(f_interp(0))
        except: zero_gamma = spot

        # --- DASHBOARD ---
        st.markdown(f"## 🏟️ {asset} Structure | Spot: {spot:.2f}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CALL WALL", f"{call_wall:.0f}")
        c2.metric("PUT WALL", f"{put_wall:.0f}")
        c3.metric("ZERO GAMMA", f"{zero_gamma:.1f}")
        c4.metric(f"NET {metric.upper()}", f"${total[metric].sum()/1e6:.1f}M")

        # --- PLOT ---
        lb, ub = spot * (1 - zoom/100), spot * (1 + zoom/100)
        # Filtriamo prima di binnare per mantenere la precisione dei muri
        plot_df = total[(total['strike'] >= lb) & (total['strike'] <= ub)].copy()
        plot_df['bin'] = (np.round(plot_df['strike'] / granularity) * granularity)
        plot_df = plot_df.groupby('bin', as_index=False).sum()

        fig = go.Figure()
        
        # Colori GexBot Style
        bar_colors = ['#00FF00' if x >= 0 else '#00BFFF' for x in plot_df[metric]]
        
        fig.add_trace(go.Bar(
            y=plot_df['bin'], x=plot_df[metric], orientation='h',
            marker_color=bar_colors, width=granularity * 0.85
        ))

        # Annotazioni
        fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
        fig.add_hline(y=zero_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO-G")
        
        # Disegniamo i muri solo se nello zoom
        if lb <= call_wall <= ub:
            fig.add_hline(y=call_wall, line_color="red", line_width=1, annotation_text="CALL WALL")
        if lb <= put_wall <= ub:
            fig.add_hline(y=put_wall, line_color="lime", line_width=1, annotation_text="PUT WALL")

        fig.update_layout(
            template="plotly_dark", height=800,
            yaxis=dict(title="STRIKE", range=[lb, ub], dtick=granularity),
            xaxis=dict(title=f"Net {metric} Dollar Exposure", gridcolor="#333"),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabella con dati grezzi per debug
        with st.expander("Analisi Strike Singoli"):
            st.write(df_calc.sort_values('openInterest', ascending=False).head(20))
