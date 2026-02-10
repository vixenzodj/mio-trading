import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE CORE ---
st.set_page_config(layout="wide", page_title="GEX PRO V47 - INSTITUTIONAL", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# --- MOTORE DI RECUPERO DATI ---
@st.cache_data(ttl=300, show_spinner=False)
def get_spot_price(ticker_str):
    try:
        t = yf.Ticker(ticker_str)
        h = t.history(period='1d')
        return h['Close'].iloc[-1] if not h.empty else 0.0
    except: return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_data(ticker_str, target_dates, spot_ref):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    # Range Istituzionale: prendiamo tutto il board per non perdere i muri lontani
    min_k, max_k = spot_ref * 0.5, spot_ref * 1.5 

    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            c = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
            p = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
            full = pd.concat([c, p])
            payload.append(full[(full['strike'] >= min_k) & (full['strike'] <= max_k)])
        except: continue
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

# --- MOTORE DI CALCOLO GRECHE (BLACK-SCHOLES) ---
def institutional_engine(df, spot, r=0.045):
    if df.empty: return df
    S, K = float(spot), df['strike'].values
    iv = np.maximum(df['impliedVolatility'].values, 0.001)
    T = np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = (norm.pdf(d1) / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi * 100
    
    # Direzionalità Dealer (Standard: Long Calls, Long Puts per OI netto)
    direction = np.where(df['type'] == 'call', 1, -1)
    df['Gamma'] = gamma * direction
    return df

# --- CALCOLO ZERO GAMMA (INTERPOLAZIONE ISTITUZIONALE) ---
def calculate_zero_gamma(total_sum):
    if total_sum.empty: return 0
    df_sorted = total_sum.sort_values('strike')
    strikes = df_sorted['strike'].values
    gamma_cum = df_sorted['Gamma'].cumsum().values
    
    # Cerchiamo il cambio di segno (Cross-over)
    try:
        f = interp1d(gamma_cum, strikes, kind='linear', fill_value="extrapolate")
        return float(f(0))
    except:
        return strikes[np.abs(gamma_cum).argmin()]

# --- UI SIDEBAR ---
st.sidebar.title("🕹️ GEX PRO V47")
asset_choice = st.sidebar.selectbox("ASSET", ["SPX", "NDX", "SPY", "QQQ", "TSLA", "NVDA", "IBIT", "MSTR"])

# Fix Ticker per Indici Real-time
ticker_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
ticker_str = ticker_map.get(asset_choice, asset_choice)

spot = get_spot_price(ticker_str)

if spot > 0:
    t_obj = yf.Ticker(ticker_str)
    all_exps = t_obj.options
    today = datetime.now()
    
    # Selezione scadenze con indicazione DTE
    dte_list = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today).days + 1} DTE | {ex}" for ex in all_exps]
    selected_exps = st.sidebar.multiselect("SCADENZE ATTIVE", dte_list, default=dte_list[:3])
    
    clean_dates = [s.split('| ')[1] for s in selected_exps]
    granularity = st.sidebar.selectbox("AGGREGAZIONE STRIKE", [1, 5, 10, 20, 50, 100], index=1)
    zoom = st.sidebar.slider("ZOOM AREA %", 2, 50, 10)

    if clean_dates:
        with st.spinner("Analisi Flussi Istituzionali..."):
            df = fetch_option_data(ticker_str, clean_dates, spot)
            
        if not df.empty:
            df['dte_years'] = df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            df = institutional_engine(df, spot)
            
            # Aggregazione per Strike
            total_sum = df.groupby('strike', as_index=False)['Gamma'].sum()
            
            # --- LIVELLI CHIAVE ---
            call_wall = total_sum.loc[total_sum['Gamma'].idxmax(), 'strike']
            put_wall = total_sum.loc[total_sum['Gamma'].idxmin(), 'strike']
            zero_gamma = calculate_zero_gamma(total_sum)
            
            # --- DASHBOARD METRICS ---
            st.markdown(f"## 🏛️ {asset_choice} Institutional Terminal | Spot: **{spot:.2f}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CALL WALL (Muro)", f"{call_wall:.0f}")
            c2.metric("PUT WALL (Muro)", f"{put_wall:.0f}")
            c3.metric("ZERO GAMMA (Pivot)", f"{zero_gamma:.2f}")
            c4.metric("NET GEX", f"${total_sum['Gamma'].sum()/1e6:.2f}M")

            # --- PREPARAZIONE GRAFICO ---
            lb, ub = spot * (1 - zoom/100), spot * (1 + zoom/100)
            plot_df = total_sum[(total_sum['strike'] >= lb) & (total_sum['strike'] <= ub)].copy()
            
            # Binning Professionale
            plot_df['bin'] = (np.round(plot_df['strike'] / granularity) * granularity)
            plot_df = plot_df.groupby('bin', as_index=False)['Gamma'].sum()

            fig = go.Figure()
            
            # Barre Gamma (Verde Neon / Blu Elettrico)
            fig.add_trace(go.Bar(
                y=plot_df['bin'], x=plot_df['Gamma'], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in plot_df['Gamma']],
                width=granularity * 0.8, name="Gamma Exposure"
            ))

            # Linee di Forza
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT PRICE", annotation_position="top right")
            fig.add_hline(y=zero_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            
            # Visualizzazione muri solo se presenti nell'area
            fig.add_hline(y=call_wall, line_color="red", line_width=1, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="lime", line_width=1, annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=800,
                yaxis=dict(title="STRIKE", range=[lb, ub], dtick=granularity, gridcolor='#333'),
                xaxis=dict(title="Gamma Exposure ($) per Strike", gridcolor='#333'),
                margin=dict(l=20, r=20, t=30, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Tabella di controllo strike (Opzionale)
            with st.expander("Vedi Dati Analitici Strike"):
                st.dataframe(total_sum.sort_values('Gamma', ascending=False).head(10), use_container_width=True)

else:
    st.error("Errore nel recupero del prezzo spot. Verifica la connessione o il ticker.")
