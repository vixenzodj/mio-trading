import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE & CACHE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V44", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# Cache Dati (TTL 5 minuti)
@st.cache_data(ttl=300)
def fetch_option_data(ticker_str, target_dates):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            # Controllo esistenza dati
            if oc.calls.empty and oc.puts.empty: continue
            
            c_df = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
            p_df = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
            payload.append(pd.concat([c_df, p_df], ignore_index=True))
        except:
            continue
            
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

@st.cache_data(ttl=60)
def get_spot_price(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        hist = t_obj.history(period='1d')
        if hist.empty: return 0.0
        return hist['Close'].iloc[-1]
    except:
        return 0.0

# --- MOTORE DI CALCOLO ---
def engine_v44(df_input, spot_price, r_rate=0.045):
    if df_input.empty or spot_price <= 0: return df_input
    df = df_input.copy()
    s = float(spot_price)
    
    k = df['strike'].to_numpy()
    # Protezione da divisione per zero e valori nulli
    v = np.maximum(df['impliedVolatility'].to_numpy(), 0.001)
    t = np.maximum(df['dte_years'].to_numpy(), 0.001)
    oi = df['openInterest'].fillna(0).to_numpy()
    
    d1 = (np.log(s/k) + (r_rate + 0.5 * v**2) * t) / (v * np.sqrt(t))
    pdf = norm.pdf(d1)
    
    # Formule Standard GEX
    gamma_val = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna_val = s * pdf * d1 / v * 0.01 * oi
    charm_val = (pdf * (r_rate / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    direction = np.where(df['type'].values == 'call', 1, -1)
    
    df['Gamma'] = gamma_val * direction
    df['Vanna'] = vanna_val * direction
    df['Charm'] = charm_val * direction
    return df

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX PRO V44")
active_t = st.sidebar.selectbox("ASSET", ["SPX", "NDX", "SPY", "QQQ", "TSLA", "NVDA", "AAPL", "MSFT", "IBIT", "COIN", "MSTR"])

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

ticker_str = fix_ticker(active_t)
t_obj_init = yf.Ticker(ticker_str)

try:
    all_exps = t_obj_init.options
    if not all_exps:
        st.error("Nessuna scadenza trovata per questo asset.")
        st.stop()
except:
    st.error("Errore connessione API. Riprova.")
    st.stop()

# Mappatura DTE
today_dt = datetime.now()
dte_options = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today_dt).days + 1} DTE ({ex})" for ex in all_exps]
selected_dte = st.sidebar.multiselect("SCADENZE ATTIVE", options=dte_options, default=dte_options[:2])

# PARAMETRI VISUALI ADATTIVI
# Step 1: Granularità Intelligente
strike_granularity = st.sidebar.selectbox("RAGGRUPPAMENTO STRIKE (Binning)", [1, 5, 10, 25, 50, 100], index=1)

# Step 2: Zoom Percentuale (Risolve il problema dei diversi prezzi)
zoom_pct = st.sidebar.slider("ZOOM AREA (+/- %)", 2, 50, 10, help="2% per indici, 20% per azioni volatili")
metric_choice = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'])

if ticker_str and selected_dte:
    try:
        spot = get_spot_price(ticker_str)
        if spot == 0:
            st.warning("Impossibile recuperare il prezzo Spot.")
            st.stop()

        target_dates = [sel.split('(')[1].replace(')', '') for sel in selected_dte]
        
        # Caricamento
        full_df = fetch_option_data(ticker_str, target_dates)
        
        if not full_df.empty:
            full_df['dte_years'] = full_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today_dt).days + 0.5) / 365
            
            # Calcolo
            processed_df = engine_v44(full_df, spot)
            
            # Aggregazione Totale per trovare i Muri e lo Zero Gamma
            total_summary = processed_df.groupby('strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
            
            # Calcolo Walls (Su tutto il dataset, non solo quello visibile)
            call_wall_strike = total_summary.loc[total_summary['Gamma'].idxmax(), 'strike']
            put_wall_strike = total_summary.loc[total_summary['Gamma'].idxmin(), 'strike'] # Gamma negativo massimo
            
            # Calcolo Zero Gamma
            total_summary['cum_sum_gamma'] = total_summary['Gamma'].cumsum()
            zero_gamma_level = total_summary.loc[total_summary['cum_sum_gamma'].abs().idxmin(), 'strike']
            
            # --- FILTRO GRAFICO (ZOOM PERCENTUALE) ---
            lower_bound = spot * (1 - zoom_pct/100)
            upper_bound = spot * (1 + zoom_pct/100)
            
            # Filtriamo i dati per il grafico
            plot_data = total_summary[(total_summary['strike'] >= lower_bound) & (total_summary['strike'] <= upper_bound)].copy()
            
            # Binning (Raggruppamento visuale)
            if strike_granularity > 1:
                plot_data['strike_bin'] = (np.round(plot_data['strike'] / strike_granularity) * strike_granularity)
                plot_data = plot_data.groupby('strike_bin', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
            else:
                plot_data['strike_bin'] = plot_data['strike']

            # --- DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} Analysis | Spot: {spot:.2f}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CALL WALL", f"{call_wall_strike:.0f}")
            c2.metric("PUT WALL", f"{put_wall_strike:.0f}")
            c3.metric("ZERO GAMMA", f"{zero_gamma_level:.0f}")
            c4.metric("NET GEX", f"${total_summary['Gamma'].sum()/1e6:.1f}M")
            
            # --- GRAFICO ---
            fig = go.Figure()
            
            # Barre
            fig.add_trace(go.Bar(
                y=plot_data['strike_bin'], x=plot_data[metric_choice], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in plot_data[metric_choice]],
                # Larghezza dinamica per evitare sovrapposizioni
                width=strike_granularity * 0.8,
                name=metric_choice
            ))
            
            # Linea SPOT
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT", annotation_position="top right")
            
            # Linea ZERO GAMMA
            fig.add_hline(y=zero_gamma_level, line_color="yellow", line_dash="dash", annotation_text="ZERO G")
            
            # Linee WALLS (Solo se rientrano nello zoom, altrimenti confondono)
            if lower_bound <= call_wall_strike <= upper_bound:
                fig.add_hline(y=call_wall_strike, line_color="red", annotation_text="CALL WALL")
            
            if lower_bound <= put_wall_strike <= upper_bound:
                fig.add_hline(y=put_wall_strike, line_color="#00ff00", annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=800,
                yaxis=dict(
                    title="STRIKE", 
                    tickmode='linear', 
                    dtick=strike_granularity, 
                    range=[lower_bound, upper_bound] # Range forzato
                ),
                xaxis=dict(title=f"Net {metric_choice} Exposure ($)"),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Nessun dato opzioni trovato per le date selezionate.")

    except Exception as e:
        st.error(f"Errore di calcolo: {e}")
