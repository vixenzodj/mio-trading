import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE STABILE ---
st.set_page_config(layout="wide", page_title="GEX PRO V45 STABLE", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# --- FUNZIONI DATI OTTIMIZZATE ---
@st.cache_data(ttl=300, show_spinner=False)
def get_spot_price(ticker_str):
    try:
        t = yf.Ticker(ticker_str)
        # Fast history retrieval
        h = t.history(period='1d')
        return h['Close'].iloc[-1] if not h.empty else 0.0
    except:
        return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_data(ticker_str, target_dates, spot_ref):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    
    # Range di sicurezza per il download (evita di caricare strike inutili)
    # Carichiamo solo strike entro +/- 40% dallo spot per risparmiare RAM
    min_strike = spot_ref * 0.6
    max_strike = spot_ref * 1.4

    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            if oc.calls.empty and oc.puts.empty: continue
            
            # Pre-filtro Calls
            c = oc.calls
            c = c[(c['strike'] >= min_strike) & (c['strike'] <= max_strike)]
            c_df = c[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
            
            # Pre-filtro Puts
            p = oc.puts
            p = p[(p['strike'] >= min_strike) & (p['strike'] <= max_strike)]
            p_df = p[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
            
            if not c_df.empty or not p_df.empty:
                payload.append(pd.concat([c_df, p_df], ignore_index=True))
        except:
            continue
            
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

# --- MOTORE DI CALCOLO VETTORIALE (LIGHT) ---
def engine_v45(df_input, spot_price, r_rate=0.045):
    if df_input.empty or spot_price <= 0: return df_input
    
    # Copia leggera
    df = df_input.copy()
    s = float(spot_price)
    
    # Conversione rapida in numpy array
    k = df['strike'].values
    v = np.maximum(df['impliedVolatility'].values, 0.001)
    t = np.maximum(df['dte_years'].values, 0.001) # Evita division by zero
    oi = df['openInterest'].fillna(0).values
    
    # Black-Scholes Vectorized
    d1 = (np.log(s/k) + (r_rate + 0.5 * v**2) * t) / (v * np.sqrt(t))
    pdf = norm.pdf(d1)
    
    # Calcolo Greche (solo 3 colonne necessarie)
    # Dollar Gamma = 0.5 * Gamma * S^2 * OI
    gamma_val = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna_val = s * pdf * d1 / v * 0.01 * oi
    charm_val = (pdf * (r_rate / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    # Segno Direzionale (Call = +, Put = -)
    is_call = (df['type'].values == 'call')
    direction = np.where(is_call, 1, -1)
    
    df['Gamma'] = gamma_val * direction
    df['Vanna'] = vanna_val * direction
    df['Charm'] = charm_val * direction
    
    return df[['strike', 'Gamma', 'Vanna', 'Charm']] # Ritorniamo solo ciò che serve

# --- INTERFACCIA UTENTE ---
st.sidebar.markdown("### ⚡ GEX PRO V45 (LITE)")
active_t = st.sidebar.selectbox("ASSET", ["SPX", "NDX", "SPY", "QQQ", "TSLA", "NVDA", "AAPL", "MSFT", "IBIT", "COIN", "MSTR", "PLTR"])

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

ticker_str = fix_ticker(active_t)

# Step 1: Prezzo Spot (Necessario per tutto il resto)
spot = get_spot_price(ticker_str)

if spot > 0:
    t_obj_init = yf.Ticker(ticker_str)
    try:
        all_exps = t_obj_init.options
    except:
        st.error("Errore API Yahoo. Riprova tra poco.")
        st.stop()
        
    today_dt = datetime.now()
    dte_opts = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today_dt).days + 1} DTE ({ex})" for ex in all_exps]
    
    # Default selection: prime 2 scadenze
    sel_lbls = st.sidebar.multiselect("SCADENZE", dte_opts, default=dte_opts[:2])
    target_dates = [x.split('(')[1].replace(')', '') for x in sel_lbls]
    
    # Controlli Visuali
    granularity = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100], index=1)
    zoom_pct = st.sidebar.slider("ZOOM %", 2, 40, 10)
    metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'], horizontal=True)

    if target_dates:
        # Caricamento Intelligente
        with st.spinner('Calcolo in corso...'):
            raw_df = fetch_option_data(ticker_str, target_dates, spot)
        
        if not raw_df.empty:
            raw_df['dte_years'] = raw_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today_dt).days + 0.5) / 365
            
            # Processing Veloce
            df_greeks = engine_v45(raw_df, spot)
            
            # Aggregazione Totale per Livelli Chiave
            total_sum = df_greeks.groupby('strike', as_index=False).sum()
            
            # Calcoli Chiave (Walls & Zero)
            # Gestione errore se dataframe vuoto dopo filtro
            if not total_sum.empty:
                call_wall = total_sum.loc[total_sum['Gamma'].idxmax(), 'strike']
                put_wall = total_sum.loc[total_sum['Gamma'].idxmin(), 'strike']
                
                total_sum['cum_gamma'] = total_sum['Gamma'].cumsum()
                # Trova il punto più vicino allo zero nel cumsum
                zero_idx = total_sum['cum_gamma'].abs().idxmin()
                zero_gamma = total_sum.loc[zero_idx, 'strike']
                
                # --- PREPARAZIONE GRAFICO (Solo dati visibili) ---
                lb = spot * (1 - zoom_pct/100)
                ub = spot * (1 + zoom_pct/100)
                
                # Taglio netto dei dati per il plot (Performance Boost)
                plot_df = total_sum[(total_sum['strike'] >= lb) & (total_sum['strike'] <= ub)].copy()
                
                # Binning
                plot_df['bin'] = (np.round(plot_df['strike'] / granularity) * granularity)
                plot_df = plot_df.groupby('bin', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
                
                # --- DASHBOARD ---
                st.markdown(f"### 🏛️ {active_t} | Spot: {spot:.2f}")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("CALL WALL", f"{call_wall:.0f}")
                k2.metric("PUT WALL", f"{put_wall:.0f}")
                k3.metric("ZERO GAMMA", f"{zero_gamma:.0f}")
                net_gex = total_sum['Gamma'].sum()
                k4.metric("NET GEX", f"${net_gex/1e6:.1f}M", delta_color="normal")
                
                # --- PLOTLY LIGHTWEIGHT ---
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=plot_df['bin'], 
                    x=plot_df[metric], 
                    orientation='h',
                    marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in plot_df[metric]],
                    width=granularity * 0.85
                ))
                
                # Linee di riferimento
                fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
                fig.add_hline(y=zero_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO-G")
                
                # Walls (solo se nel range)
                if lb <= call_wall <= ub:
                    fig.add_hline(y=call_wall, line_color="red", line_width=1, annotation_text="CW")
                if lb <= put_wall <= ub:
                    fig.add_hline(y=put_wall, line_color="lime", line_width=1, annotation_text="PW")

                fig.update_layout(
                    template="plotly_dark", 
                    height=700, 
                    margin=dict(l=40, r=40, t=40, b=40),
                    yaxis=dict(
                        title="STRIKE", 
                        range=[lb, ub], # Force range
                        dtick=granularity
                    ),
                    xaxis=dict(title=f"Net {metric} Exposure")
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Dati insufficienti dopo il filtro.")
        else:
            st.warning("Nessun dato trovato nel range dello spot.")
else:
    st.error("Impossibile recuperare il prezzo Spot. Mercato chiuso o ticker errato.")
