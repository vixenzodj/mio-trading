import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE VISUALIZZAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V46", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# CSS per scurire l'interfaccia come GexBot
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    div.stButton > button:first-child { background-color: #222; color: white; border: 1px solid #444; }
    </style>
""", unsafe_allow_html=True)

# --- FUNZIONI DATI (CACHE) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_spot_price(ticker_str):
    try:
        t = yf.Ticker(ticker_str)
        h = t.history(period='1d')
        return h['Close'].iloc[-1] if not h.empty else 0.0
    except:
        return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_option_data(ticker_str, target_dates, spot_ref):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    
    # Range ampio per catturare i muri, ma limitato per memoria
    min_strike = spot_ref * 0.5
    max_strike = spot_ref * 1.5

    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            if oc.calls.empty and oc.puts.empty: continue
            
            # Filtro rapido
            c = oc.calls[(oc.calls['strike'] >= min_strike) & (oc.calls['strike'] <= max_strike)]
            p = oc.puts[(oc.puts['strike'] >= min_strike) & (oc.puts['strike'] <= max_strike)]
            
            if not c.empty:
                c_df = c[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
                payload.append(c_df)
            if not p.empty:
                p_df = p[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
                payload.append(p_df)
        except:
            continue
            
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

# --- MOTORE DI CALCOLO ---
def engine_v46(df_input, spot_price, r_rate=0.045):
    if df_input.empty or spot_price <= 0: return df_input
    
    df = df_input.copy()
    s = float(spot_price)
    k = df['strike'].values
    v = np.maximum(df['impliedVolatility'].values, 0.001)
    t = np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    d1 = (np.log(s/k) + (r_rate + 0.5 * v**2) * t) / (v * np.sqrt(t))
    pdf = norm.pdf(d1)
    
    # Formule GEX pure
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r_rate / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    
    # Gestione Segno (Call + / Put -)
    direction = np.where(df['type'].values == 'call', 1, -1)
    
    df['Gamma'] = gamma * direction
    df['Vanna'] = vanna * direction
    df['Charm'] = charm * direction
    
    return df

# --- INTERFACCIA ---
st.sidebar.markdown("### 🧬 GEX TERMINAL REPLICA")
active_t = st.sidebar.selectbox("ASSET", ["QQQ", "SPX", "SPY", "NVDA", "TSLA", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL"])

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT", "VIX"] else s

ticker_str = fix_ticker(active_t)
spot = get_spot_price(ticker_str)

if spot > 0:
    t_obj_init = yf.Ticker(ticker_str)
    try:
        all_exps = t_obj_init.options
    except:
        st.error("Errore API. Riprova.")
        st.stop()
        
    today_dt = datetime.now()
    # Mostra DTE puliti
    dte_map = {ex: (datetime.strptime(ex, '%Y-%m-%d') - today_dt).days + 1 for ex in all_exps}
    sorted_exps = sorted(all_exps, key=lambda x: dte_map[x])
    
    # Multiselect con formattazione
    dte_opts = [f"{dte_map[ex]} DTE ({ex})" for ex in sorted_exps]
    sel_lbls = st.sidebar.multiselect("SCADENZE (Expiration)", dte_opts, default=dte_opts[:1])
    target_dates = [x.split('(')[1].replace(')', '') for x in sel_lbls]
    
    # --- CONTROLLI GRAFICI CRITICI ---
    st.sidebar.divider()
    # Questo è il parametro CHIAVE per l'effetto "GexBot"
    # Su SPX/QQQ devi usare 5 o 10 per vedere le barre piene
    granularity = st.sidebar.select_slider(
        "AGGREGAZIONE STRIKE (Binning)", 
        options=[1, 2.5, 5, 10, 25, 50, 100], 
        value=5 if spot > 1000 else 1
    )
    
    zoom_pct = st.sidebar.slider("ZOOM VISTA (%)", 1, 30, 8)
    metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm'], horizontal=True)

    if target_dates:
        with st.spinner('Scaricamento dati opzioni...'):
            raw_df = fetch_option_data(ticker_str, target_dates, spot)
        
        if not raw_df.empty:
            raw_df['dte_years'] = raw_df['exp_date'].apply(lambda x: dte_map[x] / 365.0)
            
            df_calc = engine_v46(raw_df, spot)
            
            # --- AGGREGAZIONE (La magia avviene qui) ---
            # Raggruppa per strike base
            total = df_calc.groupby('strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
            
            # Calcolo Muri GLOBALI (prima dello zoom)
            cw_idx = total['Gamma'].idxmax()
            pw_idx = total['Gamma'].idxmin()
            call_wall = total.loc[cw_idx, 'strike']
            put_wall = total.loc[pw_idx, 'strike']
            cw_val = total.loc[cw_idx, 'Gamma'] # Per sapere quanto è alta la barra
            
            # Calcolo Zero Gamma
            total['cum'] = total['Gamma'].cumsum()
            zero_gamma = total.loc[total['cum'].abs().idxmin(), 'strike']
            
            # --- FILTRO ZOOM & BINNING VISIVO ---
            lb = spot * (1 - zoom_pct/100)
            ub = spot * (1 + zoom_pct/100)
            
            # Filtra solo l'area visibile
            view_df = total[(total['strike'] >= lb) & (total['strike'] <= ub)].copy()
            
            # BINNING MATEMATICO (Arrotondamento allo strike più vicino)
            view_df['bin_strike'] = (np.round(view_df['strike'] / granularity) * granularity)
            # Somma i valori nel bin
            plot_df = view_df.groupby('bin_strike', as_index=False)[['Gamma', 'Vanna', 'Charm']].sum()
            
            # --- VISUALIZZAZIONE GEXBOT STYLE ---
            st.markdown(f"## 🏛️ {active_t} TERMINAL | Spot: {spot:.2f}")
            
            # Metriche in alto
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{call_wall:,.0f}", help="Strike con Gamma Positivo Massimo")
            m2.metric("PUT WALL", f"{put_wall:,.0f}", help="Strike con Gamma Negativo Massimo")
            m3.metric("ZERO GAMMA", f"{zero_gamma:,.0f}", delta_color="off")
            net_exp = total[metric].sum()
            m4.metric(f"NET {metric.upper()}", f"${net_exp/1e6:,.1f}M")
            
            fig = go.Figure()
            
            # Colori stile GexBot (Verde Neon vs Blu Elettrico)
            colors = ['#00FF00' if v >= 0 else '#00BFFF' for v in plot_df[metric]]
            
            # Barre Orizzontali
            fig.add_trace(go.Bar(
                y=plot_df['bin_strike'],
                x=plot_df[metric],
                orientation='h',
                marker_color=colors,
                # Larghezza dinamica per riempire lo spazio (Effetto "Muro")
                width=granularity * 0.9, 
                hovertemplate='Strike: %{y}<br>Valore: %{x:,.0f}<extra></extra>'
            ))
            
            # Linea SPOT (Ciano tratteggiata)
            fig.add_hline(y=spot, line_color="#00FFFF", line_dash="dash", line_width=1.5, annotation_text="SPOT", annotation_position="top right")
            
            # Linea ZERO GAMMA (Gialla)
            fig.add_hline(y=zero_gamma, line_color="#FFFF00", line_dash="dashdot", line_width=1, annotation_text="ZERO G")
            
            # Muri (Solo se visibili)
            if lb <= call_wall <= ub:
                fig.add_hline(y=call_wall, line_color="#FF3333", line_width=1, annotation_text=f"CW {call_wall:.0f}", annotation_position="bottom right")
            if lb <= put_wall <= ub:
                fig.add_hline(y=put_wall, line_color="#00FF00", line_width=1, annotation_text=f"PW {put_wall:.0f}", annotation_position="bottom right")

            # Layout Scuro "Pro"
            fig.update_layout(
                template="plotly_dark",
                height=800,
                paper_bgcolor='rgba(0,0,0,0)', # Sfondo trasparente per integrarsi
                plot_bgcolor='rgba(14,17,23, 1)', # Sfondo scuro app
                margin=dict(l=50, r=50, t=30, b=30),
                yaxis=dict(
                    title="STRIKE PRICE",
                    tickmode='linear',
                    dtick=granularity, # Forza i tick corretti
                    range=[lb, ub],
                    gridcolor='#333333'
                ),
                xaxis=dict(
                    title=f"Net {metric} Exposure ($)",
                    gridcolor='#333333',
                    zerolinecolor='#666666'
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Nessun dato trovato. Prova a selezionare una scadenza diversa.")
else:
    st.error("Spot price non disponibile.")
