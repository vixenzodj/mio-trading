import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V40", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE ULTRA-RAPIDO ---
def fast_engine_vectorized(df, spot, r=0.045):
    if df.empty: return df
    s = float(spot)
    k = df['strike'].values
    v = np.where(df['impliedVolatility'].values <= 0, 1e-9, df['impliedVolatility'].values)
    oi = df['openInterest'].values
    t = np.where(df['dte_years'].values <= 0, 1e-9, df['dte_years'].values)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    is_call = (df['type'].values == 'call')
    theta = (-(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    df['Gamma'] = gamma * mult
    df['Vanna'] = vanna * mult
    df['Charm'] = charm * mult
    df['Vega'] = vega
    df['Theta'] = theta
    return df

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEXBOT ENGINE V40")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        all_exps = t_obj.options
        today = datetime.now()
        
        # Calcolo DTE per ogni scadenza disponibile
        exp_data = []
        for e in all_exps:
            d = datetime.strptime(e, '%Y-%m-%d')
            dte = (d - today).days + 1
            exp_data.append({'date': e, 'dte': dte})
        
        df_exps = pd.DataFrame(exp_data)
        
        # --- FILTRO DTE (Gexbot Style) ---
        dte_choice = st.sidebar.multiselect("FILTRA PER DTE", 
                                            options=sorted(df_exps['dte'].unique()), 
                                            default=sorted(df_exps['dte'].unique())[:3])
        
        selected_dates = df_exps[df_exps['dte'].isin(dte_choice)]['date'].tolist()
        
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
        zoom_pts = st.sidebar.slider("ZOOM (Punti dallo Spot)", 50, 2000, 500)
        main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty and selected_dates:
            spot = float(hist['Close'].iloc[-1])
            
            # Scarico e unisco tutte le scadenze selezionate
            all_chains = []
            for d in selected_dates:
                chain = t_obj.option_chain(d)
                c = chain.calls.assign(type='call', exp=d)
                p = chain.puts.assign(type='put', exp=d)
                all_chains.extend([c, p])
            
            df_raw = pd.concat(all_chains, ignore_index=True)
            
            # Calcolo DTE Years
            df_raw['dte_years'] = df_raw['exp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            
            # --- PULIZIA ATOMICA ---
            df_clean = df_raw.groupby(['strike', 'type', 'dte_years'], as_index=False).agg({
                'impliedVolatility': 'mean', 'openInterest': 'sum'
            }).reset_index(drop=True)
            
            # Motore Calcolo
            df_res = fast_engine_vectorized(df_clean, spot)
            
            # Aggregazione finale per Strike (unendo tutte le scadenze)
            df_final = df_res.groupby('strike', as_index=False).sum(numeric_only=True).sort_values('strike').reset_index(drop=True)

            # --- LOGICA ZERO GAMMA ---
            # Cerchiamo il cross dello zero sulla somma cumulata nell'area intorno allo spot
            df_final['cum_gamma'] = df_final['Gamma'].cumsum()
            z_gamma = df_final.loc[df_final['cum_gamma'].abs().idxmin(), 'strike']

            # --- FILTRO ZOOM ---
            df_plot = df_final[(df_final['strike'] >= spot - zoom_pts) & (df_final['strike'] <= spot + zoom_pts)].copy()
            
            # Binning per pulizia visiva
            df_plot['bin'] = np.floor(df_plot['strike'] / strike_step) * strike_step
            df_plot = df_plot.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'})

            # --- DASHBOARD ---
            st.markdown(f"## 🏛️ {active_t} AGGREGATED TERMINAL")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SPOT", f"{spot:.2f}")
            c2.metric("ZERO GAMMA", f"{z_gamma:.0f}")
            
            g_net = df_final['Gamma'].sum()
            status = "🛡️ STABLE" if g_net > 0 else "⚠️ VOLATILE"
            c3.metric("MARKET REGIME", status, delta=f"{g_net/1e6:.1f}M", delta_color="normal")
            
            # --- GRAFICO GEXBOT STYLE ---
            fig = go.Figure()
            
            # Barre Orizzontali
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h',
                marker_color=['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]],
                width=strike_step * 0.9,
                name=main_metric
            ))

            # Linee di Supporto
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text=f"SPOT: {spot:.0f}")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO GAMMA")
            
            # Trova Call Wall e Put Wall nello zoom
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            fig.add_hline(y=call_wall, line_color="orange", line_width=1, annotation_text="MAJOR WALL")

            fig.update_layout(
                template="plotly_dark", height=850,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(dtick=strike_step, title="STRIKE", gridcolor="#222"),
                xaxis=dict(title=f"Total Net {main_metric} Exposure", zerolinecolor="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA DETTAGLIATA ---
            st.markdown("### 📊 Market Maker Inventory Details")
            st.dataframe(df_plot.sort_values('strike', ascending=False).style.format(precision=0).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Charm', 'Theta']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
