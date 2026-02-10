import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE INTERFACCIA ---
st.set_page_config(layout="wide", page_title="GEX PRO V48 - MULTI-METRIC", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="global_refresh")

# --- MOTORE DI CALCOLO PROFESSIONALE (BLACK-SCHOLES) ---
def calculate_greeks_engine(df, S, r=0.045):
    if df.empty: return df
    
    # Parametri base
    K = df['strike'].values
    iv = np.maximum(df['impliedVolatility'].values, 0.001)
    T = np.maximum(df['dte_years'].values, 0.0001)
    oi = df['openInterest'].fillna(0).values
    
    # Calcolo d1 e d2
    d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    
    # Direzionalità (Call + / Put -)
    side = np.where(df['type'] == 'call', 1, -1)
    
    # --- FORMULE ESPOSIZIONE MONETARIA (DOLLAR EXPOSURE) ---
    # Gamma: 0.5 * Gamma * S^2 * 0.01
    df['Gamma'] = (pdf / (S * iv * np.sqrt(T))) * (S**2) * 0.01 * oi * 100 * side
    
    # Vanna: dGamma / dVol -> Esposizione alla Volatilità
    df['Vanna'] = S * pdf * (d1 / iv) * 0.01 * oi * side
    
    # Charm: dDelta / dt -> Decadimento del Delta (importante per 0DTE)
    df['Charm'] = (pdf * (r / (iv * np.sqrt(T)) - d1 / (2 * T))) * oi * 100 * side
    
    # Vega: dPrice / dVol -> Esposizione se la IV sale del 1%
    df['Vega'] = S * pdf * np.sqrt(T) * 0.01 * oi * 100 # Vega è solitamente positivo per entrambi
    
    # Theta: dPrice / dt -> Decadimento temporale giornaliero
    term1 = -(S * pdf * iv) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2 * side)
    df['Theta'] = (term1 - side * term2) * (1/365) * oi * 100
    
    return df

# --- RECUPERO DATI ---
@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker_str, target_dates, spot_ref):
    t_obj = yf.Ticker(ticker_str)
    payload = []
    # Prendiamo un range molto largo per non perdere i muri (50% sopra/sotto)
    for d in target_dates:
        try:
            oc = t_obj.option_chain(d)
            c = oc.calls[['strike', 'impliedVolatility', 'openInterest']].assign(type='call', exp_date=d)
            p = oc.puts[['strike', 'impliedVolatility', 'openInterest']].assign(type='put', exp_date=d)
            payload.append(pd.concat([c, p]))
        except: continue
    return pd.concat(payload, ignore_index=True) if payload else pd.DataFrame()

# --- SIDEBAR & CONTROLLI ---
st.sidebar.title("🧬 GEX ENGINE V48")
asset = st.sidebar.selectbox("ASSET", ["SPX", "NDX", "SPY", "QQQ", "TSLA", "NVDA", "IBIT"])
ticker_map = {"SPX": "^SPX", "NDX": "^NDX"}
ticker_str = ticker_map.get(asset, asset)

t_obj = yf.Ticker(ticker_str)
h = t_obj.history(period='1d')
spot = h['Close'].iloc[-1] if not h.empty else 0

if spot > 0:
    all_exps = t_obj.options
    today = datetime.now()
    dte_opts = [f"{(datetime.strptime(ex, '%Y-%m-%d') - today).days + 1} DTE ({ex})" for ex in all_exps]
    selected_exps = st.sidebar.multiselect("SCADENZE", dte_opts, default=dte_opts[:2])
    
    # Parametri Visualizzazione
    metric = st.sidebar.radio("METRICA GRAFICO", ["Gamma", "Vanna", "Charm", "Vega", "Theta"], horizontal=True)
    granularity = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50], index=2)
    zoom = st.sidebar.slider("ZOOM AREA %", 1, 30, 10)
    
    if selected_exps:
        clean_dates = [x.split('(')[1].replace(')', '') for x in selected_exps]
        raw_df = get_data(ticker_str, clean_dates, spot)
        
        if not raw_df.empty:
            raw_df['dte_years'] = raw_df['exp_date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - today).days + 0.5) / 365
            df_greeks = calculate_greeks_engine(raw_df, spot)
            
            # Aggregazione
            total_sum = df_greeks.groupby('strike', as_index=False)[["Gamma", "Vanna", "Charm", "Vega", "Theta"]].sum()
            
            # Calcolo Muri e Zero Gamma (Interpolato)
            call_wall = total_sum.loc[total_sum['Gamma'].idxmax(), 'strike']
            put_wall = total_sum.loc[total_sum['Gamma'].idxmin(), 'strike']
            
            # Zero Gamma via interpolazione
            try:
                sorted_g = total_sum.sort_values('strike')
                f = interp1d(sorted_g['Gamma'].cumsum(), sorted_g['strike'], fill_value="extrapolate")
                zero_gamma = float(f(0))
            except: zero_gamma = spot

            # --- DASHBOARD ---
            st.markdown(f"### 🏟️ {asset} Market Structure | Spot: {spot:.2f}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("CALL WALL", f"{call_wall:.0f}")
            m2.metric("PUT WALL", f"{put_wall:.0f}")
            m3.metric("ZERO GAMMA", f"{zero_gamma:.1f}")
            m4.metric(f"NET {metric.upper()}", f"${total_sum[metric].sum()/1e6:.1f}M")
            m5.metric("TOT OI", f"{raw_df['openInterest'].sum():,.0f}")

            # --- PLOT STILE GEXBOT ---
            lb, ub = spot * (1 - zoom/100), spot * (1 + zoom/100)
            plot_df = total_sum[(total_sum['strike'] >= lb) & (total_sum['strike'] <= ub)].copy()
            plot_df['bin'] = (np.round(plot_df['strike'] / granularity) * granularity)
            plot_df = plot_df.groupby('bin', as_index=False).sum()

            fig = go.Figure()
            
            # Colori: Verde Gexbot per valori positivi, Blu per negativi
            colors = ['#00FF00' if x >= 0 else '#00BFFF' for x in plot_df[metric]]
            
            fig.add_trace(go.Bar(
                y=plot_df['bin'], x=plot_df[metric], orientation='h',
                marker_color=colors, width=granularity * 0.8,
                hovertemplate="Strike: %{y}<br>Value: %{x:,.0f}<extra></extra>"
            ))

            # Linee Tecniche
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=zero_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO-G")
            fig.add_hline(y=call_wall, line_color="red", line_width=1, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="lime", line_width=1, annotation_text="PUT WALL")

            fig.update_layout(
                template="plotly_dark", height=850,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(title="STRIKE", range=[lb, ub], dtick=granularity),
                xaxis=dict(title=f"Net {metric} Dollar Exposure", gridcolor="#333")
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA DETTAGLIATA ---
            with st.expander("Visualizza Tabella Greche per Strike"):
                st.dataframe(total_sum.sort_values('strike'), use_container_width=True)
else:
    st.error("Impossibile connettersi a Yahoo Finance o Ticker non valido.")
