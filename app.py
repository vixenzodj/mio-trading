import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V15", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE BS-INSTITUTIONAL ---
def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        
        # Dollar Gamma Exposure: 0.5 * Gamma * Spot^2 * 0.01
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * (s**2) * 0.01 * oi * 100
        # Vanna: dDelta / dVol
        vanna = s * pdf * d1 / v * 0.01 * oi
        # Charm: dDelta / dTime
        charm = (pdf * (r / (v * np.sqrt(t_yrs)) - d1 / (2 * t_yrs))) * oi * 100
        # Vega: dPrice / dVol
        vega = s * pdf * np.sqrt(t_yrs) * oi * 100
        # Theta: dPrice / dTime
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2 if row['type']=='call' else -d2)) * oi * 100
        
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0.0]*5)

def get_gamma_flip(df, spot_range):
    # Trova il punto dove il Gamma Netto Totale attraversa lo zero
    # Creiamo una simulazione di prezzi intorno allo spot attuale
    test_prices = np.linspace(df['strike'].min(), df['strike'].max(), 200)
    # Approssimazione professionale: usiamo il Gamma attuale ponderato per la distanza
    # In un modello reale ricalcoleremmo BS per ogni prezzo, qui usiamo la curva di distribuzione
    df_sorted = df.sort_values('strike')
    cumulative_gamma = df_sorted['Gamma'].cumsum()
    # Trova dove la massa del gamma passa da negativa a positiva
    zero_idx = np.abs(cumulative_gamma).idxmin()
    return float(df_sorted.loc[zero_idx, 'strike'])

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE PRO V15")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA (DTE)", exps)
    strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
    num_levels = st.sidebar.slider("VISUAL RANGE", 20, 200, 100)
    main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # Uniamo Call e Put per analisi aggregata
            calls, puts = chain.calls.copy(), chain.puts.copy()
            calls['type'], puts['type'] = 'call', 'put'
            
            c_vals = calls.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            p_vals = puts.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            
            # Creazione dataframe unico per strike
            df_calls = pd.DataFrame({'strike': calls['strike'], 'Gamma': c_vals[0], 'Vanna': c_vals[1], 'Charm': c_vals[2], 'Vega': c_vals[3], 'Theta': c_vals[4]})
            df_puts = pd.DataFrame({'strike': puts['strike'], 'Gamma': -p_vals[0], 'Vanna': -p_vals[1], 'Charm': -p_vals[2], 'Vega': p_vals[3], 'Theta': p_vals[4]})
            
            df_total = pd.concat([df_calls, df_puts]).groupby('strike').sum().reset_index()

            # --- CALCOLO LIVELLI LIVE ---
            # Zero Gamma Flip: Punto di inversione della liquidità
            z_gamma = get_gamma_flip(df_total, spot)
            
            # Muri entro il 10% del prezzo attuale (come nelle piattaforme PRO)
            df_local = df_total[(df_total['strike'] >= spot * 0.9) & (df_total['strike'] <= spot * 1.1)]
            call_wall = float(df_local.loc[df_local['Gamma'].idxmax(), 'strike'])
            put_wall = float(df_local.loc[df_local['Gamma'].idxmin(), 'strike'])

            # --- DASHBOARD ---
            st.markdown(f"## 📊 {active_t} GEX Profile | Spot: {spot:.2f}")
            
            # Bias combinato istituzionale (Gamma + Vanna + Charm)
            total_bias = df_total['Gamma'].sum() + df_total['Vanna'].sum()
            b_txt, b_col = ("LONG GAMMA (Stable)", "#00ff00") if total_bias > 0 else ("SHORT GAMMA (Volatile)", "#ff4444")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div style='border:2px solid {b_col};padding:10px;border-radius:10px;text-align:center;'>SENTIMENT<br><b style='color:{b_col};'>{b_txt}</b></div>", unsafe_allow_html=True)
            c2.metric("CALL WALL", f"{call_wall:.0f}", f"{call_wall-spot:.1f}")
            c3.metric("PUT WALL", f"{put_wall:.0f}", f"{put_wall-spot:.1f}")
            c4.metric("ZERO FLIP", f"{z_gamma:.0f}", f"Target Vol", delta_color="off")

            # --- GRAFICO ---
            df_total['bin'] = (df_total['strike'] / strike_step).round() * strike_step
            df_plot = df_total.groupby('bin').sum().reset_index().rename(columns={'bin': 'strike'})
            df_plot = df_plot[(df_plot['strike'] >= spot - (strike_step * num_levels/2)) & (df_plot['strike'] <= spot + (strike_step * num_levels/2))]

            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_plot[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_plot['strike'], x=df_plot[main_metric], orientation='h', 
                marker_color=colors, width=strike_step*0.8
            ))

            # Marcatori Professionali
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA FLIP")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=850,
                yaxis=dict(title="STRIKE", autorange=True, tickformat=".0f", gridcolor="#333"),
                xaxis=dict(title=f"Dollar {main_metric} Exposure ($)", zerolinecolor="white"),
                title=f"Analisi Quantitativa: {main_metric.upper()} Exposure"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore: {e}")
