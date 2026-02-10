import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V27", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

def calc_greeks_pro(row, spot, t_yrs, r=0.045):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * (s**2) * 0.01 * oi * 100
        vanna = s * pdf * d1 / v * 0.01 * oi
        charm = (pdf * (r / (v * np.sqrt(t_yrs)) - d1 / (2 * t_yrs))) * oi * 100
        vega = s * pdf * np.sqrt(t_yrs) * oi * 100
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2 if row['type']=='call' else -d2)) * oi * 100
        mult = 1 if row['type'] == 'call' else -1
        return pd.Series([gamma * mult, vanna * mult, charm * mult, vega, theta])
    except: return pd.Series([0.0]*5)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V27")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    exps = t_obj.options
    sel_exp = st.sidebar.selectbox("SCADENZA ATTIVA", exps)
    # Step strike cruciale per visibilità NDX
    strike_step = st.sidebar.selectbox("STEP STRIKE (Raggruppamento)", [1, 5, 10, 25, 50, 100, 250], index=4)
    zoom_range = st.sidebar.slider("ZOOM PREZZO (+/- Punti)", 50, 2500, 1200)
    main_metric = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    try:
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            c, p = chain.calls.copy(), chain.puts.copy()
            c['type'], p['type'] = 'call', 'put'
            
            c_res = c.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            p_res = p.apply(lambda r: calc_greeks_pro(r, spot, t_yrs), axis=1)
            
            cols = ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']
            df_c = pd.DataFrame(c_res.values, columns=cols); df_c['strike'] = c['strike'].values
            df_p = pd.DataFrame(p_res.values, columns=cols); df_p['strike'] = p['strike'].values
            
            df_total = pd.concat([df_c, df_p]).groupby('strike', as_index=False).sum().sort_values('strike')

            # --- LOGICA ZERO GAMMA (Equilibrio) ---
            df_total['cum_gamma_net'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[(df_total['cum_gamma_net']).abs().idxmin(), 'strike']

            # --- AGGREGAZIONE GRAFICA ---
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False)[cols].sum().rename(columns={'bin': 'strike'})
            
            # Filtro visivo dinamico
            df_view = df_plot[(df_plot['strike'] >= spot - zoom_range) & (df_plot['strike'] <= spot + zoom_range)]
            
            call_wall = df_plot.loc[df_plot['Gamma'].idxmax(), 'strike']
            put_wall = df_plot.loc[df_plot['Gamma'].idxmin(), 'strike']

            # --- BIAS CALCULATION ---
            gamma_bias = df_total['Gamma'].sum()
            vanna_bias = df_total['Vanna'].sum()
            
            # Determiniamo il Bias
            if gamma_bias > 0 and vanna_bias > 0: bias, b_col = "STRONG BULLISH", "#00ff00"
            elif gamma_bias < 0 and vanna_bias < 0: bias, b_col = "STRONG BEARISH", "#ff4444"
            else: bias, b_col = "NEUTRAL / CAUTION", "#ffff00"

            # --- 1. DASHBOARD PRINCIPALE ---
            st.markdown(f"## 🏛️ {active_t} Professional Terminal | Spot: {spot:.2f}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("PREZZO ATTUALE", f"{spot:.2f}")
            d2.metric("ZERO GAMMA", f"{z_gamma:.0f}", f"{z_gamma-spot:.1f} pts")
            d3.metric("CALL WALL", f"{call_wall:.0f}")
            d4.markdown(f"<div style='border:2px solid {b_col}; padding:8px; border-radius:10px; text-align:center;'>BIAS SISTEMA<br><b style='color:{b_col}; font-size:20px;'>{bias}</b></div>", unsafe_allow_html=True)

            # --- 2. GRAFICO GEX (Visuale corretta) ---
            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_view[main_metric]]
            
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h', 
                marker_color=colors, width=strike_step * 0.9,
                text=[f"{v/1e6:.1f}M" if abs(v)>1e5 else "" for v in df_view[main_metric]], textposition='outside'
            ))

            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text="CALL WALL")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text="PUT WALL")
            fig.add_hline(y=z_gamma, line_color="yellow", line_width=2, line_dash="dash", annotation_text="ZERO GAMMA")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dot", annotation_text="SPOT")

            fig.update_layout(
                template="plotly_dark", height=750,
                yaxis=dict(title="STRIKE PRICE", autorange=True, gridcolor="#333"),
                xaxis=dict(title=f"Esposizione Netta {main_metric} ($)", zerolinecolor="white"),
                bargap=0.02
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 3. TABELLA METRICHE SOTTO IL GRAFICO ---
            st.markdown("### 📊 Valori Dettagliati (Gamma, Vega, Theta, Vanna)")
            # Prendiamo gli strike chiave e quelli ATM
            key_levels = [call_wall, put_wall, z_gamma]
            atm_range = df_plot[(df_plot['strike'] >= spot - (strike_step*2)) & (df_plot['strike'] <= spot + (strike_step*2))]
            
            final_metrics = pd.concat([df_plot[df_plot['strike'].isin(key_levels)], atm_range]).drop_duplicates().sort_values('strike', ascending=False)
            
            def style_vals(v):
                color = '#00ff00' if v > 0 else '#ff4444' if v < 0 else 'white'
                return f'color: {color}'

            st.table(final_metrics[['strike', 'Gamma', 'Vega', 'Theta', 'Vanna']].style.format(precision=2).applymap(style_vals, subset=['Gamma', 'Vanna', 'Theta']))

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
