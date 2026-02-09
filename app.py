import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Professional GEX Terminal V5", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "QQQ", "SPY", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR"]

def fix_ticker(symbol):
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT", "VIX"] else s

# --- MOTORE CALCOLO GRECHE ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        r = 0.045
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0.0]*5)

def analyze_tf(t_obj, spot, exp_idx):
    try:
        exps = t_obj.options
        if not exps: return "N/D", "#555", "No Data"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g_net = c[0].sum() - p[0].sum()
        v_net = c[1].sum() - p[1].sum()
        if g_net > 0 and v_net > 0: return "LONG", "#00ff00", "Strong"
        if g_net < 0 and v_net < 0: return "SHORT", "#ff4444", "Strong"
        return "NEUTRAL", "#ffff00", "Volatile"
    except: return "ERR", "#555", "Error"

# --- SIDEBAR ---
st.sidebar.header("🕹️ CONTROLLO TERMINALE")
sel_ticker = st.sidebar.selectbox("ASSET PREIMPOSTATI", ["CERCA..."] + sorted(TICKER_LIST))
manual_t = st.sidebar.text_input("MANUAL TICKER", "")
active_t = manual_t if manual_t else (sel_ticker if sel_ticker != "CERCA..." else "NDX")

trade_mode = st.sidebar.selectbox("MODALITÀ", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100], index=2)
num_strikes = st.sidebar.slider("NUMERO STRIKE", 10, 100, 40)
main_metric = st.sidebar.radio("METRICA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

t_str = fix_ticker(active_t)
if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = hist['Close'].iloc[-1]
            
            # --- 1. MATRICE BIAS (Sempre visibile) ---
            st.subheader(f"📡 Radar Intelligence: {active_t.upper()}")
            m1, m2, m3 = st.columns(3)
            with st.spinner('Analizzando liquidità...'):
                b1, c1, p1 = analyze_tf(t_obj, spot, 0)
                b2, c2, p2 = analyze_tf(t_obj, spot, 2)
                b3, c3, p3 = analyze_tf(t_obj, spot, 5)
            m1.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c1}; border-radius:10px;'><h4>SCALP</h4><h2 style='color:{c1};'>{b1}</h2><p>{p1}</p></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c2}; border-radius:10px;'><h4>INTRA</h4><h2 style='color:{c2};'>{b2}</h2><p>{p2}</p></div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='text-align:center; padding:10px; border:2px solid {c3}; border-radius:10px;'><h4>SWING</h4><h2 style='color:{c3};'>{b3}</h2><p>{p3}</p></div>", unsafe_allow_html=True)

            # --- 2. ELABORAZIONE DATI ---
            exps = t_obj.options
            idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
            sel_exp = exps[min(idx, len(exps)-1)]
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            ch = t_obj.option_chain(sel_exp)
            
            calls = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            puts = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike']})
            df['Gamma'], df['Vanna'], df['Charm'] = calls[0]-puts[0], calls[1]-puts[1], calls[2]-puts[2]
            df['Vega'], df['Theta'] = calls[3]+puts[3], calls[4]+puts[4]

            # Muri (su intera catena)
            call_wall = float(df.loc[df['Gamma'].idxmax(), 'strike'])
            put_wall = float(df.loc[df['Gamma'].idxmin(), 'strike'])

            # Filtro visivo
            df_plot = df[df['strike'] % strike_step == 0].copy()
            df_plot['dist'] = abs(df_plot['strike'] - spot)
            df_z = df_plot.sort_values('dist').head(num_strikes).sort_values('strike')
            z_flip = float(df_z.loc[df_z[main_metric].abs().idxmin(), 'strike'])

            # --- 3. GRAFICO DINAMICO ---
            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_z[main_metric]]
            
            # Barre - Usiamo Strike come numero per evitare errori di concatenazione
            fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[main_metric], orientation='h', 
                                 marker_color=colors, name=main_metric))
            
            # Linee Livelli (Senza concatenazione str + int)
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text=f"CALL WALL: {call_wall}")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text=f"PUT WALL: {put_wall}")
            fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(template="plotly_dark", height=850, 
                              title=f"<b>PROFILO {main_metric.upper()} - {active_t}</b> ({sel_exp})",
                              yaxis=dict(title="STRIKE", autorange="reversed", type='linear'), bargap=0.1)
            
            st.plotly_chart(fig, use_container_width=True)

            # --- 4. RIEPILOGO ---
            st.divider()
            cols = st.columns(5)
            for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
                val = df[m].sum()
                cols[i].metric(f"Total {m}", f"{val/1e6:.2f}M" if abs(val)>1e5 else f"{val:.2f}")

    except Exception as e: st.error(f"Errore caricamento dati: {e}")
