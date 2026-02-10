import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="TERMINAL GEX PRO V10", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# DATABASE 50 TICKERS
TICKER_LIST = [
    "NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", 
    "AMD", "NFLX", "COIN", "MARA", "IBIT", "BITO", "SMCI", "AVGO", "ARM", "MU", "INTC", "ASML",
    "JPM", "GS", "BAC", "V", "MA", "DIS", "BA", "CAT", "XOM", "CVX", "TLT", "GLD", "SLV", "USO",
    "PLTR", "UBER", "ABNB", "PYPL", "SQ", "BABA", "NIO", "MSTR", "HOOD", "SHOP", "ADBE", "CRM"
]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT", "VIX", "DJI"] else s

def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = float(spot), float(row['strike']), float(row['impliedVolatility']), float(row['openInterest'])
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0.0]*5)
        r, d1 = 0.045, (np.log(s/k) + (0.045 + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2, pdf = d1 - v * np.sqrt(t_yrs), norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (r/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - r * k * np.exp(-r * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([float(gamma), float(vanna), float(charm), float(vega), float(theta)])
    except: return pd.Series([0.0]*5)

def analyze_tf(t_obj, spot, exp_idx):
    try:
        exps = t_obj.options
        if not exps: return "N/D", "#555"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g_net = c[0].sum() - p[0].sum()
        return ("LONG", "#00ff00") if g_net > 0 else ("SHORT", "#ff4444")
    except: return "ERR", "#555"

# --- SIDEBAR ---
st.sidebar.header("🕹️ DASHBOARD CONTROL")
sel_ticker = st.sidebar.selectbox("TICKER LIST (50+)", ["CERCA..."] + sorted(TICKER_LIST))
manual_t = st.sidebar.text_input("TICKER MANUALE", "")
active_t = manual_t if manual_t else (sel_ticker if sel_ticker != "CERCA..." else "NDX")

trade_mode = st.sidebar.selectbox("MODALITÀ OPERATIVA", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
strike_step = st.sidebar.selectbox("GRANULARITÀ (Step)", [1, 5, 10, 25, 50, 100, 250], index=4)
num_levels = st.sidebar.slider("ZOOM (Numero Strike)", 10, 100, 40)
main_metric = st.sidebar.radio("METRICA VISIVA", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

t_str = fix_ticker(active_t)
if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            
            # 1. RADAR BIAS
            st.subheader(f"📡 Radar Intelligence: {active_t.upper()}")
            r1, r2, r3 = st.columns(3)
            b1, c1 = analyze_tf(t_obj, spot, 0)
            b2, c2 = analyze_tf(t_obj, spot, 2)
            b3, c3 = analyze_tf(t_obj, spot, 5)
            r1.markdown(f"<div style='text-align:center;border:2px solid {c1};padding:10px;border-radius:10px;'>SCALP: <b style='color:{c1};'>{b1}</b></div>", unsafe_allow_html=True)
            r2.markdown(f"<div style='text-align:center;border:2px solid {c2};padding:10px;border-radius:10px;'>INTRA: <b style='color:{c2};'>{b2}</b></div>", unsafe_allow_html=True)
            r3.markdown(f"<div style='text-align:center;border:2px solid {c3};padding:10px;border-radius:10px;'>SWING: <b style='color:{c3};'>{b3}</b></div>", unsafe_allow_html=True)

            # 2. CALCOLO DATI
            exps = t_obj.options
            idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
            sel_exp = exps[min(idx, len(exps)-1)]
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            ch = t_obj.option_chain(sel_exp)
            
            c_vals = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            p_vals = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
            
            df = pd.DataFrame({'strike': ch.calls['strike'].astype(float)})
            for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
                df[m] = c_vals[i] - p_vals[i] if i < 3 else c_vals[i] + p_vals[i]

            # Muri (Calcolati su dati grezzi per precisione)
            call_wall = float(df.loc[df['Gamma'].idxmax(), 'strike'])
            put_wall = float(df.loc[df['Gamma'].idxmin(), 'strike'])

            # Raggruppamento per Granularità
            df['bin'] = (df['strike'] / strike_step).round() * strike_step
            df_g = df.groupby('bin')[['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']].sum().reset_index()
            df_g.rename(columns={'bin': 'strike'}, inplace=True)
            
            # Filtro Visivo
            df_g['dist'] = (df_g['strike'] - spot).abs()
            df_p = df_g.sort_values('dist').head(num_levels).sort_values('strike')
            
            # 3. GRAFICO (ASSET NUMERICO - NO ERRORI)
            fig = go.Figure()
            colors = ['#00ff00' if x >= 0 else '#00aaff' for x in df_p[main_metric]]
            
            # Barre con larghezza dinamica per evitare schiacciamento
            fig.add_trace(go.Bar(
                y=df_p['strike'], x=df_p[main_metric], orientation='h', 
                marker_color=colors, name=main_metric,
                width=strike_step * 0.8  # Occupa l'80% dello spazio dello step
            ))

            # LINEE LIVELLI (SOLO NUMERICHE)
            fig.add_hline(y=call_wall, line_color="red", line_width=3, annotation_text=f"CALL WALL: {call_wall}")
            fig.add_hline(y=put_wall, line_color="#00ff00", line_width=3, annotation_text=f"PUT WALL: {put_wall}")
            fig.add_hline(y=spot, line_color="cyan", line_width=2, line_dash="dash", annotation_text=f"SPOT: {spot:.2f}")

            fig.update_layout(
                template="plotly_dark", height=850,
                title=f"PROFILO {main_metric.upper()} - {active_t} ({sel_exp})",
                yaxis=dict(title="PREZZO STRIKE", autorange="reversed", tickformat=".0f"),
                xaxis=dict(title="Esposizione Netta", zerolinecolor="white"),
                bargap=0
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
