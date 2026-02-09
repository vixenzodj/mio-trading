import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(layout="wide", page_title="PRO GEX Intelligence Terminal", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

# --- DATABASE TICKER (50+) ---
TICKER_LIST = [
    "NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", 
    "AMD", "NFLX", "COIN", "MARA", "IBIT", "BITO", "SMCI", "AVGO", "ARM", "MU", "INTC", "ASML",
    "JPM", "GS", "BAC", "V", "MA", "DIS", "BA", "CAT", "XOM", "CVX", "TLT", "GLD", "SLV", "USO",
    "PLTR", "UBER", "ABNB", "PYPL", "SQ", "BABA", "NIO", "MSTR", "HOOD", "SHOP", "ADBE", "CRM"
]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    if s in ["NDX", "SPX", "RUT", "VIX", "DJI"]: return f"^{s}"
    return s

# --- MOTORE GRECHE ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
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
        if not exps: return "N/D", "#555", "Nessun dato"
        sel = exps[min(exp_idx, len(exps)-1)]
        t_yrs = max((datetime.strptime(sel, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
        ch = t_obj.option_chain(sel)
        c = ch.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        p = ch.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
        g, v = c[0].sum() - p[0].sum(), c[1].sum() - p[1].sum()
        if g > 0 and v > 0: return "LONG", "#00ff00", "Strong"
        if g < 0 and v < 0: return "SHORT", "#ff4444", "Strong"
        return "NEUTRAL", "#ffff00", "Volatile"
    except: return "ERR", "#555", "Errore"

# --- SIDEBAR: CONTROLLO COMPLETO ---
st.sidebar.header("🚀 CONTROLLO TERMINALE")
sel_ticker = st.sidebar.selectbox("SELEZIONA ASSET (Preimpostati)", ["CERCA..."] + sorted(TICKER_LIST))
manual_ticker = st.sidebar.text_input("OPPURE INSERISCI MANUALE", "")
active_ticker = manual_ticker if manual_ticker else (sel_ticker if sel_ticker != "CERCA..." else "NDX")

trade_mode = st.sidebar.selectbox("MODALITÀ OPERATIVA", ["SCALPING (0DTE)", "INTRADAY (Weekly)", "SWING (Monthly)"])
zoom_val = st.sidebar.slider("ZOOM AREA %", 1, 40, 5)
main_metric = st.sidebar.radio("METRICA GRAFICO PRINCIPALE", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

# --- LOGICA DI CARICAMENTO ---
t_str = fix_ticker(active_ticker)

if t_str:
    try:
        t_obj = yf.Ticker(t_str)
        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = hist['Close'].iloc[-1]
            
            # 1. BIAS MULTI-TIMEFRAME (Sempre visibile)
            st.subheader(f"📡 Radar Bias: {active_ticker.upper()}")
            m1, m2, m3 = st.columns(3)
            with st.spinner('Analisi flussi...'):
                b1, c1, p1 = analyze_tf(t_obj, spot, 0) # Scalp
                b2, c2, p2 = analyze_tf(t_obj, spot, 2) # Weekly
                b3, c3, p3 = analyze_tf(t_obj, spot, 5) # Swing
            
            m1.markdown(f"<div style='text-align:center; padding:15px; border:2px solid {c1}; border-radius:10px;'><h3>SCALP</h3><h1 style='color:{c1};'>{b1}</h1><p>{p1}</p></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='text-align:center; padding:15px; border:2px solid {c2}; border-radius:10px;'><h3>INTRA</h3><h1 style='color:{c2};'>{b2}</h1><p>{p2}</p></div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='text-align:center; padding:15px; border:2px solid {c3}; border-radius:10px;'><h3>SWING</h3><h1 style='color:{c3};'>{b3}</h1><p>{p3}</p></div>", unsafe_allow_html=True)

            # 2. GRAFICO ADATTIVO IN BASE AL TRADE MODE
            st.divider()
            exps = t_obj.options
            idx = 0 if "SCALPING" in trade_mode else (2 if "INTRADAY" in trade_mode else 5)
            sel_exp = exps[min(idx, len(exps)-1)]
            
            t_yrs_main = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            ch_data = t_obj.option_chain(sel_exp)
            
            c_m = ch_data.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs_main), axis=1)
            p_m = ch_data.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs_main), axis=1)
            
            df = pd.DataFrame({'strike': ch_data.calls['strike']})
            df['Gamma'], df['Vanna'], df['Charm'] = c_m[0]-p_m[0], c_m[1]-p_m[1], c_m[2]-p_m[2]
            df['Vega'], df['Theta'] = c_m[3]+p_m[3], c_m[4]+p_m[4]
            
            l, u = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            df_z = df[(df['strike']>=l) & (df['strike']<=u)]
            
            # Grafico
            z_flip = df_z.loc[df_z[main_metric].abs().idxmin(), 'strike']
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df_z['strike'], x=df_z[main_metric], orientation='h', 
                                 marker_color=np.where(df_z[main_metric]>=0, '#00ff00', '#00aaff')))
            fig.add_hline(y=spot, line_color="cyan", annotation_text=f"PREZZO: {spot:.2f}")
            fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text=f"ZERO {main_metric.upper()}")
            fig.update_layout(template="plotly_dark", height=700, title=f"Profilo {main_metric} - Scadenza: {sel_exp}")
            st.plotly_chart(fig, use_container_width=True)

            # 3. DASHBOARD COMPLETA METRICHE TOTALI (Sempre visibile in basso)
            st.divider()
            st.subheader("📊 Esposizione Globale (Tutte le Metriche)")
            cols = st.columns(5)
            for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
                total = df[m].sum()
                cols[i].metric(f"Total {m}", f"{total/1e6:.2f}M" if abs(total)>1e5 else f"{total:.2f}")
                
        else: st.warning("Dati non disponibili per questo ticker.")
    except Exception as e:
        st.error(f"Errore: Inserisci un ticker valido o attendi il caricamento. {e}")
