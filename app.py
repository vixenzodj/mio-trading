import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="Professional GEX Terminal", initial_sidebar_state="expanded")

# --- DIZIONARIO TICKER (Per non doverli ricordare a memoria) ---
TICKER_MAP = {
    "INDICI / ETF": ["QQQ", "SPY", "IWM", "DIA"],
    "MAGNIFICENT 7": ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META"],
    "ALTRI SEMI / TECH": ["AMD", "AVGO", "SMCI", "ARM"],
    "CRYPTO ETF": ["IBIT", "ETHW"]
}

# --- SIDEBAR: SELEZIONE STRATEGICA ---
st.sidebar.header("🕹️ CONFIGURAZIONE")

# Trasformiamo le categorie in una lista piatta per la selectbox
all_tickers = []
for cat in TICKER_MAP:
    all_tickers.extend(TICKER_MAP[cat])

selected_ticker = st.sidebar.selectbox("SELEZIONA TICKER", all_tickers, index=0)

# Opzione per ticker manuale se non in lista
manual_ticker = st.sidebar.text_input("OPPURE INSERISCI MANUALE (es. COIN)", "").upper()
ticker = manual_ticker if manual_ticker else selected_ticker

trading_mode = st.sidebar.selectbox(
    "TIPO DI TRADING",
    ["SCALPING (0DTE Focus)", "INTRADAY (Weekly Focus)", "SWING (Monthly/Institutional)"]
)

# Configurazione Modalità
if trading_mode.startswith("SCALPING"):
    refresh_rate, default_zoom, expiry_auto = 60000, 3, 0
    st_autorefresh(interval=refresh_rate, key="scalp_ref")
elif trading_mode.startswith("INTRADAY"):
    refresh_rate, default_zoom, expiry_auto = 300000, 8, 1
    st_autorefresh(interval=refresh_rate, key="intra_ref")
else:
    refresh_rate, default_zoom, expiry_auto = 900000, 20, 3
    st_autorefresh(interval=refresh_rate, key="swing_ref")

# --- MOTORE DI CALCOLO ---
def get_all_greeks(row, spot, t_yrs):
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return pd.Series([0]*5)
        d1 = (np.log(s/k) + (0.04 + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        d2 = d1 - v * np.sqrt(t_yrs)
        pdf = norm.pdf(d1)
        gamma = (pdf / (s * v * np.sqrt(t_yrs))) * oi * 100
        vanna = ((pdf * d1) / v) * oi
        charm = (pdf * ( (0.04/(v*np.sqrt(t_yrs))) - (d1/(2*t_yrs)) )) * oi
        vega = (s * pdf * np.sqrt(t_yrs)) * oi
        theta = (-(s * pdf * v) / (2 * np.sqrt(t_yrs)) - 0.04 * k * np.exp(-0.04 * t_yrs) * norm.cdf(d2)) * oi
        return pd.Series([gamma, vanna, charm, vega, theta])
    except: return pd.Series([0]*5)

@st.cache_data(ttl=60)
def load_market_data(t_symbol, exp_idx, zoom_val):
    t_obj = yf.Ticker(t_symbol)
    h = t_obj.history(period='1d')
    if h.empty: return None
    spot = h['Close'].iloc[-1]
    exps = t_obj.options
    sel_exp = exps[exp_idx]
    t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
    chain = t_obj.option_chain(sel_exp)
    c_g = chain.calls.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    p_g = chain.puts.apply(lambda r: get_all_greeks(r, spot, t_yrs), axis=1)
    df = pd.DataFrame({'strike': chain.calls['strike']})
    df['Gamma'], df['Vanna'], df['Charm'], df['Vega'], df['Theta'] = c_g[0]-p_g[0], c_g[1]-p_g[1], c_g[2]-p_g[2], c_g[3]+p_g[3], c_g[4]+p_g[4]
    l, u = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
    return spot, df[(df['strike']>=l) & (df['strike']<=u)], sel_exp, df

# --- RENDER DASHBOARD ---
try:
    ticker_obj = yf.Ticker(ticker)
    avail_exps = ticker_obj.options
    exp_sel_idx = st.sidebar.selectbox("SCADENZA DISPONIBILE", range(len(avail_exps)), index=min(expiry_auto, len(avail_exps)-1), format_func=lambda x: avail_exps[x])
    zoom_pct = st.sidebar.slider("ZOOM AREA %", 1, 50, default_zoom)
    metric_view = st.sidebar.radio("METRICA GRAFICO", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

    res = load_market_data(ticker, exp_sel_idx, zoom_pct)
    if res:
        spot, df_p, exp_d, df_f = res
        z_flip = df_p.loc[df_p[metric_view].abs().idxmin(), 'strike']

        # Grafico
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df_p['strike'], x=df_p[metric_view], orientation='h', marker_color=np.where(df_p[metric_view]>=0, '#00ff00', '#00aaff')))
        fig.add_hline(y=spot, line_color="cyan", annotation_text="SPOT")
        fig.add_hline(y=z_flip, line_dash="dash", line_color="yellow", annotation_text="ZERO FLIP")
        fig.update_layout(template="plotly_dark", height=700, title=f"ANALISI {metric_view} - {ticker}")
        st.plotly_chart(fig, use_container_width=True)

        # Pannello Statistiche e Segnali
        st.divider()
        g_total = df_f['Gamma'].sum()
        v_total = df_f['Vanna'].sum()
        
        col_sig, col_stat = st.columns(2)
        with col_sig:
            st.subheader("📡 SEGNALE STATISTICO")
            if g_total > 0:
                st.markdown(f"<h2 style='color:#00ff00;'>🟢 LONG BIAS (Bullish)</h2>", unsafe_allow_html=True)
                st.write("Le condizioni suggeriscono stabilità e recupero sui ribassi. Gamma Positivo domina.")
            else:
                st.markdown(f"<h2 style='color:#ff4444;'>🔴 SHORT BIAS (Bearish)</h2>", unsafe_allow_html=True)
                st.write("Alta probabilità di accelerazioni violente al ribasso. Gamma Negativo domina.")

        with col_stat:
            st.subheader("📈 PROIEZIONE")
            dist_flip = abs(spot - z_flip) / spot * 100
            if dist_flip < 1.0:
                st.warning(f"⚠️ DANGER ZONE: Il prezzo è al {dist_flip:.2f}% dallo Zero Gamma. Attesi movimenti erratici.")
            else:
                st.info(f"Il prezzo è in zona sicura. Supporto/Resistenza principali visibili nei picchi del grafico.")

        # Recap tutte le metriche
        m_cols = st.columns(5)
        for i, m in enumerate(['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']):
            m_cols[i].metric(f"Total {m}", f"{df_f[m].sum()/1e6:.2f}M" if abs(df_f[m].sum())>1e5 else f"{df_f[m].sum():.2f}")

except Exception as e:
    st.error(f"Seleziona un ticker valido o attendi il caricamento... ({e})")
