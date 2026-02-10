import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GEX PRO TERMINAL V39", initial_sidebar_state="expanded")
st_autorefresh(interval=60000, key="global_refresh")

TICKER_LIST = ["NDX", "SPX", "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "COIN", "MSTR", "PLTR", "IBIT"]

def fix_ticker(symbol):
    if not symbol: return None
    s = symbol.upper().strip()
    return f"^{s}" if s in ["NDX", "SPX", "RUT"] else s

# --- MOTORE VETTORIALE (Anti-Errore Label) ---
def fast_engine(df, spot, t_yrs, r=0.045):
    # Trasformiamo tutto in array NumPy subito: NumPy non ha "labels", quindi non può dare l'errore
    s = float(spot)
    k = df['strike'].to_numpy()
    v = np.where(df['impliedVolatility'].to_numpy() <= 0, 1e-9, df['impliedVolatility'].to_numpy())
    oi = df['openInterest'].to_numpy()
    types = df['type'].to_numpy()
    t = max(t_yrs, 1e-9)
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    gamma = (pdf / (s * v * np.sqrt(t))) * (s**2) * 0.01 * oi * 100
    vanna = s * pdf * d1 / v * 0.01 * oi
    charm = (pdf * (r / (v * np.sqrt(t)) - d1 / (2 * t))) * oi * 100
    vega = s * pdf * np.sqrt(t) * oi * 100
    
    is_call = (types == 'call')
    theta = (-(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(np.where(is_call, d2, -d2))) * oi * 100
    
    mult = np.where(is_call, 1, -1)
    
    # Restituiamo un DataFrame nuovo di zecca con indici puliti
    return pd.DataFrame({
        'strike': k, 'Gamma': gamma * mult, 'Vanna': vanna * mult, 
        'Charm': charm * mult, 'Vega': vega, 'Theta': theta
    }).reset_index(drop=True)

# --- SIDEBAR ---
st.sidebar.header("🕹️ GEX ENGINE V39")
active_t = st.sidebar.selectbox("ASSET", TICKER_LIST)
t_str = fix_ticker(active_t)

if t_str:
    t_obj = yf.Ticker(t_str)
    try:
        exps = t_obj.options
        sel_exp = st.sidebar.selectbox("SCADENZA", exps)
        strike_step = st.sidebar.selectbox("STEP STRIKE", [1, 5, 10, 25, 50, 100, 250], index=4)
        num_levels = st.sidebar.slider("ZOOM AREA (Punti)", 100, 2500, 1000)
        main_metric = st.sidebar.radio("METRICA GRAFICO:", ['Gamma', 'Vanna', 'Charm', 'Vega', 'Theta'])

        hist = t_obj.history(period='1d')
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            t_yrs = max((datetime.strptime(sel_exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365
            chain = t_obj.option_chain(sel_exp)
            
            # --- PULIZIA ATOMICA ---
            calls = chain.calls[['strike', 'impliedVolatility', 'openInterest']].copy().assign(type='call')
            puts = chain.puts[['strike', 'impliedVolatility', 'openInterest']].copy().assign(type='put')
            
            # Uniamo e forziamo indici univoci
            df_raw = pd.concat([calls, puts], ignore_index=True)
            df_clean = df_raw.groupby(['strike', 'type'], as_index=False).agg({
                'impliedVolatility': 'mean', 'openInterest': 'sum'
            }).reset_index(drop=True)
            
            # Calcolo Greche
            df_res = fast_engine(df_clean, spot, t_yrs)

            # --- REGIME SCORES (I CAMPANELLI) ---
            def calc_regime(series):
                net, total = series.sum(), series.abs().sum()
                return net / total if total != 0 else 0

            g_reg, v_reg, c_reg = calc_regime(df_res['Gamma']), calc_regime(df_res['Vanna']), calc_regime(df_res['Charm'])

            st.markdown(f"## 🏛️ {active_t} Institutional Monitor | Spot: {spot:.2f}")
            
            # Indicatori Superiori
            c1, c2, c3, c4 = st.columns(4)
            for col, name, val in zip([c1, c2, c3], ["GAMMA", "VANNA", "CHARM"], [g_reg, v_reg, c_reg]):
                color = "#00ff00" if val > 0 else "#ff4444"
                status = "🛡️ POSITIVO" if val > 0.5 else "⚠️ NEGATIVO" if val < -0.5 else "⚖️ NEUTRALE"
                col.markdown(f"""
                <div style="background:#1e1e1e; padding:15px; border-radius:10px; border-top: 4px solid {color}; text-align:center;">
                    <small style="color:#888">{name} REGIME</small>
                    <h2 style="margin:0; color:{color}">{val:+.2f}</h2>
                    <strong style="font-size:12px">{status}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            bias_col = "#00ff00" if g_reg > 0 else "#ff4444"
            c4.markdown(f"""
            <div style="background:{bias_col}22; padding:15px; border-radius:10px; border:1px solid {bias_col}; text-align:center; height:100%">
                <small>MARKET BIAS</small><br>
                <b style="font-size:20px; color:{bias_col}">{'BULLISH' if g_reg > 0 else 'BEARISH'}</b>
            </div>
            """, unsafe_allow_html=True)

            # --- LOGICA LIVELLI E GRAFICO ---
            df_total = df_res.groupby('strike', as_index=False).sum(numeric_only=True).sort_values('strike').reset_index(drop=True)
            df_total['cum_gamma'] = df_total['Gamma'].cumsum()
            z_gamma = df_total.loc[df_total['cum_gamma'].abs().idxmin(), 'strike']
            
            df_total['bin'] = np.floor(df_total['strike'] / strike_step) * strike_step
            df_plot = df_total.groupby('bin', as_index=False).sum(numeric_only=True).rename(columns={'bin': 'strike'}).reset_index(drop=True)
            df_view = df_plot[(df_plot['strike'] >= spot - num_levels) & (df_plot['strike'] <= spot + num_levels)].copy()

            # Plotly
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_view['strike'], x=df_view[main_metric], orientation='h',
                marker_color=['#00ff00' if x>=0 else '#00aaff' for x in df_view[main_metric]],
                width=strike_step * 0.8
            ))
            fig.add_hline(y=spot, line_color="cyan", line_dash="dot", annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="yellow", line_dash="dash", annotation_text="ZERO G")
            fig.update_layout(template="plotly_dark", height=750, margin=dict(l=0,r=0,t=20,b=0),
                              yaxis=dict(dtick=strike_step, title="STRIKE"), xaxis=dict(title=f"Net {main_metric} Exposure"))
            st.plotly_chart(fig, use_container_width=True)

            # --- TABELLA DETTAGLIATA ---
            st.markdown("### 📊 Market Maker Inventory Details")
            # Mostriamo i livelli più vicini allo spot (ATM)
            table_df = df_plot.iloc[(df_plot['strike'] - spot).abs().argsort()[:20]].sort_values('strike', ascending=False).reset_index(drop=True)
            
            st.dataframe(table_df[['strike', 'Gamma', 'Vanna', 'Charm', 'Vega', 'Theta']].style.format(precision=1).map(
                lambda x: f"color: {'#00ff00' if x > 0 else '#ff4444' if x < 0 else 'white'}",
                subset=['Gamma', 'Vanna', 'Charm', 'Theta']
            ), use_container_width=True)

    except Exception as e:
        st.error(f"Errore tecnico: {e}")
