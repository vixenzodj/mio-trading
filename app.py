import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="WAR ROOM - GREEKS TERMINAL")
st_autorefresh(interval=900000, key="datarefresh") # 15 min auto-refresh

TICKERS = [
    'QQQ', 'SPY', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL',
    'AMD', 'SMH', 'NFLX', 'ADBE', 'PYPL', 'INTC', 'ASML', 'AVGO', 'COST', 'CSCO',
    'MU', 'AMAT', 'LRCX', 'QCOM', 'TXN', 'PANW', 'SNPS', 'CDNS', 'ORCL', 'INTU',
    'BABA', 'PLTR', 'UBER', 'ABNB', 'COIN', 'MSTR', 'MARA', 'RIOT', 'SQ', 'SHOP',
    'SNOW', 'WDAY', 'TEAM', 'DDOG', 'ZS', 'OKTA', 'CRWD', 'NET', 'U', 'RBLX'
]

# --- MOTORE MATEMATICO (BLACK-SCHOLES) ---
def calculate_greeks(row, spot, t=1/365, r=0.04):
    s = spot
    k = row['strike']
    v = row['impliedVolatility']
    oi = row['openInterest']
    
    if v <= 0 or t <= 0: 
        return pd.Series([0]*5, index=['gamma', 'vanna', 'charm', 'vega', 'theta'])
    
    d1 = (np.log(s/k) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    
    # Formule Greche
    gamma = pdf / (s * v * np.sqrt(t))
    vanna = (pdf * d1) / v
    charm = (pdf * ( (r/(v*np.sqrt(t))) - (d1/(2*t)) ))
    vega = s * pdf * np.sqrt(t)
    theta = -(s * pdf * v) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2)

    # Moltiplichiamo per l'Open Interest per avere l'esposizione reale del mercato
    return pd.Series([gamma*oi, vanna*oi, charm*oi, vega*oi, theta*oi], 
                     index=['gamma', 'vanna', 'charm', 'vega', 'theta'])

def get_full_analysis(symbol):
    t_obj = yf.Ticker(symbol)
    price = t_obj.history(period='1d')['Close'].iloc[-1]
    exp = t_obj.options[0]
    opts = t_obj.option_chain(exp)
    
    calls = opts.calls.apply(lambda r: calculate_greeks(r, price), axis=1)
    puts = opts.puts.apply(lambda r: calculate_greeks(r, price), axis=1)
    
    # Net Exposure (Market Maker Side)
    # Nota: Sulle call i MM sono solitamente Short, sulle put sono Long/Short a seconda del setup
    net_gex = (calls['gamma'] - puts['gamma']).sum()
    net_vanna = (calls['vanna'] - puts['vanna']).sum()
    net_charm = (calls['charm'] - puts['charm']).sum()
    net_vega = (calls['vega'] + puts['vega']).sum()
    net_theta = (calls['theta'] + puts['theta']).sum()
    
    return {
        "price": price, "gex": net_gex, "vanna": net_vanna, 
        "charm": net_charm, "vega": net_vega, "theta": net_theta,
        "cw": opts.calls.loc[opts.calls['openInterest'].idxmax(), 'strike'],
        "pw": opts.puts.loc[opts.puts['openInterest'].idxmax(), 'strike'],
        "exp": exp
    }

# --- INTERFACCIA ---
st.title("🏛️ Terminale Greeks Avanzato (Vanna/Charm/Theta)")

# 1. SCANNER
st.header("⚡ Scanner 50 Ticker (Filtro Volatilità e Flussi)")
if st.button("Esegui Scansione"):
    data_list = []
    for s in TICKERS[:15]: # Limitato a 15 per velocità di caricamento, puoi estendere
        try:
            d = get_full_analysis(s)
            data_list.append({
                "Ticker": s, "Prezzo": round(d['price'],2), 
                "GEX Totale": round(d['gex'],2), "Vanna": round(d['vanna'],2),
                "Theta": round(d['theta'],2), "Stato": "🟢 BULL" if d['gex'] > 0 else "🔴 BEAR"
            })
        except: continue
    st.table(pd.DataFrame(data_list))

st.divider()

# 2. DETTAGLIO SINGOLO
selected = st.selectbox("Seleziona Ticker per analisi Greche Profonda:", TICKERS)
if selected:
    d = get_full_analysis(selected)
    
    # Visualizzazione Greche di II Ordine
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VANNA (Sens. Vol)", f"{d['vanna']:.2f}")
    c2.metric("CHARM (Sens. Tempo)", f"{d['charm']:.2f}")
    c3.metric("VEGA (Esposizione Vol)", f"{d['vega']:.2f}")
    c4.metric("THETA (Decadimento)", f"{d['theta']:.2f}")
    
    # Grafico di Posizionamento
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = d['price'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Posizione {selected} rispetto a Gamma Flip"},
        gauge = {
            'axis': {'range': [d['pw'], d['cw']]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [d['pw'], (d['cw']+d['pw'])/2], 'color': "red"},
                {'range': [(d['cw']+d['pw'])/2, d['cw']], 'color': "green"}
            ],
            'threshold': {'line': {'color': "yellow", 'width': 4}, 'thickness': 0.75, 'value': d['price']}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Vanna Alto:** Indica che il mercato è sensibile ai cambi di volatilità. **Charm Alto:** Indica forte pressione di acquisto/vendita verso la scadenza.")
