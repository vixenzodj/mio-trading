import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- CONFIGURAZIONE PAGINA (Full Dark Mode) ---
st.set_page_config(layout="wide", page_title="GEX TERMINAL PRO", initial_sidebar_state="expanded")
st_autorefresh(interval=300000, key="datarefresh") # Refresh 5 min

# --- CSS PER STILE DARK TERMINAL ---
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    h1, h2, h3 { color: #00ff88 !important; font-family: 'Courier New', monospace; }
    .stMetricValue { color: #00ffff !important; font-family: 'Courier New', monospace; }
    .stMetricLabel { color: #888888 !important; }
</style>
""", unsafe_allow_html=True)

# --- CALCOLI BLACK-SCHOLES & GRECHE ---
def calc_greeks(row, spot, t_yrs, r=0.04):
    """Calcola Gamma e GEX per singola opzione"""
    try:
        s, k, v, oi = spot, row['strike'], row['impliedVolatility'], row['openInterest']
        if v <= 0 or t_yrs <= 0 or oi <= 0: return 0
        
        d1 = (np.log(s/k) + (r + 0.5 * v**2) * t_yrs) / (v * np.sqrt(t_yrs))
        gamma = norm.pdf(d1) / (s * v * np.sqrt(t_yrs))
        
        # GEX = Gamma * Open Interest * Spot * 100 (Nozionale)
        # Semplificato per visualizzazione profilo: Gamma * OI
        return gamma * oi * 100
    except:
        return 0

@st.cache_data(ttl=300)
def get_gex_profile(ticker, expiry_idx, zoom_pct):
    t_obj = yf.Ticker(ticker)
    history = t_obj.history(period='1d')
    if history.empty: return None, None, None, None, None, None
    
    spot = history['Close'].iloc[-1]
    exps = t_obj.options
    sel_exp = exps[expiry_idx]
    
    # Calcolo tempo
    dt_exp = datetime.strptime(sel_exp, '%Y-%m-%d')
    t_days = (dt_exp - datetime.now()).days
    t_yrs = max(t_days, 0.5) / 365
    
    # Scarico Catena
    opts = t_obj.option_chain(sel_exp)
    calls, puts = opts.calls.copy(), opts.puts.copy()
    
    # Calcolo GEX per ogni strike
    calls['GEX'] = calls.apply(lambda x: calc_greeks(x, spot, t_yrs), axis=1)
    puts['GEX'] = puts.apply(lambda x: calc_greeks(x, spot, t_yrs), axis=1)
    
    # Unione dati per Net GEX (Call GEX - Put GEX)
    df = pd.merge(calls[['strike', 'GEX']], puts[['strike', 'GEX']], on='strike', suffixes=('_C', '_P')).fillna(0)
    df['NetGEX'] = df['GEX_C'] - df['GEX_P']
    
    # --- INDIVIDUAZIONE ZERO GAMMA (Flip Point) ---
    # Trova lo strike dove il NetGEX cambia segno più vicino al prezzo
    # Semplificazione: Strike con NetGEX assoluto più basso vicino allo spot
    near_spot = df[(df['strike'] > spot*0.9) & (df['strike'] < spot*1.1)]
    try:
        # Logica: Zero Gamma è dove il segno cambia. Cerchiamo il punto di transizione.
        # Per visualizzazione rapida: usiamo lo strike con il valore assoluto minore
        # in un range ristretto, oppure interpoliamo. Qui prendiamo il min abs value.
        zero_gamma_level = near_spot.loc[near_spot['NetGEX'].abs().idxmin(), 'strike']
    except:
        zero_gamma_level = spot # Fallback

    # --- INDIVIDUAZIONE MURI (WALLS) ---
    # Call Wall: Max GEX positivo (Resistenza)
    # Put Wall: Max GEX negativo (Supporto) - Nota: usiamo i minimi perché Put GEX è sottratto
    # Ma per visualizzarlo come OI assoluto delle Put, meglio guardare OI Put
    c_wall = calls.loc[calls['openInterest'].idxmax(), 'strike']
    p_wall = puts.loc[puts['openInterest'].idxmax(), 'strike']

    # --- FILTRO ZOOM DINAMICO ---
    # Tagliamo i dati in base allo slider
    low_bound = spot * (1 - zoom_pct/100)
    up_bound = spot * (1 + zoom_pct/100)
    
    df_view = df[(df['strike'] >= low_bound) & (df['strike'] <= up_bound)].copy()
    
    return spot, df_view, sel_exp, c_wall, p_wall, zero_gamma_level

# --- INTERFACCIA UTENTE ---
st.sidebar.title("🎛️ GEX CONTROL")
ticker_input = st.sidebar.text_input("TICKER", value="QQQ").upper()
try:
    ticker_data = yf.Ticker(ticker_input)
    avail_exps = ticker_data.options
    if not avail_exps: raise ValueError("No options")
except:
    st.error("Ticker non valido o dati opzioni assenti.")
    st.stop()

exp_idx = st.sidebar.selectbox("SCADENZA", range(len(avail_exps)), format_func=lambda x: avail_exps[x])

# LO SLIDER MAGICO (ZOOM)
zoom = st.sidebar.slider("ZOOM LEVEL (%)", 1, 20, 4, help="Più basso è il numero, più grandi sono le barre centrali.")

# --- CARICAMENTO DATI ---
spot, df, exp_date, cw, pw, zg = get_gex_profile(ticker_input, exp_idx, zoom)

if df is not None:
    # --- COSTRUZIONE GRAFICO GEXBOT ---
    
    # 1. Calcolo del massimo locale per la scala (IL SEGRETO DELLE BARRE GRANDI)
    # Troviamo il valore massimo ASSOLUTO solo tra le barre visibili
    max_val_in_view = df['NetGEX'].abs().max()
    
    # Se tutte le barre sono a zero, diamo un valore default per non rompere il grafico
    if max_val_in_view == 0: max_val_in_view = 1
    
    # Impostiamo il range dell'asse X simmetrico per centrare lo zero
    x_range = [-max_val_in_view * 1.1, max_val_in_view * 1.1]

    fig = go.Figure()

    # BARRE DINAMICHE
    # Usiamo colori condizionali: Verde se > 0, Blu GexBot se < 0
    colors = np.where(df['NetGEX'] >= 0, '#00ff00', '#00aaff') # GexBot Green & Blue
    
    fig.add_trace(go.Bar(
        y=df['strike'],
        x=df['NetGEX'],
        orientation='h',
        marker_color=colors,
        marker_line_width=0, # Barre piatte stile GexBot
        name='Net GEX',
        hovertemplate='Strike: %{y}<br>Net GEX: %{x:.2f}<extra></extra>'
    ))

    # --- AGGIUNTA LINEE (ANNOTAZIONI) ---
    
    # 1. SPOT PRICE (Ciano)
    fig.add_hline(y=spot, line_dash="dash", line_color="#00ffff", line_width=1,
                  annotation_text=f"SPOT: {spot:.2f}", annotation_position="top left", annotation_font_color="#00ffff")
    
    # 2. CALL WALL (Arancione/Oro)
    if df['strike'].min() <= cw <= df['strike'].max(): # Disegna solo se visibile
        fig.add_hline(y=cw, line_dash="dash", line_color="#ffaa00", line_width=1,
                      annotation_text=f"CALL WALL: {cw}", annotation_position="top right", annotation_font_color="#ffaa00")

    # 3. PUT WALL (Marrone/Rosso scuro)
    if df['strike'].min() <= pw <= df['strike'].max():
        fig.add_hline(y=pw, line_dash="dash", line_color="#ff4444", line_width=1,
                      annotation_text=f"PUT WALL: {pw}", annotation_position="bottom right", annotation_font_color="#ff4444")

    # 4. ZERO GAMMA / VOL TRIGGER (Giallo tratteggiato)
    if df['strike'].min() <= zg <= df['strike'].max():
        fig.add_hline(y=zg, line_dash="dashdot", line_color="#ffff00", line_width=2,
                      annotation_text=f"VOL TRIGGER (0 Gamma): {zg}", annotation_position="bottom left", annotation_font_color="#ffff00")

    # --- LAYOUT FINALE IDENTICO A GEXBOT ---
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#050505",
        paper_bgcolor="#050505",
        height=850,
        title=dict(text=f"${ticker_input} NET GEX PROFILE ({exp_date})", font=dict(color="white", size=20, family="Courier New")),
        xaxis=dict(
            range=x_range, # QUI APPLICA LO ZOOM DINAMICO
            title="NET GEX EXPOSURE",
            showgrid=True, gridcolor='rgba(255,255,255,0.1)',
            zeroline=True, zerolinecolor='white', zerolinewidth=1
        ),
        yaxis=dict(
            title="STRIKE",
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            dtick=5 if zoom < 5 else 10 # Griglia più fitta se zoom alto
        ),
        bargap=0.1, # Barre "cicciotte"
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DASHBOARD METRICHE ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPOT PRICE", f"{spot:.2f}")
    c2.metric("CALL WALL", f"{cw}")
    c3.metric("PUT WALL", f"{pw}")
    c4.metric("VOL TRIGGER", f"{zg}")

else:
    st.warning("Caricamento dati in corso...")
