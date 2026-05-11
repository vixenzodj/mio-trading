import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time as dt_time
import time  # <-- Manteniamo l'import per il delay anti-ban
import requests
import re
from bs4 import BeautifulSoup
import histdatacom
from histdatacom.options import Options
import os, zipfile, shutil, glob

LOCAL_DB_DIR = 'local_database'
os.makedirs(LOCAL_DB_DIR, exist_ok=True)

# --- 0DTE PRECISION & DYNAMIC RISK-FREE RATE ---
def get_precise_dte(exp_str):
    try:
        # Standardizzazione UTC (NY Close è circa alle 20:00/21:00 UTC)
        now_utc = datetime.utcnow()
        # Impostiamo la fine giornata alle 20:15 UTC per allinearci a Wall Street
        expiry_date = datetime.strptime(exp_str, '%Y-%m-%d').replace(hour=20, minute=15)
        diff = (expiry_date - now_utc).total_seconds() / (365.25 * 24 * 3600)
        # Floor a 0.00005 (circa 26 minuti). Salva la precisione 0DTE estrema ma impedisce la Singolarità del Gamma
        return max(diff, 0.00005) 
    except:
        return 0.00005

@st.cache_data(ttl=3600)
def get_dynamic_risk_free_rate():
    try:
        irx_info = yf.Ticker("^IRX").fast_info
        if irx_info and hasattr(irx_info, 'last_price'):
            return irx_info.last_price / 100.0
    except:
        pass
    return 0.053

DYNAMIC_R = get_dynamic_risk_free_rate()
# -----------------------------------------------

# Mappatura Ticker -> Nome per una leggibilità professionale
TICKER_NAMES = {
    # Metalli & Energia
    "GC=F": "Oro (Future)", "SI=F": "Argento (Future)", "PL=F": "Platino", "PA=F": "Palladio", "GLD": "Oro (ETF)", "SLV": "Argento (ETF)",
    "SILJ": "Silver Miners Jr", "GDX": "Gold Miners ETF", "GDXJ": "Gold Miners Jr", "CL=F": "Petrolio WTI", "BZ=F": "Petrolio Brent",
    "NG=F": "Gas Naturale", "HO=F": "Heating Oil", "RB=F": "Benzina RBOB", "XLE": "Energia USA (ETF)", "XOP": "Oil & Gas Exploration",
    "UNG": "Gas Naturale (ETF)", "DBO": "Petrolio (ETF)", "HG=F": "Rame (Future)", "CPER": "Rame (ETF)", "DBB": "Base Metals (ETF)",
    "XME": "Metals & Mining", "JJU": "Alluminio (ETF)", "JJN": "Nickel (ETF)", "PICK": "Global Mining", "LIT": "Litio (ETF)", "REMX": "Terre Rare",
    "ITB": "Homebuilders (ETF)",
    # Agricoltura
    "ZW=F": "Grano", "ZC=F": "Mais", "ZS=F": "Soia", "KC=F": "Caffè", "CT=F": "Cotone", "CC=F": "Cacao", "SB=F": "Zucchero", "DBA": "Agricoltura (ETF)", "MOO": "Agribusiness",
    # Indici & Equity
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^RUT": "Russell 2000", "^DJI": "Dow Jones", "^VIX": "Indice Volatilità", "DIA": "Dow Jones (ETF)",
    "QQQ": "Nasdaq 100 (ETF)", "IWM": "Russell 2000 (ETF)", "VTI": "US Total Market", "^GDAXI": "DAX 40", "FTSEMIB.MI": "FTSE MIB",
    "^FCHI": "CAC 40", "^IBEX": "IBEX 35", "^FTSE": "FTSE 100", "^STOXX50E": "Euro Stoxx 50", "EWG": "Germania (ETF)", "EWI": "Italia (ETF)", "EWQ": "Francia (ETF)",
    # Emergenti
    "EEM": "Emerging Markets", "VWO": "Vanguard Emerging", "MCHI": "Cina (ETF)", "EPI": "India (ETF)", "INDA": "India iShares", "EWZ": "Brasile (ETF)",
    "EWY": "Corea del Sud", "^N225": "Nikkei 225", "^HSI": "Hang Seng",
    # Forex
    "UUP": "Dollaro Index (ETF)", "EURUSD=X": "Euro/Dollaro", "JPYUSD=X": "Yen/Dollaro", "GBPUSD=X": "Sterlina/Dollaro", "AUDUSD=X": "Aussie/Dollaro",
    "CADUSD=X": "Canadese/Dollaro", "CHFUSD=X": "Franco Svizzero/Dollaro", "NZDUSD=X": "Kiwi/Dollaro", "USDCNY=X": "Dollaro/Yuan", 
    "USDBRL=X": "Dollaro/Real", "USDMXN=X": "Dollaro/Peso Mex", "USDTRY=X": "Dollaro/Lira Turca", "USDZAR=X": "Dollaro/Rand",
    # Bond & Tassi
    "TLT": "Treasury 20Y+", "IEF": "Treasury 7-10Y", "SHY": "Treasury 1-3Y", "BND": "Total Bond Market", "AGG": "Aggregate Bond",
    "LQD": "Corp Bond Inv Grade", "HYG": "High Yield Bond", "^TNX": "Rendimento 10Y USA", "^TYX": "Rendimento 30Y USA", "^FVX": "Rendimento 5Y USA",
    "BTP=F": "BTP Future", "FGBL=F": "Bund Future",
    "VNQ": "Real Estate Globale (REITs)",
    "REMX": "Terre Rare & Materiali Strategici (AI/Tech Proxy)",
    "MCHI": "Azionario Cina",
    "BTC-USD": "Bitcoin (Liquidità Speculativa)",
    "USDJPY=X": "Dollaro/Yen (Carry Trade Gauge)",
    "^VIX3M": "VIX a 3 Mesi (Coperture Istituzionali)",
    "XLU": "Utilities (Difensivo)",
    "XLK": "Tecnologia USA",
    "RSP": "S&P 500 Equal Weight (Salute Mercato)",
    "TIP": "Bond Inflazione (TIP)",
    "BND": "Total Bond Market (Stress Debito)",
    "VGK": "Europa ETF"
}

MACRO_PANELS = {
    "🟡 METALLI PREZIOSI": ["GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV", "SILJ", "GDX", "GDXJ"],
    "🛢️ ENERGIA": ["CL=F", "BZ=F", "NG=F", "HO=F", "RB=F", "XLE", "XOP", "UNG", "DBO"],
    "🏗️ METALLI INDUSTRIALI": ["HG=F", "CPER", "DBB", "XME", "JJU", "JJN", "PICK", "LIT", "REMX"],
    "🌾 AGRICOLTURA": ["ZW=F", "ZC=F", "ZS=F", "KC=F", "CT=F", "CC=F", "SB=F", "DBA", "MOO"],
    "🇺🇸 INDICI USA": ["^GSPC", "^IXIC", "^RUT", "^DJI", "^VIX", "DIA", "QQQ", "IWM", "VTI"],
    "🇪🇺 INDICI EUROPA": ["^GDAXI", "FTSEMIB.MI", "^FCHI", "^IBEX", "^FTSE", "^STOXX50E", "EWG", "EWI", "EWQ"],
    "🌍 EMERGENTI & ASIA": ["EEM", "VWO", "MCHI", "EPI", "INDA", "EWZ", "EWY", "^N225", "^HSI"],
    "💵 VALUTE (FX)": ["UUP", "EURUSD=X", "JPYUSD=X", "GBPUSD=X", "AUDUSD=X", "CADUSD=X", "CHFUSD=X", "NZDUSD=X", "USDCNY=X", "USDBRL=X", "USDMXN=X", "USDTRY=X", "USDZAR=X"],
    "📉 TASSI & BOND": ["TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "^TNX", "^TYX", "^FVX", "BTP=F", "FGBL=F"]
}

# --- STRATEGY PARAMETER GRID ---
STRATEGY_PARAM_GRID = {
    "RSI Mean Reversion": {'period': range(10, 22, 2), 'ob': range(65, 85, 5), 'os': range(20, 40, 5)},
    "MACD Crossover": {'fast': range(8, 16, 2), 'slow': range(20, 32, 2), 'signal': range(7, 12, 1)},
    "Bollinger Breakout": {'period': range(15, 30, 5), 'std_dev': [1.5, 2.0, 2.5]},
    "Golden/Death Cross": {'fast': range(40, 60, 10), 'slow': range(150, 250, 50)},
    "Stochastic Oscillator": {'k_period': range(10, 20, 2), 'ob': range(75, 95, 5), 'os': range(5, 25, 5)},
    "CCI Momentum": {'period': range(10, 30, 5)},
    "Williams %R Reversal": {'period': range(10, 30, 5)},
    "HMA Trend": {'period': range(10, 40, 5)},
    "TEMA Crossover": {'period': range(10, 40, 5)},
    "KAMA Trend": {'period': range(10, 40, 5)},
    "Aroon Oscillator": {'period': range(15, 35, 5)},
    "SuperTrend Reversal": {'period': range(7, 15, 2)},
    "Parabolic SAR": {},
    "TSI Crossover": {},
    "UO Overbought/Oversold": {},
    "Keltner Channel Breakout": {},
    "Donchian Channel Breakout": {},
    "Chaikin Volatility": {},
    "CMF Trend": {},
    "VWAP Crossover": {},
    "AD Line Trend": {},
    "Vortex Crossover": {},
    "Choppiness Index Breakout": {},
    "KST Crossover": {},
    "Coppock Curve": {},
    "Ichimoku Cloud Breakout": {},
    "Awesome Oscillator": {},
    "PPO Crossover": {},
    "Mass Index Reversal": {},
    "Ulcer Index Safety": {},
    "WMA Trend": {'period': range(10, 40, 5)},
    "TRIMA Crossover": {'period': range(10, 40, 5)},
    "CMO Reversal": {},
    "Momentum Breakout": {'period': range(5, 20, 5)},
    "BOP Trend": {},
    "TRIX Crossover": {},
    "StochRSI Reversal": {},
    "TSF Trend": {}
}

# --- CONFIGURAZIONE UI ---
st.set_page_config(layout="wide", page_title="SENTINEL GEX V63 - FULL PRO", initial_sidebar_state="expanded")

def calc_fund_metrics_v3(ticker_symbol, t_data):
    try:
        # Download dati 5 anni per Fondo e Benchmark (SPY)
        df_f = yf.download(ticker_symbol, period="5y", interval="1d")['Close']
        df_b = yf.download("SPY", period="5y", interval="1d")['Close']
        data = pd.concat([df_f, df_b], axis=1).dropna()
        data.columns = ['Fund', 'Bench']
        rets = data.pct_change().dropna()

        # 1. CAGR 5Y
        cagr = (data['Fund'].iloc[-1] / data['Fund'].iloc[0]) ** (1/5) - 1
        # 2. Volatilità Annua
        vol = rets['Fund'].std() * np.sqrt(252)
        # 3. Sharpe Ratio
        sharpe = (cagr - 0.02) / vol if vol != 0 else 0
        # 4. Max Drawdown
        dd = (data['Fund'] / data['Fund'].cummax() - 1).min()
        # 5. Beta & 6. Alpha
        cov = np.cov(rets['Fund'], rets['Bench'])[0][1]
        beta = cov / np.var(rets['Bench'])
        alpha = cagr - (0.02 + beta * (rets['Bench'].mean()*252 - 0.02))
        # 7. Tracking Error
        te = (rets['Fund'] - rets['Bench']).std() * np.sqrt(252)

        return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Max_DD": dd, "Alpha": alpha, "Beta": beta, "TE": te}
    except Exception as e:
        return None

def compute_fund_score(m):
    s = 0
    # 1. SHARPE RATIO (Max 20) - Più tollerante
    if m['Sharpe'] > 1.0: s += 20
    elif m['Sharpe'] > 0.4: s += 12  # SCHD prenderebbe 12 invece di 0
    elif m['Sharpe'] > 0.2: s += 5
    
    # 2. MAX DRAWDOWN (Max 20) - Fondamentale per il Retail
    if m['Max_DD'] > -0.12: s += 20
    elif m['Max_DD'] > -0.20: s += 15 # SCHD qui prende 15
    elif m['Max_DD'] > -0.35: s += 8
    
    # 3. ALPHA (Max 15) - Non punire troppo se vicino a zero
    if m['Alpha'] > 0.01: s += 15
    elif m['Alpha'] > -0.02: s += 10 # SCHD qui prende 10 invece di 0
    
    # 4. CAGR 5Y (Max 15)
    if m['CAGR'] > 0.10: s += 15
    elif m['CAGR'] > 0.05: s += 10 # SCHD qui prende 10
    
    # 5. VOLATILITÀ (Max 10)
    if m['Vol'] < 0.15: s += 10    # SCHD qui prende 10
    elif m['Vol'] < 0.25: s += 5
    
    # 6. BETA (Max 10) - Premiare la bassa correlazione (Beta basso = meno rischio)
    if 0.4 <= m['Beta'] <= 1.1: s += 10 # SCHD qui prende 10
    elif m['Beta'] < 1.4: s += 5
    
    # 7. TRACKING ERROR (Max 10) - Ridurre importanza per ETF di strategia
    if m['TE'] < 0.05: s += 10
    elif m['TE'] < 0.12: s += 7    # SCHD qui prende 7 invece di 0
    
    return int(s)

def get_metric_color(val, category):
    # Logica Semaforica Professionale
    if category == 'sharpe':
        return "#2ecc71" if val > 1 else "#f1c40f" if val > 0.5 else "#e74c3c"
    if category == 'drawdown':
        return "#2ecc71" if val > -0.15 else "#f1c40f" if val > -0.30 else "#e74c3c"
    if category in ['alpha', 'cagr']:
        return "#2ecc71" if val > 0.05 else "#f1c40f" if val > 0 else "#e74c3c"
    if category in ['vol', 'te']:
        return "#2ecc71" if val < 0.15 else "#f1c40f" if val < 0.25 else "#e74c3c"
    if category == 'beta':
        return "#2ecc71" if 0.5 < val < 1.3 else "#f1c40f" if val < 1.6 else "#e74c3c"
    return "#3498db"

def draw_metric_badge(label, value_str, color, status_text):
    st.markdown(f"""
        <div style="text-align: center; border: 2px solid {color}; border-radius: 20px; padding: 15px; background-color: rgba(0,0,0,0.05); margin-bottom: 10px;">
            <p style="color: gray; font-size: 0.8rem; margin: 0;">{label}</p>
            <h2 style="color: {color}; margin: 5px 0;">{value_str}</h2>
            <div style="background-color: {color}; color: white; border-radius: 10px; font-size: 0.7rem; padding: 2px 8px; display: inline-block;">
                {status_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

def calculate_dcf_value(ticker_obj):
    """Calcola il Valore Intrinseco DCF usando l'oggetto ticker passato."""
    try:
        # Recupero info e cashflow dall'oggetto già esistente
        info = ticker_obj.info
        cash_flow = ticker_obj.cashflow
        
        if 'Free Cash Flow' in cash_flow.index:
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
        elif 'Total Cash From Operating Activities' in cash_flow.index and 'Capital Expenditures' in cash_flow.index:
            fcf = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] + cash_flow.loc['Capital Expenditures'].iloc[0]
        else:
            return None

        # Parametri
        growth_rate = 0.05
        terminal_growth = 0.02
        wacc = 0.09
        shares_outstanding = info.get('sharesOutstanding')
        net_debt = info.get('totalDebt', 0) - info.get('totalCash', 0)

        if not shares_outstanding or fcf <= 0:
            return None

        pv_fcf = sum([(fcf * ((1 + growth_rate) ** i)) / ((1 + wacc) ** i) for i in range(1, 6)])
        tv = ((fcf * (1 + growth_rate)**5) * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_tv = tv / ((1 + wacc) ** 5)
        
        dcf_price = (pv_fcf + pv_tv - net_debt) / shares_outstanding
        return dcf_price if dcf_price > 0 else None
    except:
        return None

def display_correlation_matrix(tickers):
    """Genera una Heatmap di correlazione per i titoli dello scanner."""
    if len(tickers) < 2: return
    try:
        st.markdown("### 📊 Analisi di Correlazione e Rischio")
        st.write("Verifica se i titoli selezionati si muovono insieme. Correlazione > 0.70 indica un alto rischio di concentrazione.")
        data = yf.download(tickers, period="1y")['Close']
        if isinstance(data, pd.Series) or data.empty: return
        corr_matrix = data.corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

def safe_get_adj_close(tickers, period="5y"):
    """Scarica i dati risolvendo il problema di Adj Close per Indici e Futures."""
    try:
        data = yf.download(tickers, period=period, progress=False)
        if data.empty: return pd.DataFrame()
        
        # Gestione singolo ticker
        if len(tickers) == 1:
            if 'Adj Close' in data.columns and not data['Adj Close'].isna().all().all():
                df = data[['Adj Close']]
            elif 'Close' in data.columns:
                df = data[['Close']]
            else:
                return pd.DataFrame()
            df.columns = tickers # Rinomina per coerenza
        else:
            # Gestione MultiIndex: Indici e Futures NON hanno Adj Close.
            if 'Close' in data.columns.levels[0]:
                df_close = data['Close'].copy()
            else:
                return pd.DataFrame()
            
            if 'Adj Close' in data.columns.levels[0]:
                df_adj = data['Adj Close'].copy()
                # FUSIONE CHIRURGICA: Usa Adj Close, se vuoto (come per ^TNX o BTP=F) usa il Close.
                df = df_adj.combine_first(df_close)
            else:
                df = df_close
        
        # LOGICA SALVA-BOND: Riempie i buchi (max 3 giorni) prima di cancellare i NaN
        # Questo permette di allineare bond e azioni anche se hanno festività diverse
        df = df.ffill(limit=3) 
        return df
    except Exception as e:
        st.error(f"Errore nel download: {e}")
        return pd.DataFrame()

def display_macro_correlation_page():
    st.title("🕸️ Global Multi-Asset Aggregator (Full Engine)")
    st.markdown("Analisi istituzionale dei regimi e delle correlazioni inter-market.")

    # Sidebar
    st.sidebar.header("⚙️ Macro Setup")
    selected_panels = st.sidebar.multiselect("Seleziona Layer:", list(MACRO_PANELS.keys()), default=["📉 TASSI & BOND", "🇺🇸 INDICI USA"])
    custom_raw = st.sidebar.text_input("Aggiungi Ticker (es. BTC-USD):", "")
    
    final_tickers = []
    for p in selected_panels: final_tickers.extend(MACRO_PANELS[p])
    if custom_raw: final_tickers.extend([x.strip().upper() for x in custom_raw.split(",")])
    final_tickers = list(dict.fromkeys(final_tickers))

    if not final_tickers:
        st.warning("Seleziona almeno un paniere.")
        return

    with st.spinner("Sincronizzazione database..."):
        df_prices = safe_get_adj_close(final_tickers, period="5y")
        
        if df_prices.empty:
            st.error("Dati non disponibili.")
            return

        # LOGICA ROBUSTA: Calcoliamo i rendimenti SENZA dropna globale
        returns = df_prices.pct_change()
        
        # Rinominazione colonne con Nomi Estesi
        display_names = {t: f"{TICKER_NAMES.get(t, t)} ({t})" for t in df_prices.columns}
        returns_renamed = returns.rename(columns=display_names)

        # Calcolo correlazione con metodo 'pairwise' (fondamentale per i Bond)
        corr_matrix = returns_renamed.corr(method='pearson', min_periods=30)

        # Rimuove righe/colonne completamente vuote (se un ticker ha fallito del tutto)
        corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # --- FIX KEYERROR: PROTEZIONE MATRICE VUOTA ---
        if corr_matrix.empty or len(corr_matrix.columns) < 2:
            st.warning("⚠️ Dati storici insufficienti per generare la matrice su questi specifici asset (possibile mancanza di storico condiviso). Prova ad aggiungere altri strumenti o usa asset più comuni.")
            return
        # ----------------------------------------------

        st.subheader(f"📊 Matrice di Correlazione Universale ({len(corr_matrix.columns)} Asset attivi)")
        
        fig = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1]
        )
        fig.update_layout(height=1000) # Matrice più grande per gestire i tanti ticker
        st.plotly_chart(fig, use_container_width=True)

        # Rolling Correlation
        st.markdown("---")
        st.subheader("🔄 Analisi Dinamica (Rolling Correlation)")
        col1, col2 = st.columns(2)
        with col1:
            a1 = st.selectbox("Asset A", options=corr_matrix.columns, index=0)
        with col2:
            a2 = st.selectbox("Asset B", options=corr_matrix.columns, index=min(1, len(corr_matrix.columns)-1))

        window = st.slider("Finestra Mobile (Giorni)", 20, 252, 60)
        # Calcolo rolling specifico tra i due asset selezionati
        rolling_corr = returns_renamed[a1].rolling(window).corr(returns_renamed[a2])
        
        fig_line = px.line(rolling_corr, title=f"Correlazione Rolling {window}gg: {a1} vs {a2}")
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_line.update_yaxes(range=[-1.1, 1.1])
        st.plotly_chart(fig_line, use_container_width=True)

def display_macro_war_room():
    st.title("🌐 QUANTUM MACRO COMMAND CENTER V4.0")
    st.info("Sistema Istituzionale Unificato: Economia Reale, Mappa Geopolitica, Flussi Ombra e Asset Allocation.")

    # --- 1. MOTORE DATI (Sentinelle Espanse) ---
    sentinels = [
        "^GSPC", "^IXIC", "^GDAXI", "MCHI", "VGK", "EEM", "IWM",
        "TLT", "IEF", "^TNX", "^FVX", "HYG", "BND", "^IRX",
        "GC=F", "SI=F", "PL=F", "HG=F", "CL=F", "NG=F", "XOP",
        "TIP", "DBA", "XLE", "UUP", "GLD", "SLV", "GDX", "ITB",
        "REMX", "VNQ", "BTC-USD", "USDJPY=X", 
        "^VIX", "^VIX3M", "RSP", "XLK", "XLU", "XLY", "XLP",
        # EUROPA ESPANSA
        "EWG", "EWQ", "EWI", "EWP", "EWN", "EWD", "EWL",
        # DEVELOPED EX-US & EX-EU
        "EWU", "EWC", "EWA", "EWS", "EWH",
        # ASIA ESPANSA
        "EWJ", "INDA", "EWY", "EWT", "EIDO", "VNM", "EPHE",
        # EMERGING ESPANSI
        "EWZ", "EWW", "EZA", "KSA", "TUR", "ECH", "GREK", "ERUS"
    ]

    with st.spinner("Sincronizzazione Rete Quantistica Globale..."):
        df = safe_get_adj_close(sentinels, period="1y")
        if df.empty:
            st.error("Errore nel recupero dati. Verifica la connessione ai ticker.")
            return

        def get_stat(ticker, st_type="curr"):
            if ticker not in df.columns: return np.nan
            s = df[ticker].dropna()
            if s.empty: return np.nan
            if st_type == "curr": return s.iloc[-1]
            if st_type == "prev": return s.iloc[-21] if len(s) > 20 else s.iloc[0]
            if st_type == "ma50": return s.rolling(50).mean().iloc[-1] if len(s) >= 50 else s.mean()

        # --- HELPER: SPARKLINE GENERATOR ---
        def draw_sparkline(series, color="#2ecc71"):
            fig = go.Figure(go.Scatter(y=series, mode='lines', line=dict(color=color, width=2.5)))
            fig.update_layout(
                height=50, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig

        def draw_ranked_bars(ticker_map, title, period=20):
            data = []
            for name, ticker in ticker_map.items():
                if ticker in df.columns:
                    series = df[ticker].dropna()
                    if len(series) >= period:
                        perf = (series.iloc[-1] / series.iloc[-period]) - 1
                        data.append({"Asset": name, "Performance": perf})
            
            if not data: return None
            df_plot = pd.DataFrame(data).sort_values("Performance", ascending=True)
            
            fig = px.bar(df_plot, x="Performance", y="Asset", orientation='h',
                         title=title, color="Performance",
                         color_continuous_scale='RdYlGn', text_auto='.2%')
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                              showlegend=False, coloraxis_showscale=False,
                              xaxis_title="Perf. Relativa", yaxis_title="")
            return fig

        def calculate_z_score(series, window=60):
            if len(series) < window: return 0.0
            mean = series.rolling(window=window).mean().iloc[-1]
            std = series.rolling(window=window).std().iloc[-1]
            return (series.iloc[-1] - mean) / std if std != 0 else 0.0

        # --- 2. ELABORAZIONE QUANTISTICA DELLE METRICHE ---
        
        # Livello Base (Prezzi Correnti e Medie)
        us_curr, us_ma = get_stat("^GSPC"), get_stat("^GSPC", "ma50")
        re_curr, re_ma = get_stat("VNQ"), get_stat("VNQ", "ma50")
        china_curr, china_ma = get_stat("MCHI"), get_stat("MCHI", "ma50")
        eur_curr, eur_ma = get_stat("^GDAXI"), get_stat("^GDAXI", "ma50")
        eem_curr, eem_ma = get_stat("EEM"), get_stat("EEM", "ma50")
        
        # Metalli e Barometro
        hg_curr, gc_curr = get_stat("HG=F"), get_stat("GC=F")
        hg_ma, gc_ma = get_stat("HG=F", "ma50"), get_stat("GC=F", "ma50")
        cu_au_ratio = hg_curr / gc_curr if gc_curr > 0 else 0
        cu_au_ma = hg_ma / gc_ma if gc_ma > 0 else 0
        ind_growth = cu_au_ratio > cu_au_ma

        # Flussi e Salute
        breadth_ratio = get_stat("RSP") / get_stat("^GSPC")
        breadth_ma = get_stat("RSP", "ma50") / get_stat("^GSPC", "ma50")
        is_healthy_breadth = breadth_ratio > breadth_ma

        bnd_curr, bnd_ma = get_stat("BND"), get_stat("BND", "ma50")
        bond_stress = bnd_curr < bnd_ma  # Prezzi bond giù = tassi su = stress
        
        inf_expect_ratio = get_stat("TIP") / get_stat("IEF")
        inf_expect_ma = get_stat("TIP", "ma50") / get_stat("IEF", "ma50")
        inflation_fear = inf_expect_ratio > inf_expect_ma

        vix_ratio = get_stat("^VIX") / get_stat("^VIX3M")
        usdjpy_curr, usdjpy_prev = get_stat("USDJPY=X"), get_stat("USDJPY=X", "prev")
        carry_trade_risk = usdjpy_curr < usdjpy_prev
        btc_curr, btc_ma = get_stat("BTC-USD"), get_stat("BTC-USD", "ma50")
        btc_liq = btc_curr > btc_ma

        # --- 3. LIVELLO 1: MAPPA MACRO GLOBALE (I 5 Pilastri) ---
        st.header("🗺️ Mappa Macro Globale (Fisica)")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            delta_us = ((us_curr / us_ma) - 1) * 100
            st.metric("Azionario USA", f"{us_curr:,.0f}", delta=f"{delta_us:+.2f}% vs MA50", delta_color="normal" if us_curr > us_ma else "inverse")
        with c2:
            delta_re = ((re_curr / re_ma) - 1) * 100
            st.metric("Real Estate", f"${re_curr:.2f}", delta=f"{delta_re:+.2f}% vs MA50", delta_color="normal" if re_curr > re_ma else "inverse")
        with c3:
            delta_ind = ((cu_au_ratio / cu_au_ma) - 1) * 100 if cu_au_ma > 0 else 0
            st.metric("Barom. Ind. (Cu/Au)", f"{cu_au_ratio:.4f}", delta=f"{delta_ind:+.2f}% Crescita", delta_color="normal" if ind_growth else "inverse")
        with c4:
            delta_ch = ((china_curr / china_ma) - 1) * 100
            st.metric("Cina (MCHI)", f"${china_curr:.2f}", delta=f"{delta_ch:+.2f}% vs MA50", delta_color="normal" if china_curr > china_ma else "inverse")
        with c5:
            delta_eu = ((eur_curr / eur_ma) - 1) * 100
            st.metric("Europa (DAX)", f"{eur_curr:,.0f}", delta=f"{delta_eu:+.2f}% vs MA50", delta_color="normal" if eur_curr > eur_ma else "inverse")

        # --- LIVELLO 2.6: GLOBAL RELATIVE STRENGTH (20D) ---
        st.markdown("#### 🌍 Classifica Forza Relativa Globale (20gg)")
        equity_map = {
            "USA (S&P500)": "^GSPC", "USA Tech (NDX)": "^IXIC", "USA Small Caps (IWM)": "IWM",
            "Europa (VGK)": "VGK", "Germania (DAX)": "^GDAXI", "Cina (MCHI)": "MCHI", "Emergenti (EEM)": "EEM"
        }
        data_equity = []
        for name, tk in equity_map.items():
            if tk in df.columns:
                perf = (df[tk].iloc[-1] / df[tk].iloc[-20]) - 1
                data_equity.append({"Mercato": name, "Performance": perf})
        
        df_equity = pd.DataFrame(data_equity).sort_values("Performance", ascending=False)
        fig_eq = px.bar(df_equity, x="Performance", y="Mercato", orientation='h', 
                        color="Performance", color_continuous_scale='RdYlGn', text_auto='.2%')
        fig_eq.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("#### 🚀 Settori Lead, Breadth & Risk-Appetite")
        
        # Calcolo Metriche
        m_breadth = (get_stat("RSP") / get_stat("^GSPC")) - 1
        m_tech = (get_stat("XLK") / get_stat("^GSPC")) - 1
        m_risk = (get_stat("XLY") / get_stat("XLP")) - 1
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"{'🟢' if m_breadth > 0 else '🔴'} Market Breadth", f"{m_breadth:.2%}", help="RSP vs SPY")
        with c2:
            st.metric(f"{'🟢' if m_tech > 0 else '🔴'} Tech Leadership", f"{m_tech:.2%}", help="XLK vs SPY")
        with c3:
            st.metric(f"{'🟢' if m_risk > 0 else '🔴'} Risk Appetite", f"{m_risk:.2%}", help="XLY vs XLP")

        # Istogramma Settori Lead
        lead_data = pd.DataFrame([
            {"Settore": "Market Breadth", "Valore": m_breadth},
            {"Settore": "Tech Leadership", "Valore": m_tech},
            {"Settore": "Risk Appetite", "Valore": m_risk}
        ])
        fig_lead = px.bar(lead_data, x="Valore", y="Settore", orientation='h', 
                          color="Valore", color_continuous_scale='Geyser', text_auto='.2%')
        fig_lead.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_lead, use_container_width=True)

        st.markdown("#### 🏗️ Real Estate, Builders & Alpha Sectors")
        alpha_map = {
            "Real Estate (VNQ)": "VNQ", "Homebuilders (ITB)": "ITB", 
            "Small Caps (IWM)": "IWM", "Gold Miners (GDX)": "GDX"
        }
        data_alpha = []
        for name, tk in alpha_map.items():
            if tk in df.columns:
                perf = (df[tk].iloc[-1] / df[tk].iloc[-50]) - 1 # Forza relativa 50gg
                data_alpha.append({"Settore": name, "Forza": perf})
        
        df_alpha = pd.DataFrame(data_alpha).sort_values("Forza", ascending=False)
        fig_alpha = px.bar(df_alpha, x="Forza", y="Settore", orientation='h', 
                           color="Forza", color_continuous_scale='Viridis', text_auto='.2%')
        fig_alpha.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_alpha, use_container_width=True)

        st.markdown("#### 🛢️ Energy Complex & Hard Assets")
        energy_map = {
            "Oil & Gas Exp (XOP)": "XOP", "Crude Oil (WTI)": "CL=F", 
            "Natural Gas": "NG=F", "Energy ETF (XLE)": "XLE"
        }
        data_energy = []
        for name, tk in energy_map.items():
            if tk in df.columns:
                perf = (df[tk].iloc[-1] / df[tk].iloc[-20]) - 1
                data_energy.append({"Asset": name, "Performance": perf})
        
        df_energy = pd.DataFrame(data_energy).sort_values("Performance", ascending=False)
        fig_energy = px.bar(df_energy, x="Performance", y="Asset", orientation='h', 
                            color="Performance", color_continuous_scale='Tropic', text_auto='.2%')
        fig_energy.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_energy, use_container_width=True)

        st.markdown("---")

        # --- NUOVO LIVELLO 1.5: MAPPA SETTORIALE E GEOPOLITICA ---
        st.header("🌍 Geopolitica & Asset Class (Chi guida l'economia?)")
        geo1, geo2, geo3, geo4 = st.columns(4)
        
        def status_widget(name, curr, ma, suffix=""):
            if pd.isna(curr) or pd.isna(ma) or ma == 0: return f"⚪ {name}: N/D"
            delta = ((curr / ma) - 1) * 100
            emj = "🟢" if curr > ma else "🔴"
            return f"{emj} **{name}**: {curr:.2f}{suffix} ({delta:+.2f}%)"

        with geo1:
            st.markdown("#### 🗺️ Geopolitica")
            st.markdown(status_widget("USA (S&P)", us_curr, us_ma))
            st.markdown(status_widget("Europa", eur_curr, eur_ma))
            st.markdown(status_widget("Cina", china_curr, china_ma))
            st.markdown(status_widget("Emergenti (BRICS)", eem_curr, eem_ma))
        with geo2:
            st.markdown("#### 🛢️ Energia & Hard Assets")
            st.markdown(status_widget("Petrolio (WTI)", get_stat("CL=F"), get_stat("CL=F", "ma50"), "$"))
            st.markdown(status_widget("Titoli Energetici", get_stat("XLE"), get_stat("XLE", "ma50"), "$"))
            st.markdown(status_widget("Agricoltura (DBA)", get_stat("DBA"), get_stat("DBA", "ma50"), "$"))
        with geo3:
            st.markdown("#### ⛏️ Metalli")
            st.markdown(status_widget("Oro (Rifugio)", gc_curr, gc_ma, "$"))
            st.markdown(status_widget("Rame (Industria)", hg_curr, hg_ma, "$"))
            st.markdown(status_widget("Terre Rare (Tech)", get_stat("REMX"), get_stat("REMX", "ma50"), "$"))
        with geo4:
            st.markdown("#### ⚖️ Forza Relativa")
            usa_vs_china = us_curr / china_curr if china_curr > 0 else 1
            usa_vs_china_ma = us_ma / china_ma if china_ma > 0 else 1
            st.markdown(f"**USA vs Cina Ratio:** {usa_vs_china:.2f}")
            if usa_vs_china > usa_vs_china_ma:
                st.success("🇺🇸 Capitali verso USA")
            else:
                st.error("🇨🇳 Capitali verso Oriente")

        st.markdown("#### ⚡ Gerarchia delle Asset Class & Hard Assets")
        c_rank3, c_rank4, c_rank5 = st.columns(3)
        with c_rank3:
            mkt_geopol = {"Risk-ON (Nasdaq)": "^IXIC", "Safe Haven (Gold)": "GC=F", "Dollar Index": "UUP", "Bitcoin": "BTC-USD", "Bonds": "TLT"}
            st.plotly_chart(draw_ranked_bars(mkt_geopol, "Ranking Geopolitico/Asset"), use_container_width=True)
        with c_rank4:
            mkt_energy = {"Petrolio WTI": "CL=F", "Gas Naturale": "NG=F", "Oil & Gas (XOP)": "XOP", "Energy (XLE)": "XLE"}
            st.plotly_chart(draw_ranked_bars(mkt_energy, "Energia & Hard Assets"), use_container_width=True)
        with c_rank5:
            mkt_metals = {"Oro": "GC=F", "Argento": "SI=F", "Rame": "HG=F", "Platino": "PL=F"}
            st.plotly_chart(draw_ranked_bars(mkt_metals, "Metalli Ranking"), use_container_width=True)

        st.markdown("#### ⛏️ Metals Ranking & Relative Strength")
        metal_map = {
            "Oro (GC=F)": "GC=F", 
            "Argento (SI=F)": "SI=F", 
            "Platino (PL=F)": "PL=F",
            "Rame (HG=F)": "HG=F"
        }
        data_metals = []
        for name, tk in metal_map.items():
            if tk in df.columns:
                perf = (df[tk].iloc[-1] / df[tk].iloc[-20]) - 1
                data_metals.append({"Metallo": name, "Performance": perf})
        
        df_metals = pd.DataFrame(data_metals).sort_values("Performance", ascending=False)
        fig_m = px.bar(df_metals, x="Performance", y="Metallo", orientation='h', 
                       color="Performance", color_continuous_scale='Bluered', text_auto='.2%')
        fig_m.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")

        # --- 4. LIVELLO 2: SALUTE E FLUSSI OMBRA (CON GRAFICI E SEMAFORI) ---
        st.header("🧬 Analisi Molecolare e Flussi Invisibili")
        
        # Preparazione serie storiche per i grafici
        hist_breadth = (df["RSP"] / df["^GSPC"]).dropna()
        hist_bnd = df["BND"].dropna()
        hist_inf = (df["TIP"] / df["IEF"]).dropna()
        hist_btc = df["BTC-USD"].dropna()
        hist_yen = df["USDJPY=X"].dropna()
        hist_vix = (df["^VIX"] / df["^VIX3M"]).dropna()
        
        # Calcolo Z-Score
        z_br = calculate_z_score(hist_breadth)
        z_bnd = calculate_z_score(hist_bnd)
        z_inf = calculate_z_score(hist_inf)
        z_btc = calculate_z_score(hist_btc)
        z_yen = calculate_z_score(hist_yen)
        z_vix = calculate_z_score(hist_vix)

        r1, r2, r3 = st.columns(3)
        with r1:
            br_delta = breadth_ratio - breadth_ma
            tit_br = "Market Breadth 🟢" if is_healthy_breadth else "Market Breadth 🔴"
            st.metric(tit_br, f"{breadth_ratio:.3f}", delta=f"{br_delta:+.3f} vs Media", delta_color="normal" if is_healthy_breadth else "inverse")
            st.plotly_chart(draw_sparkline(hist_breadth[-60:], "#2ecc71" if is_healthy_breadth else "#e74c3c"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_br:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Indica se il mercato sale in modo sano. **Sopra la media (Verde)**: Molte aziende salgono. **Sotto (Rosso)**: Pochi giganti tengono su l'indice.")
        with r2:
            bnd_delta = ((bnd_curr / bnd_ma) - 1) * 100
            tit_bnd = "Stress Debito 🔴 ALTO" if bond_stress else "Stress Debito 🟢 BASSO"
            st.metric(tit_bnd, f"${bnd_curr:.2f}", delta=f"{bnd_delta:+.2f}%", delta_color="normal" if not bond_stress else "inverse")
            st.plotly_chart(draw_sparkline(hist_bnd[-60:], "#e74c3c" if bond_stress else "#2ecc71"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_bnd:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Mostra la salute del debito globale. Prezzo sotto la media (**Rosso**) = tassi in aumento, forte stress sistemico.")
        with r3:
            inf_delta = inf_expect_ratio - inf_expect_ma
            tit_inf = "Inflazione Attesa 🔴 ALTA" if inflation_fear else "Inflazione Attesa 🟢 OK"
            st.metric(tit_inf, f"{inf_expect_ratio:.3f}", delta=f"{inf_delta:+.3f} Spinta", delta_color="inverse" if inflation_fear else "normal")
            st.plotly_chart(draw_sparkline(hist_inf[-60:], "#e74c3c" if inflation_fear else "#2ecc71"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_inf:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Previsioni del mercato obbligazionario. Se sale (**Rosso**), i capitali temono un ritorno dell'inflazione (Tassi più alti a lungo).")

        r4, r5, r6 = st.columns(3)
        with r4:
            btc_delta = ((btc_curr / btc_ma) - 1) * 100
            tit_btc = "Bitcoin Proxy 🟢 RISK-ON" if btc_liq else "Bitcoin Proxy 🔴 RISK-OFF"
            st.metric(tit_btc, f"${btc_curr:,.0f}", delta=f"{btc_delta:+.2f}%", delta_color="normal" if btc_liq else "inverse")
            st.plotly_chart(draw_sparkline(hist_btc[-60:], "#2ecc71" if btc_liq else "#e74c3c"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_btc:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Sensore di liquidità speculativa globale. **Verde** = Nuovo denaro entra nel sistema. **Rosso** = Denaro in fuga.")
        with r5:
            yen_delta = usdjpy_curr - usdjpy_prev
            tit_yen = "Yen Carry Trade 🔴 FUGA" if carry_trade_risk else "Yen Carry Trade 🟢 STABILE"
            st.metric(tit_yen, f"¥{usdjpy_curr:.2f}", delta=f"{yen_delta:+.2f} ¥ vs Mese", delta_color="normal" if not carry_trade_risk else "inverse")
            st.plotly_chart(draw_sparkline(hist_yen[-60:], "#e74c3c" if carry_trade_risk else "#2ecc71"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_yen:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Leva globale. Se lo Yen si rafforza e il grafico crolla (**Rosso**), i fondi vendono azioni per ripagare debiti. Crash in arrivo.")
        with r6:
            vix_delta = vix_ratio - 1.0
            vix_panic = vix_ratio > 1.0
            tit_vix = "VIX Ratio (Spot/3M) 🔴 PANICO" if vix_panic else "VIX Ratio 🟢 CALMA"
            st.metric(tit_vix, f"{vix_ratio:.2f}", delta=f"{vix_delta:+.2f} (Soglia 1)", delta_color="inverse" if vix_panic else "normal")
            st.plotly_chart(draw_sparkline(hist_vix[-60:], "#e74c3c" if vix_panic else "#2ecc71"), use_container_width=True)
            st.caption(f"Z-Score (60gg): {z_vix:.2f}σ")
            with st.expander("Cos'è e come si legge?"): 
                st.write("Se supera 1.0 (**Rosso**), le banche d'affari stanno pagando un sovrapprezzo disperato per assicurarsi contro un crollo immediato.")

        # --- NUOVO LIVELLO 2.5: RADAR CORRELAZIONI ISTITUZIONALI (LIE DETECTOR) ---
        st.markdown("---")
        st.header("🔗 Radar Correlazioni (Macchina della Verità)")
        st.write("Gli istituzionali usano queste correlazioni rolling (20gg) per scoprire se il mercato sta nascondendo un'anomalia sistemica.")
        
        # Calcoli correlazione mobile a 20 giorni (Versione Blindata Anti-Crash)
        corr_eq_bnd = df["^GSPC"].rolling(20).corr(df["TLT"]).iloc[-1] if "^GSPC" in df.columns and "TLT" in df.columns else 0.0
        corr_cu_au = df["HG=F"].rolling(20).corr(df["GC=F"]).iloc[-1] if "HG=F" in df.columns and "GC=F" in df.columns else 0.0
        corr_usd_oil = df["UUP"].rolling(20).corr(df["CL=F"]).iloc[-1] if "UUP" in df.columns and "CL=F" in df.columns else 0.0

        corr1, corr2, corr3 = st.columns(3)
        
        # 1. Rischio Sistemico (Azioni vs Bond)
        with corr1:
            is_sys_risk = corr_eq_bnd > 0.3
            st.markdown(f"**Azioni (S&P) vs Bond (TLT)**")
            st.metric("Correlazione 20gg", f"{corr_eq_bnd:.2f}", delta="🔴 CRISI LIQUIDITÀ" if is_sys_risk else "🟢 NORMALE", delta_color="inverse" if is_sys_risk else "normal")
            st.caption("Normalmente decorrelati. Se salgono/scendono insieme (rosso), le banche centrali hanno perso il controllo (Stagflazione/Liquidity Crunch).")
            
        # 2. Salute Industriale (Rame vs Oro)
        with corr2:
            is_ind_risk = corr_cu_au < -0.3
            st.markdown(f"**Rame (Industria) vs Oro (Paura)**")
            st.metric("Correlazione 20gg", f"{corr_cu_au:.2f}", delta="🔴 RECESSIONE REALE" if is_ind_risk else "🟢 CRESCITA/INFLAZIONE", delta_color="inverse" if is_ind_risk else "normal")
            st.caption("Se sono fortemente inversi (Rame crolla, Oro vola), il mercato prezza una recessione industriale profonda e imminente.")
            
        # 3. Shock Inflattivo (Dollaro vs Petrolio)
        with corr3:
            is_inf_shock = corr_usd_oil > 0.3
            st.markdown(f"**Dollaro (USD) vs Petrolio (WTI)**")
            st.metric("Correlazione 20gg", f"{corr_usd_oil:.2f}", delta="🔴 SHOCK INFLATTIVO/EMERGENTI" if is_inf_shock else "🟢 DINAMICA FX NORMALE", delta_color="inverse" if is_inf_shock else "normal")
            st.caption("Normalmente inversi. Se salgono insieme (rosso), estremo dolore per l'Europa e mercati emergenti. Costi dell'energia insostenibili.")

        # --- LIVELLO 2.8: GLOBAL MACRO, LIQUIDITÀ & REGIONAL DRILL-DOWN ---
        st.markdown("---")
        st.header("🌐 Global Macro, Liquidità & Migrazione Capitali")

        # 1. Calcoli Istituzionali (Liquidità & Propensione al Rischio)
        # Liquidità Reale Proxy: (Oro / Dollaro) / Tassi Reali
        liq_proxy = (get_stat("GLD") / get_stat("UUP")) / get_stat("TIP") if get_stat("TIP") > 0 else 1
        liq_proxy_ma = (get_stat("GLD", "ma50") / get_stat("UUP", "ma50")) / get_stat("TIP", "ma50") if get_stat("TIP", "ma50") > 0 else 1
        is_liq_expanding = liq_proxy > liq_proxy_ma

        # Risk Appetite: Beni Voluttuari (XLY) vs Beni di Necessità (XLP)
        xly_xlp = get_stat("XLY") / get_stat("XLP")
        xly_xlp_ma = get_stat("XLY", "ma50") / get_stat("XLP", "ma50")
        is_risk_on = xly_xlp > xly_xlp_ma

        c_mac1, c_mac2 = st.columns(2)
        with c_mac1:
            st.metric("Liquidità Globale Netta (GLD/UUP/TIP)", "ESPANSIONE 🟢" if is_liq_expanding else "CONTRAZIONE 🔴", 
                      delta="Stampa Denaro / Svalutazione" if is_liq_expanding else "Stretta Monetaria", delta_color="normal" if is_liq_expanding else "inverse")
            st.caption("Se in espansione, le Banche Centrali stanno immettendo liquidità segreta. Fortemente Bullish per Crypto e Azioni.")
        with c_mac2:
            st.metric("Risk-Appetite Istituzionale (XLY/XLP)", "RISCHIO 🟢" if is_risk_on else "DIFESA 🔴",
                      delta="Acquisto Voluttuari" if is_risk_on else "Fuga verso Beni Rifugio", delta_color="normal" if is_risk_on else "inverse")
            st.caption("Il 'Smart Money' sta comprando Amazon/Tesla (Rischio) o Walmart/P&G (Difesa)? Anticipa i crolli azionari.")

        # 2. Interactive Regional Drill-Down (La Mappa a Schede)
        st.markdown("#### 🗺️ Esplorazione Geografica (Drill-Down)")
        st.write("Clicca su una regione per vedere esattamente quali nazioni stanno assorbendo o perdendo capitali (Performance 20gg).")
        
        tab_eu, tab_dev, tab_as, tab_em, tab_us = st.tabs([
            "🇪🇺 Europa (EU)", "🇬🇧 Developed (ex-US)", "🌏 Asia", "🌍 Emergenti", "🇺🇸 USA"
        ])

        def plot_drilldown(ticker_dict, title):
            data = []
            for name, ticker in ticker_dict.items():
                if ticker in df.columns:
                    perf = (df[ticker].iloc[-1] / df[ticker].iloc[-20]) - 1
                    if pd.isna(perf): perf = 0.0
                    data.append({"Nazione": name, "Forza Relativa": perf})
            if not data: return None
            dff = pd.DataFrame(data).sort_values("Forza Relativa", ascending=True)
            fig = px.bar(dff, x="Forza Relativa", y="Nazione", orientation='h', title=title, color="Forza Relativa", color_continuous_scale='RdYlGn', text_auto='.2%')
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Performance 1 Mese", yaxis_title="")
            return fig

        with tab_eu:
            eu_list = {
                "Germania (EWG)": "EWG", "Francia (EWQ)": "EWQ", "Italia (EWI)": "EWI", 
                "Spagna (EWP)": "EWP", "Olanda (EWN)": "EWN", "Svezia (EWD)": "EWD", 
                "Indice Area (VGK)": "VGK"
            }
            st.plotly_chart(plot_drilldown(eu_list, "Matrice Europea Core & Periferia"), use_container_width=True)

        with tab_dev:
            dev_list = {
                "UK (EWU)": "EWU", "Canada (EWC)": "EWC", "Australia (EWA)": "EWA", 
                "Svizzera (EWL)": "EWL", "Singapore (EWS)": "EWS", "Hong Kong (EWH)": "EWH"
            }
            st.plotly_chart(plot_drilldown(dev_list, "Commonwealth & Developed Hubs"), use_container_width=True)

        with tab_as:
            as_list = {
                "Giappone (EWJ)": "EWJ", "Cina (MCHI)": "MCHI", "India (INDA)": "INDA", 
                "Corea (EWY)": "EWY", "Taiwan (EWT)": "EWT", "Indonesia (EIDO)": "EIDO", 
                "Vietnam (VNM)": "VNM", "Filippine (EPHE)": "EPHE"
            }
            st.plotly_chart(plot_drilldown(as_list, "Forza Relativa Asiatica"), use_container_width=True)

        with tab_em:
            em_list = {
                "Brasile (EWZ)": "EWZ", "Messico (EWW)": "EWW", "Sud Africa (EZA)": "EZA", 
                "Arabia Saudita (KSA)": "KSA", "Turchia (TUR)": "TUR", "Cile (ECH)": "ECH", 
                "Grecia (GREK)": "GREK", "Russia (ERUS)": "ERUS", "Indice Broad (EEM)": "EEM"
            }
            st.plotly_chart(plot_drilldown(em_list, "Mercati Emergenti & Frontiera"), use_container_width=True)
        with tab_us:
            st.plotly_chart(plot_drilldown({"S&P 500 (SPY)": "^GSPC", "Nasdaq 100 (Tech)": "^IXIC", "Small Caps (Rischio)": "RSP", "Real Estate (VNQ)": "VNQ"}, "Dinamica USA Interna"), use_container_width=True)

        # --- LIVELLO 2.9: CREDIT & SYSTEMIC FRAGILITY (L'ULTIMA DIFESA) ---
        st.markdown("---")
        st.header("🏗️ Credit & Systemic Fragility (I nervi del sistema)")
        
        # Calcoli Tecnici (Esistenti)
        hist_hy_spread = (df["HYG"] / df["IEF"]).dropna()
        hy_ratio = hist_hy_spread.iloc[-1]
        z_credit = calculate_z_score(hist_hy_spread)
        
        yield_10y = get_stat("^TNX")
        yield_3m = get_stat("^IRX")
        curve_slope = (yield_10y - yield_3m) if (yield_10y and yield_3m) else 0
        
        hist_tlt_vol = df["TLT"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        bond_vol_curr = hist_tlt_vol.iloc[-1]
        bond_vol_ma = hist_tlt_vol.rolling(60).mean().iloc[-1]

        # --- NUOVA LOGICA SEMAFORICA ---
        status_hy = "🟢" if z_credit > -0.5 else "🟡" if z_credit > -1.5 else "🔴"
        status_curve = "🟢" if curve_slope > 0.5 else "🟡" if curve_slope >= 0 else "🔴"
        status_move = "🟢" if bond_vol_curr < bond_vol_ma else "🟡" if bond_vol_curr < (bond_vol_ma * 1.15) else "🔴"

        c_frag1, c_frag2, c_frag3 = st.columns(3)
        
        with c_frag1:
            st.metric(f"{status_hy} High Yield Spread 🛡️", f"{hy_ratio:.3f}", 
                      delta=f"Z-Score: {z_credit:.2f}σ", 
                      delta_color="normal" if z_credit > -1 else "inverse")
            st.plotly_chart(draw_sparkline(hist_hy_spread[-60:], "#2ecc71" if status_hy=="🟢" else "#f1c40f" if status_hy=="🟡" else "#e74c3c"), use_container_width=True)
            st.caption("Se lo Z-Score scende sotto -1.5, le aziende High Yield sono in difficoltà: rischio fallimenti in aumento.")

        with c_frag2:
            st.metric(f"{status_curve} Yield Curve (10Y-3M) 📉", f"{curve_slope:.2f}%", 
                      delta="INVERSIONE" if curve_slope < 0 else "NORMALE", 
                      delta_color="inverse" if curve_slope < 0 else "normal")
            hist_curve = (df["^TNX"] - df["^IRX"]).dropna()
            st.plotly_chart(draw_sparkline(hist_curve[-60:], "#2ecc71" if status_curve=="🟢" else "#f1c40f" if status_curve=="🟡" else "#e74c3c"), use_container_width=True)
            st.caption("La curva invertita (negativa) ha previsto il 100% delle ultime recessioni.")

        with c_frag3:
            st.metric(f"{status_move} Bond Volatility (MOVE) ⚡", f"{bond_vol_curr:.1f}%", 
                      delta="ALERT" if bond_vol_curr > bond_vol_ma else "STABILE", 
                      delta_color="inverse" if bond_vol_curr > bond_vol_ma else "normal")
            st.plotly_chart(draw_sparkline(hist_tlt_vol[-60:].dropna(), "#2ecc71" if status_move=="🟢" else "#f1c40f" if status_move=="🟡" else "#e74c3c"), use_container_width=True)
            st.caption("VIX dei Bond. Se esplode, i fondi pensione vendono azioni per coprire i margini sui bond.")

        # --- 5. LIVELLO 3: IL CERVELLO CIO (ASSET ALLOCATION QUANTISTICA) ---
        st.markdown("---")
        st.header("🎯 Strategia Operativa Suggerita (CIO View Integrata)")
        
        # --- CALCOLO SYSTEMIC HEALTH SCORE (0-100) AGGIORNATO ---
        health_score = 50 # Base Neutral
        
        # 1. Macro, Liquidità & Flussi Istituzionali (NUOVO)
        if is_liq_expanding: health_score += 10
        else: health_score -= 10
        if is_risk_on: health_score += 10
        else: health_score -= 10
        
        # 2. Analisi Molecolare & Flussi Ombra
        if is_healthy_breadth: health_score += 5
        else: health_score -= 10
        if not bond_stress: health_score += 5
        else: health_score -= 10
        if not inflation_fear: health_score += 5
        if btc_liq: health_score += 5
        if carry_trade_risk: health_score -= 15
        else: health_score += 5
        if vix_ratio > 1: health_score -= 15
        else: health_score += 5
            
        # 3. Analisi Mappa Fisica & Correlazioni
        if ind_growth: health_score += 10
        if is_sys_risk: health_score -= 10
        if is_ind_risk: health_score -= 10
        
        # 4. Analisi Nervi del Sistema (Fragility)
        if z_credit < -1.5: health_score -= 10
        elif z_credit > 0: health_score += 5
        if curve_slope < 0: health_score -= 10
        if bond_vol_curr > bond_vol_ma: health_score -= 10

        # 5. Fattori Aggiuntivi (CIO View Finale)
        dev_tickers = ["EWC", "EWA", "EWL", "EWU"]
        dev_perfs = [(get_stat(tk) / get_stat(tk, "ma50")) - 1 for tk in dev_tickers if get_stat(tk, "ma50") > 0]
        if dev_perfs and sum(dev_perfs)/len(dev_perfs) > 0:
            health_score += 5  # Developed Strength
            
        re_perf_cio = (get_stat("VNQ") / get_stat("VNQ", "ma50")) - 1 if get_stat("VNQ", "ma50") > 0 else 0
        if re_perf_cio < 0:
            health_score -= 10  # Real Estate Stress
            
        # Normalizzazione Score (0-100)
        health_score = max(0, min(100, health_score))
        
        # --- ASSET ALLOCATION DINAMICA (RISK MANAGER) ---
        if health_score < 35:
            regime = "🔴 RISK-OFF (Preservation)"
            weights = {'Cash / USD': 50, 'Bonds (TLT/IEF)': 30, 'Gold (Safe Haven)': 15, 'Defensive Equity': 5}
            color_seq = px.colors.sequential.Reds_r
        elif health_score < 65:
            regime = "🟡 NEUTRAL (Transition)"
            weights = {'Cash / USD': 20, 'Bonds (TLT/IEF)': 30, 'Gold / Commodities': 10, 'Broad Equity (SPY)': 40}
            color_seq = px.colors.sequential.YlOrBr
        else:
            regime = "🟢 RISK-ON (Expansion)"
            weights = {'Cash / USD': 5, 'Bonds (TLT/IEF)': 15, 'Broad Equity (SPY)': 45, 'Tech / High Beta': 25, 'Crypto / Speculative': 10}
            color_seq = px.colors.sequential.Greens_r

        # --- RENDERIZZAZIONE DASHBOARD CIO ---
        col_charts1, col_charts2 = st.columns(2)
        
        with col_charts1:
            # Gauge / Istogramma Salute Sistemica
            fig_health = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                title = {'text': "Systemic Health Score", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white", 'thickness': 0.2},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 35], 'color': "rgba(231, 76, 60, 0.8)"},
                        {'range': [35, 65], 'color': "rgba(241, 196, 15, 0.8)"},
                        {'range': [65, 100], 'color': "rgba(46, 204, 113, 0.8)"}],
                }
            ))
            fig_health.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_health, use_container_width=True)

        with col_charts2:
            # Grafico a Torta (Risk Manager Portfolio)
            labels = list(weights.keys())
            values = list(weights.values())
            fig_pie = px.pie(names=labels, values=values, hole=0.4, title=f"Allocazione Portafoglio Suggerita", color_discrete_sequence=color_seq)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- SUGGERIMENTI OPERATIVI E TESTUALI ---
        st.markdown(f"### Valutazione di Mercato Attuale: **{regime}**")
        
        strat_col, list_col = st.columns([1.5, 1])
        with strat_col:
            if health_score < 35:
                st.error("### 🚨 DE-LEVERAGING / CRASH RISK")
                st.write("**Diagnosi Integrata:** Le metriche indicano stress grave. I rendimenti sono invertiti, lo Yen prezza deleverage, o c'è panico nei bond. Liquidità in fuga.")
                buys = ["Cash (Dollari - UUP)", "Bond a breve scadenza (SHY)", "Indici Volatilità (VIX)"]
                sells = ["Tutto l'azionario (USA, EU, Cina)", "Bitcoin e Crypto", "Real Estate (VNQ)", "Titoli Growth ad alto Beta"]
            
            elif health_score < 65:
                if inflation_fear and not ind_growth:
                    st.warning("### 🏗️ TRAPPOLA STAGFLATTIVA (Stagflation)")
                    st.write("**Diagnosi Integrata:** L'economia reale rallenta ma l'inflazione attesa sale. Scenario pessimo per l'azionario tradizionale.")
                    buys = ["Metalli Preziosi (Oro, Argento)", "Materia Prima Agricola (DBA)", "Bond indicizzati all'inflazione (TIP)"]
                    sells = ["Tecnologia USA (XLK)", "Real Estate (VNQ)", "Bond tradizionali a lungo termine (TLT)"]
                else:
                    st.info("### ⚖️ FASE DI TRANSIZIONE (Stock Picking Market)")
                    st.write("**Diagnosi Integrata:** Le metriche si annullano a vicenda. Assenza di trend macro chiaro. Rotazione continua tra settori.")
                    buys = ["Aziende Value con dividendi sicuri", "Settori difensivi (Utilities, Staples)", "Mercati esteri sottovalutati"]
                    sells = ["Asset iper-comprati a livello tecnico", "Aziende fortemente indebitate"]

            else:
                st.success("### 🚀 ESPANSIONE FULL RISK-ON (Goldilocks)")
                st.write("**Diagnosi Integrata:** Allineamento quantistico perfetto. Crescita reale, partecipazione di mercato ampia (Breadth), tassi stabili e iniezioni di liquidità (BTC).")
                buys = ["Azionario Tecnologia & Semiconduttori (XLK)", "Terre Rare (REMX)", "Bitcoin e Crypto", "Small Caps USA (IWM)"]
                sells = ["Asset Difensivi (Utilities - XLU)", "Dollaro Cash (UUP)", "Protezioni / VIX"]

        with list_col:
            st.markdown("#### ✅ FOCUS ACCUMULO")
            for b in buys: st.write(f"- {b}")
            st.markdown("#### ❌ FOCUS DISTRIBUZIONE")
            for s in sells: st.write(f"- {s}")

# --- CORE QUANT ENGINE ---
def calculate_gex_at_price(price, df, r=DYNAMIC_R, q=0.0):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.00005) # Allineato al Floor 0DTE
    # Calibrazione Istituzionale: 80% OI + 20% Volume per matchare get_greeks_pro
    exposure_size = (df['openInterest'].fillna(0).values * 0.8) + (df['volume'].fillna(0).values * 0.2)
    d1 = (np.log(price/K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = (norm.pdf(d1) * np.exp(-q * T)) / (price * iv * np.sqrt(T))
    side = np.where(df['type'] == 'call', 1, -1)
    return np.sum(gamma * exposure_size * 100 * price * side)

def calculate_0g_dynamic(price, df, r=DYNAMIC_R, q=0.0):
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    T = np.maximum(df['dte_years'].values, 0.00005) # Allineato al Floor 0DTE
    exposure_size = df['volume'].fillna(0).values 
    d1 = (np.log(price/K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    gamma = (norm.pdf(d1) * np.exp(-q * T)) / (price * iv * np.sqrt(T))
    side = np.where(df['type'] == 'call', 1, -1)
    return np.sum(gamma * exposure_size * 100 * price * side)

def bs_price(S, K, T, r, q, iv, option_type='call'):
    d1 = (np.log(S/K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def find_iv(target_price, S, K, T, r, q, option_type):
    if target_price <= 0 or T <= 0: return np.nan
    try:
        return brentq(lambda x: bs_price(S, K, T, r, q, x, option_type) - target_price, 0.0001, 5.0, xtol=1e-5)
    except:
        return np.nan

@st.cache_data(ttl=60)
def get_greeks_pro(df, S, r=DYNAMIC_R, q=0.0):
    df = df.copy()
    if df.empty: return df
    
    # 1. Pre-processing Istituzionale
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['openInterest'] = df['openInterest'].fillna(0)
    df['volume'] = df['volume'].fillna(0)
    
    # 2. Separazione Vettoriale per Calcolo Skew Puro (Evita derivate nulle su strike identici C/P)
    df_c = df[df['type'] == 'call'].sort_values('strike').copy()
    df_p = df[df['type'] == 'put'].sort_values('strike').copy()
    
    # 3. Curva di Volatilità e Pendenza Indipendenti
    for d in [df_c, df_p]:
        if not d.empty:
            d['iv_working'] = pd.to_numeric(d['impliedVolatility'], errors='coerce').replace(0, np.nan).interpolate().bfill().ffill()
            d['iv_working'] = np.maximum(d['iv_working'], 0.01) # Protezione anti-zero
            d['skew_slope'] = d['iv_working'].diff() / d['strike'].diff().replace(0, np.nan)
            d['skew_slope'] = d['skew_slope'].bfill().ffill().clip(-0.005, 0.005)
        
    # Ricostruzione Ordine
    df = pd.concat([df_c, df_p]).sort_values('strike').reset_index(drop=True)

    # 4. Motore Greco Vettorializzato (Con Sweet Spot 0DTE)
    T = np.maximum(df['dte_years'].values, 0.00005) if 'dte_years' in df.columns else 0.00005
    iv = df['iv_working'].values
    K = df['strike'].values
    d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf = norm.pdf(d1)
    cdf = norm.cdf(d1)

    # Formule Standard BS
    gamma_bs = np.exp(-q * T) * pdf / (S * iv * np.sqrt(T))
    vega_bs = S * np.exp(-q * T) * pdf * np.sqrt(T)
    vanna = (vega_bs / S) * (1 - d1 / (iv * np.sqrt(T)))
    vomma = vega_bs * (d1 * d2 / iv)

    # 5. Calcolo Shadow Gamma (Corretto per la pendenza IV)
    # Questa è la formula "maniacale" per l'esposizione reale
    df['Gamma_Adj'] = gamma_bs + (2 * vanna * df['skew_slope']) + (vomma * (df['skew_slope']**2))
    
    # 6. Conversione in Esposizione Monetaria (Notional GEX & Cross-Greeks)
    # Calcolo base dell'esposizione volumetrica
    oi_vol = (df['openInterest'] * 0.8) + (df['volume'] * 0.2)
    df['type_sign'] = df['type'].map({'call': 1, 'put': -1})
    
    # GEX: Dollari di esposizione per 1% mossa del sottostante
    df['GEX_Total'] = df['Gamma_Adj'] * (S**2) * 0.01 * oi_vol * 100 * df['type_sign']
    
    # --- IL DROPNA È STATO SPOSTATO ALLA FINE PER EVITARE VALUEERROR ---

    # Cross-Greeks Scaling Istituzionale
    side = np.where(df['type'] == 'call', 1, -1)
    charm_raw = -np.exp(-q * T) * (pdf * ((r - q) / (iv * np.sqrt(T)) - d2 / (2 * T)) + side * q * norm.cdf(d1 * side))
    speed_raw = -(gamma_bs / S) * (d1 / (iv * np.sqrt(T)) + 1)
    
    # Calcolo Greche (Scaling puro, nessuna alterazione formule)
    df['Vanna'] = vanna * 0.01 * S * oi_vol * 100 * df['type_sign'] 
    df['Charm'] = S * charm_raw * (1/252.0) * oi_vol * 100 * df['type_sign']
    df['Vega'] = vega_bs * 0.01 * oi_vol * 100
    
    term1 = -(S * np.exp(-q * T) * pdf * iv) / (2 * np.sqrt(T))
    term2 = side * r * K * np.exp(-r * T) * norm.cdf(d2 * side)
    term3 = side * q * S * np.exp(-q * T) * norm.cdf(d1 * side)
    df['Theta'] = (term1 - term2 + term3) * (1/252.0) * oi_vol * 100
    
    df['Vomma'] = vomma * 0.0001 * oi_vol * 100 
    df['Speed'] = speed_raw * oi_vol * 100 * (S**3) * 0.0001 * df['type_sign']
    df['Gamma'] = df['GEX_Total']
    
    df['Delta'] = np.exp(-q * T) * np.where(df['type'] == 'call', norm.cdf(d1), norm.cdf(d1) - 1)
    df['DEX'] = df['Delta'] * S * 0.01 * oi_vol * 100
    
    # Garantisce che tutte le colonne esistano prima del ritorno
    for col in ['Gamma', 'Vanna', 'Vomma', 'Charm', 'Speed', 'Vega', 'Theta', 'DEX']:
        if col not in df.columns:
            df[col] = 0.0

    # 7. PULIZIA FINALE (Spostata qui per proteggere l'integrità dei vettori)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['GEX_Total'])
    
    return df

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker, dates):
    t = yf.Ticker(ticker)
    frames = []
    for d in dates:
        try:
            oc = t.option_chain(d)
            frames.append(pd.concat([oc.calls.assign(type='call', exp=d), oc.puts.assign(type='put', exp=d)]))
        except: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# --- FUNZIONE PROTETTIVA PER LO SCANNER (Evita Ban IP da Yahoo) ---
@st.cache_data(ttl=300, show_spinner=False) # Aggiorna max ogni 5 minuti per ticker
def fetch_scanner_ticker(t_name, expiry_mode_str, today_str):
    try:
        t_obj = yf.Ticker(t_name)
        hist = t_obj.history(period='5d')
        if hist.empty: return None
        px = hist['Close'].iloc[-1]
        opts = t_obj.options
        if not opts: return None
        target_opt = opts[0] if "0-1 DTE" in expiry_mode_str else (opts[2] if len(opts) > 2 else opts[0])
        oc = t_obj.option_chain(target_opt)
        df_scan = pd.concat([oc.calls.assign(type='call'), oc.puts.assign(type='put')])
        
        # Conversione stringa a datetime
        today_obj = datetime.strptime(today_str, '%Y-%m-%d')
        dte_years = get_precise_dte(target_opt)
        df_scan['dte_years'] = dte_years
        df_scan = df_scan[(df_scan['strike'] > px*0.7) & (df_scan['strike'] < px*1.3)]
        
        return px, df_scan, dte_years
    except:
        return None

@st.cache_data(ttl=20, show_spinner=False)
def fetch_yahoo_history(symbol, timeframe, start_str=None, end_str=None, period=None):
    try:
        tf_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1H": "1h", "1D": "1d"}
        tf = tf_map.get(timeframe, "1m")
        df = yf.download(symbol, period=period if period else None, start=start_str, end=end_str, interval=tf, progress=False, prepost=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        col_map = {'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}
        df.rename(columns=col_map, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.ffill().bfill(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def fetch_alpaca_history(symbol, timeframe, start_str, end_str):
    # GESTIONE INDICI CON PREZZI "REALI" (Proxy ETF + Scaling)
    index_map = {
        "SPX": ("SPY", "^GSPC"),
        "NDX": ("QQQ", "^NDX"),
        "RUT": ("IWM", "^RUT"),
        "DJI": ("DIA", "^DJI"),
        "VIX": ("VIXY", "^VIX")
    }
    
    is_index = symbol.upper() in index_map
    alpaca_sym = index_map[symbol.upper()][0] if is_index else symbol.upper().replace("^", "")
    real_index_sym = index_map[symbol.upper()][1] if is_index else None
    
    headers = {
        "APCA-API-KEY-ID": st.session_state.get("alpaca_api_key", "PKQVMHYR25JUXQVLTEEBEKVIMV"),
        "APCA-API-SECRET-KEY": st.session_state.get("alpaca_secret_key", "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7")
    }
    
    tf_map = {"1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1H": "1Hour", "1D": "1Day"}
    tf = tf_map.get(timeframe, "1Day")
    
    now_utc = datetime.utcnow()
    safe_end_dt = now_utc - timedelta(minutes=20)
    
    try:
        req_end_obj = datetime.strptime(end_str, '%Y-%m-%d')
        if req_end_obj.date() >= now_utc.date():
            final_end = safe_end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            final_end = end_str + "T23:59:59Z"
    except:
        final_end = end_str + "T23:59:59Z"

    url = f"https://data.alpaca.markets/v2/stocks/{alpaca_sym}/bars"
    
    all_bars = []
    next_token = None
    
    # Progress Bar per download lunghi
    p_bar = st.progress(0, text="Scaricamento dati storici (Pagination)...")
    
    while True:
        params = {
            "start": start_str + "T00:00:00Z",
            "end": final_end,
            "timeframe": tf,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "iex",
            "page_token": next_token
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if "bars" in data and data["bars"]:
                    all_bars.extend(data["bars"])
                    next_token = data.get("next_page_token")
                    
                    # Update progress (fake visual feedback)
                    current_len = len(all_bars)
                    p_bar.progress(min(current_len % 100, 90), text=f"Scaricati {current_len} records...")
                    
                    if not next_token:
                        break
                else:
                    break
            else:
                st.error(f"Errore Alpaca API ({response.status_code}): {response.text}")
                break
        except Exception as e:
            st.error(f"Errore Connessione Alpaca: {e}")
            break
            
    p_bar.empty()
    
    if all_bars:
        df = pd.DataFrame(all_bars)
        df['t'] = pd.to_datetime(df['t'])
        df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        
        # SCALING LOGIC PER GLI INDICI
        if is_index and real_index_sym:
            try:
                # Scarica il prezzo reale dell'indice da Yahoo Finance (solo l'ultimo giorno per calcolare il ratio)
                real_idx_data = yf.download(real_index_sym, period="5d", progress=False)
                if not real_idx_data.empty:
                    real_close = real_idx_data['Close'].iloc[-1].item()
                    etf_close = df['Close'].iloc[-1]
                    ratio = real_close / etf_close
                    
                    # Moltiplica i prezzi dell'ETF per il ratio
                    for col in ['Open', 'High', 'Low', 'Close']:
                        df[col] = df[col] * ratio
            except Exception as e:
                st.warning(f"Impossibile scalare l'indice {symbol}: {e}")
                
        return df
        
    return pd.DataFrame()

def fetch_alpaca_crypto(symbol, timeframe, start_str, end_str):
    clean_sym = symbol.upper().replace('-USD', '').replace('USD', '').replace('/', '')
    alpaca_sym = f"{clean_sym}/USD"
    headers = {
        "APCA-API-KEY-ID": st.session_state.get("alpaca_api_key", "PKQVMHYR25JUXQVLTEEBEKVIMV"),
        "APCA-API-SECRET-KEY": st.session_state.get("alpaca_secret_key", "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7")
    }
    tf_map = {"1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1H": "1Hour", "1D": "1Day"}
    tf = tf_map.get(timeframe, "1Day")
    
    url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    all_bars = []
    page_token = None
    
    while True:
        params = {
            "symbols": alpaca_sym,
            "timeframe": tf,
            "start": start_str,
            "end": end_str,
            "limit": 10000
        }
        if page_token:
            params["page_token"] = page_token
            
        try:
            import requests
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                bars = data.get("bars", {}).get(alpaca_sym, [])
                all_bars.extend(bars)
                page_token = data.get("next_page_token")
                if not page_token:
                    break
            else:
                st.error(f"Errore Alpaca Crypto API ({response.status_code}): {response.text}")
                break
        except Exception as e:
            st.error(f"Errore Connessione Alpaca Crypto: {e}")
            break
            
    if all_bars:
        df = pd.DataFrame(all_bars)
        df['t'] = pd.to_datetime(df['t'])
        df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
        
    return pd.DataFrame()

# --- DATA FETCHING ENHANCED ---
def fetch_data_smart(ticker, timeframe, start_date, end_date, target_tz="America/New_York"):
    import requests
    from datetime import timedelta
    
    df = pd.DataFrame()
    
    # Determine asset type
    crypto_list = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE']
    is_crypto = "-USD" in ticker or any(ticker.startswith(c) and (ticker.endswith("USD") or len(ticker) == len(c)) for c in crypto_list)
    is_forex = ("=X" in ticker or (len(ticker) == 6 and ticker.isalpha())) and not is_crypto
    is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
    is_stock = not (is_forex or is_index or is_crypto)
    
    days_requested = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    clean_ticker = ticker.replace('=X', '').replace('^', '')

    # Helper to standardize dataframe
    def process_dataframe(df, ticker, start_date, end_date):
        if df.empty:
            return df
            
        # --- EXISTING NORMALIZATION LOGIC ---
        # Standardize columns
        rename_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[c] = cl.capitalize()
            elif cl in ['date', 'timestamp', 'time', 'datetime']:
                rename_map[c] = 'datetime'
                
        df.rename(columns=rename_map, inplace=True)
        
        if 'datetime' not in df.columns:
            if df.index.name and str(df.index.name).lower() in ['date', 'timestamp', 'time', 'datetime']:
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
            else:
                st.error("❌ Errore: Colonna data non trovata nel file CSV.")
                return pd.DataFrame()
                
        # Force numeric on OHLC and ensure they are float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                
        # Drop rows where Close is NaN
        if 'Close' in df.columns:
            df.dropna(subset=['Close'], inplace=True)
            
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        
        # Filter by date
        df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]
        
        if not df.empty and len(df) > 1:
            df.sort_values('datetime', ascending=True, inplace=True)
            diffs = df['datetime'].diff()
            max_gap = diffs.max()
            if pd.notnull(max_gap) and max_gap > pd.Timedelta(days=3):
                st.warning(f"⚠️ Attenzione: Rilevato un buco temporale nei dati di {max_gap.days} giorni.")
                
            df.set_index('datetime', drop=False, inplace=True)
            
        return df

    # ROUTE 1: Forex/Derivative -> histdatacom
    if is_forex:
        try:
            # Setup directory temporanea
            temp_dir = "./data_forex"
            extract_dir = "./data_extracted"
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(extract_dir, exist_ok=True)

            options = Options()
            options.pairs = {clean_ticker.lower()}
            options.formats = {"ascii"}
            options.timeframes = {"1-minute-bar-quotes"}
            options.start_yearmonth = pd.to_datetime(start_date).strftime("%Y-%m")
            options.end_yearmonth = pd.to_datetime(end_date).strftime("%Y-%m")
            
            # Chiediamo SOLO il download dell'archivio (No extract_csvs per evitare datatable)
            options.download_data_archives = True  
            options.data_directory = temp_dir
            
            # Esecuzione download
            histdatacom(options)
            
            # ESTRAZIONE MANUALE (Standard Python - No dependencies)
            zip_files = glob.glob(os.path.join(temp_dir, "**", "*.zip"), recursive=True)
            if not zip_files:
                st.error("❌ Nessun file ZIP scaricato da HistData.")
            else:
                for zf in zip_files:
                    with zipfile.ZipFile(zf, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                
                # LETTURA CSV
                csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
                if csv_files:
                    # Carichiamo i file (Solitamente il formato HistData è senza header)
                    df_list = []
                    for f in csv_files:
                        tmp_df = pd.read_csv(f, header=None, sep=';', engine='python')
                        df_list.append(tmp_df)
                    df_raw = pd.concat(df_list, ignore_index=True)
                    
                    # Assegnazione colonne standard (Format: YYYYMMDD HHMMSS;O;H;L;C;V)
                    df_raw.columns = ['datetime_str', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df_raw['datetime'] = pd.to_datetime(df_raw['datetime_str'], format='%Y%m%d %H%M%S')
                    
                    df = process_dataframe(df_raw, ticker, start_date, end_date)
                    
                    # --- FIX VOLUME SINTETICO FOREX ---
                    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                        df['Volume'] = ((df['High'] - df['Low']) * 100000).clip(lower=1)
                    
                    # Resampling per Timeframe custom
                    if not df.empty and timeframe not in ['1m', '1Min']:
                        resample_map = {
                            '5m': '5T', '5Min': '5T',
                            '15m': '15T', '15Min': '15T',
                            '1h': 'h', '1H': 'h',
                            '1d': 'D', '1D': 'D'
                        }
                        freq = resample_map.get(timeframe, 'D')
                        
                        df = df.resample(freq).agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                        
                        df['datetime'] = df.index

                    st.success(f"✅ Forex: {ticker} caricato con successo ({len(df)} righe).")
                    
                    # PULIZIA (Rimuoviamo tutto per liberare spazio su Streamlit)
                    shutil.rmtree(temp_dir)
                    shutil.rmtree(extract_dir)
                else:
                    st.error("❌ Nessun file CSV trovato negli archivi estratti.")
        except Exception as e:
            st.error(f"❌ Errore critico Forex: {e}")

    # ROUTE 2: Non-Forex -> Alpaca / Yahoo Finance
    if not is_forex:
        # ENGINE 1: Alpaca (Primary for Stocks/Indices/Crypto)
        try:
            tf_alpaca = timeframe
            if timeframe == "1d" or timeframe == "1D": tf_alpaca = "1Day"
            elif timeframe == "1h" or timeframe == "1H": tf_alpaca = "1Hour"
            elif timeframe == "15m" or timeframe == "15Min": tf_alpaca = "15Min"
            elif timeframe == "5m" or timeframe == "5Min": tf_alpaca = "5Min"
            elif timeframe == "1m" or timeframe == "1Min": tf_alpaca = "1Min"
            
            if is_crypto:
                df = fetch_alpaca_crypto(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            else:
                df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            st.error(f"Alpaca fetch failed: {e}")
        
        # ENGINE 2: yfinance (Fallback)
        if df.empty:
            try:
                tf_yf = "1d"
                if timeframe == "1m" or timeframe == "1Min": tf_yf = "1m"
                elif timeframe == "5m" or timeframe == "5Min": tf_yf = "5m"
                elif timeframe == "15m" or timeframe == "15Min": tf_yf = "15m"
                elif timeframe == "1h" or timeframe == "1H": tf_yf = "1h"
                elif timeframe == "1d" or timeframe == "1D": tf_yf = "1d"
                
                actual_start = start_date
                if tf_yf in ["1m"] and days_requested > 7:
                    actual_start = end_date - timedelta(days=7)
                    st.warning("⚠️ yfinance supporta solo 7 giorni per il timeframe 1m. Date troncate.")
                elif tf_yf in ["5m", "15m"] and days_requested > 60:
                    actual_start = end_date - timedelta(days=60)
                    st.warning(f"⚠️ yfinance supporta solo 60 giorni per il timeframe {tf_yf}. Date troncate.")
                elif tf_yf == "1h" and days_requested > 730:
                    actual_start = end_date - timedelta(days=730)
                    st.warning(f"⚠️ yfinance supporta solo 730 giorni per il timeframe 1h. Date troncate.")
                
                df_yf = yf.download(ticker, start=actual_start, end=end_date, interval=tf_yf, progress=False)
                if not df_yf.empty:
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)
                    df_yf.reset_index(inplace=True)
                    df = process_dataframe(df_yf, ticker, start_date, end_date)
                    if not df.empty:
                        st.warning("⚠️ Dati presi da Yahoo Finance (Limiti applicati).")
            except Exception as e:
                st.error(f"❌ Errore Yahoo Finance: {e}")

    # ROUTE 3: Local Database (If everything fails)
    if df.empty:
        try:
            possible_files = [f"{clean_ticker}.csv", f"{clean_ticker}.CSV", f"{clean_ticker.lower()}.csv"]
            local_path = None
            for pf in possible_files:
                p = os.path.join(LOCAL_DB_DIR, pf)
                if os.path.exists(p):
                    local_path = p
                    break
                    
            if local_path:
                df_local = pd.read_csv(local_path)
                df = process_dataframe(df_local, ticker, start_date, end_date)
                if not df.empty:
                    st.success("📂 Dati recuperati dal Database Locale.")
        except Exception as e:
            st.error(f"❌ Errore lettura Database Locale: {e}")
            
    # ENGINE 4: Fatal Error
    if df.empty:
        st.error("❌ ERRORE CRITICO: Dati non trovati in nessun motore (Alpaca, HistData, Locale, Yahoo). Per favore, carica un file CSV manualmente usando l'apposito uploader per testare questo asset.")
        st.stop()
            
    if not df.empty:
        cols = df.select_dtypes(include=['float64']).columns
        if not cols.empty:
            df[cols] = df[cols].astype('float32')
            
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            try:
                if df['datetime'].dt.tz is None:
                    df['datetime'] = df['datetime'].dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert(target_tz).dt.tz_localize(None)
                else:
                    df['datetime'] = df['datetime'].dt.tz_convert(target_tz).dt.tz_localize(None)
            except Exception as e:
                # Fallback estremo: se la conversione fallisce, manteniamo il tempo originale per non bloccare il sistema
                st.warning(f"⚠️ Avviso Timezone: {e}. Utilizzo tempo originale.")
            
        # --- FIX AMBIGUITÀ PANDAS ---
        # Resettiamo l'indice PRIMA di ordinare. Questo elimina l'eventuale indice 'datetime'
        # mantenendo intatta e sicura la colonna 'datetime', permettendo il sort senza crash.
        df.reset_index(drop=True, inplace=True)
        
        if 'datetime' in df.columns:
            df.sort_values('datetime', inplace=True)
            
        df.ffill().bfill(inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

@st.cache_data(ttl=60)
def get_whale_intelligence(ticker):
    # 1. Normalizzazione e Routing Proxy
    clean_ticker = ticker.upper().replace("^", "")
    route_map = {"NDX": "QQQ", "SPX": "SPY", "RUT": "IWM"}
    target_ticker = route_map.get(clean_ticker, ticker)
    is_index = clean_ticker in route_map

    try:
        # Download dati ETF (Sorgente Volumi)
        df_etf = yf.download(target_ticker, period="5d", interval="1m", progress=False, prepost=True)
        if df_etf.empty: return {"Whale_Price": None, "Whale_Intensity": 0, "Confluence_Status": "N/A", "Whale_Bias": "NEUTRAL", "Volume_Buyers": 0, "Volume_Sellers": 0}
        if df_etf.index.tz is not None: df_etf.index = df_etf.index.tz_localize(None)
        
        if isinstance(df_etf.columns, pd.MultiIndex): df_etf.columns = df_etf.columns.get_level_values(0)
        df_etf.rename(columns={str(c): str(c).capitalize() for c in df_etf.columns}, inplace=True)

        # 2. Calcolo Dinamico della Ratio (Index / ETF) - ROLLING MINUTE-BY-MINUTE
        ratio_val = 1.0
        if is_index:
            idx_sym = "^" + clean_ticker if not ticker.startswith("^") else ticker
            df_idx = yf.download(idx_sym, period="5d", interval="1m", progress=False, prepost=True) # Allineato a 5d come l'ETF
            if not df_idx.empty:
                if df_idx.index.tz is not None: df_idx.index = df_idx.index.tz_localize(None)
                if isinstance(df_idx.columns, pd.MultiIndex): df_idx.columns = df_idx.columns.get_level_values(0)
                # Calcolo Vettoriale (Candela per Candela)
                idx_aligned, etf_aligned = df_idx['Close'].align(df_etf['Close'], join='inner')
                df_etf['Dynamic_Ratio'] = (idx_aligned / etf_aligned).replace([np.inf, -np.inf], np.nan)
                # Molto importante: riempiamo i buchi e garantiamo che ci sia sempre un valore
                df_etf['Dynamic_Ratio'] = df_etf['Dynamic_Ratio'].ffill().bfill()
                # Se dopo la pulizia è ancora vuoto, usa la ratio dell'ultimo minuto
                if df_etf['Dynamic_Ratio'].isnull().all():
                    df_etf['Dynamic_Ratio'] = df_idx['Close'].iloc[-1] / df_etf['Close'].iloc[-1]
                ratio_val = df_etf['Dynamic_Ratio'].iloc[-1]
            else:
                # Fallback
                ticker_info = yf.Ticker(idx_sym).fast_info
                etf_info = yf.Ticker(target_ticker).fast_info
                ratio_val = ticker_info.last_price / etf_info.last_price
                df_etf['Dynamic_Ratio'] = ratio_val
        else:
            df_etf['Dynamic_Ratio'] = 1.0

        # 3. Identificazione Whale Level sull'ETF (Punti di Accumulazione)
        vol_mean = df_etf['Volume'].mean()
        vol_sd = df_etf['Volume'].std()
        
        # Filtro: Candele Anomale (Volume > 2.5 SD)
        anomalies = df_etf[df_etf['Volume'] > (vol_mean + 2.5 * vol_sd)].copy()
        
        intensity = 20
        whale_price_etf = df_etf['Close'].iloc[-1] # Default
        vol_buyers = 0
        vol_sellers = 0
        whale_bias = "NEUTRAL"
        
        if not anomalies.empty:
            # 1. Time-Weighted VWAP (Ultime 4 ore = 3x peso)
            last_timestamp = df_etf.index[-1]
            delta_hours = (last_timestamp - anomalies.index).total_seconds() / 3600
            anomalies['Time_Weight'] = np.where(delta_hours <= 4.0, 3.0, 1.0)
            anomalies['Weighted_Volume'] = anomalies['Volume'] * anomalies['Time_Weight']
            
            # Calcolo VWAP Globale
            global_vwap = (anomalies['Close'] * anomalies['Weighted_Volume']).sum() / anomalies['Weighted_Volume'].sum()
            current_etf_price = df_etf['Close'].iloc[-1]
            
            # 2. Adaptive Cluster (Se distanza > 1.5%)
            if abs(current_etf_price - global_vwap) / global_vwap > 0.015:
                # Filtra anomalie vicine al prezzo (entro 2%)
                lower_bound = current_etf_price * 0.98
                upper_bound = current_etf_price * 1.02
                proximate_anomalies = anomalies[(anomalies['Close'] >= lower_bound) & (anomalies['Close'] <= upper_bound)].copy()
                
                if not proximate_anomalies.empty:
                    # Raggruppamento in Cluster (ogni 0.5% di distanza)
                    cluster_step = current_etf_price * 0.005
                    proximate_anomalies['Cluster'] = (proximate_anomalies['Close'] / cluster_step).round() * cluster_step
                    
                    # Prendi il cluster dell'anomalia più recente
                    last_proximate_cluster = proximate_anomalies.iloc[-1]['Cluster']
                    target_anomalies = proximate_anomalies[proximate_anomalies['Cluster'] == last_proximate_cluster]
                else:
                    # Se non ci sono cluster vicini, usa l'anomalia più recente in assoluto
                    target_anomalies = anomalies.iloc[[-1]]
            else:
                # Usa tutte le anomalie se il prezzo è vicino al VWAP globale
                target_anomalies = anomalies
                
            # Calcolo Whale-VWAP scalato millimetricamente minuto per minuto
            idx_closes = target_anomalies['Close'] * target_anomalies['Dynamic_Ratio']
            final_whale_price = float((idx_closes * target_anomalies['Weighted_Volume']).sum() / target_anomalies['Weighted_Volume'].sum())
            
            # Logica Direzionale (Lee-Ready) sul target scelto
            mid_price = (target_anomalies['High'] + target_anomalies['Low']) / 2
            buyers_mask = target_anomalies['Close'] > mid_price
            
            vol_buyers = target_anomalies.loc[buyers_mask, 'Volume'].sum()
            vol_sellers = target_anomalies.loc[~buyers_mask, 'Volume'].sum()
            
            if vol_buyers > vol_sellers:
                whale_bias = "BULLISH"
            elif vol_sellers > vol_buyers:
                whale_bias = "BEARISH"
            else:
                whale_bias = "NEUTRAL"
            
            # Intensità basata sul numero di anomalie nel target
            intensity = 70 if len(target_anomalies) > 3 else 50
            
        # Confluenza basata sulla vicinanza al prezzo attuale
        current_price = df_etf['Close'].iloc[-1] * ratio_val
        confluence = "High" if abs(current_price - final_whale_price) / current_price < 0.003 else "Mid"

        # Mostra i diamanti/marker solo per le ultime 24 ore
        if not anomalies.empty:
            cutoff_24h = datetime.now(anomalies.index.tz) - timedelta(hours=24)
            anomalies = anomalies[anomalies.index >= cutoff_24h]

        return {
            "Whale_Price": final_whale_price,
            "Whale_Intensity": intensity,
            "Confluence_Status": confluence,
            "Whale_Bias": whale_bias,
            "Volume_Buyers": float(vol_buyers),
            "Volume_Sellers": float(vol_sellers),
            "Error": None,
            "df_whale": df_etf.copy() if not df_etf.empty else pd.DataFrame(), # FIX: Esportiamo l'intero ETF per la Dynamic Ratio
            "vol_mean": float(vol_mean),
            "vol_sd": float(vol_sd),
            "ratio": float(ratio_val) # FIX: Reintegrata per permettere il rendering corretto sugli indici
        }
    except Exception as e:
        return {"Whale_Price": None, "Whale_Intensity": 0, "Confluence_Status": "N/A", "Whale_Bias": "NEUTRAL", "Volume_Buyers": 0, "Volume_Sellers": 0, "Error": str(e)}

import streamlit.components.v1 as components

def display_seasonality_and_calendar():
    st.title("📅 INSTITUTIONAL SEASONALITY & MACRO CALENDAR")
    st.info("Motore quantitativo per il vantaggio statistico storico e tracking degli eventi macroeconomici.")
    
    tab_seas, tab_cal = st.tabs(["📊 Multi-Asset Seasonality Engine", "📆 Global Economic Calendar"])
    
    with tab_seas:
        # 1. CATEGORIE ISTITUZIONALI PREDEFINITE
        st.markdown("### 🔍 Setup Asset & Historical Data")
        cats = {
            "Indici Equity (ETF)": ["SPY (S&P 500)", "QQQ (Nasdaq)", "DIA (Dow Jones)", "IWM (Small Cap)", "VGK (Europa)", "EEM (Emerging)", "^VIX (Volatilità)"],
            "Forex (Valute)": [
                "EURUSD=X (Euro)", "JPYUSD=X (Yen)", "GBPUSD=X (Sterlina)", "USDCHF=X (Franco Svizzero)", 
                "AUDUSD=X (Australiano)", "USDCAD=X (Canadese)", "NZDUSD=X (Neozelandese)",
                "EURJPY=X (Euro/Yen)", "EURGBP=X (Euro/Sterlina)", "GBPJPY=X (Sterlina/Yen)", "EURCHF=X (Euro/Franco)"
            ],
            "Metalli Preziosi & Ind.": ["GC=F (Oro)", "SI=F (Argento)", "PL=F (Platino)", "PA=F (Palladio)", "HG=F (Rame)", "ALI=F (Alluminio)"],
            "Materie Prime (Agricoltura)": [
                "KC=F (Caffè)", "CORN (Mais)", "WEAT (Grano)", "SOYB (Soia)", "ZL=F (Olio di Soia)", 
                "ZO=F (Avena)", "ZR=F (Riso Grezzo)", "KE=F (Grano Rosso Inv.)", "MG=F (Grano Primavera)", 
                "CT=F (Cotone)", "SB=F (Zucchero)", "CC=F (Cacao)", "DBA (Agricoltura G.)"
            ],
            "Settore Zootecnico (Livestock)": [
                "LE=F (Bovini Vivi)", "GF=F (Bovini da Ingrasso)", "HE=F (Maiali Magri)"
            ],
            "Settore Energetico": ["CL=F (Crude Oil)", "NG=F (Natural Gas)", "BZ=F (Brent)", "RB=F (Benzina)", "HO=F (Heating Oil)", "XLE (Energy ETF)"],
            "Bond & Tassi": ["TLT (20Y Bond)", "IEF (7-10Y Bond)", "HYG (High Yield)", "LQD (Corporate)", "^TNX (10Y Yield)"],
            "Custom (Inserimento Manuale)": []
        }
        
        col1, col2 = st.columns(2)
        with col1:
            cat_choice = st.selectbox("Seleziona Categoria Asset", list(cats.keys()))
        with col2:
            if cat_choice == "Custom (Inserimento Manuale)":
                ticker = st.text_input("Inserisci Ticker", "AAPL").upper().strip()
            else:
                ticker_raw = st.selectbox("Seleziona Ticker", cats[cat_choice])
                ticker = ticker_raw.split(" (")[0] # Estrae solo la parte prima della parentesi (es. KC=F)
                
        if not ticker:
            st.warning("Seleziona o inserisci un ticker valido.")
            return

        with st.spinner(f"Scaricamento di tutto lo storico disponibile per {ticker}..."):
            try:
                # 2. SCARICAMENTO TUTTI I DATI ESISTENTI (MAX)
                df_raw = yf.download(ticker, period="max", progress=False)
                if df_raw.empty:
                    st.error(f"Nessun dato trovato per {ticker}.")
                    return
                
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)
                
                if 'Close' not in df_raw.columns:
                    st.error("Dati Close mancanti.")
                    return
                    
                df_close = df_raw[['Close']].dropna()
                first_date = df_close.index.min().strftime('%Y-%m-%d')
                last_date = df_close.index.max().strftime('%Y-%m-%d')
                total_years = (df_close.index.max() - df_close.index.min()).days / 365.25
                
                st.success(f"✅ Storico acquisito: {total_years:.1f} Anni di dati (Dal {first_date} al {last_date})")
                
                # 3. ELABORAZIONE RENDIMENTI MENSILI
                # Resample fine mese e calcolo variazione %
                monthly_data = df_close.resample('ME').last().pct_change() * 100
                monthly_data = monthly_data.dropna()
                monthly_data['Year'] = monthly_data.index.year
                monthly_data['Month'] = monthly_data.index.month
                
                # 4. FUNZIONE CALCOLO CURVA STAGIONALE
                current_year = datetime.now().year
                def get_seasonality_curve(data, years_back=None):
                    if years_back:
                        data = data[data['Year'] >= (current_year - years_back)]
                    return data.groupby('Month')['Close'].mean()

                s_all = get_seasonality_curve(monthly_data)
                s_20 = get_seasonality_curve(monthly_data, 20)
                s_10 = get_seasonality_curve(monthly_data, 10)
                s_5 = get_seasonality_curve(monthly_data, 5)
                
                months_labels = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
                
                # 5. PLOT GRAFICO COMPARATIVO
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=months_labels, y=s_all.values, mode='lines+markers', name=f'All-Time Storico ({total_years:.0f} Anni)', line=dict(color='white', width=4)))
                if total_years >= 20:
                    fig.add_trace(go.Scatter(x=months_labels, y=s_20.values, mode='lines', name='Ultimi 20 Anni', line=dict(color='gold', width=2, dash='dash')))
                if total_years >= 10:
                    fig.add_trace(go.Scatter(x=months_labels, y=s_10.values, mode='lines', name='Ultimi 10 Anni', line=dict(color='cyan', width=2, dash='dot')))
                if total_years >= 5:
                    fig.add_trace(go.Scatter(x=months_labels, y=s_5.values, mode='lines', name='Ultimi 5 Anni', line=dict(color='magenta', width=2, dash='longdash')))

                fig.update_layout(
                    title=f"Analisi Stagionalità Sovrapposta: {ticker}",
                    xaxis_title="Mese dell'Anno",
                    yaxis_title="Rendimento Medio Mensile (%)",
                    template="plotly_dark",
                    hovermode="x unified",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. MOTORE STATISTICO (TABELLA RIEPILOGATIVA)
                st.markdown("### 🧠 Macchina Statistica (Dati All-Time)")
                st.write("Analisi matematica per singolo mese. Il **Win Rate** indica quante volte storicamente il mese ha chiuso in positivo.")
                
                stats_data = []
                for m in range(1, 13):
                    m_data = monthly_data[monthly_data['Month'] == m]['Close']
                    if len(m_data) > 0:
                        win_rate = (len(m_data[m_data > 0]) / len(m_data)) * 100
                        avg_ret = m_data.mean()
                        # Identificazione Bias Direzionale
                        bias = "🟢 BULLISH (Long)" if avg_ret > 0 else "🔴 BEARISH (Short)"
                        stats_data.append({
                            "Mese": months_labels[m-1],
                            "Previsione Storica": bias,
                            "Win Rate (%)": f"{int(round(win_rate, 0))}%",
                            "Rend. Medio (%)": f"{avg_ret:.2f}%",
                            "Max Gain (%)": f"{m_data.max():.2f}%",
                            "Max Loss (%)": f"{m_data.min():.2f}%",
                            "Campione (Anni)": len(m_data)
                        })
                
                df_stats = pd.DataFrame(stats_data)
                
                # Trasformazione temporanea per la formattazione (rimozione simbolo % per calcolo gradiente)
                df_viz = df_stats.copy()
                df_viz['Rend. Medio (%)'] = df_viz['Rend. Medio (%)'].str.replace('%', '').astype(float)
                
                # Applichiamo lo stile con gradiente sulla colonna del Rendimento Medio
                # Usiamo la colonna numerica originale o pulita per il calcolo del colore
                styled_df = df_viz.style.background_gradient(
                    subset=['Rend. Medio (%)'], 
                    cmap='RdYlGn', # Scala da Rosso (Red) a Verde (Green)
                    vmin=-3.0,     # Range minimo per il colore pieno
                    vmax=3.0       # Range massimo per il colore pieno
                ).format({'Rend. Medio (%)': '{:.2f}%'}, precision=2)

                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Executive Summary Finale (Aggiornato con Percentuali)
                st.markdown("### 🏆 Seasonal Alpha Ranking")
                df_numeric = pd.DataFrame([
                    {"Mese": months_labels[m-1], "Avg": monthly_data[monthly_data['Month'] == m]['Close'].mean()} 
                    for m in range(1, 13)
                ])
                top_3 = df_numeric.sort_values("Avg", ascending=False).head(3)
                worst_3 = df_numeric.sort_values("Avg", ascending=True).head(3)
                
                c_best, c_worst = st.columns(2)
                with c_best:
                    st.success(
                        f"**Mesi Migliori (Long):**\n\n"
                        f"1. {top_3.iloc[0]['Mese']} ({top_3.iloc[0]['Avg']:.2f}%)\n"
                        f"2. {top_3.iloc[1]['Mese']} ({top_3.iloc[1]['Avg']:.2f}%)\n"
                        f"3. {top_3.iloc[2]['Mese']} ({top_3.iloc[2]['Avg']:.2f}%)"
                    )
                with c_worst:
                    st.error(
                        f"**Mesi Peggiori (Short):**\n\n"
                        f"1. {worst_3.iloc[0]['Mese']} ({worst_3.iloc[0]['Avg']:.2f}%)\n"
                        f"2. {worst_3.iloc[1]['Mese']} ({worst_3.iloc[1]['Avg']:.2f}%)\n"
                        f"3. {worst_3.iloc[2]['Mese']} ({worst_3.iloc[2]['Avg']:.2f}%)"
                    )

            except Exception as e:
                st.error(f"Errore durante l'elaborazione della stagionalità: {e}")

        # ==========================================
        # NUOVA SEZIONE: C.O.T. REPORT (CFTC)
        # ==========================================
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 🏛️ Commitment of Traders (COT) - Smart Money Tracker")
        
        # Estrazione base ticker
        base_ticker = ticker.split(" (")[0].strip() if "(" in ticker else ticker.strip()
        
        # 1. MAPPING VELOCE (Per gli asset più comuni, per velocità) - include vecchio e nuovo
        fast_mapping = {
            "EURUSD=X": "099741", "JPYUSD=X": "097741", "GBPUSD=X": "096742", 
            "AUDUSD=X": "232741", "USDCAD=X": "090741", "USDCHF=X": "092741",
            "GC=F": "088691", "SI=F": "084691", "HG=F": "085692", "PL=F": "076651", "PA=F": "075651",
            "CL=F": "067651", "NG=F": "023651", "BZ=F": "06765T",
            "CORN": "002602", "WEAT": "001602", "SOYB": "005602", "KC=F": "083731", "SB=F": "080732",
            "LE=F": "057642", "HE=F": "054642", "GF=F": "061641",
            "SPY": "13874A", "QQQ": "209742", "DIA": "124603",
            "TLT": "043602", "^TNX": "044601", "IEF": "044601"
        }

        cftc_code = None
        
        if base_ticker in fast_mapping:
            cftc_code = fast_mapping[base_ticker]
        else:
            # 2. AUTO-DISCOVERY TRAMITE RICERCA YFINANCE -> CFTC
            with st.spinner(f"Ricerca codice CFTC dinamico per {base_ticker}..."):
                try:
                    # Otteniamo info sull'asset per estrarre il nome
                    yf_info = yf.Ticker(base_ticker).info
                    search_term = yf_info.get('shortName', base_ticker)
                    # Se non lo trova, fallback sul base_ticker
                    if not search_term or pd.isna(search_term):
                         search_term = base_ticker
                    # Pulizia nome per ricerca migliore
                    search_term = search_term.split(" ")[0].upper() # Es. "Copper Futures" -> "COPPER"

                    # Configurazione Header Istituzionali (Necessari per evitare il blocco CFTC)
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/json'
                    }
                    
                    # Chiamata al catalogo mercati CFTC per trovare il codice
                    discovery_url = f"https://publicreporting.cftc.gov/resource/j7ed-idbc.json?$where=upper(market_and_exchange_names) like '%25{search_term}%25'&$limit=1"
                    cat_response = requests.get(discovery_url, headers=headers, timeout=10)
                    
                    if cat_response.status_code == 200:
                        catalog = cat_response.json()
                        # Cerchiamo il nome dell'asset all'interno del nome del mercato CFTC
                        for market in catalog:
                            if "market_and_exchange_names" in market and search_term in market["market_and_exchange_names"].upper():
                                cftc_code = market.get("cftc_contract_market_code")
                                st.success(f"Trovato codice CFTC correlato: {cftc_code} ({market['market_and_exchange_names']})")
                                break
                except Exception as e:
                    st.write(f"Discovery fallita: {e}")

        if cftc_code:
            with st.spinner("Estrazione dati COT Report dal database governativo CFTC..."):
                try:
                    # Configurazione Header Istituzionali (Necessari per evitare il blocco CFTC)
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/json'
                    }
                    
                    # URL Dataset Legacy (più stabile per analisi storica)
                    # ID Dataset: jun7-fc8e (Legacy Combined)
                    cftc_url = f"https://publicreporting.cftc.gov/resource/jun7-fc8e.json?cftc_contract_market_code={cftc_code}&$order=report_date_as_yyyy_mm_dd DESC&$limit=104"
                    
                    response = requests.get(cftc_url, headers=headers, timeout=20) # Timeout aumentato a 20s
                    
                    if response.status_code == 200:
                        cot_data = response.json()
                        if cot_data:
                            df_cot = pd.DataFrame(cot_data)
                            # Conversione colonne con gestione errori
                            df_cot['date'] = pd.to_datetime(df_cot['report_date_as_yyyy_mm_dd'])
                            
                            # Identificazione colonne corrette per il dataset Legacy
                            # Nota: I nomi delle colonne possono variare tra 'noncomm_positions_long_all' e 'non_commercial_long_all'
                            col_mapping = {
                                'long': 'noncomm_positions_long_all' if 'noncomm_positions_long_all' in df_cot.columns else 'non_commercial_long_all',
                                'short': 'noncomm_positions_short_all' if 'noncomm_positions_short_all' in df_cot.columns else 'non_commercial_short_all',
                                'comm_long': 'comm_positions_long_all' if 'comm_positions_long_all' in df_cot.columns else 'commercial_long_all',
                                'comm_short': 'comm_positions_short_all' if 'comm_positions_short_all' in df_cot.columns else 'commercial_short_all',
                                'retail_long': 'nonrept_positions_long_all' if 'nonrept_positions_long_all' in df_cot.columns else 'nonreportable_positions_long_all',
                                'retail_short': 'nonrept_positions_short_all' if 'nonrept_positions_short_all' in df_cot.columns else 'nonreportable_positions_short_all',
                                'open_interest': 'open_interest_all' if 'open_interest_all' in df_cot.columns else 'open_interest'
                            }
                            
                            for c in col_mapping.values():
                                if c in df_cot.columns:
                                    df_cot[c] = pd.to_numeric(df_cot[c], errors='coerce').fillna(0)

                            # Calcolo Posizioni Nette
                            df_cot['Net_NonComm'] = df_cot[col_mapping['long']] - df_cot[col_mapping['short']]
                            df_cot['Net_Comm'] = df_cot[col_mapping['comm_long']] - df_cot[col_mapping['comm_short']]
                            
                            if col_mapping['retail_long'] in df_cot.columns:
                                df_cot['Net_Retail'] = df_cot[col_mapping['retail_long']] - df_cot[col_mapping['retail_short']]
                            else:
                                df_cot['Net_Retail'] = 0
                            
                            df_cot = df_cot.sort_values('date')
                            
                            # Creazione Grafico Istituzionale Plotly
                            fig_cot = go.Figure()
                            
                            # Linea Zero di equilibrio
                            fig_cot.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                            
                            # Tracciato Commercials
                            fig_cot.add_trace(go.Scatter(x=df_cot['date'], y=df_cot['Net_Comm'], mode='lines', name='Commercials (Smart Money/Hedgers)', line=dict(color='#2ecc71', width=2), fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'))
                            # Tracciato Hedge Funds
                            fig_cot.add_trace(go.Scatter(x=df_cot['date'], y=df_cot['Net_NonComm'], mode='lines', name='Large Specs (Hedge Funds)', line=dict(color='#e74c3c', width=2)))
                            # Tracciato Retail
                            fig_cot.add_trace(go.Scatter(x=df_cot['date'], y=df_cot['Net_Retail'], mode='lines', name='Retail (Small Specs)', line=dict(color='#f1c40f', width=1.5, dash='dot')))
                            
                            fig_cot.update_layout(
                                title=f"Posizionamento Netto COT - Ultimi 2 Anni ({base_ticker} - Codice: {cftc_code})",
                                xaxis_title="Data (Weekly)",
                                yaxis_title="Contratti Netti (Long - Short)",
                                template="plotly_dark",
                                hovermode="x unified",
                                height=450,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=0, r=0, t=50, b=0)
                            )
                            st.plotly_chart(fig_cot, use_container_width=True)
                            
                            # --- START INSTITUTIONAL DASHBOARD ---
                            st.markdown("#### 📊 Wall Street Intelligence Dashboard")
                            
                            # 1. CALCOLO METRICHE AVANZATE (Delta, Percentili e OI)
                            latest = df_cot.iloc[-1]
                            prev_1w = df_cot.iloc[-2] if len(df_cot) > 1 else latest
                            prev_4w = df_cot.iloc[-5] if len(df_cot) > 4 else latest

                            # Delta 1 Settimana
                            d1w_comm = latest['Net_Comm'] - prev_1w['Net_Comm']
                            d1w_spec = latest['Net_NonComm'] - prev_1w['Net_NonComm']
                            d1w_oi = latest[col_mapping['open_interest']] - prev_1w[col_mapping['open_interest']]
                            
                            # Delta 1 Mese (4W)
                            d4w_comm = latest['Net_Comm'] - prev_4w['Net_Comm']
                            d4w_spec = latest['Net_NonComm'] - prev_4w['Net_NonComm']

                            # COT Index (Percentile a 2 anni)
                            comm_net_min = df_cot['Net_Comm'].min()
                            comm_net_max = df_cot['Net_Comm'].max()
                            cot_index_comm = ((latest['Net_Comm'] - comm_net_min) / (comm_net_max - comm_net_min)) * 100 if comm_net_max != comm_net_min else 50
                            
                            # Normalizzazione: Commercial Net % of Open Interest (CP/OI)
                            cpoi_latest = (latest['Net_Comm'] / latest[col_mapping['open_interest']]) * 100 if latest[col_mapping['open_interest']] > 0 else 0

                            # --- LOGICA DI SCORE ISTITUZIONALE (CONFLUENCE SCORE) ---
                            score = 0
                            max_score = 6
                            
                            # 1. Valutazione COT Index (Peso: 2)
                            if cot_index_comm > 80: score += 2
                            elif cot_index_comm > 60: score += 1
                            elif cot_index_comm < 20: score -= 2
                            elif cot_index_comm < 40: score -= 1

                            # 2. Valutazione Delta 1W Smart Money (Peso: 1)
                            if d1w_comm > 0: score += 1
                            else: score -= 1

                            # 3. Valutazione Delta 4W Smart Money (Peso: 1)
                            if d4w_comm > 0: score += 1
                            else: score -= 1

                            # 4. Valutazione Open Interest & Liquidità (Peso: 1)
                            # Se OI sale insieme al posizionamento Smart Money, il segnale è forte
                            if d1w_oi > 0 and d1w_comm > 0: score += 1
                            elif d1w_oi > 0 and d1w_comm < 0: score -= 1

                            # 5. Valutazione CP/OI (Peso: 1)
                            if cpoi_latest > 10: score += 1
                            elif cpoi_latest < -10: score -= 1

                            # Normalizzazione in percentuale (0% Bearish - 100% Bullish)
                            bias_percent = ((score + max_score) / (max_score * 2)) * 100
                            
                            # Definizione Semaforo e Colore
                            if bias_percent >= 70: 
                                signal_text, signal_col, signal_emoji = "STRONG BUY (Institutional Accumulation)", "#2ecc71", "🟢"
                            elif bias_percent >= 55: 
                                signal_text, signal_col, signal_emoji = "NEUTRAL/BULLISH (Mild Buying)", "#f1c40f", "🟡"
                            elif bias_percent <= 30: 
                                signal_text, signal_col, signal_emoji = "STRONG SELL (Institutional Distribution)", "#e74c3c", "🔴"
                            elif bias_percent <= 45: 
                                signal_text, signal_col, signal_emoji = "NEUTRAL/BEARISH (Mild Selling)", "#e67e22", "🟠"
                            else: 
                                signal_text, signal_col, signal_emoji = "NEUTRAL (No Clear Bias)", "#95a5a6", "⚪"

                            # --- VISUALIZZAZIONE BIAS DASHBOARD ---
                            st.markdown(f"""
                            <div style="background-color: {signal_col}22; padding: 20px; border-radius: 10px; border: 2px solid {signal_col}; margin-bottom: 20px;">
                                <h3 style="margin:0; color:{signal_col};">{signal_emoji} Institutional Bias Score: {bias_percent:.1f}%</h3>
                                <p style="margin:5px 0 0 0; font-size: 1.2rem; font-weight: bold;">Sentiment: {signal_text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 2. VISUALIZZAZIONE METRICHE CHIAVE (Con Funzioni di Aiuto Tooltip)
                            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                            
                            # Net Comm
                            r1c1.metric("Net Comm (Smart Money)", f"{int(latest['Net_Comm']):,}", 
                                       f"{int(d1w_comm):,} (1W)", 
                                       delta_color="normal",
                                       help="Posizione netta (Long - Short) dei Commercial Traders (produttori/utilizzatori fisici). È il 'Denaro Intelligente' che anticipa le grandi inversioni di trend.")
                            
                            # Net Specs
                            r1c2.metric("Net Specs (Hedge Funds)", f"{int(latest['Net_NonComm']):,}", 
                                       f"{int(d1w_spec):,} (1W)", 
                                       delta_color="inverse",
                                       help="Posizione netta dei Large Speculators (Hedge Funds). Solitamente seguono il trend e tendono a sovraesporsi pericolosamente alla fine di un ciclo di mercato.")
                            
                            # COT Index
                            idx_color = "normal" if cot_index_comm > 50 else "inverse"
                            r1c3.metric("COT Index (Comm)", f"{cot_index_comm:.1f}%", 
                                       delta="BULLISH" if cot_index_comm > 60 else "BEARISH" if cot_index_comm < 40 else "NEUTRAL",
                                       delta_color=idx_color,
                                       help="Normalizza la posizione attuale rispetto al range degli ultimi 2 anni. >80% = Estremo Buy Storico; <20% = Estremo Sell Storico.")

                            # CP/OI Weight
                            cpoi_status = "⚠️ EXTREME" if abs(cpoi_latest) > 15 else "STABLE"
                            r1c4.metric("Comm/OI Weight", f"{cpoi_latest:.1f}%", 
                                       delta=cpoi_status, 
                                       delta_color="off",
                                       help="Percentuale di Open Interest controllata dagli Smart Money. Valori oltre il ±15-20% indicano un mercato saturo (Overcrowded) pronto a invertire.")

                            st.markdown("---")
                            r2c1, r2c2, r2c3 = st.columns([2, 1, 1])
                            
                            # Open Interest
                            oi_delta_label = f"{int(d1w_oi):,} (New Liquidity)" if d1w_oi > 0 else f"{int(d1w_oi):,} (Liquidation)"
                            r2c1.metric("Open Interest Totale (OI)", f"{int(latest[col_mapping['open_interest']]):,}", 
                                       oi_delta_label, 
                                       delta_color="normal",
                                       help="Rappresenta il numero totale di contratti attivi. Se cresce conferma il trend; se cala indica che i trader stanno chiudendo le posizioni (mancanza di forza).")
                            
                            # Delta Mensili
                            r2c2.metric("Delta 4W Comm", f"{int(d4w_comm):,}", "Contratti", 
                                       delta_color="normal",
                                       help="Variazione netta dei contratti degli Smart Money nell'ultimo mese. Mostra se stanno accumulando o distribuendo in modo costante.")
                            
                            r2c3.metric("Delta 4W Specs", f"{int(d4w_spec):,}", "Contratti", 
                                       delta_color="inverse",
                                       help="Variazione netta dei contratti dei Large Speculators nell'ultimo mese. Indica se i fondi stanno rincorrendo il prezzo o perdendo interesse.")

                            # 3. ISTOGRAMMA DELTA VOLUMI (Flussi di Liquidità)
                            st.markdown("##### 🌊 Analisi della Liquidità (Delta Contratti Mensile)")
                            
                            # Prepariamo i dati per l'istogramma degli ultimi 12 report
                            df_delta = df_cot.tail(12).copy()
                            df_delta['Comm_Delta'] = df_delta['Net_Comm'].diff()
                            df_delta['Spec_Delta'] = df_delta['Net_NonComm'].diff()
                            
                            fig_delta = go.Figure()
                            fig_delta.add_trace(go.Bar(
                                x=df_delta['date'], y=df_delta['Comm_Delta'],
                                name='Flusso Smart Money', marker_color='#2ecc71', opacity=0.7
                            ))
                            fig_delta.add_trace(go.Bar(
                                x=df_delta['date'], y=df_delta['Spec_Delta'],
                                name='Flusso Hedge Funds', marker_color='#e74c3c', opacity=0.7
                            ))
                            
                            fig_delta.update_layout(
                                barmode='group', height=300, template="plotly_dark",
                                margin=dict(l=0, r=0, t=20, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_delta, use_container_width=True)

                            # --- 5. GRAFICO DI CONFLUENZA: PREZZO VS SMART MONEY ---
                            st.markdown("##### 📈 Divergenza Prezzo vs. Smart Money (Commercials)")
                            
                            # Allineamento dati: Prendiamo i prezzi settimanali per matchare il COT
                            df_price_weekly = df_close.resample('W-TUE').last().dropna()
                            # Uniamo i dati su data comune
                            df_merged = pd.merge(df_price_weekly, df_cot[['date', 'Net_Comm']], left_index=True, right_on='date', how='inner')
                            
                            if not df_merged.empty:
                                fig_div = go.Figure()
                                
                                # Asse Y Sinistro: Prezzo
                                fig_div.add_trace(go.Scatter(
                                    x=df_merged['date'], y=df_merged['Close'],
                                    name='Prezzo Asset', line=dict(color='#3498db', width=2),
                                    yaxis="y1"
                                ))
                                
                                # Asse Y Destro: Net Commercials
                                fig_div.add_trace(go.Scatter(
                                    x=df_merged['date'], y=df_merged['Net_Comm'],
                                    name='Net Comm (Smart Money)', line=dict(color='#2ecc71', width=3),
                                    yaxis="y2", fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.1)'
                                ))
                                
                                fig_div.update_layout(
                                    template="plotly_dark", height=500,
                                    hovermode="x unified",
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    yaxis=dict(title="Prezzo Asset", side="left", showgrid=False),
                                    yaxis2=dict(title="Posizionamento Smart Money", side="right", overlaying="y", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                                )
                                st.plotly_chart(fig_div, use_container_width=True)
                                st.caption("🔍 **Analisi Divergenze:** Se il Prezzo (Blu) scende ma lo Smart Money (Verde) sale, siamo in accumulazione (Bullish). Se il Prezzo sale ma lo Smart Money scende, siamo in distribuzione (Bearish).")
                            else:
                                st.info("Dati insufficienti per generare il grafico di divergenza prezzo/COT.")

                            # 4. TABELLA ANOMALIE
                            with st.expander("🔍 Ispezione Anomalie Dati (Raw Data Analysis)"):
                                # Preparazione dati
                                df_inspec = df_cot.tail(15)[['date', 'Net_Comm', 'Net_NonComm', col_mapping['open_interest']]].copy()
                                df_inspec = df_inspec.rename(columns={
                                    'Net_Comm': 'Smart Money (Net)',
                                    'Net_NonComm': 'Hedge Funds (Net)',
                                    col_mapping['open_interest']: 'Open Interest'
                                })
                                df_inspec['Variazione Comm %'] = df_inspec['Smart Money (Net)'].pct_change() * 100
                                
                                # Ordinamento decrescente
                                df_inspec_sorted = df_inspec.sort_values('date', ascending=False)
                                
                                # Applichiamo formattazione estetica senza rompere pd.NA/inf
                                styled_df = df_inspec_sorted.style.format({
                                    'Smart Money (Net)': "{:,.0f}",
                                    'Hedge Funds (Net)': "{:,.0f}",
                                    'Open Interest': "{:,.0f}",
                                    'Variazione Comm %': "{:+.1f}%"
                                }, na_rep="-").background_gradient(cmap='RdYlGn', subset=['Variazione Comm %'])
                                
                                st.dataframe(styled_df, use_container_width=True)
                                st.caption("💡 **Tip:** Cerca variazioni superiori al ±10% per identificare entrate/uscite aggressive degli istituzionali.")
                            
                            st.caption("💡 **Guida alla Lettura:** I **Commercials (Verde)** sono i produttori e operatori del settore fisico; agiscono spesso come *Smart Money contrarian* (es. comprano quando i prezzi crollano). I **Large Specs (Rosso)** sono i grandi fondi che seguono il trend. Quando queste due linee raggiungono divergenze estreme, indicano probabili inversioni di mercato a medio termine.")
                        else:
                            st.warning("Dati storici COT non disponibili attualmente per questo contratto.")
                    else:
                        st.error("Impossibile connettersi al database governativo CFTC in questo momento.")
                except Exception as e:
                    st.error(f"Errore durante l'elaborazione dei dati COT: {e}")
        else:
            st.info(f"Il motore Auto-Discovery non ha trovato un codice CFTC associato a {base_ticker}. Il report COT è disponibile per i principali Futures fisici e valutari.")

    with tab_cal:
        st.markdown("### 📆 Global Economic Calendar (Real-Time)")
        st.write("Monitora i dati macroeconomici (CPI, NFP, Riunioni FED/BCE) in tempo reale. Filtra per importanza.")
        
        # Widget TradingView istituzionale
        calendar_html = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
          {
          "colorTheme": "dark",
          "isTransparent": true,
          "width": "100%",
          "height": "600",
          "locale": "it",
          "importanceFilter": "-1,0,1",
          "currencyFilter": "USD,EUR,GBP,JPY,AUD,CAD,CHF,CNY"
        }
          </script>
        </div>
        """
        components.html(calendar_html, height=620)

# --- NAVIGAZIONE ---
st.sidebar.markdown("## 🔑 API KEYS")
st.session_state.alpaca_api_key = st.sidebar.text_input("Alpaca API Key ID", value=st.session_state.get("alpaca_api_key", "PKQVMHYR25JUXQVLTEEBEKVIMV"), type="password")
st.session_state.alpaca_secret_key = st.sidebar.text_input("Alpaca Secret Key", value=st.session_state.get("alpaca_secret_key", "EeZLG3n9NN7uxPCjVSZkQEScgBDjrVE4jiGeabTngeK7"), type="password")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📁 DATABASE LOCALE")
uploaded_file = st.sidebar.file_uploader("Carica file CSV (Database Locale)", type=['csv'])
if uploaded_file is not None:
    file_path = os.path.join(LOCAL_DB_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File {uploaded_file.name} salvato permanentemente nel Database Locale.")
st.sidebar.markdown("---")

st.sidebar.markdown("## 🧭 SISTEMA")
menu = st.sidebar.radio("Seleziona Vista:", ["🌍 WAR ROOM (Dashboard Globale)", "🏟️ DASHBOARD SINGOLA", "🔥 SCANNER HOT TICKERS", "🔙 BACKTESTING STRATEGIA", "🛠️ STRATEGY BUILDER", "🏛️ BLOOMBERG TERMINAL (Inst.)", "🔍 GLOBAL SCANNER (Alpha)", "🕸️ Macro & Correlazione", "📅 SEASONALITY & CALENDAR"])

with st.sidebar.expander("🤖 SEGUGIO DIGITALE (Estrai Ticker)", expanded=False):
    st.write("Estrai automaticamente i ticker da Testo o da Link Web.")
    
    metodo = st.radio("Scegli la fonte:", ["🌐 Inserisci Link (URL)", "📝 Copia-Incolla Testo", "📚 Download S&P 500"])
    
    # Dizionario di parole comuni da ignorare per evitare falsi positivi
    blacklist = {"THE", "AND", "FOR", "INC", "CORP", "LTD", "NYSE", "NASDAQ", "USD", "EUR", "NEW", "ETF", "LLC", "PLC", "SPA", "HOLDING", "GROUP", "INDEX", "MARKET", "STOCK", "SHARE", "BUY", "SELL"}

    if metodo == "🌐 Inserisci Link (URL)":
        url_input = st.text_input("Incolla l'URL dell'articolo o sito web:")
        if st.button("Cerca Ticker nel Link 🕵️‍♂️"):
            if url_input:
                try:
                    # Camuffiamo la richiesta per sembrare un normale browser Chrome su Windows
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    }
                    
                    response = requests.get(url_input, headers=headers, timeout=10)
                    
                    # Controlliamo se il sito ci ha bloccato (es. Errore 403 Forbidden)
                    if response.status_code == 403:
                        st.error("Accesso Negato: Questo sito usa scudi anti-bot severi (es. Cloudflare). Usa il metodo 'Copia-Incolla' per aggirarlo.")
                    elif response.status_code == 200:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Estraiamo tutto il testo visibile
                        testo_estratto = soup.get_text(separator=' ')
                        
                        # Cerchiamo parole maiuscole da 1 a 5 lettere
                        estratti = re.findall(r'\b[A-Z]{1,5}\b', testo_estratto)
                        tickers_puliti = list(set([t for t in estratti if t not in blacklist and len(t) > 1]))
                        
                        if tickers_puliti:
                            st.success(f"Trovati {len(tickers_puliti)} potenziali Ticker!")
                            st.code(", ".join(tickers_puliti))
                            st.info("👆 Copia questa lista e incollala in 'Custom List'")
                        else:
                            st.warning("Nessun ticker rilevato nel testo del sito.")
                    else:
                        st.warning(f"Il server ha risposto con codice errore: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Impossibile leggere il link. Errore: {e}")

    elif metodo == "📝 Copia-Incolla Testo":
        testo_sporco = st.text_area("Incolla il testo (es. da Finviz o PDF):")
        if st.button("Estrai dal Testo ✂️"):
            if testo_sporco:
                estratti = re.findall(r'\b[A-Z]{1,5}\b', testo_sporco)
                tickers_puliti = list(set([t for t in estratti if t not in blacklist and len(t) > 1]))
                
                if tickers_puliti:
                    st.success(f"Trovati {len(tickers_puliti)} ticker!")
                    st.code(", ".join(tickers_puliti))
                else:
                    st.warning("Nessun ticker trovato.")

    elif metodo == "📚 Download S&P 500":
        st.write("Scarica la lista Ufficiale Live da Wikipedia (Metodo Nativo Anti-Crash).")
        if st.button("Estrai S&P 500 📥"):
            try:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                
                # Camuffamento per bypassare i blocchi
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                risposta = requests.get(url, headers=headers, timeout=10)
                
                if risposta.status_code == 200:
                    # BYPASS LXML: Usiamo BeautifulSoup con il parser nativo di Python
                    soup = BeautifulSoup(risposta.text, 'html.parser')
                    
                    # Cerchiamo la tabella specifica con l'ID di Wikipedia
                    table = soup.find('table', {'id': 'constituents'})
                    
                    tickers_sp = []
                    # Iteriamo sulle righe saltando l'intestazione [1:]
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if cols:
                            # Il ticker è nella prima colonna, togliamo spazi e sistemiamo il punto per Yahoo
                            ticker = cols[0].text.strip().replace(".", "-")
                            tickers_sp.append(ticker)
                    
                    if tickers_sp:
                        st.success(f"Scaricati con successo {len(tickers_sp)} ticker dell'S&P 500!")
                        st.code(", ".join(tickers_sp))
                        st.info("👆 Copia questa lista e incollala in 'Custom List'")
                    else:
                        st.error("Errore: Impossibile trovare i dati nella tabella. La struttura del sito potrebbe essere cambiata.")
                else:
                    st.error(f"Wikipedia ha rifiutato la connessione. Codice Errore: {risposta.status_code}")
                    
            except Exception as e:
                st.error(f"Errore critico durante l'estrazione: {e}")

# --- REFRESH CONFIG ---
# Dashboard: refresh ogni 1 minuto (60000 ms)
# Scanner: refresh ogni 5 minuti (300000 ms) per evitare Rate Limit
if menu == "🏟️ DASHBOARD SINGOLA":
    st_autorefresh(interval=60000, key="sentinel_dash_refresh")
elif menu == "🔥 SCANNER HOT TICKERS":
    st_autorefresh(interval=300000, key="sentinel_scan_refresh")
# --------------------------

today = datetime.now()
today_str_format = today.strftime('%Y-%m-%d') # Per la cache

if menu == "🏟️ DASHBOARD SINGOLA":
    if 'ticker_list' not in st.session_state:
        st.session_state.ticker_list = ["NDX", "SPX", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "MSTR"]
    
    new_asset = st.sidebar.text_input("➕ CARICA TICKER", "").upper().strip()
    if new_asset and new_asset not in st.session_state.ticker_list:
        st.session_state.ticker_list.insert(0, new_asset)
        st.rerun()

    asset = st.sidebar.selectbox("SELEZIONA ASSET", st.session_state.ticker_list)
    t_map = {"SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT"}
    current_ticker = t_map.get(asset, asset)

    default_gran = 1.0
    if "NDX" in asset: default_gran = 25.0
    elif "SPX" in asset: default_gran = 10.0
    elif any(x in asset for x in ["NVDA", "MSTR", "SMCI"]): default_gran = 5.0
    
    ticker_obj = yf.Ticker(current_ticker)
    h = ticker_obj.history(period='1d')
    if h.empty: st.stop()
    spot = h['Close'].iloc[-1]

    try:
        available_dates = ticker_obj.options
    except Exception as e:
        st.error("⚠️ Yahoo Finance ti ha temporaneamente bloccato per troppe richieste (Rate Limit). Cambia rete/IP o attendi 10 minuti prima di riprovare.")
        st.stop()

    all_dates_info = []
    for d in available_dates:
        try:
            dt_obj = datetime.strptime(d, '%Y-%m-%d')
            dte = (dt_obj - today).days + 1
            if dte >= 0:
                all_dates_info.append({"label": f"{dte} DTE | {d}", "date": d, "dte": dte})
        except: continue
    
    date_labels = [x['label'] for x in all_dates_info]
    selected_dte_labels = st.sidebar.multiselect("SCADENZE", date_labels, default=date_labels[:1])
    metric = st.sidebar.radio("METRICA", ["Gamma", "Vanna", "Charm", "Vega", "Theta"])
    gran = st.sidebar.select_slider("GRANULARITÀ", options=[0.5, 1, 2.5, 5, 10, 25, 50, 100], value=default_gran)
    
    # Determina lo zoom predefinito in base all'asset
    ticker_up = current_ticker.upper().strip()
    is_index_or_etf = ticker_up.startswith('^') or ticker_up in ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV']
    
    # Pesatura dello zoom: 3% per indici/etf, 10% per azioni
    default_zoom_level = 3.0 if is_index_or_etf else 10.0
    
    # Configurazione Slider: Min 1.0, Max 20.0, Default dinamico
    zoom_val = st.sidebar.slider("Zoom Predefinito (%)", 1.0, 20.0, default_zoom_level)

    if selected_dte_labels:
        target_dates = [label.split('| ')[1] for label in selected_dte_labels]
        raw_data = fetch_data(current_ticker, target_dates)
        
        if not raw_data.empty:
            raw_data = raw_data[(raw_data['volume'] > 0) | (raw_data['openInterest'] > 0)].copy()
            raw_data['dte_years'] = raw_data['exp'].apply(get_precise_dte)
            
            # Recupero Dividend Yield dinamico (fallback a 0.0 se non disponibile)
            try:
                div_yield = ticker_obj.info.get('dividendYield', 0.0)
                if div_yield is None: div_yield = 0.0
            except:
                div_yield = 0.0

            # Funzione sicura di ricalcolo IV sul Mid-Price
            def refine_iv_vectorized(row):
                try:
                    b, a = row.get('bid', 0), row.get('ask', 0)
                    if b > 0 and a > 0:
                        mid = (b + a) / 2
                        K_val, T_val, opt_type = row['strike'], max(row['dte_years'], 0.00005), row['type']
                        def bs_price(x_iv):
                            d1_tmp = (np.log(spot/K_val) + (DYNAMIC_R - div_yield + 0.5 * x_iv**2) * T_val) / (x_iv * np.sqrt(T_val))
                            d2_tmp = d1_tmp - x_iv * np.sqrt(T_val)
                            if opt_type == 'call': 
                                return spot * np.exp(-div_yield*T_val) * norm.cdf(d1_tmp) - K_val * np.exp(-DYNAMIC_R*T_val) * norm.cdf(d2_tmp)
                            else: 
                                return K_val * np.exp(-DYNAMIC_R*T_val) * norm.cdf(-d2_tmp) - spot * np.exp(-div_yield*T_val) * norm.cdf(-d1_tmp)
                        return brentq(lambda x: bs_price(x) - mid, 0.001, 5.0)
                except: pass
                return row['impliedVolatility']

            # Applica il raffinamento alla colonna impliedVolatility originale
            raw_data['impliedVolatility'] = raw_data.apply(refine_iv_vectorized, axis=1)
            mean_iv = raw_data['impliedVolatility'].mean()
            dte_ref = (datetime.strptime(target_dates[0], '%Y-%m-%d') - today).days + 0.5
            
            if 'prev_iv' not in st.session_state:
                st.session_state.prev_iv = mean_iv
            iv_change = mean_iv - st.session_state.prev_iv
            st.session_state.prev_iv = mean_iv

            # --- MODIFICA ASIMMETRICA DS (SKEW DRIVEN) ---
            # 1. Estrazione IV Specifica (Skew) su 0DTE/1DTE
            try:
                skew_date = available_dates[0]
                skew_data = fetch_data(current_ticker, [skew_date])
                
                if not skew_data.empty:
                    # Target: 2% OTM (~25 Delta proxy)
                    c_target = spot * 1.02
                    p_target = spot * 0.98
                    
                    c_skew = skew_data[skew_data['type'] == 'call']
                    p_skew = skew_data[skew_data['type'] == 'put']
                    
                    # Trova lo strike più vicino al target
                    c_iv = c_skew.iloc[(c_skew['strike'] - c_target).abs().argmin()]['impliedVolatility'] if not c_skew.empty else mean_iv
                    p_iv = p_skew.iloc[(p_skew['strike'] - p_target).abs().argmin()]['impliedVolatility'] if not p_skew.empty else mean_iv
                else:
                    c_iv = p_iv = mean_iv
            except:
                c_iv = p_iv = mean_iv

            # 2. Calcolo Fixed 1-Day Move (1/252)
            one_day_factor = np.sqrt(1/252)
            
            # 3. Creazione delle 4 Linee Asimmetriche
            sd1_up = spot * (1 + (c_iv * one_day_factor))
            sd2_up = spot * (1 + (c_iv * 2 * one_day_factor))
            sd1_down = spot * (1 - (p_iv * one_day_factor))
            sd2_down = spot * (1 - (p_iv * 2 * one_day_factor))
            
            skew_factor = p_iv / c_iv if c_iv > 0 else 1.0
            # ---------------------------------------------

            # CALCOLO 0-GAMMA ORIGINALE
            try: z_gamma = brentq(calculate_gex_at_price, spot * 0.85, spot * 1.15, args=(raw_data, DYNAMIC_R, div_yield))
            except: z_gamma = spot 

            # CALCOLO 0-GAMMA DINAMICO (SOLO VOLUMI)
            try: z_gamma_dyn = brentq(calculate_0g_dynamic, spot * 0.85, spot * 1.15, args=(raw_data, DYNAMIC_R, div_yield))
            except: z_gamma_dyn = spot

            df = get_greeks_pro(raw_data, spot, r=DYNAMIC_R, q=div_yield)
            
            # --- LOGICA DI AGGREGAZIONE MATEMATICA (Binning Dinamico) ---
            # Usiamo floor division per forzare ogni contratto nel proprio bin matematico
            pivot_series = (df['strike'] // gran) * gran
            
            # Aggregazione Totale su Pivot
            # --- PROTEZIONE AGGREGAZIONE FAIL-SAFE ---
            if not df.empty:
                # Costruzione dinamica del dizionario di aggregazione per evitare KeyError
                target_cols = ['Gamma', 'Vanna', 'Vomma', 'Charm', 'Speed', 'Vega', 'Theta', 'DEX']
                actual_agg = {col: 'sum' for col in target_cols if col in df.columns}
                
                if actual_agg:
                    agg = df.groupby(pivot_series).agg(actual_agg).reset_index()
                else:
                    agg = pd.DataFrame(columns=[pivot_series] + target_cols)
            else:
                st.warning("⚠️ Dati insufficienti per questa scadenza (Dati sporchi o assenti su Yahoo Finance).")
                agg = pd.DataFrame(columns=[pivot_series, 'Gamma', 'Vanna', 'Vomma', 'Charm', 'Speed', 'Vega', 'Theta', 'DEX'])
            
            # Rinomina la colonna pivot
            if 'strike' not in agg.columns:
                agg.rename(columns={'index': 'strike', agg.columns[0]: 'strike'}, inplace=True)
            
            lo, hi = spot * (1 - zoom_val/100), spot * (1 + zoom_val/100)
            visible_agg = agg[(agg['strike'] >= lo) & (agg['strike'] <= hi)]
            
            # Calcolo Muri basato sui dati aggregati
            if not agg.empty:
                c_wall = agg.loc[agg['Gamma'].idxmax(), 'strike']
                p_wall = agg.loc[agg['Gamma'].idxmin(), 'strike']
                v_trigger = agg.loc[agg['Vanna'].abs().idxmax(), 'strike']
            else:
                c_wall = p_wall = v_trigger = spot

            st.subheader(f"🏟️ {asset} Quant Terminal | Spot: {spot:.2f}")

            net_gamma, net_vanna, net_charm = agg['Gamma'].sum(), agg['Vanna'].sum(), agg['Charm'].sum()
            direction = "NEUTRALE"; bias_color = "gray"
            
            if net_gamma < 0 and net_vanna < 0:
                direction = "☢️ PERICOLO ESTREMO (Crash Risk / Short Gamma & Vanna)"; bias_color = "#8B0000"
            elif net_gamma < 0:
                direction = "🔴 SHORT GAMMA BIAS (Espansione Volatilità)"; bias_color = "#FF4136"
            elif spot < z_gamma:
                direction = "🟠 PRESSIONE SOTTO ZERO GAMMA (Vulnerabilità)"; bias_color = "#FF851B"
            elif net_gamma > 0 and net_charm < 0:
                direction = "🚀 BULLISH FLOW (Charm Support / Long Gamma)"; bias_color = "#2ECC40"
            else:
                direction = "🔵 LONG GAMMA / STABILITÀ (Contrazione Volatilità)"; bias_color = "#0074D9"
            
            st.markdown(f"### 📊 Real-Time Metric Regime")
            net_vomma = agg['Vomma'].sum() if 'Vomma' in agg.columns else 0
            net_speed = agg['Speed'].sum() if 'Speed' in agg.columns else 0
            c_reg1, c_reg2, c_reg3, c_reg4, c_reg5, c_reg6 = st.columns(6)
            c_reg1.metric("Net Gamma", f"{net_gamma:,.0f}", delta=f"{'LONG' if net_gamma > 0 else 'SHORT'}")
            c_reg2.metric("Net Vanna", f"{net_vanna:,.0f}", delta=f"{'STABLE' if net_vanna > 0 else 'UNSTABLE'}")
            c_reg3.metric("Net Charm", f"{net_charm:,.0f}", delta=f"{'SUPPORT' if net_charm < 0 else 'DECAY'}")
            c_reg4.metric("SKEW FACTOR (P/C)", f"{skew_factor:.2f}x")
            c_reg5.metric("Net Vomma", f"{net_vomma:,.0f}", help="Risk of IV explosion")
            c_reg6.metric("Net Speed", f"{net_speed:,.0f}", help="Gamma Acceleration / Squeeze Risk")

            st.markdown(f"""
                <div style='background-color:{bias_color}; padding:15px; border-radius:10px; text-align:center; margin-top: 10px; margin-bottom: 25px;'>
                    <b style='color:white; font-size:24px;'>MARKET BIAS: {direction}</b>
                </div>
                """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CALL WALL", f"{c_wall:.0f}")
            m2.metric("ZERO GAMMA (STA/DYN)", f"{z_gamma:.0f} / {z_gamma_dyn:.0f}")
            m3.metric("PUT WALL", f"{p_wall:.0f}")
            m4.metric("EXPECTED 1SD", f"±{spot*one_day_factor*mean_iv:.2f}")

            st.markdown("---")
            
            def get_dist(target, spot):
                d = ((target - spot) / spot) * 100
                color = "#00FF41" if d > 0 else "#FF4136"
                return f"<span style='color:{color};'>{d:+.2f}%</span>"

            sd_up_pct = ((sd1_up - spot) / spot) * 100
            sd_dn_pct = ((sd1_down - spot) / spot) * 100

            st.markdown(f"""
                <div style='background-color:rgba(30, 30, 30, 0.8); padding:10px; border-radius:5px; border: 1px solid #444; margin-bottom: 20px; display: flex; justify-content: space-around;'>
                    <div><b>📍 DIST. CW:</b> {get_dist(c_wall, spot)}</div>
                    <div><b>📍 DIST. 0G-DYN:</b> {get_dist(z_gamma_dyn, spot)}</div>
                    <div><b>📍 DIST. VT:</b> {get_dist(v_trigger, spot)}</div>
                    <div><b>📍 DIST. PW:</b> {get_dist(p_wall, spot)}</div>
                    <div><b>📍 1SD UP/DN:</b> <span style='color:#FFA500;'>{sd_up_pct:+.2f}% / {sd_dn_pct:+.2f}%</span></div>
                </div>
                """, unsafe_allow_html=True)

            # --- INIZIO HUD QUANTISTICO (VERSIONE BILANCIATA PRO) ---
            with st.expander("🔍 🧠 HUD QUANTISTICO: SENTIMENT & CONFLUENZA GREEKS (Clicca per espandere)"):
                whale_data = get_whale_intelligence(current_ticker)
                
                if whale_data.get("Error"):
                    st.warning("Whale Data: N/A - Switch to ETF for analysis")
                else:
                    is_idx = whale_data.get('is_index', False)
                    proxy = whale_data.get('proxy_etf', '')
                    ratio = whale_data.get('ratio', 1.0)
                    
                    idx_note = f"<br><span style='font-size:0.8em; color:#aaa;'>*Price converted from {proxy} (Ratio: {ratio:.2f})</span>" if is_idx else ""
                    
                    bias = whale_data.get('Whale_Bias', 'NEUTRAL')
                    bias_icon = "🟢" if bias == "BULLISH" else ("🔴" if bias == "BEARISH" else "⚪")
                    w_price = whale_data['Whale_Price']
                    intensity = whale_data['Whale_Intensity']
                    
                    if intensity >= 70:
                        if spot > w_price and bias == "BULLISH":
                            status_html = "<span style='color:#2ECC40; font-weight:bold;'>🛡️ STRONG ACCUMULATION</span>"
                        elif spot < w_price and bias == "BEARISH":
                            status_html = "<span style='color:#FF4136; font-weight:bold;'>⚠️ HEAVY DISTRIBUTION</span>"
                        else:
                            status_html = "<span style='color:#2ECC40; font-weight:bold;'>🔥 CONFLUENZA ISTITUZIONALE</span>"
                    elif intensity <= 30:
                        status_html = "<span style='color:#0074D9; font-weight:bold;'>💨 RETAIL DRIVEN</span>"
                    else:
                        status_html = "<span style='color:#FFDC00; font-weight:bold;'>⚖️ ZONA MISTA</span>"

                    st.markdown(f"""
                    <div style='background-color:rgba(20, 20, 20, 0.8); padding:15px; border-radius:5px; border: 1px solid #555; display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                        <div><b style='color:DeepSkyBlue; font-size:1.1em;'>🐳 W-LVL:</b> <span style='font-size:1.2em; font-weight:bold;'>${w_price:.2f}</span> <span style='font-size:1.1em;'>{bias_icon} {bias}</span>{idx_note}</div>
                        <div><b>Score:</b> {intensity}%</div>
                        <div>{status_html}</div>
                    </div>
                    <div style='text-align: right; font-size: 0.8em; color: #888; margin-top: -10px; margin-bottom: 15px;'>Dati triangolati da: FINRA Weekly + IEX Real-Time Tape</div>
                    """, unsafe_allow_html=True)
                    
                import math 
                
                # 1. LOGICA MATEMATICA ORIGINALE (4, 3, 3) - INVARIATA
                pos_score = 4 if (spot > z_gamma and spot > z_gamma_dyn) else (-4 if (spot < z_gamma and spot < z_gamma_dyn) else 0)
                vanna_score = 3 if net_vanna > 0 else -3
                charm_score = 3 if net_charm < 0 else -3
                total_ss = pos_score + vanna_score + charm_score
                
                hud_color = "#2ECC40" if total_ss >= 5 else ("#FF4136" if total_ss <= -5 else "#FFDC00")
                
                # 2. NUOVO BILANCIAMENTO "ANTI-OSCURAMENTO"
                # Usiamo un moltiplicatore più dolce (800 invece di 5000) e mettiamo un tetto (CAP)
                p_dist_raw = (abs(spot - z_gamma_dyn) / spot) * 800
                p_intensity = min(15, p_dist_raw) # Il prezzo non può pesare più di "15 punti" nella torta totale
                
                # Le Greche rimangono logaritmiche per gestire i milioni/miliardi senza esplodere
                v_intensity = math.log10(abs(net_vanna) + 1)
                c_intensity = math.log10(abs(net_charm) + 1)
                
                total_intensity = p_intensity + v_intensity + c_intensity
                
                if total_intensity > 0:
                    p_w = int((p_intensity / total_intensity) * 100)
                    v_w = int((v_intensity / total_intensity) * 100)
                    c_w = 100 - p_w - v_w 
                else:
                    p_w, v_w, c_w = 40, 30, 30

                # 3. TESTI COMPLETI ORIGINALI (Invariati)
                pos_text = "🟢 SOPRA entrambi 0-G (Pieno controllo acquirenti)" if pos_score == 4 else ("🔴 SOTTO entrambi 0-G (Pieno controllo venditori)" if pos_score == -4 else "🟡 Divergenza OI vs Volumi (Fase incerta)")
                vanna_text = "🟢 Stabile (Nessuno Squeeze Imminente)" if vanna_score == 3 else "🔴 Pericolo Squeeze (Dealer costretti a comprare/vendere in corsa)"
                charm_text = "🔵 Supporto Passivo (Il tempo aiuta i Long)" if charm_score == 3 else "🔴 Flusso in Uscita (Il tempo pesa sul prezzo)"

                # 4. LOGICA SEGNALI E RISCHIO (Invariata)
                abs_ss = abs(total_ss)
                if total_ss >= 8:
                    res_sig, res_strat, res_target = "🚀 STRONG BUY", "Long Call / Bull Call Spread", "Call Wall"
                elif total_ss <= -8:
                    res_sig, res_strat, res_target = "☢️ STRONG SELL", "Long Put / Bear Put Spread", "Put Wall"
                elif total_ss >= 4:
                    res_sig, res_strat, res_target = "🟢 BUY ON DIP", "Bull Put Spread (Credit)", "+1 SD Line"
                elif total_ss <= -4:
                    res_sig, res_strat, res_target = "🔴 SELL ON RALLY", "Bear Call Spread (Credit)", "-1 SD Line"
                else:
                    res_sig, res_strat, res_target = "⚖️ NEUTRAL", "Wait / Iron Condor", "Gamma Flip Zone"

                res_risk = "2.0% (ALTO)" if abs_ss >= 8 else ("1.0% (MEDIO)" if abs_ss >= 4 else "0.0% (NO TRADE)")
                res_rr = "1:3+" if abs_ss >= 8 else ("1:2" if abs_ss >= 4 else "N/A")

                # 5. INTERFACCIA (Testi lunghi + Percentuali dinamiche)
                st.markdown(f"""
<div style='background-color:rgba(15,15,15,0.9); padding:20px; border: 2px solid {hud_color}; border-radius:10px;'>
<h2 style='text-align:center; color:{hud_color}; margin-top:0;'>SENTIMENT SCORE: {total_ss} / 10</h2>
<h3 style='text-align:center; color:white; margin-bottom:15px;'>AZIONE: <span style='color:{hud_color};'>{res_sig}</span></h3>
<hr style='border-color:#333;'>
<div style='display:flex; justify-content:space-between; text-align:center;'>
<div style='width:30%;'>
<h4 style='color:white;'>⚡ Forza Prezzo ({p_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Confluenza 0G Statico / Dinamico</i></p>
<b style='font-size:13px; color:white;'>{pos_text}</b>
</div>
<div style='width:30%;'>
<h4 style='color:white;'>🌪️ Forza Vanna ({v_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Rischio accelerazione Volatilità</i></p>
<b style='font-size:13px; color:white;'>{vanna_text}</b>
</div>
<div style='width:30%;'>
<h4 style='color:white;'>⏳ Forza Charm ({c_w}%)</h4>
<p style='color:lightgray; font-size:11px;'><i>Supporto/Pressione legati al Tempo</i></p>
<b style='font-size:13px; color:white;'>{charm_text}</b>
</div>
</div>
<hr style='border-color:#333; margin-top:20px;'>
<div style='display:flex; justify-content:space-between; text-align:center; background:rgba(255,255,255,0.05); padding:15px; border-radius:8px;'>
<div style='width:33%;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>STRATEGIA</p>
<b style='color:white;'>{res_strat}</b>
</div>
<div style='width:33%; border-left:1px solid #444; border-right:1px solid #444;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>RISCHIO CONSIGLIATO</p>
<b style='color:white;'>{res_risk}</b>
</div>
<div style='width:33%;'>
<p style='color:#FFDC00; margin:0; font-size:12px; font-weight:bold;'>TARGET / R:R</p>
<b style='color:white;'>{res_target} ({res_rr})</b>
</div>
</div>
</div>
""", unsafe_allow_html=True)
            # --- FINE NUOVO HUD ---

            # --- CALCOLO STORICO E CURTOSI ---
            try:
                hist_1m = ticker_obj.history(period='1mo')
                hv_20d = hist_1m['Close'].pct_change().std() * np.sqrt(252) * 100 if not hist_1m.empty else 0.0
                
                iv_atm = raw_data.iloc[(raw_data['strike'] - spot).abs().argsort()[:1]]['impliedVolatility'].values[0]
                iv_put_tail = raw_data[raw_data['strike'] <= spot * 0.9]['impliedVolatility'].mean()
                iv_call_tail = raw_data[raw_data['strike'] >= spot * 1.1]['impliedVolatility'].mean()
                # Se mancano code, fallback su ATM
                iv_put_tail = iv_put_tail if pd.notna(iv_put_tail) else iv_atm
                iv_call_tail = iv_call_tail if pd.notna(iv_call_tail) else iv_atm
                
                kurtosis_ratio = (iv_put_tail + iv_call_tail) / (2 * iv_atm)
                kurt_pct = min(max((kurtosis_ratio - 1.0) * 200, 0), 100) # Normalizza 0-100%
                
                kurt_color = "#2ecc71" if kurt_pct < 30 else ("#f1c40f" if kurt_pct < 60 else "#e74c3c")
            except:
                hv_20d, kurt_pct, kurt_color = 0.0, 0, "gray"

            # --- UI LAYOUT AGGIORNATO ---
            col_view, col_vol = st.columns([1.8, 1.2])
            with col_view:
                view_mode = st.radio("👁️ VISTA GRAFICO:", [
                    "📊 Standard", 
                    "🌪️ Vanna Overlay", 
                    "⚡ DEX Overlay", 
                    "📉 Puro DEX"
                ], horizontal=True)
                
            with col_vol:
                v1, v2, v3 = st.columns([1.2, 1.2, 1])
                with v1:
                    st.metric("📈 DYN IV", f"{mean_iv*100:.1f}%", delta=f"{iv_change*100:.1f}%", delta_color="inverse")
                with v2:
                    st.metric("🏛️ HIST VOL", f"{hv_20d:.1f}%", help="Volatilità Storica 20gg")
                with v3:
                    st.markdown(f"""
                        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:-5px;">
                            <span style="font-size:10px; color:#aaa;">TAIL RISK</span>
                            <div style="width: 45px; height: 45px; border-radius: 50%; border: 4px solid {kurt_color}; display:flex; align-items:center; justify-content:center; background-color:rgba(0,0,0,0.2);">
                                <span style="font-size:12px; font-weight:bold; color:{kurt_color};">{int(kurt_pct)}%</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            fig = go.Figure()

            if view_mode == "📊 Standard":
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg[metric], 
                    orientation='h', 
                    marker=dict(color=['#00FF41' if x >= 0 else '#FF4136' for x in visible_agg[metric]]),
                    name=metric
                ))
                xaxis_title = f"Net {metric} Exposure"
                
            elif view_mode == "🌪️ Vanna Overlay":
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg['Gamma'], 
                    orientation='h', 
                    marker=dict(color='rgba(100, 100, 100, 0.3)', line=dict(width=0)), 
                    name="Gamma (Background)",
                    xaxis="x1"
                ))
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], 
                    x=visible_agg['Vanna'], 
                    orientation='h', 
                    marker=dict(
                        color=['#00FFFF' if x >= 0 else '#FF00FF' for x in visible_agg['Vanna']], 
                        line=dict(color='white', width=1)
                    ),
                    width=gran * 0.4, 
                    name="Vanna (Focus)",
                    xaxis="x2"
                ))
                fig.update_layout(
                    xaxis=dict(title="Gamma Exposure", side="bottom", showgrid=False),
                    xaxis2=dict(title="Vanna Exposure (Scaled)", side="top", overlaying="x", showgrid=False, zerolinecolor="white"),
                    barmode='overlay'
                )
                xaxis_title = "Vanna vs Gamma Overlay"
                
            elif view_mode == "⚡ DEX Overlay":
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], x=visible_agg['Gamma'], orientation='h', 
                    marker=dict(color='rgba(100, 100, 100, 0.3)', line=dict(width=0)), 
                    name="Gamma (Background)", xaxis="x1"
                ))
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], x=visible_agg['DEX'], orientation='h', 
                    marker=dict(color=['#00FFCC' if x >= 0 else '#FF3366' for x in visible_agg['DEX']], line=dict(color='white', width=1)),
                    width=gran * 0.4, name="Delta Exposure (DEX)", xaxis="x2"
                ))
                fig.update_layout(
                    xaxis=dict(title="Gamma Exposure", side="bottom", showgrid=False),
                    xaxis2=dict(title="Delta Exposure (DEX)", side="top", overlaying="x", showgrid=False, zerolinecolor="white"),
                    barmode='overlay'
                )
                xaxis_title = "Gamma vs DEX Overlay"
                
            elif view_mode == "📉 Puro DEX":
                fig.add_trace(go.Bar(
                    y=visible_agg['strike'], x=visible_agg['DEX'], orientation='h', 
                    marker=dict(color=['#00FFCC' if x >= 0 else '#FF3366' for x in visible_agg['DEX']]),
                    name="DEX Netto"
                ))
                xaxis_title = "Pure Delta Exposure (DEX)"

            for strike in visible_agg['strike']:
                fig.add_hline(y=strike, line_width=0.3, line_dash="dot", line_color="rgba(255,255,255,0.2)")

            fig.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
            fig.add_hline(y=z_gamma, line_color="#FFD700", line_width=2, line_dash="dash", annotation_text="0-G STATIC")
            fig.add_hline(y=z_gamma_dyn, line_color="#00BFFF", line_width=2, line_dash="dot", annotation_text="0-G DYNAMIC (VOL)")
            fig.add_hline(y=c_wall, line_color="#32CD32", line_width=2, annotation_text="CW")
            fig.add_hline(y=p_wall, line_color="#FF4500", line_width=2, annotation_text="PW")
            fig.add_hline(y=v_trigger, line_color="#FF00FF", line_width=2, line_dash="longdash", annotation_text="VANNA TRIGGER")
            
            if 'whale_data' in locals() and not whale_data.get("Error") and whale_data.get("Whale_Price"):
                bias_str = whale_data.get('Whale_Bias', '')
                leg_name = f"🐳 W-LVL ({bias_str})" if bias_str else "🐳 W-LVL"
                fig.add_hline(y=whale_data["Whale_Price"], line=dict(color='DeepSkyBlue', width=2, dash='dot'), annotation_text="🐳 W-LVL", name=leg_name)
            
            # --- VISUALIZZAZIONE LINEE ASIMMETRICHE ---
            fig.add_hline(y=sd1_up, line_color="#FFA500", line_dash="dash", annotation_text=f"+1SD (Call IV) {sd1_up:.2f}")
            fig.add_hline(y=sd1_down, line_color="#FFA500", line_dash="dash", annotation_text=f"-1SD (Put IV) {sd1_down:.2f}")
            fig.add_hline(y=sd2_up, line_color="#FF0000", line_dash="solid", annotation_text=f"+2SD {sd2_up:.2f}")
            fig.add_hline(y=sd2_down, line_color="#FF0000", line_dash="solid", annotation_text=f"-2SD {sd2_down:.2f}")

            # Nota Skew Factor nella legenda (dummy trace)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'), 
                                     name=f"Skew Factor: {skew_factor:.2f}x (Put/Call IV)", showlegend=True))

            # --- FIX FINALE: LEGENDA SOPRA IL DOPPIO ASSE ---
            fig.update_layout(
                template="plotly_dark", 
                height=850, 
                margin=dict(l=0, r=0, t=100, b=0), # Aumentato 't' a 100 per far stare asse + legenda
                yaxis=dict(range=[lo, hi], dtick=gran),
                legend=dict(
                    orientation="h",        # Legenda orizzontale
                    yanchor="bottom",
                    y=1.12,                 # Alzata a 1.12 per non toccare l'asse Vanna
                    xanchor="left",
                    x=0.01,                 
                    bgcolor="rgba(0,0,0,0)" 
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- NUOVE FUNZIONALITÀ VISIVE AVANZATE ---
            tab_iv, tab_price, tab_squeeze = st.tabs(["🔥 Analisi Volatilità Implicita (Heatmap)", "📈 Price Action vs Muri Quant", "🌪️ Squeeze Radar (Vomma & Speed)"])

            with tab_iv:
                if 'raw_data' in locals() and not raw_data.empty:
                    # 1. TRASFORMAZIONE DATI & 2. FILTRO QUALITÀ
                    iv_data = raw_data.copy()
                    iv_data['impliedVolatility'] = iv_data['impliedVolatility'] * 100
                    iv_data = iv_data[iv_data['impliedVolatility'] >= 0.1]

                    fig_iv = go.Figure(data=go.Heatmap(
                        z=iv_data['impliedVolatility'],
                        x=iv_data['exp'],
                        y=iv_data['strike'],
                        colorscale='Viridis',
                        hoverongaps=False,
                        zsmooth="best",
                        colorbar=dict(ticksuffix="%", tickformat=".1f")
                    ))
                    fig_iv.update_layout(
                        title="Heatmap Volatilità Implicita",
                        xaxis_title="Scadenza",
                        yaxis_title="Strike Price",
                        yaxis=dict(range=[spot * 0.85, spot * 1.15], autorange=False),
                        template="plotly_dark",
                        height=600
                    )
                    
                    fig_iv.add_hline(
                        y=spot, 
                        line_dash="dash", 
                        line_color="white", 
                        line_width=2, 
                        annotation_text=f"SPOT: {spot:.2f}", 
                        annotation_position="bottom right"
                    )
                    
                    st.plotly_chart(fig_iv, use_container_width=True)
                    
                    st.markdown("---")
                    skew_selected_date = st.selectbox("Seleziona Scadenza per Grafico Skew Curve", raw_data['exp'].unique())
                    
                    if skew_selected_date:
                        skew_df = raw_data[raw_data['exp'] == skew_selected_date].copy()
                        skew_df = skew_df.sort_values(by='strike', ascending=True)
                        
                        fig_skew = go.Figure()
                        fig_skew.add_trace(go.Scatter(
                            x=skew_df['strike'], 
                            y=skew_df['impliedVolatility'] * 100, 
                            mode='lines+markers', 
                            line=dict(color='cyan', width=3, shape='spline'), 
                            marker=dict(color='white', size=8, line=dict(color='cyan', width=2)), 
                            name='IV Skew'
                        ))
                        
                        fig_skew.add_vline(
                            x=spot, 
                            line_dash="dash", 
                            line_color="white", 
                            line_width=2, 
                            annotation_text=f"SPOT: {spot:.2f}", 
                            annotation_position="top left"
                        )
                        
                        fig_skew.update_layout(
                            template="plotly_dark",
                            title="Analisi Skew (Volatilità per Strike) - Stile Quant",
                            xaxis_title="Strike Price",
                            yaxis_title="Volatilità Implicita (%)",
                            xaxis=dict(range=[spot * 0.85, spot * 1.15])
                        )
                        st.plotly_chart(fig_skew, use_container_width=True)

            with tab_price:
                # --- INIZIO MIGLIORIE UX ---
                st.markdown("### 🧱 Livelli Quantistici Chiave")
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("🟢 CALL WALL", f"{c_wall:.0f}")
                c_m2.metric("🟡 ZERO GAMMA (STA/DYN)", f"{z_gamma:.0f} / {z_gamma_dyn:.0f}")
                c_m3.metric("🔴 PUT WALL", f"{p_wall:.0f}")
                
                # Legenda HTML personalizzata (fuori dal grafico Plotly per non interferire)
                st.markdown("""
                <div style="display: flex; gap: 15px; padding: 10px; background: #1E2329; border: 1px solid #333; border-radius: 8px; margin-bottom: 10px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #00ff88;">█</span> <span style="color: lightgray; font-size: 14px;">Prezzo</span></div>
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #00FFCC;">▲</span> <span style="color: lightgray; font-size: 14px;">Whale Buy (Accumulo)</span></div>
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #FF3366;">▼</span> <span style="color: lightgray; font-size: 14px;">Whale Sell (Distribuzione)</span></div>
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #FFFF00;">--</span> <span style="color: lightgray; font-size: 14px;">Zero Gamma</span></div>
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #32CD32;">--</span> <span style="color: lightgray; font-size: 14px;">Call Wall</span></div>
                    <div style="display: flex; align-items: center; gap: 5px;"><span style="color: #FF4500;">--</span> <span style="color: lightgray; font-size: 14px;">Put Wall</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                # --- FINE MIGLIORIE UX ---

                # 1. Inizializzazione chiavi robuste
                def init_slider(key, default_val):
                    if key not in st.session_state:
                        st.session_state[key] = default_val

                init_slider(f"visibili_{current_ticker}", 100)
                init_slider(f"offset_{current_ticker}", 0)
                init_slider(f"proiezione_{current_ticker}", 30)
                init_slider(f"padding_{current_ticker}", 1.0)
                init_slider("force_zoom_update", True) # Flag per aggiornare assi solo se si muove lo slider

                # 2. Callback: Salva valore e forza l'aggiornamento degli assi
                def update_slider(key):
                    st.session_state[key] = st.session_state[f"{key}_widget"]
                    st.session_state["force_zoom_update"] = True

                df_price = fetch_yahoo_history(current_ticker, "1Min", period="1d")
                
                if df_price is not None and not df_price.empty:
                    gamma_mode = st.toggle("Attiva Visualizzazione Gamma/Volatilità Live", value=False)

                    with st.expander("🔍 Controlli Zoom e Vista Grafico", expanded=True):
                        col_z1, col_z2, col_z3, col_z4 = st.columns(4)
                        with col_z1:
                            st.slider("Candele Visibili", 5, len(df_price), value=st.session_state[f"visibili_{current_ticker}"], key=f"visibili_{current_ticker}_widget", on_change=update_slider, args=(f"visibili_{current_ticker}",))
                            visibili = st.session_state[f"visibili_{current_ticker}"]
                        with col_z2:
                            st.slider("Sposta (Offset)", 0, len(df_price), value=st.session_state[f"offset_{current_ticker}"], key=f"offset_{current_ticker}_widget", on_change=update_slider, args=(f"offset_{current_ticker}",))
                            offset = st.session_state[f"offset_{current_ticker}"]
                        with col_z3:
                            st.slider("Futuro (Min)", 0, 120, value=st.session_state[f"proiezione_{current_ticker}"], key=f"proiezione_{current_ticker}_widget", on_change=update_slider, args=(f"proiezione_{current_ticker}",))
                            proiezione = st.session_state[f"proiezione_{current_ticker}"]
                        with col_z4:
                            st.slider("Padding %", 0.1, 10.0, value=st.session_state[f"padding_{current_ticker}"], key=f"padding_{current_ticker}_widget", on_change=update_slider, args=(f"padding_{current_ticker}",))
                            padding = st.session_state[f"padding_{current_ticker}"]

                    # --- CALCOLO LIMITI VISTA ---
                    end_idx = len(df_price) - offset
                    start_idx = max(0, end_idx - visibili)
                    
                    t_start = df_price['datetime'].iloc[start_idx]
                    t_end_data = df_price['datetime'].iloc[end_idx - 1] if end_idx > 0 else df_price['datetime'].iloc[0]
                    t_end_final = t_end_data + pd.Timedelta(minutes=proiezione)
                    x_range = [t_start, t_end_final]
                    
                    subset = df_price.iloc[start_idx:end_idx]
                    if not subset.empty:
                        prezzo_min = subset['Low'].min() * (1 - padding/100)
                        prezzo_max = subset['High'].max() * (1 + padding/100)
                    else:
                        prezzo_min = df_price['Low'].min() * (1 - padding/100)
                        prezzo_max = df_price['High'].max() * (1 + padding/100)

                    fig_key = f"fig_price_{current_ticker}_v5"
                    
                    # 1. Creazione Figura Base ALL-IN-ONE (Zero distruzioni, zero memory leak)
                    if fig_key not in st.session_state:
                        fig = go.Figure()
                        # Traccia 0: Price
                        fig.add_trace(go.Candlestick(name="Price"))
                        # Tracce 1 e 2: Whale Markers
                        fig.add_trace(go.Scatter(name='Whale Buy', mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00FFCC', line=dict(width=2, color='white'))))
                        fig.add_trace(go.Scatter(name='Whale Sell', mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF3366', line=dict(width=2, color='white'))))
                        
                        # Pre-allocazione Muri Quantistici Continui (Tracce 3-8)
                        fig.add_trace(go.Scatter(name="0-G STATIC", mode="lines+text", line=dict(color="yellow", width=2, dash="dash"), textfont=dict(color="yellow", size=12)))
                        fig.add_trace(go.Scatter(name="0-G DYNAMIC", mode="lines+text", line=dict(color="white", width=2, dash="dot"), textfont=dict(color="white", size=12)))
                        fig.add_trace(go.Scatter(name="CW", mode="lines+text", line=dict(color="#32CD32", width=2), textfont=dict(color="#32CD32", size=12)))
                        fig.add_trace(go.Scatter(name="PW", mode="lines+text", line=dict(color="#FF4500", width=2), textfont=dict(color="#FF4500", size=12)))
                        fig.add_trace(go.Scatter(name="VANNA TRIGGER", mode="lines+text", line=dict(color="#FF00FF", width=2, dash="longdash"), textfont=dict(color="#FF00FF", size=12)))
                        fig.add_trace(go.Scatter(name="WHALE LEVEL", mode="lines+text", line=dict(color="DeepSkyBlue", width=2, dash="dot"), textfont=dict(color="DeepSkyBlue", size=12)))
                        
                        fig.update_layout(
                            title=f"Price Action (1m) vs Muri Quant - {current_ticker}",
                            xaxis_title="Data/Ora", yaxis_title="Prezzo",
                            template="plotly_dark",
                            xaxis=dict(rangeslider=dict(visible=False), type='date', fixedrange=False),
                            yaxis=dict(fixedrange=False), # SBLOCCA ZOOM E PAN MANUALE ASSE Y
                            height=850, 
                            uirevision=current_ticker, 
                            dragmode='pan', hovermode='x unified',
                            showlegend=False,
                            paper_bgcolor='#0E1117',
                            plot_bgcolor='#0E1117',
                            margin=dict(l=0, r=50, t=30, b=0)
                        )
                        st.session_state[fig_key] = fig
                    
                    fig_price = st.session_state[fig_key]
                    
                    # 2. Aggiornamento Dati Candele In-Place
                    fig_price.data[0].x = df_price['datetime']
                    fig_price.data[0].open = df_price['Open']
                    fig_price.data[0].high = df_price['High']
                    fig_price.data[0].low = df_price['Low']
                    fig_price.data[0].close = df_price['Close']
                    fig_price.data[0].increasing = dict(line=dict(color='#00ff88'), fillcolor='#00A666')
                    fig_price.data[0].decreasing = dict(line=dict(color='#ff3333'), fillcolor='#EF4F4F')
                    
                    # 3. Gestione Zoom Preservato
                    if st.session_state["force_zoom_update"]:
                        fig_price.update_xaxes(range=x_range, autorange=False)
                        fig_price.update_yaxes(range=[prezzo_min, prezzo_max], autorange=False)
                        st.session_state["force_zoom_update"] = False

                    # 4. Aggiornamento Sicuro Muri Orizzontali (Senza usare add_hline che causa il crash)
                    x_muri = [df_price['datetime'].iloc[0], df_price['datetime'].iloc[-1]] if not df_price.empty else [None, None]
                    
                    def update_wall(idx, y_val, label):
                        fig_price.data[idx].x = x_muri
                        fig_price.data[idx].y = [y_val, y_val]
                        fig_price.data[idx].text = ["", label]
                        fig_price.data[idx].textposition = "top left"

                    update_wall(3, z_gamma, "0-G STATIC")
                    update_wall(4, z_gamma_dyn, "0-G DYNAMIC")
                    update_wall(5, c_wall, "CW")
                    update_wall(6, p_wall, "PW")
                    update_wall(7, v_trigger, "VANNA TRIGGER")
                    
                    # 5. WHALE INTELLIGENCE OVERLAYS
                    if 'whale_data' in locals() and not whale_data.get("Error") and whale_data.get("Whale_Price"):
                        bias_str = whale_data.get('Whale_Bias', '')
                        leg_name = f"🐳 W-LVL ({bias_str})" if bias_str else "🐳 W-LVL"
                        update_wall(8, whale_data["Whale_Price"], leg_name)
                        
                        if 'df_whale' in whale_data and not whale_data['df_whale'].empty:
                            df_price.set_index('datetime', inplace=True)
                            df_whale_data = whale_data['df_whale'].copy()
                            
                            if df_price.index.tz is not None: df_price.index = df_price.index.tz_localize(None)
                            if df_whale_data.index.tz is not None: df_whale_data.index = df_whale_data.index.tz_localize(None)
                            
                            df_whale_ratio = df_whale_data[['Dynamic_Ratio']]
                            df_price = df_price.merge(df_whale_ratio, left_index=True, right_index=True, how='left')
                            df_price['Dynamic_Ratio'] = df_price['Dynamic_Ratio'].ffill().bfill()
                            
                            if df_price['Dynamic_Ratio'].isnull().all():
                                df_price['Dynamic_Ratio'] = whale_data.get('ratio', 1.0)
                                
                            df_price.reset_index(inplace=True)

                            vol_mean = whale_data.get('vol_mean', 0)
                            vol_sd = whale_data.get('vol_sd', 0)
                            whale_candles = whale_data['df_whale']
                            whale_candles = whale_candles[whale_candles['Volume'] > (vol_mean + 2.5 * vol_sd)].copy()
                            
                            if not whale_candles.empty:
                                cutoff_24h = datetime.now(whale_candles.index.tz) - timedelta(hours=24)
                                whale_candles = whale_candles[whale_candles.index >= cutoff_24h]
                            
                            mid_prices = (whale_candles['High'] + whale_candles['Low']) / 2
                            buy_candles = whale_candles[whale_candles['Close'] > mid_prices]
                            sell_candles = whale_candles[whale_candles['Close'] <= mid_prices]
                            
                            df_price['Whale_Buy'] = False
                            df_price['Whale_Sell'] = False
                            
                            if not buy_candles.empty:
                                x_buy = buy_candles.index.tz_localize(None) if buy_candles.index.tz is not None else buy_candles.index
                                df_price.loc[df_price['datetime'].isin(x_buy), 'Whale_Buy'] = True
                            
                            if not sell_candles.empty:
                                x_sell = sell_candles.index.tz_localize(None) if sell_candles.index.tz is not None else sell_candles.index
                                df_price.loc[df_price['datetime'].isin(x_sell), 'Whale_Sell'] = True
                                
                            buy_points = df_price[df_price['Whale_Buy']]
                            sell_points = df_price[df_price['Whale_Sell']]
                            
                            price_sample = df_price['Close'].iloc[-1]
                            current_ratio = df_price['Dynamic_Ratio'].iloc[-1]
                            scaling_factor = 1.0 if (price_sample > 1000 and current_ratio > 5) else df_price['Dynamic_Ratio']
                            
                            if not buy_points.empty:
                                y_buy = buy_points['Low'] * (scaling_factor if isinstance(scaling_factor, float) else buy_points['Dynamic_Ratio']) * 0.9992
                                fig_price.data[1].x = buy_points['datetime']
                                fig_price.data[1].y = y_buy
                                fig_price.data[1].hovertemplate = "<b>Whale Buy</b><br>Prezzo: %{y:.2f}<br>%{x}<extra></extra>"
                            else:
                                fig_price.data[1].x = [None]
                                fig_price.data[1].y = [None]
                                
                            if not sell_points.empty:
                                y_sell = sell_points['High'] * (scaling_factor if isinstance(scaling_factor, float) else sell_points['Dynamic_Ratio']) * 1.0008
                                fig_price.data[2].x = sell_points['datetime']
                                fig_price.data[2].y = y_sell
                                fig_price.data[2].hovertemplate = "<b>Whale Sell</b><br>Prezzo: %{y:.2f}<br>%{x}<extra></extra>"
                            else:
                                fig_price.data[2].x = [None]
                                fig_price.data[2].y = [None]
                    else:
                        update_wall(8, None, "")
                    
                    # 6. PULIZIA CONTROLLATA SHAPES (Evita il crash per la Gamma Mode)
                    fig_price.layout.shapes = tuple()
                    
                    if gamma_mode and 'df' in locals() and not df.empty:
                        xray_agg = df.groupby('strike')['Gamma'].sum().reset_index()
                        max_gex = xray_agg['Gamma'].abs().max()
                        if max_gex > 0:
                            for _, row in xray_agg.iterrows():
                                s_val, g_val = row['strike'], row['Gamma']
                                if s_val < df_price['Low'].min() * 0.8 or s_val > df_price['High'].max() * 1.2: continue
                                intensita = min(1.0, abs(g_val) / max_gex)
                                if intensita > 0.01:
                                    colorscale = px.colors.sequential.Viridis
                                    base_color = colorscale[int(intensita * (len(colorscale) - 1))]
                                    rgba_color = base_color.replace('rgb', 'rgba').replace(')', f', {0.1 + intensita*0.5})')
                                    # Usa il metodo nativo sicuro per non interferire con le traces
                                    fig_price.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=s_val * 0.9998, y1=s_val * 1.0002, fillcolor=rgba_color, line_width=0, layer="below")

                    st.plotly_chart(
                        fig_price, 
                        use_container_width=True, 
                        key=f"fixed_chart_{current_ticker}_render_v5", 
                        theme=None, 
                        config={'scrollZoom': True, 'displayModeBar': False}
                    )
                else:
                    st.warning("Dati intraday non disponibili per il grafico Price Action.")

            with tab_squeeze:
                st.markdown("### 🌪️ Squeeze Radar: Vomma & Speed")
                col_v, col_s = st.columns(2)
                
                # Applica un filtro rigoroso per escludere i livelli con Volume = 0 dai grafici
                plot_df = visible_agg[visible_agg['volume'] > 0]
                
                with col_v:
                    fig_vomma = go.Figure()
                    fig_vomma.add_trace(go.Bar(
                        x=plot_df['Vomma'],
                        y=plot_df['strike'],
                        orientation='h',
                        marker_color='#FF00FF',
                        name='Vomma'
                    ))
                    fig_vomma.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
                    fig_vomma.add_hline(y=c_wall, line_color="#32CD32", line_width=2, line_dash="dot", annotation_text="CW")
                    fig_vomma.add_hline(y=p_wall, line_color="#FF4500", line_width=2, line_dash="dot", annotation_text="PW")
                    fig_vomma.add_hline(y=v_trigger, line_color="#FF00FF", line_width=2, line_dash="dot", annotation_text="VT")
                    fig_vomma.update_layout(
                        title="Vomma Profile",
                        xaxis_title="Net Vomma",
                        yaxis_title="Strike",
                        template="plotly_dark",
                        height=600,
                        yaxis=dict(range=[lo, hi], dtick=gran)
                    )
                    st.plotly_chart(fig_vomma, use_container_width=True)
                
                with col_s:
                    fig_speed = go.Figure()
                    fig_speed.add_trace(go.Bar(
                        x=plot_df['Speed'],
                        y=plot_df['strike'],
                        orientation='h',
                        marker_color='#FFFF00',
                        name='Speed'
                    ))
                    fig_speed.add_hline(y=spot, line_color="#00FFFF", line_width=3, annotation_text="SPOT")
                    fig_speed.add_hline(y=c_wall, line_color="#32CD32", line_width=2, line_dash="dot", annotation_text="CW")
                    fig_speed.add_hline(y=p_wall, line_color="#FF4500", line_width=2, line_dash="dot", annotation_text="PW")
                    fig_speed.add_hline(y=v_trigger, line_color="#FF00FF", line_width=2, line_dash="dot", annotation_text="VT")
                    fig_speed.update_layout(
                        title="Speed Profile",
                        xaxis_title="Net Speed",
                        yaxis_title="Strike",
                        template="plotly_dark",
                        height=600,
                        yaxis=dict(range=[lo, hi], dtick=gran)
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)

elif menu == "🔥 SCANNER HOT TICKERS":
    st.title("🔥 Professional Market Scanner (50 Tickers)")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 AGGIORNA SCANNER", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        expiry_mode = st.selectbox("📅 SELEZIONE SCADENZE:", ["0-1 DTE (Scalping/Intraday)", "Prossima Scadenza Mensile (Swing)"])
    
    tickers_50 = ["^NDX", "^SPX", "^RUT", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "AMD", "AVGO", "MU", "INTC", "QCOM", "ARM", "TSM", "SMCI", "MSTR", "COIN", "MARA", "RIOT", "CLSK", "BITO", "PLTR", "SNOW", "U", "DKNG", "HOOD", "SHOP", "SQ", "PYPL", "ROKU", "JPM", "GS", "BAC", "V", "MA", "LLY", "UNH", "PFE", "XOM", "CVX", "DIS", "BA"]
    
    scan_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t_name in enumerate(tickers_50):
        status_text.text(f"Scansione in profondità: {t_name} ({i+1}/{len(tickers_50)})")
        
        # --- UTILIZZO FUNZIONE PROTETTA ---
        data_pack = fetch_scanner_ticker(t_name, expiry_mode, today_str_format)
        
        time.sleep(0.5) 
        
        if data_pack is None:
            progress_bar.progress((i + 1) / len(tickers_50))
            continue
            
        px, df_scan, dte_years = data_pack
        
        try:
            # Calcolo 0-G Statico e Dinamico
            try: zg_val = brentq(calculate_gex_at_price, px*0.75, px*1.25, args=(df_scan,))
            except: zg_val = px
            try: zg_dyn = brentq(calculate_0g_dynamic, px*0.75, px*1.25, args=(df_scan,))
            except: zg_dyn = px

            # Calcolo Greche Scanner
            df_scan_greeks = get_greeks_pro(df_scan, px)
            net_vanna_scan = df_scan_greeks['Vanna'].sum() if not df_scan_greeks.empty else 0
            net_charm_scan = df_scan_greeks['Charm'].sum() if not df_scan_greeks.empty else 0
            
            # Motore di Scoring Confluenza
            p_score = 4 if (px > zg_val and px > zg_dyn) else (-4 if (px < zg_val and px < zg_dyn) else 0)
            v_score = 3 if net_vanna_scan > 0 else -3
            c_score = 3 if net_charm_scan < 0 else -3
            ss = p_score + v_score + c_score

            v_icon = "🟢" if net_vanna_scan > 0 else "🔴"
            c_icon = "🔵" if net_charm_scan < 0 else "🔴"
            
            # Cluster/Market Regime
            if ss >= 8: verdict = "🚀 CONFLUENZA FULL LONG"
            elif ss <= -8: verdict = "☢️ CRASH RISK / FULL SHORT"
            elif px > zg_val and px < zg_dyn: verdict = "⚠️ DISTRIBUZIONE (Volumi in uscita)"
            elif px < zg_val and px > zg_dyn: verdict = "🔥 SHORT SQUEEZE IN ATTO"
            elif net_vanna_scan < 0 and px > zg_dyn: verdict = "🌪️ GAMMA SQUEEZE (Alta Volatilità)"
            else: verdict = "⚖️ NEUTRO / RANGE BOUND"

            # --- MODIFICA ASIMMETRICA DS (IDENTICA ALLA DASHBOARD) ---
            try:
                mean_iv = df_scan['impliedVolatility'].mean()
                
                if not df_scan.empty:
                    c_target = px * 1.02
                    p_target = px * 0.98
                    c_skew = df_scan[df_scan['type'] == 'call']
                    p_skew = df_scan[df_scan['type'] == 'put']
                    
                    def clean_iv(val):
                        if val is None: return mean_iv / 100 if mean_iv > 1 else mean_iv
                        return val / 100 if val > 1.5 else val

                    raw_c_iv = c_skew.iloc[(c_skew['strike'] - c_target).abs().argmin()]['impliedVolatility'] if not c_skew.empty else mean_iv
                    raw_p_iv = p_skew.iloc[(p_skew['strike'] - p_target).abs().argmin()]['impliedVolatility'] if not p_skew.empty else mean_iv
                    
                    c_iv = clean_iv(raw_c_iv)
                    p_iv = clean_iv(raw_p_iv)
                else:
                    c_iv = p_iv = clean_iv(mean_iv)
            except:
                c_iv = p_iv = 0.15 # Fallback prudenziale

            # 2. Calcolo Fixed 1-Day Move (1/252)
            one_day_factor = np.sqrt(1/252)
            
            # 3. Creazione delle 4 Linee Asimmetriche per lo Scanner
            sd1_up = px * (1 + (c_iv * one_day_factor))
            sd2_up = px * (1 + (c_iv * 2 * one_day_factor))
            sd1_down = px * (1 - (p_iv * one_day_factor))
            sd2_down = px * (1 - (p_iv * 2 * one_day_factor))
            
            # 4. Skew Factor
            skew_factor = p_iv / c_iv if c_iv > 0 else 1.0

            # --- MOTORE OPPORTUNITÀ MEAN REVERSION ---
            if px <= sd2_down:
                reversion_signal = "💎 BUY REVERSION (2DS)"
                rev_score = 2
            elif px <= sd1_down:
                reversion_signal = "🟢 BUY REVERSION (1DS)"
                rev_score = 1
            elif px >= sd2_up:
                reversion_signal = "💀 SELL REVERSION (2DS)"
                rev_score = -2
            elif px >= sd1_up:
                reversion_signal = "🟠 SELL REVERSION (1DS)"
                rev_score = -1
            else:
                reversion_signal = "---"
                rev_score = 0
            # ---------------------------------------------

            dist_zg_pct = ((px - zg_val) / px) * 100
            is_above_0g = px > zg_val
            near_sd_up = abs(px - sd1_up) / px < 0.005
            near_sd_down = abs(px - sd1_down) / px < 0.005
            
            if not is_above_0g: 
                if near_sd_down: status_label = "🔴 < 0G | TEST -1SD (Bounce?)"
                elif px < sd1_down: status_label = "⚫ < 0G | SOTTO -1SD (Short Ext)"
                else: status_label = "🔻 SOTTO 0G (Short Bias)"
            else: 
                if near_sd_up: status_label = "🟡 > 0G | TEST +1SD (Breakout?)"
                elif px > sd1_up: status_label = "🟢 > 0G | SOPRA +1SD (Long Ext)"
                elif near_sd_down: status_label = "🟢 > 0G | DIP BUY (Test -1SD)"
                else: status_label = "✅ SOPRA 0G (Long Bias)"
            
            if abs(dist_zg_pct) < 0.3: status_label = "🔥 FLIP IMMINENTE (0G)"
            
            scan_results.append({
                "Ticker": t_name.replace("^", ""), 
                "Score": int(ss),                  
                "Verdict (Regime)": verdict,      
                "Greche V|C": f"V:{v_icon} C:{c_icon}",
                "Prezzo": round(px, 2), 
                "0-G Static": round(zg_val, 2), 
                "0-G Dynamic": round(zg_dyn, 2), # --- NUOVA COLONNA DINAMICA ---
                "1SD Range": f"{sd1_down:.0f} - {sd1_up:.0f}", 
                "2SD Range": f"{sd2_down:.0f} - {sd2_up:.0f}", # --- NUOVA COLONNA 2SD ---
                "Dist. 0G %": round(dist_zg_pct, 2), 
                "OPPORTUNITÀ": reversion_signal,  
                "Analisi": status_label, 
                "_rev_score": rev_score,          
                "_sort_score": -ss,                 
                "_sort_dist": abs(dist_zg_pct)
            })
        except: pass
        progress_bar.progress((i + 1) / len(tickers_50))
    
    if scan_results:
        final_df = pd.DataFrame(scan_results).sort_values(by=["_sort_score", "_sort_dist"]).drop(columns=["_sort_score", "_sort_dist", "_rev_score"])
        
        def color_logic_pro(row):
            styles = [''] * len(row)
            
            # --- Colore per Score ---
            if 'Score' in row.index:
                score_idx = row.index.get_loc('Score')
                val_score = row['Score']
                if val_score >= 8: styles[score_idx] = 'background-color: #2ECC40; color: white; font-weight: bold'
                elif val_score <= -8: styles[score_idx] = 'background-color: #8B0000; color: white; font-weight: bold'
                elif val_score > 0: styles[score_idx] = 'color: #2ECC40; font-weight: bold'
                elif val_score < 0: styles[score_idx] = 'color: #FF4136; font-weight: bold'
            
            # --- Colore per OPPORTUNITÀ ---
            if 'OPPORTUNITÀ' in row.index:
                opp_idx = row.index.get_loc('OPPORTUNITÀ')
                val_opp = row['OPPORTUNITÀ']
                if "💎 BUY" in val_opp:
                    styles[opp_idx] = 'background-color: #00FF00; color: black; font-weight: bold; border: 1px solid white'
                elif "🟢 BUY" in val_opp:
                    styles[opp_idx] = 'color: #00FF00; font-weight: bold'
                elif "💀 SELL" in val_opp:
                    styles[opp_idx] = 'background-color: #FF0000; color: white; font-weight: bold; border: 1px solid white'
                elif "🟠 SELL" in val_opp:
                    styles[opp_idx] = 'color: #FF0000; font-weight: bold'

            # --- Colore per Analisi ---
            if 'Analisi' in row.index:
                analisi_idx = row.index.get_loc('Analisi')
                val_ana = row['Analisi']
                if "🔥" in val_ana: styles[analisi_idx] = 'background-color: #8B0000; color: white'
                elif "🔴" in val_ana: styles[analisi_idx] = 'color: #FF4136; font-weight: bold'
                elif "🟢" in val_ana: styles[analisi_idx] = 'color: #2ECC40; font-weight: bold'
                elif "🟡" in val_ana: styles[analisi_idx] = 'color: #FFDC00'
                elif "✅" in val_ana: styles[analisi_idx] = 'color: #0074D9'

            return styles

        st.dataframe(final_df.style.apply(color_logic_pro, axis=1), use_container_width=True, height=800)
        
        # Visualizzazione Matrice di Correlazione per i risultati dello scanner
        if not final_df.empty and len(final_df) > 1:
            st.markdown("---")
            tickers_per_corr = final_df['Ticker'].tolist()
            display_correlation_matrix(tickers_per_corr)

elif menu == "🔙 BACKTESTING STRATEGIA":
    st.title("🛠️ Professional Backtesting Suite")
    
    st.sidebar.markdown("---")
    tz_choice = st.sidebar.selectbox("🌍 Fuso Orario", ["America/New_York", "UTC", "Europe/Rome"], index=0)
    st.sidebar.markdown("### 🛡️ Risk & Robustness")
    friction_pct = st.sidebar.slider("Execution Friction (%)", 0.00, 0.50, 0.00, 0.01)

    def normalize_key(d, possible_keys):
        for k in d.keys():
            if k.lower() in [pk.lower() for pk in possible_keys]:
                return d[k]
        return None

    def apply_friction_post_process(trades_list, initial_capital, friction_pct):
        if not trades_list:
            return trades_list, [initial_capital]
            
        new_trades = []
        balance = initial_capital
        equity_curve = [balance]
        
        for t in trades_list:
            t_copy = dict(t)
            t_type = str(normalize_key(t_copy, ['type', 'Type']) or '').upper()
            price = normalize_key(t_copy, ['price', 'Price', 'Entry Price', 'Exit Price']) or 0
            pnl = normalize_key(t_copy, ['pnl', 'PnL']) or 0
            
            friction_multiplier = 1 - (friction_pct / 100)
            new_price = price * friction_multiplier
            pnl = pnl * friction_multiplier
            t_copy['price'] = new_price
            t_copy['pnl'] = pnl
            balance += pnl
            t_copy['balance'] = balance
            equity_curve.append(balance)
            
            new_trades.append(t_copy)
                
        return new_trades, equity_curve

    def calculate_advanced_metrics(trades_list):
        fallback = {'expectancy': 0, 'profit_factor': 0, 'max_drawdown': 0, 'win_rate': 0, 'total_profit_abs': 0, 'max_dd_abs': 0}
        if not trades_list: return fallback
        df = pd.DataFrame(trades_list)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()] # Elimina colonne con lo stesso nome
        if 'pnl' not in df.columns: return fallback
        
        # FIX: Filtriamo solo le righe che rappresentano la chiusura di un trade (PnL diverso da zero)
        # o che contengono la stringa 'EXIT' nel tipo.
        mask = df['pnl'] != 0
        if 'direction' in df.columns:
            mask = mask | df['direction'].astype(str).str.contains('EXIT', case=False, na=False)
        if 'type' in df.columns:
            mask = mask | df['type'].astype(str).str.contains('EXIT', case=False, na=False)
            
        exits = df[mask]
            
        if exits.empty: return fallback
        
        wins = exits[exits['pnl'] > 0]['pnl']
        losses = exits[exits['pnl'] < 0]['pnl']
        
        win_rate = len(wins) / len(exits)
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
        total_profit_abs = exits['pnl'].sum()
        
        # Calcolo Drawdown (rimane invariato)
        bal_col = 'balance' if 'balance' in df.columns else None
        max_dd, max_dd_abs = 0, 0
        if bal_col:
            curve = df[bal_col].tolist()
            peak = curve[0]
            for val in curve:
                if val > peak: peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                dd_abs = peak - val
                if dd > max_dd: max_dd = dd
                if dd_abs > max_dd_abs: max_dd_abs = dd_abs
                
        return {
            'expectancy': expectancy, 
            'profit_factor': profit_factor, 
            'max_drawdown': max_dd * 100, 
            'win_rate': win_rate * 100, 
            'total_profit_abs': total_profit_abs, 
            'max_dd_abs': max_dd_abs
        }

    def run_monte_carlo(trades_list, initial_capital, simulations=1000):
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        if not trades_list:
            return None
            
        df_res = pd.DataFrame(trades_list)
        if 'pnl' in df_res.columns:
            pnls = df_res[df_res['pnl'].notna()]['pnl'].values
        else:
            return None
            
        n_trades = len(pnls)
        if n_trades == 0:
            return None
            
        # Fixed forward-horizon
        sim_length = min(50, n_trades)
        
        # Vectorized Monte Carlo: Sample with replacement
        random_indices = np.random.randint(0, n_trades, size=(simulations, sim_length))
        simulated_pnls = pnls[random_indices]
        
        # Calculate equity curves
        equity_curves = np.cumsum(simulated_pnls, axis=1) + initial_capital
        
        # Prepend initial capital to the beginning of each curve
        starting_capital = np.full((simulations, 1), initial_capital)
        equity_curves = np.hstack((starting_capital, equity_curves))
        
        # Calculate median curve
        median_curve = np.median(equity_curves, axis=0)
        
        # Calculate quantitative analytics
        final_balances = equity_curves[:, -1]
        prob_profit = (np.sum(final_balances > initial_capital) / simulations) * 100
        
        # Risk of Ruin: equity drops below initial_capital * 0.80 at any point
        ruin_threshold = initial_capital * 0.80
        ruined_simulations = np.any(equity_curves < ruin_threshold, axis=1)
        risk_of_ruin = (np.sum(ruined_simulations) / simulations) * 100
        
        median_final_balance = np.median(final_balances)
        
        # Visualization with Plotly
        fig = go.Figure()
        
        # Performance optimization: Plot all 1000 lines as a single trace separated by NaNs
        # This prevents Plotly from crashing the browser when rendering 1000 individual traces
        x_base = np.arange(sim_length + 1)
        x_all = np.tile(np.append(x_base, np.nan), simulations)
        y_all = np.hstack((equity_curves, np.full((simulations, 1), np.nan))).flatten()
        
        # Add all simulated paths (Gray, low opacity)
        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add Median Curve (Gold, bold)
        fig.add_trace(go.Scatter(
            x=x_base,
            y=median_curve,
            mode='lines',
            line=dict(color='gold', width=3),
            name='Median (50th Percentile)'
        ))
        
        fig.update_layout(
            title='🔬 Monte Carlo Robustness Analysis (Forward 50 Trades)',
            xaxis_title='Trade Number',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig, prob_profit, risk_of_ruin, median_final_balance
    
    # Engine Selection
    engine_choice = st.radio("Seleziona Motore di Backtesting:", 
                             ["🧬 MOTORE A: GEX & Options Hybrid Simulator", 
                              "📈 MOTORE B: Technical Strategy Hub (Pure Trading)"], 
                             horizontal=True)
    
    # Common Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        # Ticker Selection with Predefined List + Custom
        PREDEFINED_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META", "NFLX"]
        ticker_select = st.selectbox("Seleziona Ticker", ["Seleziona..."] + PREDEFINED_TICKERS + ["Inserisci Manualmente"])
        
        if ticker_select == "Inserisci Manualmente":
            ticker = st.text_input("Inserisci Simbolo Ticker", value="SPY").upper()
        elif ticker_select != "Seleziona...":
            ticker = ticker_select
        else:
            ticker = "SPY" # Default

    with c2: timeframe = st.selectbox("Timeframe", ["1D", "1H", "15Min", "5Min"], index=0)
    with c3: 
        start_date = st.date_input("Data Inizio", value=datetime.now() - timedelta(days=252*2))
    with c4: 
        end_date = st.date_input("Data Fine", value=datetime.now())
        initial_capital = st.number_input("Capitale Iniziale ($)", value=10000)

    # Session State for Data Verification
    if 'backtest_data' not in st.session_state:
        st.session_state.backtest_data = None
    if 'backtest_ticker' not in st.session_state:
        st.session_state.backtest_ticker = None


    # Data Verification Step
    st.markdown("---")
    if st.button("🔍 Verifica Disponibilità Dati Storici"):
        with st.spinner(f"Ricerca dati storici per {ticker} dal {start_date} al {end_date}..."):
            df_check = fetch_data_smart(ticker, timeframe, start_date, end_date, target_tz=tz_choice)
            
            if not df_check.empty:
                # Check actual date range
                min_date = df_check['datetime'].min().date()
                max_date = df_check['datetime'].max().date()
                count = len(df_check)
                
                st.success(f"✅ Dati Trovati! {count} candele disponibili.")
                st.info(f"📅 Range Disponibile: {min_date} -> {max_date}")
                
                if min_date > start_date:
                    st.warning(f"⚠️ Attenzione: I dati iniziano dal {min_date}, successivi alla data richiesta {start_date}.")
                
                st.session_state.backtest_data = df_check
                st.session_state.backtest_ticker = ticker
            else:
                st.error(f"❌ Nessun dato trovato per {ticker} nel range selezionato. Prova a cambiare date o ticker.")
                st.session_state.backtest_data = None

    # --- BACKTESTING ENGINE & VISUALIZER ---
    class TechnicalIndicators:
        # --- TREND ---
        @staticmethod
        def sma(series, period): return series.rolling(period).mean()
        @staticmethod
        def ema(series, period): return series.ewm(span=period, adjust=False).mean()
        @staticmethod
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        @staticmethod
        def hma(series, period):
            half_length = int(period / 2)
            sqrt_length = int(np.sqrt(period))
            wmaf = TechnicalIndicators.wma(series, half_length)
            wmas = TechnicalIndicators.wma(series, period)
            return TechnicalIndicators.wma(2 * wmaf - wmas, sqrt_length)
        @staticmethod
        def tema(series, period):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            ema3 = TechnicalIndicators.ema(ema2, period)
            return 3 * ema1 - 3 * ema2 + ema3
        @staticmethod
        def dema(series, period):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            return 2 * ema1 - ema2
        @staticmethod
        def kama(series, period=10, pow1=2, pow2=30):
            change = abs(series - series.shift(period))
            volatility = series.diff().abs().rolling(window=period).sum()
            er = change / volatility
            sc = (er * (2.0 / (pow1 + 1) - 2.0 / (pow2 + 1)) + 2.0 / (pow2 + 1)) ** 2
            kama = [series.values[period-1]]
            for i in range(period, len(series)):
                kama.append(kama[-1] + sc.values[i] * (series.values[i] - kama[-1]))
            return pd.Series(kama, index=series.index[period-1:])

        @staticmethod
        def macd(series, fast=12, slow=26, signal=9):
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd, signal_line
        
        @staticmethod
        def adx(df, period=14):
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = TechnicalIndicators.atr(df, period)
            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            return dx.ewm(alpha=1/period).mean()

        @staticmethod
        def aroon(df, period=25):
            aroon_up = 100 * df['High'].rolling(period + 1).apply(lambda x: x.argmax()) / period
            aroon_down = 100 * df['Low'].rolling(period + 1).apply(lambda x: x.argmin()) / period
            return aroon_up, aroon_down

        @staticmethod
        def cci(df, period=20):
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (tp - sma) / (0.015 * mad)

        @staticmethod
        def supertrend(df, period=10, multiplier=3):
            hl2 = (df['High'] + df['Low']) / 2
            atr = TechnicalIndicators.atr(df, period)
            upper = hl2 + (multiplier * atr)
            lower = hl2 - (multiplier * atr)
            return upper, lower

        @staticmethod
        def parabolic_sar(df, af=0.02, max_af=0.2):
            high, low = df['High'], df['Low']
            sar = [low[0]]
            ep = high[0]
            acc = af
            trend = 1 # 1 for long, -1 for short
            
            for i in range(1, len(df)):
                prev_sar = sar[-1]
                if trend == 1:
                    curr_sar = prev_sar + acc * (ep - prev_sar)
                    curr_sar = min(curr_sar, low[i-1])
                    if i > 1: curr_sar = min(curr_sar, low[i-2])
                    
                    if low[i] < curr_sar:
                        trend = -1
                        curr_sar = ep
                        ep = low[i]
                        acc = af
                    else:
                        if high[i] > ep:
                            ep = high[i]
                            acc = min(acc + af, max_af)
                else:
                    curr_sar = prev_sar + acc * (ep - prev_sar)
                    curr_sar = max(curr_sar, high[i-1])
                    if i > 1: curr_sar = max(curr_sar, high[i-2])
                    
                    if high[i] > curr_sar:
                        trend = 1
                        curr_sar = ep
                        ep = high[i]
                        acc = af
                    else:
                        if low[i] < ep:
                            ep = low[i]
                            acc = min(acc + af, max_af)
                sar.append(curr_sar)
            return pd.Series(sar, index=df.index)

        # --- MOMENTUM ---
        @staticmethod
        def rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        @staticmethod
        def stochastic(df, k_period=14, d_period=3):
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            return k, d

        @staticmethod
        def williams_r(df, period=14):
            highest_high = df['High'].rolling(period).max()
            lowest_low = df['Low'].rolling(period).min()
            return ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100

        @staticmethod
        def roc(series, period=12):
            return ((series - series.shift(period)) / series.shift(period)) * 100
        
        @staticmethod
        def tsi(series, r=25, s=13):
            m = series.diff()
            m1 = m.ewm(span=r).mean().ewm(span=s).mean()
            m2 = abs(m).ewm(span=r).mean().ewm(span=s).mean()
            return 100 * (m1 / m2)

        @staticmethod
        def uo(df, p1=7, p2=14, p3=28):
            bp = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
            tr = TechnicalIndicators.atr(df, 1) # True Range 1-period approximation
            avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
            avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
            avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
            return 100 * (4*avg1 + 2*avg2 + avg3) / 7

        # --- VOLATILITY ---
        @staticmethod
        def bollinger_bands(series, period=20, std_dev=2):
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            return sma + (std * std_dev), sma - (std * std_dev)

        @staticmethod
        def atr(df, period=14):
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            return true_range.rolling(period).mean()

        @staticmethod
        def keltner_channels(df, period=20, mult=2):
            ema = TechnicalIndicators.ema(df['Close'], period)
            atr = TechnicalIndicators.atr(df, 10)
            return ema + (mult * atr), ema - (mult * atr)

        @staticmethod
        def donchian_channels(df, period=20):
            return df['High'].rolling(period).max(), df['Low'].rolling(period).min()

        @staticmethod
        def chaikin_volatility(df, period=10, roc_period=10):
            hl = df['High'] - df['Low']
            ema_hl = TechnicalIndicators.ema(hl, period)
            return TechnicalIndicators.roc(ema_hl, roc_period)

        # --- VOLUME ---
        @staticmethod
        def obv(df):
            return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        @staticmethod
        def mfi(df, period=14):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
            return 100 - (100 / (1 + positive_flow / negative_flow))

        @staticmethod
        def cmf(df, period=20):
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_volume = mf_multiplier * df['Volume']
            return mf_volume.rolling(period).sum() / df['Volume'].rolling(period).sum()

        @staticmethod
        def vwap(df):
            v = df['Volume'].values
            tp = (df['High'] + df['Low'] + df['Close']).values / 3
            return df.assign(vwap=(tp * v).cumsum() / v.cumsum())['vwap']
        
        @staticmethod
        def ad_line(df):
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            return (clv * df['Volume']).cumsum()

        # --- NEW INDICATORS (BATCH 2) ---
        @staticmethod
        def vortex(df, period=14):
            tr = TechnicalIndicators.atr(df, 1).rolling(period).sum()
            vm_plus = abs(df['High'] - df['Low'].shift(1)).rolling(period).sum()
            vm_minus = abs(df['Low'] - df['High'].shift(1)).rolling(period).sum()
            return vm_plus / tr, vm_minus / tr

        @staticmethod
        def chop(df, period=14):
            tr = TechnicalIndicators.atr(df, 1).rolling(period).sum()
            r = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
            return 100 * np.log10(tr / r) / np.log10(period)

        @staticmethod
        def kst(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15):
            roc1 = TechnicalIndicators.roc(df['Close'], r1).rolling(n1).mean()
            roc2 = TechnicalIndicators.roc(df['Close'], r2).rolling(n2).mean()
            roc3 = TechnicalIndicators.roc(df['Close'], r3).rolling(n3).mean()
            roc4 = TechnicalIndicators.roc(df['Close'], r4).rolling(n4).mean()
            return (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)

        @staticmethod
        def coppock(df, wma_period=10, roc1=14, roc2=11):
            roc_sum = TechnicalIndicators.roc(df['Close'], roc1) + TechnicalIndicators.roc(df['Close'], roc2)
            return roc_sum.ewm(span=wma_period).mean()

        @staticmethod
        def ichimoku(df):
            nine_period_high = df['High'].rolling(window=9).max()
            nine_period_low = df['Low'].rolling(window=9).min()
            tenkan_sen = (nine_period_high + nine_period_low) / 2

            period26_high = df['High'].rolling(window=26).max()
            period26_low = df['Low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2

            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            period52_high = df['High'].rolling(window=52).max()
            period52_low = df['Low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

            chikou_span = df['Close'].shift(-26)

            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        
        @staticmethod
        def ao(df):
            mp = (df['High'] + df['Low']) / 2
            return mp.rolling(5).mean() - mp.rolling(34).mean()

        @staticmethod
        def ppo(df, fast=12, slow=26):
            fast_ema = TechnicalIndicators.ema(df['Close'], fast)
            slow_ema = TechnicalIndicators.ema(df['Close'], slow)
            return ((fast_ema - slow_ema) / slow_ema) * 100

        @staticmethod
        def mass_index(df, period=25, ema_period=9):
            high_low = df['High'] - df['Low']
            ema1 = high_low.ewm(span=ema_period).mean()
            ema2 = ema1.ewm(span=ema_period).mean()
            ratio = ema1 / ema2
            return ratio.rolling(period).sum()

        @staticmethod
        def ulcer_index(df, period=14):
            close = df['Close']
            max_close = close.rolling(period).max()
            drawdown = 100 * ((close - max_close) / max_close)
            sq_drawdown = drawdown ** 2
            return np.sqrt(sq_drawdown.rolling(period).mean())

        # --- NEW INDICATORS (BATCH 3) ---
        @staticmethod
        def wma(series, period=9):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        @staticmethod
        def trima(series, period=18):
            return TechnicalIndicators.sma(TechnicalIndicators.sma(series, period), period) # Approx

        @staticmethod
        def cmo(series, period=14):
            # Chande Momentum Oscillator
            diff = series.diff()
            pos = diff.where(diff > 0, 0)
            neg = abs(diff.where(diff < 0, 0))
            sum_pos = pos.rolling(period).sum()
            sum_neg = neg.rolling(period).sum()
            return 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)

        @staticmethod
        def mom(series, period=10):
            return series.diff(period)

        @staticmethod
        def bop(df):
            # Balance of Power
            return (df['Close'] - df['Open']) / (df['High'] - df['Low'])

        @staticmethod
        def trix(series, period=15):
            ema1 = TechnicalIndicators.ema(series, period)
            ema2 = TechnicalIndicators.ema(ema1, period)
            ema3 = TechnicalIndicators.ema(ema2, period)
            return ema3.pct_change() * 100

        @staticmethod
        def stochrsi(series, period=14):
            rsi = TechnicalIndicators.rsi(series, period)
            min_rsi = rsi.rolling(period).min()
            max_rsi = rsi.rolling(period).max()
            return (rsi - min_rsi) / (max_rsi - min_rsi)

        @staticmethod
        def stddev(series, period=20):
            return series.rolling(period).std()

        @staticmethod
        def tsf(series, period=14):
            # Time Series Forecast (Linear Regression Forecast)
            # Simplified: Linear Reg at end point
            x = np.arange(period)
            def linreg_pred(y):
                if len(y) < period: return np.nan
                slope, intercept, _, _, _ = stats.linregress(x, y)
                return slope * (period - 1) + intercept
            return series.rolling(period).apply(linreg_pred, raw=True)

        @staticmethod
        def process_indicator(data):
            """Standardize input into clean float32 array"""
            if isinstance(data, pd.Series):
                return data.astype('float32').fillna(0).values
            elif isinstance(data, pd.DataFrame):
                return data.astype('float32').fillna(0).values
            elif isinstance(data, np.ndarray):
                return np.nan_to_num(data.astype('float32'))
            return np.array(data, dtype='float32')

        @classmethod
        def get_registry(cls):
            return {
                "RSI": {"func": cls.rsi, "params": {"period": 14}, "input": "Close", "outputs": ["RSI"]},
                "MACD": {"func": cls.macd, "params": {"fast": 12, "slow": 26, "signal": 9}, "input": "Close", "outputs": ["MACD", "MACD_Signal"]},
                "Bollinger": {"func": cls.bollinger_bands, "params": {"period": 20, "std_dev": 2}, "input": "Close", "outputs": ["BB_Upper", "BB_Lower"]},
                "ATR": {"func": cls.atr, "params": {"period": 14}, "input": "df", "outputs": ["ATR"]},
                "SMA200": {"func": cls.sma, "params": {"period": 200}, "input": "Close", "outputs": ["SMA200"]},
                "SMA50": {"func": cls.sma, "params": {"period": 50}, "input": "Close", "outputs": ["SMA50"]},
                "EMA20": {"func": cls.ema, "params": {"period": 20}, "input": "Close", "outputs": ["EMA20"]},
                "Stoch_K": {"func": cls.stochastic, "params": {"k_period": 14, "d_period": 3}, "input": "df", "outputs": ["Stoch_K", "Stoch_D"]},
                "ADX": {"func": cls.adx, "params": {"period": 14}, "input": "df", "outputs": ["ADX"]},
                "CCI": {"func": cls.cci, "params": {"period": 20}, "input": "df", "outputs": ["CCI"]},
                "WilliamsR": {"func": cls.williams_r, "params": {"period": 14}, "input": "df", "outputs": ["WilliamsR"]},
                "ROC": {"func": cls.roc, "params": {"period": 12}, "input": "Close", "outputs": ["ROC"]},
                "OBV": {"func": cls.obv, "params": {}, "input": "df", "outputs": ["OBV"]},
                "MFI": {"func": cls.mfi, "params": {"period": 14}, "input": "df", "outputs": ["MFI"]},
                "HMA20": {"func": cls.hma, "params": {"period": 20}, "input": "Close", "outputs": ["HMA20"]},
                "TEMA20": {"func": cls.tema, "params": {"period": 20}, "input": "Close", "outputs": ["TEMA20"]},
                "DEMA20": {"func": cls.dema, "params": {"period": 20}, "input": "Close", "outputs": ["DEMA20"]},
                "KAMA20": {"func": cls.kama, "params": {"period": 20}, "input": "Close", "outputs": ["KAMA20"]},
                "Aroon": {"func": cls.aroon, "params": {"period": 25}, "input": "df", "outputs": ["Aroon_Up", "Aroon_Down"]},
                "SuperTrend": {"func": cls.supertrend, "params": {"period": 10, "multiplier": 3}, "input": "df", "outputs": ["SuperTrend_Upper", "SuperTrend_Lower"]},
                "Parabolic_SAR": {"func": cls.parabolic_sar, "params": {}, "input": "df", "outputs": ["Parabolic_SAR"]},
                "TSI": {"func": cls.tsi, "params": {}, "input": "Close", "outputs": ["TSI"]},
                "UO": {"func": cls.uo, "params": {}, "input": "df", "outputs": ["UO"]},
                "KC": {"func": cls.keltner_channels, "params": {}, "input": "df", "outputs": ["KC_Upper", "KC_Lower"]},
                "DC": {"func": cls.donchian_channels, "params": {}, "input": "df", "outputs": ["DC_Upper", "DC_Lower"]},
                "Chaikin_Vol": {"func": cls.chaikin_volatility, "params": {}, "input": "df", "outputs": ["Chaikin_Vol"]},
                "CMF": {"func": cls.cmf, "params": {}, "input": "df", "outputs": ["CMF"]},
                "VWAP": {"func": cls.vwap, "params": {}, "input": "df", "outputs": ["VWAP"]},
                "AD_Line": {"func": cls.ad_line, "params": {}, "input": "df", "outputs": ["AD_Line"]},
                "Vortex": {"func": cls.vortex, "params": {}, "input": "df", "outputs": ["Vortex_Plus", "Vortex_Minus"]},
                "Chop": {"func": cls.chop, "params": {}, "input": "df", "outputs": ["Chop_Index"]},
                "KST": {"func": cls.kst, "params": {}, "input": "df", "outputs": ["KST"]},
                "Coppock": {"func": cls.coppock, "params": {}, "input": "df", "outputs": ["Coppock"]},
                "Ichimoku": {"func": cls.ichimoku, "params": {}, "input": "df", "outputs": ["Tenkan", "Kijun", "SpanA", "SpanB", "Chikou"]},
                "AO": {"func": cls.ao, "params": {}, "input": "df", "outputs": ["AO"]},
                "PPO": {"func": cls.ppo, "params": {}, "input": "df", "outputs": ["PPO"]},
                "Mass_Index": {"func": cls.mass_index, "params": {}, "input": "df", "outputs": ["Mass_Index"]},
                "Ulcer_Index": {"func": cls.ulcer_index, "params": {}, "input": "df", "outputs": ["Ulcer_Index"]},
                "WMA20": {"func": cls.wma, "params": {"period": 20}, "input": "Close", "outputs": ["WMA20"]},
                "TRIMA20": {"func": cls.trima, "params": {"period": 20}, "input": "Close", "outputs": ["TRIMA20"]},
                "CMO": {"func": cls.cmo, "params": {}, "input": "Close", "outputs": ["CMO"]},
                "MOM10": {"func": cls.mom, "params": {"period": 10}, "input": "Close", "outputs": ["MOM10"]},
                "BOP": {"func": cls.bop, "params": {}, "input": "df", "outputs": ["BOP"]},
                "TRIX": {"func": cls.trix, "params": {}, "input": "Close", "outputs": ["TRIX"]},
                "StochRSI": {"func": cls.stochrsi, "params": {}, "input": "Close", "outputs": ["StochRSI"]},
                "STDDEV": {"func": cls.stddev, "params": {}, "input": "Close", "outputs": ["STDDEV"]},
                "TSF": {"func": cls.tsf, "params": {}, "input": "Close", "outputs": ["TSF"]},
            }

    class StrategyLib:
        @staticmethod
        def get_signal_func(strategy_name):
            strategies = {
                "RSI Mean Reversion": StrategyLib.rsi_mean_reversion,
                "MACD Crossover": StrategyLib.macd_crossover,
                "Bollinger Breakout": StrategyLib.bollinger_breakout,
                "Golden/Death Cross": StrategyLib.golden_death_cross,
                "Stochastic Oscillator": StrategyLib.stochastic_oscillator,
                "CCI Momentum": StrategyLib.cci_momentum,
                "Williams %R Reversal": StrategyLib.williams_r_reversal,
                "HMA Trend": StrategyLib.hma_trend,
                "TEMA Crossover": StrategyLib.tema_crossover,
                "KAMA Trend": StrategyLib.kama_trend,
                "Aroon Oscillator": StrategyLib.aroon_oscillator,
                "SuperTrend Reversal": StrategyLib.supertrend_reversal,
                "Parabolic SAR": StrategyLib.parabolic_sar_strategy,
                "TSI Crossover": StrategyLib.tsi_crossover,
                "UO Overbought/Oversold": StrategyLib.uo_strategy,
                "Keltner Channel Breakout": StrategyLib.keltner_channel_breakout,
                "Donchian Channel Breakout": StrategyLib.donchian_channel_breakout,
                "Chaikin Volatility": StrategyLib.chaikin_volatility_strategy,
                "CMF Trend": StrategyLib.cmf_trend,
                "VWAP Crossover": StrategyLib.vwap_crossover,
                "AD Line Trend": StrategyLib.ad_line_trend,
                "Vortex Crossover": StrategyLib.vortex_crossover,
                "Choppiness Index Breakout": StrategyLib.choppiness_index_breakout,
                "KST Crossover": StrategyLib.kst_crossover,
                "Coppock Curve": StrategyLib.coppock_curve,
                "Ichimoku Cloud Breakout": StrategyLib.ichimoku_cloud_breakout,
                "Awesome Oscillator": StrategyLib.awesome_oscillator,
                "PPO Crossover": StrategyLib.ppo_crossover,
                "Mass Index Reversal": StrategyLib.mass_index_reversal,
                "Ulcer Index Safety": StrategyLib.ulcer_index_safety,
                "WMA Trend": StrategyLib.wma_trend,
                "TRIMA Crossover": StrategyLib.trima_crossover,
                "CMO Reversal": StrategyLib.cmo_reversal,
                "Momentum Breakout": StrategyLib.momentum_breakout,
                "BOP Trend": StrategyLib.bop_trend,
                "TRIX Crossover": StrategyLib.trix_crossover,
                "StochRSI Reversal": StrategyLib.stochrsi_reversal,
                "TSF Trend": StrategyLib.tsf_trend,
            }
            return strategies.get(strategy_name)

        @staticmethod
        def rsi_mean_reversion(df, params, cache=None):
            p = params.get('period', 14)
            if cache is not None and ('RSI', p) in cache:
                rsi = cache[('RSI', p)]
            else:
                rsi = TechnicalIndicators.rsi(df['Close'], p)
                if cache is not None: cache[('RSI', p)] = rsi
            
            prev = rsi.shift(1)
            curr = rsi
            os, ob = params.get('os', 30), params.get('ob', 70)
            long_sig = (prev < os) & (curr > os)
            short_sig = (prev > ob) & (curr < ob)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def macd_crossover(df, params, cache=None):
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            sig = params.get('signal', 9)
            
            key = ('MACD', fast, slow, sig)
            if cache is not None and key in cache:
                macd, signal_line = cache[key]
            else:
                macd, signal_line = TechnicalIndicators.macd(df['Close'], fast, slow, sig)
                if cache is not None: cache[key] = (macd, signal_line)

            long_sig = (macd.shift(1) < signal_line.shift(1)) & (macd > signal_line)
            short_sig = (macd.shift(1) > signal_line.shift(1)) & (macd < signal_line)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def bollinger_breakout(df, params, cache=None):
            p = params.get('period', 20)
            std = params.get('std_dev', 2)
            
            key = ('BB', p, std)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.bollinger_bands(df['Close'], p, std)
                if cache is not None: cache[key] = (upper, lower)

            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_lower, prev_upper = lower.shift(1), upper.shift(1)
            long_sig = (prev_close < prev_lower) & (curr_close > prev_lower)
            short_sig = (prev_close > prev_upper) & (curr_close < prev_upper)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def golden_death_cross(df, params, cache=None):
            fast_p = params.get('fast', 50)
            slow_p = params.get('slow', 200)
            
            key_fast = ('SMA', fast_p)
            if cache is not None and key_fast in cache:
                fast_ma = cache[key_fast]
            else:
                fast_ma = TechnicalIndicators.sma(df['Close'], fast_p)
                if cache is not None: cache[key_fast] = fast_ma
                
            key_slow = ('SMA', slow_p)
            if cache is not None and key_slow in cache:
                slow_ma = cache[key_slow]
            else:
                slow_ma = TechnicalIndicators.sma(df['Close'], slow_p)
                if cache is not None: cache[key_slow] = slow_ma

            long_sig = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)
            short_sig = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def stochastic_oscillator(df, params, cache=None):
            k_p = params.get('k_period', 14)
            d_p = params.get('d_period', 3)
            
            key = ('Stoch', k_p, d_p)
            if cache is not None and key in cache:
                k, d = cache[key]
            else:
                k, d = TechnicalIndicators.stochastic(df, k_p, d_p)
                if cache is not None: cache[key] = (k, d)

            prev_k, curr_k = k.shift(1), k
            os, ob = params.get('os', 20), params.get('ob', 80)
            long_sig = (prev_k < os) & (curr_k > os)
            short_sig = (prev_k > ob) & (curr_k < ob)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cci_momentum(df, params, cache=None):
            p = params.get('period', 20)
            key = ('CCI', p)
            if cache is not None and key in cache:
                cci = cache[key]
            else:
                cci = TechnicalIndicators.cci(df, p)
                if cache is not None: cache[key] = cci
                
            prev_cci, curr_cci = cci.shift(1), cci
            long_sig = (prev_cci < -100) & (curr_cci > -100)
            short_sig = (prev_cci > 100) & (curr_cci < 100)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def williams_r_reversal(df, params, cache=None):
            p = params.get('period', 14)
            key = ('WilliamsR', p)
            if cache is not None and key in cache:
                wr = cache[key]
            else:
                wr = TechnicalIndicators.williams_r(df, p)
                if cache is not None: cache[key] = wr
                
            prev_wr, curr_wr = wr.shift(1), wr
            long_sig = (prev_wr < -80) & (curr_wr > -80)
            short_sig = (prev_wr > -20) & (curr_wr < -20)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def hma_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('HMA', p)
            if cache is not None and key in cache:
                hma = cache[key]
            else:
                hma = TechnicalIndicators.hma(df['Close'], p)
                if cache is not None: cache[key] = hma
                
            long_sig = (hma.shift(1) > hma.shift(2)) & (df['Close'].shift(1) < hma.shift(1)) & (df['Close'] > hma)
            short_sig = (hma.shift(1) < hma.shift(2)) & (df['Close'].shift(1) > hma.shift(1)) & (df['Close'] < hma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tema_crossover(df, params, cache=None):
            p = params.get('period', 20)
            key = ('TEMA', p)
            if cache is not None and key in cache:
                tema = cache[key]
            else:
                tema = TechnicalIndicators.tema(df['Close'], p)
                if cache is not None: cache[key] = tema
                
            long_sig = (df['Close'].shift(1) < tema.shift(1)) & (df['Close'] > tema)
            short_sig = (df['Close'].shift(1) > tema.shift(1)) & (df['Close'] < tema)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def kama_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('KAMA', p)
            if cache is not None and key in cache:
                kama = cache[key]
            else:
                kama = TechnicalIndicators.kama(df['Close'], p)
                if cache is not None: cache[key] = kama
                
            prev_kama, curr_kama = kama.shift(1), kama
            long_sig = prev_kama < curr_kama
            short_sig = prev_kama > curr_kama
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def aroon_oscillator(df, params, cache=None):
            p = params.get('period', 25)
            key = ('Aroon', p)
            if cache is not None and key in cache:
                aroon_up, aroon_down = cache[key]
            else:
                aroon_up, aroon_down = TechnicalIndicators.aroon(df, p)
                if cache is not None: cache[key] = (aroon_up, aroon_down)
                
            prev_up, curr_up = aroon_up.shift(1), aroon_up
            prev_down, curr_down = aroon_down.shift(1), aroon_down
            long_sig = (prev_up < prev_down) & (curr_up > curr_down)
            short_sig = (prev_up > prev_down) & (curr_up < curr_down)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def supertrend_reversal(df, params, cache=None):
            p = params.get('period', 10)
            m = params.get('multiplier', 3)
            key = ('SuperTrend', p, m)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.supertrend(df, p, m)
                if cache is not None: cache[key] = (upper, lower)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_upper, curr_upper = upper.shift(1), upper
            prev_lower, curr_lower = lower.shift(1), lower
            long_sig = (prev_close < prev_upper) & (curr_close > curr_lower)
            short_sig = (prev_close > prev_lower) & (curr_close < curr_upper)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def parabolic_sar_strategy(df, params, cache=None):
            key = ('Parabolic_SAR',)
            if cache is not None and key in cache:
                sar = cache[key]
            else:
                sar = TechnicalIndicators.parabolic_sar(df)
                if cache is not None: cache[key] = sar
                
            prev_sar, curr_sar = sar.shift(1), sar
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_sar > prev_close) & (curr_sar < curr_close)
            short_sig = (prev_sar < prev_close) & (curr_sar > curr_close)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tsi_crossover(df, params, cache=None):
            key = ('TSI',)
            if cache is not None and key in cache:
                tsi = cache[key]
            else:
                tsi = TechnicalIndicators.tsi(df['Close'])
                if cache is not None: cache[key] = tsi
                
            prev_tsi, curr_tsi = tsi.shift(1), tsi
            long_sig = (prev_tsi < 0) & (curr_tsi > 0)
            short_sig = (prev_tsi > 0) & (curr_tsi < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def uo_strategy(df, params, cache=None):
            key = ('UO',)
            if cache is not None and key in cache:
                uo = cache[key]
            else:
                uo = TechnicalIndicators.uo(df)
                if cache is not None: cache[key] = uo
                
            prev_uo, curr_uo = uo.shift(1), uo
            long_sig = (prev_uo < 30) & (curr_uo > 30)
            short_sig = (prev_uo > 70) & (curr_uo < 70)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def keltner_channel_breakout(df, params, cache=None):
            key = ('KC',)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.keltner_channels(df)
                if cache is not None: cache[key] = (upper, lower)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_upper, curr_upper = upper.shift(1), upper
            prev_lower, curr_lower = lower.shift(1), lower
            long_sig = (prev_close < prev_upper) & (curr_close > curr_upper)
            short_sig = (prev_close > prev_lower) & (curr_close < curr_lower)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def donchian_channel_breakout(df, params, cache=None):
            key = ('DC',)
            if cache is not None and key in cache:
                upper, lower = cache[key]
            else:
                upper, lower = TechnicalIndicators.donchian_channels(df)
                if cache is not None: cache[key] = (upper, lower)
                
            curr_close = df['Close']
            prev_upper = upper.shift(1)
            prev_lower = lower.shift(1)
            long_sig = curr_close >= prev_upper
            short_sig = curr_close <= prev_lower
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def chaikin_volatility_strategy(df, params, cache=None):
            key = ('Chaikin_Vol',)
            if cache is not None and key in cache:
                cv = cache[key]
            else:
                cv = TechnicalIndicators.chaikin_volatility(df)
                if cache is not None: cache[key] = cv
                
            prev_cv, curr_cv = cv.shift(1), cv
            long_sig = (prev_cv < 0) & (curr_cv > 0)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cmf_trend(df, params, cache=None):
            key = ('CMF',)
            if cache is not None and key in cache:
                cmf = cache[key]
            else:
                cmf = TechnicalIndicators.cmf(df)
                if cache is not None: cache[key] = cmf
                
            prev_cmf, curr_cmf = cmf.shift(1), cmf
            long_sig = (prev_cmf < 0) & (curr_cmf > 0)
            short_sig = (prev_cmf > 0) & (curr_cmf < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def vwap_crossover(df, params, cache=None):
            key = ('VWAP',)
            if cache is not None and key in cache:
                vwap = cache[key]
            else:
                vwap = TechnicalIndicators.vwap(df)
                if cache is not None: cache[key] = vwap
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_vwap, curr_vwap = vwap.shift(1), vwap
            long_sig = (prev_close < prev_vwap) & (curr_close > curr_vwap)
            short_sig = (prev_close > prev_vwap) & (curr_close < curr_vwap)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ad_line_trend(df, params, cache=None):
            key = ('AD_Line',)
            if cache is not None and key in cache:
                ad = cache[key]
            else:
                ad = TechnicalIndicators.ad_line(df)
                if cache is not None: cache[key] = ad
                
            prev_ad, curr_ad = ad.shift(1), ad
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_ad < curr_ad) & (prev_close > curr_close)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def vortex_crossover(df, params, cache=None):
            key = ('Vortex',)
            if cache is not None and key in cache:
                vp, vm = cache[key]
            else:
                vp, vm = TechnicalIndicators.vortex(df)
                if cache is not None: cache[key] = (vp, vm)
                
            prev_vp, curr_vp = vp.shift(1), vp
            prev_vm, curr_vm = vm.shift(1), vm
            long_sig = (prev_vp < prev_vm) & (curr_vp > curr_vm)
            short_sig = (prev_vp > prev_vm) & (curr_vp < curr_vm)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def choppiness_index_breakout(df, params, cache=None):
            key = ('Chop',)
            if cache is not None and key in cache:
                chop = cache[key]
            else:
                chop = TechnicalIndicators.chop(df)
                if cache is not None: cache[key] = chop
                
            prev_chop, curr_chop = chop.shift(1), chop
            long_sig = (prev_chop > 61.8) & (curr_chop < 61.8)
            short_sig = (prev_chop < 38.2) & (curr_chop > 38.2)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def kst_crossover(df, params, cache=None):
            key = ('KST',)
            if cache is not None and key in cache:
                kst = cache[key]
            else:
                kst = TechnicalIndicators.kst(df)
                if cache is not None: cache[key] = kst
                
            prev_kst, curr_kst = kst.shift(1), kst
            long_sig = (prev_kst < 0) & (curr_kst > 0)
            short_sig = (prev_kst > 0) & (curr_kst < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def coppock_curve(df, params, cache=None):
            key = ('Coppock',)
            if cache is not None and key in cache:
                cop = cache[key]
            else:
                cop = TechnicalIndicators.coppock(df)
                if cache is not None: cache[key] = cop
                
            prev_cop, curr_cop = cop.shift(1), cop
            long_sig = (prev_cop < 0) & (curr_cop > 0)
            short_sig = (prev_cop > 0) & (curr_cop < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ichimoku_cloud_breakout(df, params, cache=None):
            key = ('Ichimoku',)
            if cache is not None and key in cache:
                tenkan, kijun, span_a, span_b, chikou = cache[key]
            else:
                tenkan, kijun, span_a, span_b, chikou = TechnicalIndicators.ichimoku(df)
                if cache is not None: cache[key] = (tenkan, kijun, span_a, span_b, chikou)
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_span_a, curr_span_a = span_a.shift(1), span_a
            curr_span_b = span_b
            long_sig = (prev_close < prev_span_a) & (curr_close > curr_span_a) & (curr_close > curr_span_b)
            short_sig = (prev_close > prev_span_a) & (curr_close < curr_span_a) & (curr_close < curr_span_b)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def awesome_oscillator(df, params, cache=None):
            key = ('AO',)
            if cache is not None and key in cache:
                ao = cache[key]
            else:
                ao = TechnicalIndicators.ao(df)
                if cache is not None: cache[key] = ao
                
            prev_ao, curr_ao = ao.shift(1), ao
            long_sig = (prev_ao < 0) & (curr_ao > 0)
            short_sig = (prev_ao > 0) & (curr_ao < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ppo_crossover(df, params, cache=None):
            key = ('PPO',)
            if cache is not None and key in cache:
                ppo = cache[key]
            else:
                ppo = TechnicalIndicators.ppo(df)
                if cache is not None: cache[key] = ppo
                
            prev_ppo, curr_ppo = ppo.shift(1), ppo
            long_sig = (prev_ppo < 0) & (curr_ppo > 0)
            short_sig = (prev_ppo > 0) & (curr_ppo < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def mass_index_reversal(df, params, cache=None):
            key = ('Mass_Index',)
            if cache is not None and key in cache:
                mi = cache[key]
            else:
                mi = TechnicalIndicators.mass_index(df)
                if cache is not None: cache[key] = mi
                
            prev_mi, curr_mi = mi.shift(1), mi
            long_sig = (prev_mi > 27) & (curr_mi < 27)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def ulcer_index_safety(df, params, cache=None):
            key = ('Ulcer_Index',)
            if cache is not None and key in cache:
                ui = cache[key]
            else:
                ui = TechnicalIndicators.ulcer_index(df)
                if cache is not None: cache[key] = ui
                
            prev_ui, curr_ui = ui.shift(1), ui
            long_sig = (prev_ui > 5) & (curr_ui < 5)
            short_sig = pd.Series(False, index=df.index)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def wma_trend(df, params, cache=None):
            p = params.get('period', 20)
            key = ('WMA', p)
            if cache is not None and key in cache:
                wma = cache[key]
            else:
                wma = TechnicalIndicators.wma(df['Close'], p)
                if cache is not None: cache[key] = wma
                
            prev_wma, curr_wma = wma.shift(1), wma
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_wma < curr_wma) & (prev_close > curr_wma)
            short_sig = (prev_wma > curr_wma) & (prev_close < curr_wma)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def trima_crossover(df, params, cache=None):
            p = params.get('period', 20)
            key = ('TRIMA', p)
            if cache is not None and key in cache:
                trima = cache[key]
            else:
                trima = TechnicalIndicators.trima(df['Close'], p)
                if cache is not None: cache[key] = trima
                
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            prev_trima, curr_trima = trima.shift(1), trima
            long_sig = (prev_close < prev_trima) & (curr_close > curr_trima)
            short_sig = (prev_close > prev_trima) & (curr_close < curr_trima)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def cmo_reversal(df, params, cache=None):
            key = ('CMO',)
            if cache is not None and key in cache:
                cmo = cache[key]
            else:
                cmo = TechnicalIndicators.cmo(df['Close'])
                if cache is not None: cache[key] = cmo
                
            prev_cmo, curr_cmo = cmo.shift(1), cmo
            long_sig = (prev_cmo < -50) & (curr_cmo > -50)
            short_sig = (prev_cmo > 50) & (curr_cmo < 50)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def momentum_breakout(df, params, cache=None):
            p = params.get('period', 10)
            key = ('MOM', p)
            if cache is not None and key in cache:
                mom = cache[key]
            else:
                mom = TechnicalIndicators.mom(df['Close'], p)
                if cache is not None: cache[key] = mom
                
            prev_mom, curr_mom = mom.shift(1), mom
            long_sig = (prev_mom < 0) & (curr_mom > 0)
            short_sig = (prev_mom > 0) & (curr_mom < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def bop_trend(df, params, cache=None):
            key = ('BOP',)
            if cache is not None and key in cache:
                bop = cache[key]
            else:
                bop = TechnicalIndicators.bop(df)
                if cache is not None: cache[key] = bop
                
            prev_bop, curr_bop = bop.shift(1), bop
            long_sig = (prev_bop < 0) & (curr_bop > 0)
            short_sig = (prev_bop > 0) & (curr_bop < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def trix_crossover(df, params, cache=None):
            key = ('TRIX',)
            if cache is not None and key in cache:
                trix = cache[key]
            else:
                trix = TechnicalIndicators.trix(df['Close'])
                if cache is not None: cache[key] = trix
                
            prev_trix, curr_trix = trix.shift(1), trix
            long_sig = (prev_trix < 0) & (curr_trix > 0)
            short_sig = (prev_trix > 0) & (curr_trix < 0)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def stochrsi_reversal(df, params, cache=None):
            key = ('StochRSI',)
            if cache is not None and key in cache:
                srsi = cache[key]
            else:
                srsi = TechnicalIndicators.stochrsi(df['Close'])
                if cache is not None: cache[key] = srsi
                
            prev_srsi, curr_srsi = srsi.shift(1), srsi
            long_sig = (prev_srsi < 0.2) & (curr_srsi > 0.2)
            short_sig = (prev_srsi > 0.8) & (curr_srsi < 0.8)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

        @staticmethod
        def tsf_trend(df, params, cache=None):
            key = ('TSF',)
            if cache is not None and key in cache:
                tsf = cache[key]
            else:
                tsf = TechnicalIndicators.tsf(df['Close'])
                if cache is not None: cache[key] = tsf
                
            prev_tsf, curr_tsf = tsf.shift(1), tsf
            prev_close, curr_close = df['Close'].shift(1), df['Close']
            long_sig = (prev_tsf < curr_tsf) & (prev_close > curr_tsf)
            short_sig = (prev_tsf > curr_tsf) & (prev_close < curr_tsf)
            return long_sig.reindex(df.index, fill_value=False), short_sig.reindex(df.index, fill_value=False)

    class BacktestEngine:
        def __init__(self, ticker, start_date, end_date, timeframe, initial_capital=10000, target_tz="America/New_York"):
            self.ticker = ticker
            self.start_date = start_date
            self.end_date = end_date
            self.timeframe = timeframe
            self.initial_capital = initial_capital
            self.target_tz = target_tz
            self.df = pd.DataFrame()

        def fetch_data(self):
            self.df = fetch_data_smart(self.ticker, self.timeframe, self.start_date, self.end_date, target_tz=self.target_tz)
            return not self.df.empty

        def add_technical_indicators(self):
            if self.df.empty: return
            
            registry = TechnicalIndicators.get_registry()
            
            for name, config in registry.items():
                try:
                    # Prepare Input
                    if config['input'] == 'df':
                        data = self.df
                    else:
                        data = self.df[config['input']]
                    
                    # Execute Calculation
                    result = config['func'](data, **config['params'])
                    
                    # Handle Outputs
                    if isinstance(result, tuple):
                        for i, out_col in enumerate(config['outputs']):
                            if i < len(result):
                                self.df[out_col] = result[i]
                    else:
                        if config['outputs']:
                            self.df[config['outputs'][0]] = result
                            
                except Exception as e:
                    pass

            # Memory Optimization: Convert all float64 to float32
            # This is crucial for 50k+ candles on Streamlit Cloud
            cols = self.df.select_dtypes(include=['float64']).columns
            if not cols.empty:
                self.df[cols] = self.df[cols].astype('float32')
            
            # Data Integrity: Use ffill().bfill() instead of dropna to keep full dataframe length
            self.df.ffill().bfill(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        def add_gex_levels(self, sensitivity=1.5):
            if self.df.empty: return
            
            import numpy as np
            
            # --- NUOVA LOGICA PROXY GEX (VWAP & Volume-Weighted SD) ---
            # 1. Calcolo del Prezzo Tipico e VP (Volume * Price)
            self.df['Typical_Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3.0
            self.df['VP'] = self.df['Typical_Price'] * self.df['Volume']
            
            # 2. Logica differenziata: Anchored Daily per Intraday, Rolling per Daily
            if self.timeframe.upper() in ['1D', '1DAY']:
                # Rolling VWAP (20 periodi) per grafici giornalieri
                window = 20
                cum_vp = self.df['VP'].rolling(window=window, min_periods=1).sum()
                cum_vol = self.df['Volume'].rolling(window=window, min_periods=1).sum()
                self.df['ZeroGamma'] = cum_vp / cum_vol
                
                # Calcolo Varianza e Deviazione Standard ponderata ai volumi
                self.df['Price_Diff_Sq'] = ((self.df['Typical_Price'] - self.df['ZeroGamma']) ** 2) * self.df['Volume']
                vwap_var = self.df['Price_Diff_Sq'].rolling(window=window, min_periods=1).sum() / cum_vol
                vwap_sd = np.sqrt(vwap_var)
                
            else:
                # Anchored VWAP giornaliero per grafici Intraday (1H, 15Min, 5Min, 1Min)
                self.df['Date'] = self.df['datetime'].dt.date
                daily_groups = self.df.groupby('Date')
                
                cum_vp = daily_groups['VP'].cumsum()
                cum_vol = daily_groups['Volume'].cumsum()
                self.df['ZeroGamma'] = cum_vp / cum_vol
                
                # Calcolo Varianza e Deviazione Standard ancorata alla giornata
                self.df['Price_Diff_Sq'] = ((self.df['Typical_Price'] - self.df['ZeroGamma']) ** 2) * self.df['Volume']
                cum_price_diff_sq = daily_groups['Price_Diff_Sq'].cumsum()
                
                vwap_var = cum_price_diff_sq / cum_vol
                vwap_sd = np.sqrt(vwap_var)

            # --- DEFINIZIONE MURI (WALLS) DINAMICI ---
            self.df['CallWall'] = self.df['ZeroGamma'] + (vwap_sd * sensitivity)
            self.df['PutWall'] = self.df['ZeroGamma'] - (vwap_sd * sensitivity)
            
            # Pulizia colonne temporanee per ottimizzare la RAM
            cols_to_drop = ['Typical_Price', 'VP', 'Date', 'Price_Diff_Sq']
            self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)
            
            # FFill e BFill per sicurezza
            self.df['ZeroGamma'] = self.df['ZeroGamma'].ffill().bfill()
            self.df['CallWall'] = self.df['CallWall'].ffill().bfill()
            self.df['PutWall'] = self.df['PutWall'].ffill().bfill()

        def optimize_strategy(self, strategy_type, param_ranges, risk_reward, risk_per_trade, time_ranges=None):
            import itertools
            import time

            # 1. Data Preparation (Vectorized & Float32)
            # Ensure base indicators are present (ATR is critical for SL/TP)
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()

            # Work with a copy to avoid side effects, convert to float32 for memory/speed
            # We use numpy arrays directly for the optimization loop
            opens = self.df['Open'].values.astype(np.float32)
            highs = self.df['High'].values.astype(np.float32)
            lows = self.df['Low'].values.astype(np.float32)
            closes = self.df['Close'].values.astype(np.float32)
            atrs = self.df['ATR'].values.astype(np.float32)
            dates = self.df['datetime'].values
            n_candles = len(closes)

            # 2. Parameter & Time Setup
            keys = list(param_ranges.keys()) if param_ranges else []
            values = [param_ranges[k] for k in keys] if keys else []
            param_combinations = list(itertools.product(*values)) if values else [()]

            time_combinations = [(None, None)]
            if time_ranges:
                start_times = time_ranges.get('start_times', [])
                end_times = time_ranges.get('end_times', [])
                if start_times and end_times:
                    time_combinations = list(itertools.product(start_times, end_times))

            total_iterations = len(param_combinations) * len(time_combinations)
            
            # Results containers
            best_win_rate = 0.0
            best_pnl = -float('inf')
            best_params = {}
            best_time_config = {'start': None, 'end': None}
            results = []

            # UI Feedback
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time_perf = time.time()
            counter = 0

            # Memoization for expensive indicator calculations
            # Key: (IndicatorName, ParamValue), Value: NumpyArray
            indicator_cache = {}

            # 3. Optimization Loop
            for t_start, t_end in time_combinations:
                # Pre-calculate Time Mask
                time_mask = np.ones(n_candles, dtype=bool)
                if t_start and t_end:
                    # Vectorized Time Filtering
                    # Convert datetime objects to minutes from midnight for fast comparison
                    # We assume dates are numpy datetime64[ns]
                    dt_index = pd.to_datetime(dates)
                    minutes = dt_index.hour * 60 + dt_index.minute
                    start_min = t_start.hour * 60 + t_start.minute
                    end_min = t_end.hour * 60 + t_end.minute
                    time_mask = (minutes >= start_min) & (minutes <= end_min)

                for params_tuple in param_combinations:
                    counter += 1
                    params = dict(zip(keys, params_tuple)) if keys else {}

                    # Update UI periodically
                    if counter % 10 == 0 or counter == total_iterations:
                        progress = counter / total_iterations
                        elapsed = time.time() - start_time_perf
                        rem = (elapsed / progress) - elapsed if progress > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"AI Optimization: {counter}/{total_iterations} | Best WR: {best_win_rate:.1f}%")

                    # --- A. Signal Generation (Vectorized) ---
                    signals = np.zeros(n_candles, dtype=np.int8) # 0: None, 1: Long, -1: Short

                    try:
                        # Use StrategyLib for signal generation
                        signal_func = StrategyLib.get_signal_func(strategy_type)
                        if signal_func:
                            import inspect
                            sig_params = inspect.signature(signal_func).parameters
                            if 'cache' in sig_params:
                                long_sig, short_sig = signal_func(self.df, params, cache=indicator_cache)
                            else:
                                long_sig, short_sig = signal_func(self.df, params)
                            
                            # Handle potential NaNs in signals
                            if isinstance(long_sig, pd.Series):
                                long_sig = long_sig.fillna(False).values
                            if isinstance(short_sig, pd.Series):
                                short_sig = short_sig.fillna(False).values
                                
                            print(f'Testing {strategy_type} with {params} -> Raw signals: {np.sum(long_sig | short_sig)}')
                                
                            signals[long_sig] = 1
                            signals[short_sig] = -1
                        else:
                            # Fallback or continue
                            continue

                    except Exception as e:
                        print(f"Strategy Error in {strategy_type} with params {params}: {e}")
                        continue

                    raw_signal_count = np.sum(signals != 0)

                    # Apply Time Mask
                    signals[~time_mask] = 0
                    final_signal_count = np.sum(signals != 0)

                    print(f"DEBUG: Params {params} generated {final_signal_count} signals")
                    if final_signal_count == 0:
                        if raw_signal_count == 0:
                            print(f"  -> Reason: Indicator produced 0 signals (possibly all NaNs or no crossovers).")
                        else:
                            print(f"  -> Reason: Time filter blocked all {raw_signal_count} signals.")
                        continue

                    # --- B. Vectorized Trade Simulation (Isolation & Next-Bar) ---
                    # Identify potential entry points (Signal at T -> Entry at T+1)
                    sig_indices = np.where(signals != 0)[0]
                    if len(sig_indices) == 0: continue

                    trade_pnl = []
                    last_exit_idx = -1

                    # Fast Loop over Signals (NOT Candles)
                    for idx in sig_indices:
                        entry_idx = idx + 1
                        if entry_idx >= n_candles: break
                        
                        # Trade Isolation
                        if entry_idx <= last_exit_idx: continue

                        # Setup
                        direction = signals[idx]
                        entry_price = opens[entry_idx] # Entry at Open T+1
                        atr_val = atrs[idx] # ATR at Signal Candle
                        
                        if np.isnan(atr_val) or atr_val == 0:
                            atr_val = entry_price * 0.01  # Fallback: 1% del prezzo se ATR è rotto

                        sl_dist = atr_val * 1.5 # Fixed as per original
                        
                        if direction == 1: # Long
                            sl = entry_price - sl_dist
                            tp = entry_price + (sl_dist * risk_reward)
                            
                            # Vectorized Exit Search
                            future_lows = lows[entry_idx:]
                            future_highs = highs[entry_idx:]
                            
                            sl_hit = future_lows <= sl
                            tp_hit = future_highs >= tp
                            
                            # Find first occurrence
                            first_sl = np.argmax(sl_hit) if sl_hit.any() else n_candles
                            first_tp = np.argmax(tp_hit) if tp_hit.any() else n_candles
                            
                            # Determine Outcome
                            if first_sl == n_candles and first_tp == n_candles:
                                last_exit_idx = n_candles # Held till end
                            elif first_sl <= first_tp: # SL hit first or same candle (Conservative)
                                trade_pnl.append(-sl_dist)
                                last_exit_idx = entry_idx + first_sl
                            else: # TP hit first
                                trade_pnl.append(sl_dist * risk_reward)
                                last_exit_idx = entry_idx + first_tp
                                
                        else: # Short
                            sl = entry_price + sl_dist
                            tp = entry_price - (sl_dist * risk_reward)
                            
                            future_lows = lows[entry_idx:]
                            future_highs = highs[entry_idx:]
                            
                            sl_hit = future_highs >= sl
                            tp_hit = future_lows <= tp
                            
                            first_sl = np.argmax(sl_hit) if sl_hit.any() else n_candles
                            first_tp = np.argmax(tp_hit) if tp_hit.any() else n_candles
                            
                            if first_sl == n_candles and first_tp == n_candles:
                                last_exit_idx = n_candles
                            elif first_sl <= first_tp:
                                trade_pnl.append(-sl_dist)
                                last_exit_idx = entry_idx + first_sl
                            else:
                                trade_pnl.append(sl_dist * risk_reward)
                                last_exit_idx = entry_idx + first_tp

                    # --- C. Metrics & Best Selection ---
                    if trade_pnl:
                        pnl_arr = np.array(trade_pnl)
                        wins = np.sum(pnl_arr > 0)
                        count = len(pnl_arr)
                        wr = (wins / count) * 100
                        tot_pnl = np.sum(pnl_arr)
                        
                        if wr > best_win_rate:
                            best_win_rate = wr
                            best_pnl = tot_pnl
                            best_params = params
                            best_time_config = {'start': t_start, 'end': t_end}
                        elif wr == best_win_rate and tot_pnl > best_pnl:
                            best_pnl = tot_pnl
                            best_params = params
                            best_time_config = {'start': t_start, 'end': t_end}
                            
                        results.append({
                            'params': params,
                            'time_config': {'start': t_start, 'end': t_end},
                            'win_rate': wr,
                            'trades': count,
                            'pnl': tot_pnl
                        })

            progress_bar.empty()
            status_text.empty()
            return best_params, best_time_config, best_win_rate, results

        def optimize_hybrid_strategy(self, param_ranges, time_ranges=None):
            best_win_rate = 0
            best_params = {}
            best_time_config = {'start': None, 'end': None}
            results = []
            
            keys = list(param_ranges.keys())
            import itertools
            values = [param_ranges[k] for k in keys]
            param_combinations = list(itertools.product(*values))
            
            time_combinations = [(None, None)]
            if time_ranges:
                start_times = time_ranges.get('start_times', [])
                end_times = time_ranges.get('end_times', [])
                if start_times and end_times:
                    time_combinations = list(itertools.product(start_times, end_times))

            total_iterations = len(param_combinations) * len(time_combinations)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            counter = 0
            start_time_perf = time.time()

            for t_start, t_end in time_combinations:
                if t_start and t_end and t_start >= t_end: continue

                for combo in param_combinations:
                    counter += 1
                    # combo: (long_trigger, short_trigger, sensitivity, rr, risk_pct)
                    # Map keys to args
                    p = dict(zip(keys, combo))
                    
                    if counter % 5 == 0:
                        progress = counter / total_iterations
                        elapsed = time.time() - start_time_perf
                        rem = (elapsed / progress) - elapsed if progress > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Ottimizzazione GEX... {counter}/{total_iterations} (Rimanente: {rem:.1f}s)")

                    # Recalculate GEX levels if sensitivity changes? 
                    # Optimization: GEX levels depend on sensitivity. 
                    # Ideally we pre-calculate, but here we might need to re-run add_gex_levels if sensitivity changes.
                    # However, add_gex_levels modifies self.df in place. This is tricky for optimization loop.
                    # Solution: Calculate GEX walls dynamically inside run_hybrid or reset DF.
                    # Better: Pre-calculate GEX for all sensitivity levels in ranges? No, memory heavy.
                    # Acceptable: Re-calculate GEX inside loop (slower) or clone DF.
                    # Let's clone DF for safety or modify run_hybrid to accept walls directly?
                    # For now, let's re-run add_gex_levels. It's fast enough for vector ops.
                    
                    self.add_gex_levels(p['sensitivity'])
                    
                    trades, _ = self.run_hybrid_strategy(
                        p['long_trigger'], p['short_trigger'], p['rr'], p['risk_pct'], 
                        start_time=t_start, end_time=t_end, entry_mode=p.get('entry_mode', 'Standard')
                    )
                    
                    if trades:
                        df_res = pd.DataFrame(trades)
                        win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
                        
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_params = p
                            best_time_config = {'start': t_start, 'end': t_end}
                            
                        results.append({
                            'params': p,
                            'time_config': {'start': t_start, 'end': t_end},
                            'win_rate': win_rate,
                            'trades': len(trades),
                            'pnl': df_res['pnl'].sum()
                        })

            progress_bar.empty()
            status_text.empty()
            return best_params, best_time_config, best_win_rate, results

        def run_hybrid_strategy(self, long_trigger, short_trigger, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard"):
            trades = []
            balance = self.initial_capital
            equity_curve = [balance]
            position = None

            for i in range(len(self.df)):
                if i < 200: continue # Warmup
                curr = self.df.iloc[i]
                prev = self.df.iloc[i-1]
                
                # Schedule Check
                if start_time and end_time:
                    curr_time = curr['datetime'].time()
                    if not (start_time <= curr_time <= end_time):
                        equity_curve.append(balance)
                        continue

                # Exit Logic
                if position:
                    exit_res = None
                    if position['type'] == 'long':
                        if curr['Low'] <= position['sl']: exit_res, exit_price = 'LOSS', position['sl']
                        elif curr['High'] >= position['tp']: exit_res, exit_price = 'WIN', position['tp']
                    else:
                        if curr['High'] >= position['sl']: exit_res, exit_price = 'LOSS', position['sl']
                        elif curr['Low'] <= position['tp']: exit_res, exit_price = 'WIN', position['tp']
                    
                    if exit_res:
                        pnl = (exit_price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - exit_price) * position['size']
                        balance += pnl
                        trades.append({'time': curr['datetime'], 'type': f'EXIT {exit_res}', 'price': exit_price, 'pnl': pnl, 'balance': balance, 'Logica': f"Exit: {exit_res}"})
                        position = None
                    equity_curve.append(balance)
                    continue

                # Entry Logic (Hybrid GEX)
                signal = None
                # Long
                if long_trigger == "Bounce Put Wall" and prev['Low'] <= prev['PutWall'] and curr['Close'] > prev['PutWall']: signal = 'long'
                elif long_trigger == "Breakout 0-Gamma" and prev['Close'] < prev['ZeroGamma'] and curr['Close'] > prev['ZeroGamma']: signal = 'long'
                elif long_trigger == "Breakout Call Wall" and prev['Close'] < prev['CallWall'] and curr['Close'] > prev['CallWall']: signal = 'long'
                
                # Short
                if short_trigger == "Bounce Call Wall" and prev['High'] >= prev['CallWall'] and curr['Close'] < prev['CallWall']: signal = 'short'
                elif short_trigger == "Breakdown 0-Gamma" and prev['Close'] > prev['ZeroGamma'] and curr['Close'] < prev['ZeroGamma']: signal = 'short'
                elif short_trigger == "Breakdown Put Wall" and prev['Close'] > prev['PutWall'] and curr['Close'] < prev['PutWall']: signal = 'short'

                if signal:
                    entry_price = curr['Close']
                    sl_dist = curr['ATR'] * 1.5
                    
                    if entry_mode == "Retest":
                        entry_price = (curr['High'] + curr['Low']) / 2
                    
                    risk_amt = balance * (risk_per_trade / 100)
                    logic_str = f"Trigger: {long_trigger if signal == 'long' else short_trigger} | RR: {risk_reward} -> Signal: {signal.upper()}"
                    
                    if signal == 'long':
                        sl = entry_price - sl_dist
                        tp = entry_price + (sl_dist * risk_reward)
                        size = risk_amt / (entry_price - sl) if (entry_price - sl) > 0 else 0
                        if size > 0:
                            position = {'type': 'long', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY LONG', 'price': entry_price, 'pnl': 0, 'balance': balance, 'Logica': logic_str})
                    else:
                        sl = entry_price + sl_dist
                        tp = entry_price - (sl_dist * risk_reward)
                        size = risk_amt / (sl - entry_price) if (sl - entry_price) > 0 else 0
                        if size > 0:
                            position = {'type': 'short', 'entry': entry_price, 'sl': sl, 'tp': tp, 'size': size}
                            trades.append({'time': curr['datetime'], 'type': 'ENTRY SHORT', 'price': entry_price, 'pnl': 0, 'balance': balance, 'Logica': logic_str})
                
                equity_curve.append(balance)
                
            return trades, equity_curve

        def execute_trades_agnostic(self, signals, risk_reward, risk_per_trade, sl_atr_mult=1.5, start_time=None, end_time=None, strategy_name="", params=None):
            # Prepare Data Arrays
            opens = self.df['Open'].values
            highs = self.df['High'].values
            lows = self.df['Low'].values
            closes = self.df['Close'].values
            datetimes = self.df['datetime'].values
            atrs = self.df['ATR'].values if 'ATR' in self.df.columns else np.zeros_like(closes)
            
            n_candles = len(closes)
            trades = []
            
            # Equity Curve Initialization
            equity_curve = np.full(n_candles, self.initial_capital, dtype=np.float32)
            balance = self.initial_capital
            
            last_exit_idx = -1
            
            # Diagnostics
            diag = {
                'total_signals': 0,
                'skipped_isolation': 0,
                'skipped_time': 0,
                'skipped_invalid_atr': 0,
                'skipped_size': 0,
                'executed': 0
            }
            
            # Get indices where signal is not 0 (up to n-2)
            sig_indices = np.where(signals[:-1] != 0)[0]
            diag['total_signals'] = len(sig_indices)
            
            for idx in sig_indices:
                entry_idx = idx + 1
                
                # Trade Isolation
                if entry_idx <= last_exit_idx:
                    diag['skipped_isolation'] += 1
                    continue
                
                # Time Check
                if start_time and end_time:
                    dt = pd.Timestamp(datetimes[idx])
                    t = dt.time()
                    if not (start_time <= t <= end_time):
                        diag['skipped_time'] += 1
                        continue

                direction = signals[idx] # 1 or -1
                entry_price = opens[entry_idx]
                entry_time = datetimes[entry_idx]
                atr_val = atrs[idx]
                
                if np.isnan(atr_val) or atr_val == 0: 
                    atr_val = entry_price * 0.01
                    diag['skipped_invalid_atr'] += 1
                
                sl_dist = atr_val * sl_atr_mult
                
                # Position Sizing
                risk_amount = balance * (risk_per_trade / 100.0)
                size = risk_amount / sl_dist if sl_dist > 0 else 0
                
                if size <= 0: 
                    diag['skipped_size'] += 1
                    continue

                if direction == 1: # Long
                    sl = entry_price - sl_dist
                    tp = entry_price + (sl_dist * risk_reward)
                    
                    future_lows = lows[entry_idx:]
                    future_highs = highs[entry_idx:]
                    
                    sl_hit_mask = future_lows <= sl
                    tp_hit_mask = future_highs >= tp
                    
                    first_sl = np.argmax(sl_hit_mask) if sl_hit_mask.any() else n_candles
                    first_tp = np.argmax(tp_hit_mask) if tp_hit_mask.any() else n_candles
                    
                    if first_sl == n_candles and first_tp == n_candles:
                        last_exit_idx = n_candles
                        continue
                    elif first_sl <= first_tp:
                        exit_idx = entry_idx + first_sl
                        exit_price = sl
                        exit_type = "SL"
                    else:
                        exit_idx = entry_idx + first_tp
                        exit_price = tp
                        exit_type = "TP"
                        
                    pnl = (exit_price - entry_price) * size
                        
                else: # Short
                    sl = entry_price + sl_dist
                    tp = entry_price - (sl_dist * risk_reward)
                    
                    future_lows = lows[entry_idx:]
                    future_highs = highs[entry_idx:]
                    
                    sl_hit_mask = future_highs >= sl
                    tp_hit_mask = future_lows <= tp
                    
                    first_sl = np.argmax(sl_hit_mask) if sl_hit_mask.any() else n_candles
                    first_tp = np.argmax(tp_hit_mask) if tp_hit_mask.any() else n_candles
                    
                    if first_sl == n_candles and first_tp == n_candles:
                        last_exit_idx = n_candles
                        continue
                    elif first_sl <= first_tp:
                        exit_idx = entry_idx + first_sl
                        exit_price = sl
                        exit_type = "SL"
                    else:
                        exit_idx = entry_idx + first_tp
                        exit_price = tp
                        exit_type = "TP"
                        
                    pnl = (entry_price - exit_price) * size

                # Record Trade
                balance += pnl
                logic_str = f"Trigger: {strategy_name} | Params: {params} -> Signal: {'LONG' if direction == 1 else 'SHORT'}"
                trades.append({
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time': datetimes[exit_idx],
                    'Exit Price': exit_price,
                    'pnl': pnl,
                    'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                    'Direction': 'long' if direction == 1 else 'short',
                    'Status': exit_type,
                    'type': 'ENTRY LONG' if direction == 1 else 'ENTRY SHORT',
                    'time': entry_time,
                    'price': entry_price,
                    'Logica': logic_str
                })
                
                if exit_idx < n_candles:
                    equity_curve[exit_idx:] = balance
                
                last_exit_idx = exit_idx
                diag['executed'] += 1

            return trades, equity_curve, diag

        def run_technical_strategy(self, strategy_type, params, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard", sl_atr_mult=1.5):
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()
            
            n_candles = len(self.df)
            signal_series = np.zeros(n_candles, dtype=np.int8)
            
            try:
                signal_func = StrategyLib.get_signal_func(strategy_type)
                if signal_func:
                    import inspect
                    sig_params = inspect.signature(signal_func).parameters
                    if 'cache' in sig_params:
                        long_sig, short_sig = signal_func(self.df, params, cache={})
                    else:
                        long_sig, short_sig = signal_func(self.df, params)
                    
                    signal_series[long_sig] = 1
                    signal_series[short_sig] = -1
            except Exception as e:
                st.error(f"Signal Gen Error: {e}")
                return [], []

            trades, equity, diag = self.execute_trades_agnostic(signal_series, risk_reward, risk_per_trade, sl_atr_mult, start_time, end_time, strategy_name=strategy_type, params=params)
            
            if len(trades) == 0:
                st.warning(f"⚠️ Nessuna operazione eseguita. Diagnostica: {diag}")
                
            return trades, equity

        def run_technical_strategy_old(self, strategy_type, params, risk_reward, risk_per_trade, start_time=None, end_time=None, entry_mode="Standard", sl_atr_mult=1.5):
            trades = []
            balance = self.initial_capital
            equity_curve = [balance] * len(self.df)
            
            if 'ATR' not in self.df.columns:
                self.add_technical_indicators()
            
            # Convert to records for speed
            records = self.df.to_dict('records')
            n_candles = len(records)
            
            position = None # {type, entry_price, sl, tp, size, entry_time}
            
            # --- Pre-calculate Signals ---
            signal_series = np.zeros(n_candles, dtype=np.int8)
            try:
                signal_func = StrategyLib.get_signal_func(strategy_type)
                if signal_func:
                    # Check if accepts cache
                    import inspect
                    sig_params = inspect.signature(signal_func).parameters
                    if 'cache' in sig_params:
                        long_sig, short_sig = signal_func(self.df, params, cache={})
                    else:
                        long_sig, short_sig = signal_func(self.df, params)
                    
                    signal_series[long_sig] = 1
                    signal_series[short_sig] = -1
            except Exception as e:
                print(f"Signal Gen Error: {e}")

            # Loop from 1 to n (need prev candle for signal)
            for i in range(1, n_candles):
                prev = records[i-1]
                curr = records[i]
                
                # Update Equity (Carry forward)
                equity_curve[i] = balance 
                
                # 1. Manage Open Position
                if position:
                    exit_type = None
                    exit_price = 0.0
                    
                    # Check SL/TP against Current High/Low
                    if position['type'] == 'long':
                        if curr['Low'] <= position['sl']:
                            exit_type = 'SL'
                            exit_price = position['sl']
                        elif curr['High'] >= position['tp']:
                            exit_type = 'TP'
                            exit_price = position['tp']
                    else: # Short
                        if curr['High'] >= position['sl']:
                            exit_type = 'SL'
                            exit_price = position['sl']
                        elif curr['Low'] <= position['tp']:
                            exit_type = 'TP'
                            exit_price = position['tp']
                            
                    if exit_type:
                        # Execute Exit
                        if position['type'] == 'long':
                            pnl = (exit_price - position['entry']) * position['size']
                        else:
                            pnl = (position['entry'] - exit_price) * position['size']
                            
                        balance += pnl
                        equity_curve[i] = balance
                        
                        trades.append({
                            'Entry Time': position['time'],
                            'Entry Price': position['entry'],
                            'Exit Time': curr['datetime'],
                            'Exit Price': exit_price,
                            'pnl': pnl,
                            'Return %': (pnl / (position['entry'] * position['size'])) * 100 if position['size'] > 0 else 0,
                            'Direction': position['type'],
                            'Status': exit_type,
                            # Compatibility fields for Visualizer
                            'type': f"ENTRY {position['type'].upper()}", 
                            'time': position['time'], 
                            'price': position['entry']
                        })
                        position = None
                        continue # Isolation: No new entry on same bar as exit
                
                # 2. Check for New Entry (if no position)
                if position is None:
                    # Time Schedule Check (based on Signal Time)
                    if start_time and end_time:
                        t_obj = prev['datetime'].time()
                        if not (start_time <= t_obj <= end_time):
                            continue
                    
                    # Check Pre-calculated Signal (Signal at T (prev) -> Entry at T+1 (curr))
                    # signal_series[i-1] corresponds to signal generated at prev candle
                    sig_val = signal_series[i-1]
                    signal = 'long' if sig_val == 1 else 'short' if sig_val == -1 else None
                        
                    if signal:
                        # Entry Setup
                        entry_price = curr['Open'] # Entry at Open T+1
                        atr = prev['ATR'] # ATR at Signal Candle
                        
                        if atr > 0:
                            sl_dist = atr * sl_atr_mult
                            risk_amt = balance * (risk_per_trade / 100)
                            
                            if signal == 'long':
                                sl = entry_price - sl_dist
                                tp = entry_price + (sl_dist * risk_reward)
                                size = risk_amt / sl_dist
                            else:
                                sl = entry_price + sl_dist
                                tp = entry_price - (sl_dist * risk_reward)
                                size = risk_amt / sl_dist
                                
                            position = {
                                'type': signal,
                                'entry': entry_price,
                                'sl': sl,
                                'tp': tp,
                                'size': size,
                                'time': curr['datetime']
                            }
                            
                            # Check for Immediate Exit (Same Candle)
                            imm_exit = None
                            imm_price = 0.0
                            
                            if signal == 'long':
                                if curr['Low'] <= sl:
                                    imm_exit = 'SL'; imm_price = sl
                                elif curr['High'] >= tp:
                                    imm_exit = 'TP'; imm_price = tp
                            else:
                                if curr['High'] >= sl:
                                    imm_exit = 'SL'; imm_price = sl
                                elif curr['Low'] <= tp:
                                    imm_exit = 'TP'; imm_price = tp
                                    
                            if imm_exit:
                                if signal == 'long': pnl = (imm_price - entry_price) * size
                                else: pnl = (entry_price - imm_price) * size
                                    
                                balance += pnl
                                equity_curve[i] = balance
                                
                                trades.append({
                                    'Entry Time': curr['datetime'],
                                    'Entry Price': entry_price,
                                    'Exit Time': curr['datetime'],
                                    'Exit Price': imm_price,
                                    'pnl': pnl,
                                    'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                                    'Direction': signal,
                                    'Status': imm_exit,
                                    'type': f"ENTRY {signal.upper()}",
                                    'time': curr['datetime'],
                                    'price': entry_price
                                })
                                position = None
                            
            return trades, equity_curve

    class Visualizer:
        @staticmethod
        def plot_tradingview_clone(df, trades, engine_type="Hybrid", strategy_name=""):
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df['datetime'],
                            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            name='Price'), row=1, col=1)
            
            # Indicators based on Engine
            if engine_type == "Hybrid":
                if 'ZeroGamma' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['ZeroGamma'], name='Zero Gamma', line=dict(color='orange', width=1)), row=1, col=1)
                if 'CallWall' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['CallWall'], name='Call Wall', line=dict(color='green', dash='dash')), row=1, col=1)
                if 'PutWall' in df.columns: fig.add_trace(go.Scatter(x=df['datetime'], y=df['PutWall'], name='Put Wall', line=dict(color='red', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['datetime'], y=[0]*len(df), showlegend=False, opacity=0), row=2, col=1)
            else:
                # Technical Indicators Visualization
                if 'SMA200' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=2)), row=1, col=1)
                if 'SMA50' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['SMA50'], name='SMA 50', line=dict(color='cyan', width=1)), row=1, col=1)
                
                if "Bollinger" in strategy_name and 'BB_Upper' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

                # Row 2 Oscillators
                if "RSI" in strategy_name and 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                elif "MACD" in strategy_name and 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=2, col=1)
                    fig.add_bar(x=df['datetime'], y=df['MACD'] - df['MACD_Signal'], name='Histogram', row=2, col=1)
                elif "Stochastic" in strategy_name and 'Stoch_K' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Stoch_K'], name='Stoch K', line=dict(color='blue', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Stoch_D'], name='Stoch D', line=dict(color='orange', width=1)), row=2, col=1)
                    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
                elif "CCI" in strategy_name and 'CCI' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['CCI'], name='CCI', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
                elif "Williams" in strategy_name and 'WilliamsR' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['WilliamsR'], name='Williams %R', line=dict(color='purple', width=1)), row=2, col=1)
                    fig.add_hline(y=-20, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-80, line_dash="dash", line_color="green", row=2, col=1)
                elif "Aroon" in strategy_name and 'Aroon_Up' in df.columns:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Aroon_Up'], name='Aroon Up', line=dict(color='green', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Aroon_Down'], name='Aroon Down', line=dict(color='red', width=1)), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df['datetime'], y=[0]*len(df), showlegend=False, opacity=0), row=2, col=1)

            # Signals
            buy_signals = [t for t in trades if 'ENTRY LONG' in t['type']]
            sell_signals = [t for t in trades if 'ENTRY SHORT' in t['type']]
            
            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in buy_signals], 
                    y=[t['price'] for t in buy_signals],
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy Signal'
                ), row=1, col=1)
            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in sell_signals], 
                    y=[t['price'] for t in sell_signals],
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell Signal'
                ), row=1, col=1)

            fig.update_layout(
                title=f"TradingView Clone - {engine_type} Strategy ({strategy_name})",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
                height=800,
                dragmode='pan',
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Crosshairs
            fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", row=1, col=1)
            fig.update_yaxes(showspikes=True, spikecolor="gray", spikemode="across", row=1, col=1)
            fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", row=2, col=1)
            fig.update_yaxes(showspikes=True, spikecolor="gray", spikemode="across", row=2, col=1)
            
            return fig

    engine = BacktestEngine(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe, initial_capital, target_tz=tz_choice)

    # Use verified data if available
    if st.session_state.backtest_data is not None and st.session_state.backtest_ticker == ticker:
        engine.df = st.session_state.backtest_data.copy()
        data_ready = True
    else:
        data_ready = False

    if "MOTORE A" in engine_choice:
        st.subheader("🧬 Configurazione Strategia Ibrida GEX")
        
        # Schedule & Entry Mode
        with st.expander("⚙️ Opzioni Avanzate (Orari & Ingresso)"):
            c1, c2, c3 = st.columns(3)
            with c1: use_schedule = st.checkbox("Abilita Orari di Trading")
            with c2: start_t = st.time_input("Inizio Sessione", dt_time(9, 30))
            with c3: end_t = st.time_input("Fine Sessione", dt_time(16, 0))
            
            entry_mode = st.selectbox("Modalità di Ingresso", ["Standard", "Breakout (Close)", "Retest"])
            
        col1, col2 = st.columns(2)
        with col1:
            long_trigger = st.selectbox("Trigger Long", ["Rimbalzo Put Wall", "Breakout 0-Gamma", "Breakout Call Wall", "Nessuno"])
            # Map Italian to English for logic
            long_trigger_map = {"Rimbalzo Put Wall": "Bounce Put Wall", "Breakout 0-Gamma": "Breakout 0-Gamma", "Breakout Call Wall": "Breakout Call Wall", "Nessuno": "None"}
            long_trigger_en = long_trigger_map[long_trigger]

        with col2:
            short_trigger = st.selectbox("Trigger Short", ["Rimbalzo Call Wall", "Breakdown 0-Gamma", "Breakdown Put Wall", "Nessuno"])
            short_trigger_map = {"Rimbalzo Call Wall": "Bounce Call Wall", "Breakdown 0-Gamma": "Breakdown 0-Gamma", "Breakdown Put Wall": "Breakdown Put Wall", "Nessuno": "None"}
            short_trigger_en = short_trigger_map[short_trigger]
        
        sensitivity = st.slider("Sensibilità GEX", 0.5, 3.0, 1.5, 0.1)
        rr = st.slider("Rischio:Rendimento", 1.0, 5.0, 2.0, 0.5)
        risk_pct = st.slider("Rischio per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        
        if data_ready:
            if st.button("🚀 Avvia Simulazione GEX"):
                with st.spinner("Calcolo Livelli GEX e Simulazione..."):
                    # Data already in engine.df
                    engine.add_technical_indicators() # Need ATR
                    engine.add_gex_levels(sensitivity)
                    
                    s_time = start_t if use_schedule else None
                    e_time = end_t if use_schedule else None
                    
                    trades, equity = engine.run_hybrid_strategy(long_trigger_en, short_trigger_en, rr, risk_pct, s_time, e_time, entry_mode)
                    st.session_state['trades'] = trades
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    st.subheader("Professional Risk Dashboard")
                    
                    current_trades = st.session_state.get('trades', [])
                    if not current_trades:
                        st.warning("Esegui il backtest per vedere le analisi avanzate")
                    else:
                        # Apply Friction
                        adjusted_trades, adjusted_equity = apply_friction_post_process(current_trades, initial_capital, friction_pct)
                        
                        # Calculate Metrics
                        metrics = calculate_advanced_metrics(adjusted_trades)
                        
                        # Metric Cards
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)
                        
                        with col1:
                            expectancy = metrics['expectancy']
                            st.metric(
                                label="Mathematical Expectancy",
                                value=f"${expectancy:.2f}",
                                delta="Positive" if expectancy > 0 else "Negative",
                                delta_color="normal" if expectancy > 0 else "inverse"
                            )
                            
                        with col2:
                            profit_factor = metrics['profit_factor']
                            st.metric(
                                label="Profit Factor",
                                value=f"{profit_factor:.2f}",
                                delta="Good" if profit_factor > 1.5 else "Needs Improvement",
                                delta_color="normal" if profit_factor > 1.5 else "inverse"
                            )
                            
                        with col3:
                            win_rate = metrics['win_rate']
                            st.metric(
                                label="Win Rate (%)",
                                value=f"{win_rate:.2f}%",
                                delta="Profitable" if win_rate > 50 else "Unprofitable",
                                delta_color="normal" if win_rate > 50 else "inverse"
                            )
                            
                        with col4:
                            max_dd = metrics['max_drawdown']
                            st.metric(
                                label="Max Drawdown (%)",
                                value=f"{max_dd:.2f}%",
                                delta="High Risk" if max_dd < -20 else "Acceptable",
                                delta_color="inverse" if max_dd < -20 else "normal"
                            )
                            
                        with col5:
                            total_profit_abs = metrics.get('total_profit_abs', 0)
                            st.metric(
                                label="Total Net Profit ($)",
                                value=f"${total_profit_abs:.2f}",
                                delta="Positive" if total_profit_abs > 0 else "Negative",
                                delta_color="normal" if total_profit_abs > 0 else "inverse"
                            )
                            
                        with col6:
                            max_dd_abs = metrics.get('max_dd_abs', 0)
                            st.metric(
                                label="Max Drawdown ($)",
                                value=f"${max_dd_abs:.2f}",
                                delta="High Loss" if max_dd_abs > initial_capital * 0.2 else "Acceptable",
                                delta_color="inverse" if max_dd_abs > initial_capital * 0.2 else "normal"
                            )
                            
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, adjusted_trades, "Hybrid"), use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['drawline']})
                        st.line_chart(adjusted_equity)
                        
                        # Monte Carlo Expander
                        with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                            mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                            if mc_res:
                                mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                                st.plotly_chart(mc_fig, use_container_width=True)
                                
                                st.subheader('🔬 Validazione Statistica Long-Term')
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                                with c2:
                                    st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                                with c3:
                                    st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                                
                                if prob_profit > 75:
                                    st.success('✅ Strategia Robusta')
                                elif prob_profit < 60:
                                    st.warning('⚠️ Strategia Fragile (Flop)')
                                    
                                if risk_of_ruin > 10:
                                    st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                    
                                if len(adjusted_trades) < 30:
                                    st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                            else:
                                st.warning("Not enough data for Monte Carlo simulation.")
                        
                        st.subheader("📝 Dettaglio Operazioni (Explainability)")
                        df_res = pd.DataFrame(adjusted_trades)
                        st.data_editor(df_res, use_container_width=True, hide_index=True)
        else:
            st.info("⚠️ Esegui prima la 'Verifica Disponibilità Dati Storici' per abilitare la simulazione.")

    else: # MOTORE B
        st.subheader("📈 Configurazione Hub Strategie Tecniche")
        
        # Schedule & Entry Mode
        with st.expander("⚙️ Opzioni Avanzate (Orari & Ingresso)"):
            c1, c2, c3 = st.columns(3)
            with c1: use_schedule = st.checkbox("Abilita Orari di Trading", value=st.session_state.get('use_schedule', False))
            with c2: start_t = st.time_input("Inizio Sessione", value=st.session_state.get('start_time', dt_time(9, 30)))
            with c3: end_t = st.time_input("Fine Sessione", value=st.session_state.get('end_time', dt_time(16, 0)))
            
            entry_mode = st.selectbox("Modalità di Ingresso", ["Standard", "Breakout (Close)", "Retest"], index=["Standard", "Breakout (Close)", "Retest"].index(st.session_state.get('entry_mode', "Standard")))
        
        # Strategy Selection
        strategies_list = [
            "RSI Mean Reversion", "MACD Crossover", "Bollinger Breakout", "Golden/Death Cross", 
            "Stochastic Oscillator", "CCI Momentum", "Williams %R Reversal",
            "HMA Trend", "TEMA Crossover", "KAMA Trend", "Aroon Oscillator",
            "SuperTrend Reversal", "Parabolic SAR", "TSI Crossover", "UO Overbought/Oversold",
            "Keltner Channel Breakout", "Donchian Channel Breakout", "Chaikin Volatility",
            "CMF Trend", "VWAP Crossover", "AD Line Trend",
            "Vortex Crossover", "Choppiness Index Breakout", "KST Crossover", "Coppock Curve",
            "Ichimoku Cloud Breakout", "Awesome Oscillator", "PPO Crossover", "Mass Index Reversal",
            "Ulcer Index Safety",
            "WMA Trend", "TRIMA Crossover", "CMO Reversal", "Momentum Breakout", "BOP Trend",
            "TRIX Crossover", "StochRSI Reversal", "TSF Trend"
        ]
        # Restore strategy selection if exists
        saved_strat_idx = 0
        if st.session_state.get('strategy_type') in strategies_list:
            saved_strat_idx = strategies_list.index(st.session_state.get('strategy_type'))
            
        strategy_type = st.selectbox("Seleziona Tipo Strategia", strategies_list, index=saved_strat_idx)
        
        # Dynamic Parameters with Session State Persistence
        params = {}
        col1, col2, col3 = st.columns(3)
        
        if strategy_type == "RSI Mean Reversion":
            with col1: params['period'] = st.number_input("Periodo RSI", value=int(st.session_state.get('period_rsi', 14)))
            with col2: params['ob'] = st.number_input("Ipercomprato", value=int(st.session_state.get('ob_rsi', 70)))
            with col3: params['os'] = st.number_input("Ipervenduto", value=int(st.session_state.get('os_rsi', 30)))
        elif strategy_type == "Stochastic Oscillator":
            with col1: params['k_period'] = st.number_input("Periodo K", value=int(st.session_state.get('k_stoch', 14)))
            with col2: params['ob'] = st.number_input("Ipercomprato", value=int(st.session_state.get('ob_stoch', 80)))
            with col3: params['os'] = st.number_input("Ipervenduto", value=int(st.session_state.get('os_stoch', 20)))
        elif strategy_type == "CCI Momentum":
             with col1: params['period'] = st.number_input("Periodo CCI", value=int(st.session_state.get('period_cci', 20)))
        elif strategy_type == "Williams %R Reversal":
             with col1: params['period'] = st.number_input("Periodo Williams %R", value=int(st.session_state.get('period_williams', 14)))
        elif strategy_type == "Bollinger Breakout":
            with col1: params['period'] = st.number_input("Periodo BB", value=int(st.session_state.get('period_bb', 20)))
            with col2: params['std_dev'] = st.number_input("Dev. Std", value=float(st.session_state.get('std_bb', 2.0)))
        elif strategy_type == "MACD Crossover":
            with col1: params['fast'] = st.number_input("Fast Period", value=int(st.session_state.get('fast_macd', 12)))
            with col2: params['slow'] = st.number_input("Slow Period", value=int(st.session_state.get('slow_macd', 26)))
            with col3: params['signal'] = st.number_input("Signal Period", value=int(st.session_state.get('signal_macd', 9)))
        elif strategy_type == "Golden/Death Cross":
            with col1: params['fast'] = st.number_input("Fast MA", value=int(st.session_state.get('fast_gd', 50)))
            with col2: params['slow'] = st.number_input("Slow MA", value=int(st.session_state.get('slow_gd', 200)))
        else:
            # Generic fallback for others to avoid errors if params needed
            with col1: params['period'] = st.number_input("Periodo", value=int(st.session_state.get('period_generic', 14)))
        
        rr = st.slider("Rischio:Rendimento", 1.0, 5.0, float(st.session_state.get('rr', 2.0)), 0.5)
        risk_pct = st.slider("Rischio per Trade (%)", 0.5, 5.0, float(st.session_state.get('risk_pct', 1.0)), 0.1)
        
        if data_ready:
            c_run, c_opt = st.columns([2, 1])
            
            # Check Auto-Run Flag
            auto_run = st.session_state.get('run_backtest_auto', False)
            
            if c_run.button("🚀 Avvia Backtest Tecnico") or auto_run:
                if auto_run: st.session_state['run_backtest_auto'] = False # Reset flag
                
                with st.spinner("Calcolo Indicatori e Simulazione..."):
                    # Data already in engine.df
                    engine.add_technical_indicators()
                    
                    s_time = start_t if use_schedule else None
                    e_time = end_t if use_schedule else None
                    
                    trades, equity = engine.run_technical_strategy(strategy_type, params, rr, risk_pct, s_time, e_time, entry_mode)
                    st.session_state['trades'] = trades
                    
                    # Results
                    st.success(f"Simulazione Completata. Totale Operazioni: {len(trades)}")
                    
                    # Optimization Feedback: Actual Processed Range
                    if not engine.df.empty:
                        min_date = engine.df['datetime'].min()
                        max_date = engine.df['datetime'].max()
                        st.caption(f"📅 Range Effettivo Processato: {min_date} -> {max_date} ({len(engine.df)} candele)")
                    
                    st.subheader("Professional Risk Dashboard")
                    
                    current_trades = st.session_state.get('trades', [])
                    if not current_trades:
                        st.warning("Esegui il backtest per vedere le analisi avanzate")
                    else:
                        # Apply Friction
                        adjusted_trades, adjusted_equity = apply_friction_post_process(current_trades, initial_capital, friction_pct)
                        
                        # Calculate Metrics
                        metrics = calculate_advanced_metrics(adjusted_trades)
                        
                        # Metric Cards
                        col1, col2, col3 = st.columns(3)
                        col4, col5, col6 = st.columns(3)
                        
                        with col1:
                            expectancy = metrics['expectancy']
                            st.metric(
                                label="Mathematical Expectancy",
                                value=f"${expectancy:.2f}",
                                delta="Positive" if expectancy > 0 else "Negative",
                                delta_color="normal" if expectancy > 0 else "inverse"
                            )
                            
                        with col2:
                            profit_factor = metrics['profit_factor']
                            st.metric(
                                label="Profit Factor",
                                value=f"{profit_factor:.2f}",
                                delta="Good" if profit_factor > 1.5 else "Needs Improvement",
                                delta_color="normal" if profit_factor > 1.5 else "inverse"
                            )
                            
                        with col3:
                            win_rate = metrics['win_rate']
                            st.metric(
                                label="Win Rate (%)",
                                value=f"{win_rate:.2f}%",
                                delta="Profitable" if win_rate > 50 else "Unprofitable",
                                delta_color="normal" if win_rate > 50 else "inverse"
                            )
                            
                        with col4:
                            max_dd = metrics['max_drawdown']
                            st.metric(
                                label="Max Drawdown (%)",
                                value=f"{max_dd:.2f}%",
                                delta="High Risk" if max_dd < -20 else "Acceptable",
                                delta_color="inverse" if max_dd < -20 else "normal"
                            )
                            
                        with col5:
                            total_profit_abs = metrics.get('total_profit_abs', 0)
                            st.metric(
                                label="Total Net Profit ($)",
                                value=f"${total_profit_abs:.2f}",
                                delta="Positive" if total_profit_abs > 0 else "Negative",
                                delta_color="normal" if total_profit_abs > 0 else "inverse"
                            )
                            
                        with col6:
                            max_dd_abs = metrics.get('max_dd_abs', 0)
                            st.metric(
                                label="Max Drawdown ($)",
                                value=f"${max_dd_abs:.2f}",
                                delta="High Loss" if max_dd_abs > initial_capital * 0.2 else "Acceptable",
                                delta_color="inverse" if max_dd_abs > initial_capital * 0.2 else "normal"
                            )
                            
                        # Charts
                        st.plotly_chart(Visualizer.plot_tradingview_clone(engine.df, adjusted_trades, "Technical", strategy_type), use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['drawline']})
                        st.line_chart(adjusted_equity)
                        
                        # Monte Carlo Expander
                        with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                            mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                            if mc_res:
                                mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                                st.plotly_chart(mc_fig, use_container_width=True)
                                
                                st.subheader('🔬 Validazione Statistica Long-Term')
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                                with c2:
                                    st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                                with c3:
                                    st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                                
                                if prob_profit > 75:
                                    st.success('✅ Strategia Robusta')
                                elif prob_profit < 60:
                                    st.warning('⚠️ Strategia Fragile (Flop)')
                                    
                                if risk_of_ruin > 10:
                                    st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                    
                                if len(adjusted_trades) < 30:
                                    st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                            else:
                                st.warning("Not enough data for Monte Carlo simulation.")
                        
                        st.subheader("📝 Dettaglio Operazioni (Explainability)")
                        df_res = pd.DataFrame(adjusted_trades)
                        st.data_editor(df_res, use_container_width=True, hide_index=True)
            
            if c_opt.button("🧠 Ottimizza Strategia (AI)"):
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # --- FAST OPTIMIZATION LOGIC ---
                # 1. Prepare Data (Downsample if needed)
                engine.add_technical_indicators() # Ensure ATR and other base indicators are present
                opt_df = engine.df.copy()
                if len(opt_df) > 10000:
                    status_text.text(f"⚠️ Dati estesi ({len(opt_df)} candele). Ottimizzazione in corso su tutto il dataset...")
                    # No truncation as per user request

                
                # 2. Define Ranges
                opt_config = STRATEGY_PARAM_GRID.get(strategy_type, {})
                
                rr_ranges = [1.5, 2.0, 2.5, 3.0]
                
                # 3. Generate Combinations
                import itertools
                keys = list(opt_config.keys())
                values = [opt_config[k] for k in keys]
                param_combos = list(itertools.product(*values)) if values else [()]
                
                total_steps = len(param_combos) * len(rr_ranges)
                step = 0
                best_res = {'wr': 0, 'pnl': -float('inf'), 'params': {}, 'rr': 0}
                
                # 4. Fast Loop
                opt_cache = {}
                for p_vals in param_combos:
                    curr_p = dict(zip(keys, p_vals))
                    
                    # Generate Signals (Vectorized)
                    sigs = pd.Series(0, index=opt_df.index)
                    
                    try:
                        signal_func = StrategyLib.get_signal_func(strategy_type)
                        if signal_func:
                            long_sig, short_sig = signal_func(opt_df, curr_p, cache=opt_cache)
                            long_sig = long_sig.fillna(False)
                            short_sig = short_sig.fillna(False)
                            sigs = np.where(long_sig, 1, np.where(short_sig, -1, 0))
                    except Exception as e:
                        continue

                    # Get Entry Indices
                    entry_idxs = np.where(sigs != 0)[0]
                    if len(entry_idxs) == 0: continue
                    
                    # 5. Fast Trade Outcome Loop
                    # We assume fixed RR and ATR based SL
                    atr_arr = opt_df['ATR'].values
                    close_arr = opt_df['Close'].values
                    high_arr = opt_df['High'].values
                    low_arr = opt_df['Low'].values
                    times = opt_df['datetime'].values
                    
                    for rr_val in rr_ranges:
                        step += 1
                        if step % 20 == 0:
                            progress_bar.progress(min(step/total_steps, 1.0))
                            status_text.text(f"Scansione AI... WR: {best_res['wr']:.1f}%")
                            
                        wins = 0
                        losses = 0
                        pnl = 0
                        
                        for idx in entry_idxs:
                            if idx >= len(opt_df) - 1: continue
                            
                            entry_price = close_arr[idx]
                            direction = sigs[idx] # 1 or -1
                            atr = atr_arr[idx]
                            if np.isnan(atr): continue
                            
                            sl_dist = atr * 1.5
                            
                            if direction == 1: # Long
                                sl = entry_price - sl_dist
                                tp = entry_price + (sl_dist * rr_val)
                                # Look forward max 50 bars
                                for fwd in range(idx+1, min(idx+51, len(opt_df))):
                                    if low_arr[fwd] <= sl:
                                        losses += 1
                                        pnl -= sl_dist
                                        break
                                    elif high_arr[fwd] >= tp:
                                        wins += 1
                                        pnl += (sl_dist * rr_val)
                                        break
                            else: # Short
                                sl = entry_price + sl_dist
                                tp = entry_price - (sl_dist * rr_val)
                                for fwd in range(idx+1, min(idx+51, len(opt_df))):
                                    if high_arr[fwd] >= sl:
                                        losses += 1
                                        pnl -= sl_dist
                                        break
                                    elif low_arr[fwd] <= tp:
                                        wins += 1
                                        pnl += (sl_dist * rr_val)
                                        break
                        
                        total_trades = wins + losses
                        if total_trades > 0:
                            wr = (wins / total_trades) * 100
                            
                            # Selection Logic: Max WR, then Max PnL
                            if wr > best_res['wr']:
                                best_res = {'wr': wr, 'pnl': pnl, 'params': curr_p, 'rr': rr_val}
                            elif wr == best_res['wr'] and pnl > best_res['pnl']:
                                best_res = {'wr': wr, 'pnl': pnl, 'params': curr_p, 'rr': rr_val}

                progress_bar.empty()
                status_text.empty()
                
                if best_res['wr'] > 0:
                    st.success(f"🏆 Ottimizzazione Completata! WR: {best_res['wr']:.1f}%")
                    
                    # Save to Session State
                    for k, v in best_res['params'].items():
                        if strategy_type == "RSI Mean Reversion":
                            if k == 'period': st.session_state['period_rsi'] = v
                            elif k == 'ob': st.session_state['ob_rsi'] = v
                            elif k == 'os': st.session_state['os_rsi'] = v
                        elif strategy_type == "Stochastic Oscillator":
                            if k == 'k_period': st.session_state['k_stoch'] = v
                            elif k == 'ob': st.session_state['ob_stoch'] = v
                            elif k == 'os': st.session_state['os_stoch'] = v
                        elif strategy_type == "CCI Momentum":
                            if k == 'period': st.session_state['period_cci'] = v
                        elif strategy_type == "Williams %R Reversal":
                            if k == 'period': st.session_state['period_williams'] = v
                        elif strategy_type == "Bollinger Breakout":
                            if k == 'period': st.session_state['period_bb'] = v
                            elif k == 'std_dev': st.session_state['std_bb'] = v
                        elif strategy_type == "MACD Crossover":
                            if k == 'fast': st.session_state['fast_macd'] = v
                            elif k == 'slow': st.session_state['slow_macd'] = v
                            elif k == 'signal': st.session_state['signal_macd'] = v
                        elif strategy_type == "Golden/Death Cross":
                            if k == 'fast': st.session_state['fast_gd'] = v
                            elif k == 'slow': st.session_state['slow_gd'] = v
                        else:
                            if k == 'period': st.session_state['period_generic'] = v
                            
                    st.session_state['rr'] = best_res['rr']
                    st.session_state['run_backtest_auto'] = True
                    
                    st.rerun()
                else:
                    st.error("Nessun risultato valido trovato.")

        else:
            st.info("⚠️ Esegui prima la 'Verifica Disponibilità Dati Storici' per abilitare la simulazione.")

elif menu == "🛠️ STRATEGY BUILDER":
    st.title("🛠️ Strategy Builder (No-Code)")
    
    st.sidebar.markdown("---")
    tz_choice = st.sidebar.selectbox("🌍 Fuso Orario", ["America/New_York", "UTC", "Europe/Rome"], index=0)
    st.sidebar.markdown("### ⚙️ Impostazioni Base")
    
    # Time Filters UI
    start_time = st.sidebar.time_input("Start Trading Time", value=datetime.strptime("09:30", "%H:%M").time())
    end_time = st.sidebar.time_input("End Trading Time", value=datetime.strptime("16:00", "%H:%M").time())
    eod_close = st.sidebar.checkbox("Close all at EOD", value=True)
    
    # Core Strategies UI
    st.sidebar.markdown("### 🧠 Strategie Core")
    core_strategies = st.sidebar.multiselect(
        "Seleziona Strategie",
        ["ORB (Opening Range Breakout)", "VWAP Reversion", "Volume Profile"],
        default=["ORB (Opening Range Breakout)"]
    )
    
    orb_duration = 15
    vwap_sd_level = 2
    vp_mode = 'breakout'
    
    if "ORB (Opening Range Breakout)" in core_strategies:
        orb_duration = st.sidebar.selectbox("ORB Duration (min)", [5, 15, 30, 60])
    if "VWAP Reversion" in core_strategies:
        vwap_sd_level = st.sidebar.selectbox("VWAP SD Level", [1, 2, 3], index=1)
    if "Volume Profile" in core_strategies:
        vp_mode = st.sidebar.selectbox("Volume Profile Mode", ["breakout", "rejection"])

    st.sidebar.markdown("### 📊 Filtri Tecnici (pandas_ta)")
    num_ta_filters = st.sidebar.number_input("Numero di Filtri TA", min_value=0, max_value=5, value=0)
    ta_filters = []
    
    TA_INDICATORS_MAP = {
        # MOMENTUM
        "stoch": {"k": 14, "d": 3, "smooth_k": 3},
        "stochrsi": {"length": 14, "rsi_length": 14, "k": 3, "d": 3},
        "cci": {"length": 20},
        "willr": {"length": 14},
        "mfi": {"length": 14},
        "tsi": {"fast": 13, "slow": 25},
        "uo": {"fast": 7, "medium": 14, "slow": 28},
        "roc": {"length": 10},
        "mom": {"length": 10},
        "cmo": {"length": 14},
        "cg": {"length": 10},
        "bop": {},
        # TREND
        "wma": {"length": 50},
        "hma": {"length": 20},
        "tema": {"length": 20},
        "dema": {"length": 20},
        "kama": {"length": 10},
        "alma": {"length": 9, "sigma": 6.0, "offset": 0.85},
        "supertrend": {"length": 7, "multiplier": 3.0},
        "adx": {"length": 14},
        "aroon": {"length": 14},
        "chop": {"length": 14},
        "psar": {"af0": 0.02, "af": 0.02, "max_af": 0.2},
        "qstick": {"length": 14},
        "decay": {"length": 14},
        # VOLATILITA E CANALI
        "kc": {"length": 20, "scalar": 2.0},
        "dc": {"length": 20},
        "natr": {"length": 14},
        "massi": {"fast": 9, "slow": 25},
        "true_range": {},
        # VOLUME
        "vwap": {},
        "obv": {},
        "cmf": {"length": 20},
        "ad": {},
        "pvt": {},
        "efi": {"length": 14}
    }

    full_indicator_list = ["RSI", "MACD", "EMA", "SMA", "ATR", "Bollinger Bands"] + [k.upper() for k in TA_INDICATORS_MAP.keys()]

    for i in range(num_ta_filters):
        st.sidebar.markdown(f"**Filtro {i+1}**")
        
        indicator_choice = st.sidebar.selectbox(
            f"Indicatore {i+1}", 
            full_indicator_list, 
            key=f"ind_choice_{i}"
        )
        
        params_dict = {}
        output_col = ""
        ind_name = ""
        
        output_col = "TARGET_IND" # Il motore userà sempre questo alias sicuro

        if indicator_choice == "RSI":
            ind_name = "rsi"
            length = st.sidebar.number_input(f"Length {i+1}", min_value=1, value=14, key=f"rsi_len_{i}")
            params_dict = {"length": length}
            default_long_val, default_short_val = 30.0, 70.0
            default_op_long, default_op_short = "<", ">"
            
        elif indicator_choice == "MACD":
            ind_name = "macd"
            c1, c2, c3 = st.sidebar.columns(3)
            with c1: fast = st.number_input(f"Fast", min_value=1, value=12, key=f"macd_fast_{i}")
            with c2: slow = st.number_input(f"Slow", min_value=1, value=26, key=f"macd_slow_{i}")
            with c3: signal = st.number_input(f"Signal", min_value=1, value=9, key=f"macd_sig_{i}")
            params_dict = {"fast": fast, "slow": slow, "signal": signal}
            default_long_val, default_short_val = 0.0, 0.0
            default_op_long, default_op_short = ">", "<"
            
        elif indicator_choice in ["EMA", "SMA", "ATR"]:
            ind_name = indicator_choice.lower()
            length = st.sidebar.number_input(f"Length {i+1}", min_value=1, value=50 if ind_name!="atr" else 14, key=f"{ind_name}_len_{i}")
            params_dict = {"length": length}
            default_long_val, default_short_val = 0.0 if ind_name!="atr" else 1.0, 0.0 if ind_name!="atr" else 1.0
            default_op_long, default_op_short = ">", "<" if ind_name!="atr" else ">"
            
        elif indicator_choice in ["Bollinger Bands", "KC", "DC"]:
            ind_name = "bbands" if indicator_choice == "Bollinger Bands" else indicator_choice.lower()
            c1, c2 = st.sidebar.columns(2)
            with c1: length = st.number_input(f"Length", min_value=1, value=20, key=f"{ind_name}_len_{i}")
            
            if ind_name != "dc":
                with c2: std = st.number_input(f"Multiplier", min_value=0.1, value=2.0, step=0.1, key=f"{ind_name}_std_{i}")
                if ind_name == "bbands": params_dict = {"length": length, "std": std}
                else: params_dict = {"length": length, "scalar": std}
            else:
                params_dict = {"length": length}
                
            band_choice = st.sidebar.selectbox(f"Banda da confrontare {i+1}", ["Lower (BBL)", "Middle/Basis (BBM)", "Upper (BBU)"], key=f"{ind_name}_band_{i}")
            if "Lower" in band_choice: params_dict['_band'] = "Lower"
            elif "Upper" in band_choice: params_dict['_band'] = "Upper"
            else: params_dict['_band'] = "Middle"
            
            default_long_val, default_short_val = 0.0, 0.0
            default_op_long, default_op_short = "<", ">"
            
        else:
            ind_name = indicator_choice.lower()
            base_params = TA_INDICATORS_MAP.get(ind_name, {})
            
            if base_params:
                cols = st.sidebar.columns(len(base_params))
                for col, (param_key, param_val) in zip(cols, base_params.items()):
                    with col:
                        if isinstance(param_val, float): params_dict[param_key] = st.number_input(f"{param_key.capitalize()}", value=float(param_val), step=0.01, key=f"{ind_name}_{param_key}_{i}")
                        else: params_dict[param_key] = st.number_input(f"{param_key.capitalize()}", value=int(param_val), step=1, key=f"{ind_name}_{param_key}_{i}")
            
            if ind_name == "stoch":
                scelta = st.sidebar.selectbox(f"Linea Stocastico {i+1}", ["K", "D"], key=f"stoch_line_{i}")
                params_dict['_line'] = scelta
                
            default_long_val, default_short_val = 0.0, 0.0
            default_op_long, default_op_short = ">", "<"

        compare_mode = st.sidebar.radio(
            f"Tipo di confronto {i+1}", 
            ["Indicatore vs Valore Fisso", "Prezzo vs Indicatore"], 
            horizontal=True,
            key=f"comp_mode_{i}"
        )

        c_long, c_short = st.sidebar.columns(2)
        
        with c_long:
            st.markdown("🟢 **LONG**")
            op_long = st.selectbox("Operatore", [">", "<", ">=", "<=", "=="], index=[">", "<", ">=", "<=", "=="].index(default_op_long), key=f"op_long_{i}")
            
            if compare_mode == "Indicatore vs Valore Fisso":
                thresh_long = st.number_input("Soglia", value=float(default_long_val), key=f"thresh_long_{i}")
                cond_long = f"{output_col} {op_long} {thresh_long}"
            else:
                price_col_long = st.selectbox("Prezzo", ["Close", "Open", "High", "Low"], index=0, key=f"price_long_{i}")
                cond_long = f"{price_col_long} {op_long} {output_col}"
                
        with c_short:
            st.markdown("🔴 **SHORT**")
            op_short = st.selectbox("Operatore", ["<", ">", "<=", ">=", "=="], index=["<", ">", "<=", ">=", "=="].index(default_op_short), key=f"op_short_{i}")
            
            if compare_mode == "Indicatore vs Valore Fisso":
                thresh_short = st.number_input("Soglia", value=float(default_short_val), key=f"thresh_short_{i}")
                cond_short = f"{output_col} {op_short} {thresh_short}"
            else:
                price_col_short = st.selectbox("Prezzo", ["Close", "Open", "High", "Low"], index=0, key=f"price_short_{i}")
                cond_short = f"{price_col_short} {op_short} {output_col}"

        ta_filters.append({
            'name': ind_name,
            'params': params_dict,
            'condition': {'long': cond_long, 'short': cond_short}
        })
        
    st.sidebar.markdown("### 🔀 Logica Combinatoria")
    combinatorial_logic = st.sidebar.radio("Come combinare i segnali?", ["OR (Basta un segnale)", "AND (Tutti allineati)"])
        
    # Ticker and Date Range
    st.sidebar.markdown("### 📈 Selezione Asset")
    
    import json
    import os
    
    ASSETS_FILE = "strategy_assets.json"
    
    def load_assets():
        if os.path.exists(ASSETS_FILE):
            try:
                with open(ASSETS_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"Personalizzati": [], "Preferiti": []}

    def save_assets(data):
        with open(ASSETS_FILE, "w") as f:
            json.dump(data, f)
            
    user_assets = load_assets()
    
    categories = {
        "Forex": ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'],
        "Azioni": ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META'],
        "Crypto": ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE'],
        "Indici (ETF)": ['SPY', 'QQQ', 'DIA', 'IWM', 'VXX'],
        "Materie Prime (ETF)": ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
        "⭐ Preferiti": user_assets.get("Preferiti", []),
        "➕ Personalizzati": user_assets.get("Personalizzati", [])
    }
    
    selected_category = st.sidebar.selectbox("Categoria", list(categories.keys()), key="strategy_builder_category_select")
    
    ticker_options = categories[selected_category]
    if not ticker_options:
        ticker_options = ["Nessun ticker"]
        
    selected_ticker = st.sidebar.selectbox("Ticker", ticker_options, key="strategy_builder_ticker_select")
    
    st.sidebar.markdown("#### Gestione Ticker")
    new_ticker = st.sidebar.text_input("Nuovo Ticker (es. AAPL)", key="strategy_builder_new_ticker").upper().strip()
    
    col_add, col_rem, col_fav = st.sidebar.columns(3)
    
    if col_add.button("➕", key="btn_add_ticker"):
        if new_ticker and new_ticker not in user_assets["Personalizzati"]:
            user_assets["Personalizzati"].append(new_ticker)
            save_assets(user_assets)
            st.rerun()
            
    if col_rem.button("➖", key="btn_rem_ticker"):
        if selected_ticker in user_assets["Personalizzati"]:
            user_assets["Personalizzati"].remove(selected_ticker)
            save_assets(user_assets)
            st.rerun()
        elif selected_ticker in user_assets["Preferiti"]:
            user_assets["Preferiti"].remove(selected_ticker)
            save_assets(user_assets)
            st.rerun()
            
    if col_fav.button("⭐", key="btn_fav_ticker"):
        if selected_ticker and selected_ticker != "Nessun ticker" and selected_ticker not in user_assets["Preferiti"]:
            user_assets["Preferiti"].append(selected_ticker)
            save_assets(user_assets)
            st.rerun()
            
    if new_ticker:
        ticker = new_ticker
    else:
        ticker = selected_ticker if selected_ticker != "Nessun ticker" else "SPY"
        
    # Determine Data Engine
    crypto_list = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE']
    is_crypto = "-USD" in ticker or any(ticker.startswith(c) and (ticker.endswith("USD") or len(ticker) == len(c)) for c in crypto_list)
    is_forex = ("=X" in ticker or (len(ticker) == 6 and ticker.isalpha())) and not is_crypto
    is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
    is_stock = not (is_forex or is_index or is_crypto)
    
    if is_crypto:
        engine_str = "Alpaca Crypto v1beta3"
    elif is_forex:
        engine_str = "HistData (Forex)"
    else:
        engine_str = "Alpaca Stocks/ETF v2"
        
    st.sidebar.markdown(f"**Asset:** `{ticker}`")
    st.sidebar.markdown(f"**Data Engine:** `{engine_str}`")
        
    st.sidebar.markdown("### 🛡️ Risk Management")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
    rr_ratio = st.sidebar.slider("Target Risk/Reward (R:R)", 1.0, 5.0, 2.0, 0.1)
    sl_mode = st.sidebar.selectbox("Stop Loss Mode", ["Fixed %", "Candle Low/High"])
    fixed_sl_pct = 1.0
    if sl_mode == "Fixed %":
        fixed_sl_pct = st.sidebar.slider("Fixed Stop Loss (%)", 0.1, 5.0, 1.0, 0.1)
    
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with c2: end_date = st.date_input("End Date", value=datetime.now())
    
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)
    
    # Duplicate necessary functions
    def normalize_key(d, possible_keys):
        for k in d.keys():
            if k.lower() in [pk.lower() for pk in possible_keys]:
                return d[k]
        return None

    def apply_friction_post_process(trades_list, initial_capital, friction_pct):
        if not trades_list:
            return trades_list, [initial_capital]
            
        new_trades = []
        balance = initial_capital
        equity_curve = [balance]
        
        for t in trades_list:
            t_copy = dict(t)
            t_type = str(normalize_key(t_copy, ['type', 'Type']) or '').upper()
            price = normalize_key(t_copy, ['price', 'Price', 'Entry Price', 'Exit Price']) or 0
            pnl = normalize_key(t_copy, ['pnl', 'PnL']) or 0
            
            friction_multiplier = 1 - (friction_pct / 100)
            new_price = price * friction_multiplier
            pnl = pnl * friction_multiplier
            t_copy['price'] = new_price
            t_copy['pnl'] = pnl
            balance += pnl
            t_copy['balance'] = balance
            equity_curve.append(balance)
            
            new_trades.append(t_copy)
                
        return new_trades, equity_curve

    def calculate_advanced_metrics(trades_list, equity_curve):
        fallback = {'expectancy': 0, 'profit_factor': 0, 'max_drawdown': 0, 'win_rate': 0, 'total_profit_abs': 0, 'max_dd_abs': 0}
        if not trades_list: return fallback
        
        df = pd.DataFrame(trades_list)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()] # Elimina colonne con lo stesso nome
        if 'pnl' not in df.columns: return fallback
        
        # FIX: Filtriamo solo le righe che rappresentano la chiusura di un trade (PnL diverso da zero)
        # o che contengono la stringa 'EXIT' nel tipo.
        mask = df['pnl'] != 0
        if 'direction' in df.columns:
            mask = mask | df['direction'].astype(str).str.contains('EXIT', case=False, na=False)
        if 'type' in df.columns:
            mask = mask | df['type'].astype(str).str.contains('EXIT', case=False, na=False)
            
        exits = df[mask]
        if exits.empty: return fallback
            
        wins = exits[exits['pnl'] > 0]['pnl']
        losses = exits[exits['pnl'] < 0]['pnl']
        
        win_rate = len(wins) / len(exits)
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
        total_profit_abs = exits['pnl'].sum()
        
        # FIX DRAWDOWN
        max_dd = 0
        max_dd_abs = 0
        if equity_curve:
            peak = equity_curve[0]
            for val in equity_curve:
                if val > peak: peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                dd_abs = peak - val
                if dd > max_dd: max_dd = dd
                if dd_abs > max_dd_abs: max_dd_abs = dd_abs
                
        return {'expectancy': expectancy, 'profit_factor': profit_factor, 'max_drawdown': max_dd * 100, 'win_rate': win_rate * 100, 'total_profit_abs': total_profit_abs, 'max_dd_abs': max_dd_abs}

    def run_monte_carlo(trades_list, initial_capital, simulations=1000):
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        if not trades_list:
            return None
            
        df_res = pd.DataFrame(trades_list)
        if 'pnl' in df_res.columns:
            pnls = df_res[df_res['pnl'].notna()]['pnl'].values
        else:
            return None
            
        n_trades = len(pnls)
        if n_trades == 0:
            return None
            
        sim_length = min(50, n_trades)
        
        random_indices = np.random.randint(0, n_trades, size=(simulations, sim_length))
        simulated_pnls = pnls[random_indices]
        
        equity_curves = np.cumsum(simulated_pnls, axis=1) + initial_capital
        
        starting_capital = np.full((simulations, 1), initial_capital)
        equity_curves = np.hstack((starting_capital, equity_curves))
        
        median_curve = np.median(equity_curves, axis=0)
        
        final_balances = equity_curves[:, -1]
        prob_profit = (np.sum(final_balances > initial_capital) / simulations) * 100
        
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (running_max - equity_curves) / running_max
        ruined_simulations = np.any(drawdowns > 0.20, axis=1)
        risk_of_ruin = (np.sum(ruined_simulations) / simulations) * 100
        
        median_final_balance = np.median(final_balances)
        
        fig = go.Figure()
        
        x_base = np.arange(sim_length + 1)
        x_all = np.tile(np.append(x_base, np.nan), simulations)
        y_all = np.hstack((equity_curves, np.full((simulations, 1), np.nan))).flatten()
        
        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_base,
            y=median_curve,
            mode='lines',
            line=dict(color='gold', width=3),
            name='Median (50th Percentile)'
        ))
        
        fig.update_layout(
            title='🔬 Monte Carlo Robustness Analysis (Forward 50 Trades)',
            xaxis_title='Trade Number',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig, prob_profit, risk_of_ruin, median_final_balance

    class InstitutionalIndicators:
        @staticmethod
        def vwap_sd(df):
            """
            Calcola il VWAP e le Bande di Deviazione Standard (1, 2, 3 SD).
            Per il Forex, usa df['Volume'] come Tick Volume.
            """
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                df['VWAP'] = df['Close']
                for i in range(1, 4):
                    df[f'VWAP_Upper_{i}SD'] = df['Close']
                    df[f'VWAP_Lower_{i}SD'] = df['Close']
                return df

            df['Date'] = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VP'] = df['Typical_Price'] * df['Volume']
            
            daily_groups = df.groupby('Date')
            df['Cum_VP'] = daily_groups['VP'].cumsum()
            df['Cum_Vol'] = daily_groups['Volume'].cumsum()
            
            df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
            
            df['Price_Diff_Sq'] = ((df['Typical_Price'] - df['VWAP']) ** 2) * df['Volume']
            df['Cum_Price_Diff_Sq'] = daily_groups['Price_Diff_Sq'].cumsum()
            df['VWAP_Variance'] = df['Cum_Price_Diff_Sq'] / df['Cum_Vol']
            df['VWAP_SD'] = np.sqrt(df['VWAP_Variance'])
            
            for i in range(1, 4):
                df[f'VWAP_Upper_{i}SD'] = df['VWAP'] + (i * df['VWAP_SD'])
                df[f'VWAP_Lower_{i}SD'] = df['VWAP'] - (i * df['VWAP_SD'])
                
            df.drop(columns=['Date', 'Typical_Price', 'VP', 'Cum_VP', 'Cum_Vol', 'Price_Diff_Sq', 'Cum_Price_Diff_Sq', 'VWAP_Variance', 'VWAP_SD'], inplace=True, errors='ignore')
            return df

        @staticmethod
        def volume_profile_intraday(df, bins=50, value_area_pct=0.70):
            """
            Calcola il Volume Profile Intraday: POC, VAH, VAL.
            Per il Forex, usa df['Volume'] come Tick Volume.
            """
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                df['POC'] = df['Close']
                df['VAH'] = df['High']
                df['VAL'] = df['Low']
                return df

            df['Date'] = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
            df['POC'] = np.nan
            df['VAH'] = np.nan
            df['VAL'] = np.nan

            for date, group in df.groupby('Date'):
                min_price = group['Low'].min()
                max_price = group['High'].max()
                
                if min_price == max_price:
                    df.loc[group.index, 'POC'] = min_price
                    df.loc[group.index, 'VAH'] = min_price
                    df.loc[group.index, 'VAL'] = min_price
                    continue

                price_bins = np.linspace(min_price, max_price, bins)
                volume_profile = np.zeros(bins - 1)

                for _, row in group.iterrows():
                    idx = np.digitize((row['High'] + row['Low'] + row['Close']) / 3, price_bins) - 1
                    idx = min(max(idx, 0), bins - 2)
                    volume_profile[idx] += row['Volume']

                poc_idx = np.argmax(volume_profile)
                poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
                
                total_volume = np.sum(volume_profile)
                va_volume = volume_profile[poc_idx]
                upper_idx = poc_idx
                lower_idx = poc_idx
                
                while va_volume < total_volume * value_area_pct:
                    upper_vol = volume_profile[upper_idx + 1] if upper_idx < bins - 2 else 0
                    lower_vol = volume_profile[lower_idx - 1] if lower_idx > 0 else 0
                    
                    if upper_vol == 0 and lower_vol == 0:
                        break
                        
                    if upper_vol >= lower_vol:
                        upper_idx += 1
                        va_volume += upper_vol
                    else:
                        lower_idx -= 1
                        va_volume += lower_vol

                vah_price = price_bins[upper_idx + 1]
                val_price = price_bins[lower_idx]

                df.loc[group.index, 'POC'] = poc_price
                df.loc[group.index, 'VAH'] = vah_price
                df.loc[group.index, 'VAL'] = val_price

            df.drop(columns=['Date'], inplace=True, errors='ignore')
            return df

    class StrategyFactory:
        @staticmethod
        def get_vwap_reversion_signal(df, params):
            """
            Genera segnali di mean reversion basati sulle bande di deviazione standard del VWAP.
            params: dict con chiave 'sd_level' (default 2)
            """
            sd_level = params.get('sd_level', 2)
            signals = pd.Series(0, index=df.index)
            
            upper_col = f'VWAP_Upper_{sd_level}SD'
            lower_col = f'VWAP_Lower_{sd_level}SD'
            
            if upper_col not in df.columns or lower_col not in df.columns:
                return signals
                
            # Long: Il prezzo tocca/scende sotto la banda inferiore e chiude sopra (rejection)
            long_cond = (df['Low'] < df[lower_col]) & (df['Close'] > df[lower_col])
            
            # Short: Il prezzo tocca/sale sopra la banda superiore e chiude sotto (rejection)
            short_cond = (df['High'] > df[upper_col]) & (df['Close'] < df[upper_col])
            
            signals.loc[long_cond] = 1
            signals.loc[short_cond] = -1
            
            return signals

        @staticmethod
        def get_volumeprofile_signal(df, params):
            """
            Genera segnali di breakout o rejection dal POC e dai bordi della Value Area (VAH, VAL).
            params: dict con chiave 'mode' ('breakout' o 'rejection')
            """
            mode = params.get('mode', 'breakout')
            signals = pd.Series(0, index=df.index)
            
            if not all(col in df.columns for col in ['POC', 'VAH', 'VAL']):
                return signals
                
            if mode == 'breakout':
                # Breakout Long: Chiusura sopra VAH (con apertura sotto o uguale)
                long_cond = (df['Close'] > df['VAH']) & (df['Open'] <= df['VAH'])
                # Breakout Short: Chiusura sotto VAL (con apertura sopra o uguale)
                short_cond = (df['Close'] < df['VAL']) & (df['Open'] >= df['VAL'])
            else:
                # Rejection Long: Minimo sotto VAL ma chiusura sopra VAL
                long_cond = (df['Low'] < df['VAL']) & (df['Close'] > df['VAL'])
                # Rejection Short: Massimo sopra VAH ma chiusura sotto VAH
                short_cond = (df['High'] > df['VAH']) & (df['Close'] < df['VAH'])
                
            signals.loc[long_cond] = 1
            signals.loc[short_cond] = -1
            
            return signals

        @staticmethod
        def get_orb_signal(df, params):
            """
            Genera segnali basati sull'Opening Range Breakout (ORB).
            params: dict con chiave 'orb_duration' (minuti, default 15)
            """
            orb_duration = params.get('orb_duration', 15)
            signals = pd.Series(0, index=df.index)
            
            # Identifica la colonna datetime o l'indice
            if isinstance(df.index, pd.DatetimeIndex):
                dt_series = df.index.to_series()
            elif 'datetime' in df.columns:
                dt_series = pd.to_datetime(df['datetime'])
            else:
                return signals
                
            # Creiamo un DataFrame temporaneo per raggruppare per giorno in modo efficiente
            df_temp = pd.DataFrame({
                'dt': dt_series, 
                'Open': df['Open'], 
                'High': df['High'], 
                'Low': df['Low'], 
                'Close': df['Close']
            }, index=df.index)
            df_temp['Date'] = df_temp['dt'].dt.date
            
            for date, group in df_temp.groupby('Date'):
                if group.empty:
                    continue
                    
                # Definisci il periodo ORB (primi 'orb_duration' minuti della sessione per quel giorno)
                start_time = group['dt'].iloc[0]
                end_orb_time = start_time + pd.Timedelta(minutes=orb_duration)
                
                orb_mask = (group['dt'] >= start_time) & (group['dt'] < end_orb_time)
                orb_data = group[orb_mask]
                
                if orb_data.empty:
                    continue
                    
                orb_high = orb_data['High'].max()
                orb_low = orb_data['Low'].min()
                
                # Dati successivi all'ORB
                post_orb_mask = group['dt'] >= end_orb_time
                
                # Breakout Long: Chiusura sopra l'ORB High
                long_cond = post_orb_mask & (group['Close'] > orb_high) & (group['Open'] <= orb_high)
                # Breakout Short: Chiusura sotto l'ORB Low
                short_cond = post_orb_mask & (group['Close'] < orb_low) & (group['Open'] >= orb_low)
                
                # Assegna i segnali usando gli indici originali
                signals.loc[group[long_cond].index] = 1
                signals.loc[group[short_cond].index] = -1
                
            return signals

        @staticmethod
        def get_ta_signal(df, indicator_name, params, condition):
            """
            Genera segnali utilizzando pandas_ta in modo AGNOSTICO ai nomi.
            """
            signals = pd.Series(0, index=df.index)
            try:
                # Estraiamo parametri speciali di routing
                target_band = params.pop('_band', None)
                target_line = params.pop('_line', None)
                
                # Calcola l'indicatore
                ind_func = getattr(df.ta, indicator_name)
                ind_result = ind_func(**params)
                
                # --- NORMALIZZAZIONE COLONNA TARGET ---
                target_col_name = None
                if isinstance(ind_result, pd.Series):
                    ind_result.name = "TARGET_IND"
                elif isinstance(ind_result, pd.DataFrame):
                    # Logica per trovare la colonna giusta se l'indicatore ne genera molte
                    if target_band == 'Lower': target_col_name = [c for c in ind_result.columns if 'l' in c.lower()][-1]
                    elif target_band == 'Upper': target_col_name = [c for c in ind_result.columns if 'u' in c.lower()][-1]
                    elif target_band == 'Middle': target_col_name = [c for c in ind_result.columns if 'm' in c.lower()][-1]
                    elif indicator_name == 'macd': target_col_name = ind_result.columns[0] # Linea MACD principale
                    elif indicator_name == 'stoch': target_col_name = ind_result.columns[1] if target_line == 'D' else ind_result.columns[0]
                    else: target_col_name = ind_result.columns[0] # Fallback
                    
                    ind_result = ind_result[[target_col_name]].rename(columns={target_col_name: "TARGET_IND"})
                
                # Unisci al df originale usando il nome sicuro
                temp_df = pd.concat([df, ind_result], axis=1)
                
                # --- VALUTAZIONE CONDIZIONI ---
                if isinstance(condition, dict):
                    if condition.get('long'):
                        signals.loc[temp_df.eval(condition['long'])] = 1
                    if condition.get('short'):
                        signals.loc[temp_df.eval(condition['short'])] = -1
            except Exception as e:
                pass # Silenzia errori anomali per non bloccare i test
            return signals


    def run_custom_strategy(df, start_time, end_time, eod_close, initial_capital, risk_per_trade, rr_ratio, sl_mode, fixed_sl_pct):
        trades = []
        if df.empty or 'Master_Signal' not in df.columns:
            return trades
            
        df = df.copy()
        
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        # Ensure no data is dropped after indicator calculations (Deep data support)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        in_position = False
        entry_price = 0
        entry_time = None
        position_type = None
        size = 0
        sl_price = 0
        tp_price = 0
        
        grouped = df.groupby('date')
        
        for date, group in grouped:
            group = group.sort_values('datetime').reset_index(drop=True)
            
            # Use itertuples for performance on large datasets
            for row in group.itertuples():
                current_time = row.time
                current_datetime = row.datetime
                
                is_trading_time = start_time <= current_time <= end_time
                
                if in_position:
                    exit_triggered = False
                    exit_type = ""
                    exit_price = 0
                    
                    if position_type == 'LONG':
                        if row.Low <= sl_price:
                            exit_triggered = True
                            exit_type = "SL"
                            exit_price = sl_price
                        elif row.High >= tp_price:
                            exit_triggered = True
                            exit_type = "TP"
                            exit_price = tp_price
                    elif position_type == 'SHORT':
                        if row.High >= sl_price:
                            exit_triggered = True
                            exit_type = "SL"
                            exit_price = sl_price
                        elif row.Low <= tp_price:
                            exit_triggered = True
                            exit_type = "TP"
                            exit_price = tp_price
                            
                    if not exit_triggered and eod_close and current_time >= end_time:
                        exit_triggered = True
                        exit_type = "EOD"
                        exit_price = row.Close
                    elif not exit_triggered and not is_trading_time:
                        exit_triggered = True
                        exit_type = "Out of Time"
                        exit_price = row.Close
                    
                    if exit_triggered:
                        pnl = (exit_price - entry_price) if position_type == 'LONG' else (entry_price - exit_price)
                        pnl *= size
                        
                        trades.append({
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': current_datetime,
                            'Exit Price': exit_price,
                            'pnl': pnl,
                            'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                            'Direction': 'long' if position_type == 'LONG' else 'short',
                            'Status': exit_type,
                            'type': f'EXIT {exit_type}',
                            'time': current_datetime,
                            'price': exit_price,
                            'Logica': f"Exit: {exit_type}"
                        })
                        in_position = False
                
                if not in_position and is_trading_time:
                    if row.Master_Signal == 1:
                        in_position = True
                        position_type = 'LONG'
                        entry_price = row.Close
                        entry_time = current_datetime
                        
                        if sl_mode == "Fixed %":
                            sl_price = entry_price * (1 - fixed_sl_pct / 100)
                        else:
                            sl_price = row.Low
                            if sl_price >= entry_price:
                                sl_price = entry_price * 0.999
                                
                        tp_price = entry_price + ((entry_price - sl_price) * rr_ratio)
                        
                        risk_amount = initial_capital * (risk_per_trade / 100)
                        risk_per_unit = entry_price - sl_price
                        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                        
                    elif row.Master_Signal == -1:
                        in_position = True
                        position_type = 'SHORT'
                        entry_price = row.Close
                        entry_time = current_datetime
                        
                        if sl_mode == "Fixed %":
                            sl_price = entry_price * (1 + fixed_sl_pct / 100)
                        else:
                            sl_price = row.High
                            if sl_price <= entry_price:
                                sl_price = entry_price * 1.001
                                
                        tp_price = entry_price - ((sl_price - entry_price) * rr_ratio)
                        
                        risk_amount = initial_capital * (risk_per_trade / 100)
                        risk_per_unit = sl_price - entry_price
                        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                        
        return trades

    if st.button("🚀 Esegui Strategia Custom"):
        # LOG FOREX
        if ticker in categories.get("Forex", []):
            st.info(f"📥 Connessione a HistData. Download e decompressione di {ticker} in corso... (Potrebbe richiedere alcuni secondi)")
            
        with st.spinner("Fetching data and running strategy..."):
            df = fetch_data_smart(ticker, timeframe, start_date, end_date, target_tz=tz_choice)
            if not df.empty:
                # Calcolo indicatori istituzionali se necessari
                if "VWAP Reversion" in core_strategies:
                    df = InstitutionalIndicators.vwap_sd(df)
                if "Volume Profile" in core_strategies:
                    df = InstitutionalIndicators.volume_profile_intraday(df)
                
                # Generazione segnali
                all_signals = []
                
                if "ORB (Opening Range Breakout)" in core_strategies:
                    sig = StrategyFactory.get_orb_signal(df, {'orb_duration': orb_duration})
                    all_signals.append(sig)
                
                if "VWAP Reversion" in core_strategies:
                    sig = StrategyFactory.get_vwap_reversion_signal(df, {'sd_level': vwap_sd_level})
                    all_signals.append(sig)
                    
                if "Volume Profile" in core_strategies:
                    sig = StrategyFactory.get_volumeprofile_signal(df, {'mode': vp_mode})
                    all_signals.append(sig)
                    
                for ta_f in ta_filters:
                    sig = StrategyFactory.get_ta_signal(df, ta_f['name'], ta_f['params'], ta_f['condition'])
                    all_signals.append(sig)
                
                # Combinazione segnali
                df['Master_Signal'] = 0
                if all_signals:
                    signals_df = pd.concat(all_signals, axis=1)
                    if combinatorial_logic.startswith("OR"):
                        long_mask = (signals_df == 1).any(axis=1)
                        short_mask = (signals_df == -1).any(axis=1)
                        df.loc[long_mask, 'Master_Signal'] = 1
                        df.loc[short_mask & ~long_mask, 'Master_Signal'] = -1
                    else: # AND
                        long_mask = (signals_df == 1).all(axis=1)
                        short_mask = (signals_df == -1).all(axis=1)
                        df.loc[long_mask, 'Master_Signal'] = 1
                        df.loc[short_mask, 'Master_Signal'] = -1
                
                trades = run_custom_strategy(df, start_time, end_time, eod_close, initial_capital, risk_per_trade, rr_ratio, sl_mode, fixed_sl_pct)
                
                if trades:
                    friction_pct = 0.0
                    adjusted_trades, adjusted_equity = apply_friction_post_process(trades, initial_capital, friction_pct)
                    
                    st.subheader("📊 Risultati Strategia")
                    # Passiamo la equity_curve alla nuova funzione
                    metrics = calculate_advanced_metrics(adjusted_trades, adjusted_equity)
                    
                    # 5 Colonne per aggiungere il Total Trades
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Tot. Operazioni", len(adjusted_trades))
                    col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    col3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                    col5.metric("Total Profit", f"${metrics['total_profit_abs']:.2f}")
                    
                    st.line_chart(adjusted_equity)
                    
                    with st.expander('🔍 Analisi di Robustezza e Stress Test', expanded=True):
                        mc_res = run_monte_carlo(adjusted_trades, initial_capital)
                        if mc_res:
                            mc_fig, prob_profit, risk_of_ruin, median_final_balance = mc_res
                            st.plotly_chart(mc_fig, use_container_width=True)
                            
                            st.subheader('🔬 Validazione Statistica Long-Term')
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric('Probabilità di Profitto (Prossimi 50 Trade)', f"{prob_profit:.1f}%")
                            with c2:
                                st.metric('Rischio di Rovina (Max DD > 20%)', f"{risk_of_ruin:.1f}%")
                            with c3:
                                st.metric('Rendimento Mediano Stimato', f"${median_final_balance:.2f}")
                            
                            if prob_profit > 75:
                                st.success('✅ Strategia Robusta')
                            elif prob_profit < 60:
                                st.warning('⚠️ Strategia Fragile (Flop)')
                                
                            if risk_of_ruin > 10:
                                st.error('⚠️ Rischio di Rovina Elevato: La strategia potrebbe bruciare il conto.')
                                
                            if len(adjusted_trades) < 30:
                                st.warning('⚠️ Low Sample Size: Results might be overly optimistic.')
                        else:
                            st.warning("Not enough data for Monte Carlo simulation.")
                            
                    st.subheader("📝 Dettaglio Operazioni")
                    st.dataframe(pd.DataFrame(adjusted_trades))
                else:
                    st.warning("Nessun trade generato con questi parametri.")
            else:
                st.error("Errore nel recupero dei dati storici.")

elif menu == "🏛️ BLOOMBERG TERMINAL (Inst.)":
    st.markdown("""<style>
        .metric-card { background: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
        .status-pill { padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }
    </style>""", unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def get_terminal_data(ticker):
        t = yf.Ticker(ticker)
        try:
            # Recupero info e valuta
            inf = t.info
            currency = inf.get('financialCurrency', inf.get('currency', '$'))
            sym = '€' if currency == 'EUR' else ('$' if currency == 'USD' else currency + " ")
            
            return {
                "info": inf,
                "sym": sym,
                "history": t.history(period="2y"),
                "fin": t.financials,
                "q_fin": t.quarterly_financials,  # <-- AGGIUNTO
                "bs": t.balance_sheet,
                "q_bs": t.quarterly_balance_sheet, # <-- AGGIUNTO
                "cf": t.cashflow,
                "q_cf": t.quarterly_cashflow,      # <-- AGGIUNTO
                "inst": t.institutional_holders,
                "insider": t.insider_transactions,
                "news": t.news
            }
        except: return None

    def format_big_num(val, sym):
        if pd.isna(val) or not isinstance(val, (int, float)): return "N/A"
        abs_v = abs(val)
        sign = "-" if val < 0 else ""
        if abs_v >= 1e12: return f"{sign}{sym}{abs_v/1e12:.2f}T"
        if abs_v >= 1e9: return f"{sign}{sym}{abs_v/1e9:.2f}B"
        if abs_v >= 1e6: return f"{sign}{sym}{abs_v/1e6:.2f}M"
        return f"{sign}{sym}{abs_v:,.0f}"

    t_code = st.sidebar.text_input("Ticker Bloomberg:", value="AAPL").upper().strip()
    
    # --- NORMALIZZAZIONE TICKER GLOBALE ---
    if t_code:
        # Mappa rapida indici comuni
        idx_map = {"SPX": "^GSPC", "NDX": "^IXIC", "DJI": "^DJI", "DAX": "^GDAXI", "FTSEMIB": "FTSEMIB.MI"}
        if t_code in idx_map:
            t_code = idx_map[t_code]
        
        # Fallback: se non è un'azione (EQUITY) e non ha il prefisso, prova ad aggiungerlo
        try:
            check = yf.Ticker(t_code).info
            if not check or 'quoteType' not in check:
                if not t_code.startswith("^"):
                    t_code = "^" + t_code
        except:
            if not t_code.startswith("^"):
                t_code = "^" + t_code
    
    if t_code:
        data = get_terminal_data(t_code)
        if data and data["info"]:
            inf = data["info"]
            s = data["sym"]
            
            # --- HEADER / PRE-CALCOLI ---
            c_price = inf.get('currentPrice', 0)
            ticker_data = yf.Ticker(t_code)
            
            # 1. INIZIALIZZAZIONE DI SICUREZZA
            z_val, f_score, m_score, val_dcf, margin_graham, ev_val, roic, wacc, roic_wacc_spread, eps_growth, final_rating, magic_ey, ev_ebitda, tr_eps = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1
            
            # --- NUOVO BLOCCO CALCOLO ISTITUZIONALE ---
            try:
                bs = data["bs"] if data["bs"] is not None else pd.DataFrame()
                cf = data["cf"] if data["cf"] is not None else pd.DataFrame()
                is_ = data["fin"] if data["fin"] is not None else pd.DataFrame()
                
                # Z-Score (Proxy)
                z_val = (inf.get('currentRatio', 0) * 1.2) + (inf.get('profitMargins', 0) * 3.3)
                
                # F-Score (9/9)
                if not bs.empty and not cf.empty and not is_.empty and bs.shape[1] >= 2 and cf.shape[1] >= 1 and is_.shape[1] >= 2:
                    try:
                        net_inc_curr = is_.loc['Net Income'].iloc[0] if 'Net Income' in is_.index else 0
                        net_inc_prev = is_.loc['Net Income'].iloc[1] if 'Net Income' in is_.index else 0
                        cfo_curr = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
                        ta_curr = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else 1
                        ta_prev = bs.loc['Total Assets'].iloc[1] if 'Total Assets' in bs.index else 1
                        roa_curr = net_inc_curr / ta_curr if ta_curr != 0 else 0
                        roa_prev = net_inc_prev / ta_prev if ta_prev != 0 else 0
                        
                        f_score += 1 if roa_curr > 0 else 0
                        f_score += 1 if cfo_curr > 0 else 0
                        f_score += 1 if roa_curr > roa_prev else 0
                        f_score += 1 if cfo_curr > net_inc_curr else 0
                        
                        lt_curr = bs.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in bs.index else 0
                        lt_prev = bs.loc['Long Term Debt'].iloc[1] if 'Long Term Debt' in bs.index else 0
                        f_score += 1 if lt_curr < lt_prev else 0
                        
                        ca_curr = bs.loc['Current Assets'].iloc[0] if 'Current Assets' in bs.index else 0
                        ca_prev = bs.loc['Current Assets'].iloc[1] if 'Current Assets' in bs.index else 0
                        cl_curr = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index else 1
                        cl_prev = bs.loc['Current Liabilities'].iloc[1] if 'Current Liabilities' in bs.index else 1
                        cr_curr = ca_curr / cl_curr if cl_curr != 0 else 0
                        cr_prev = ca_prev / cl_prev if cl_prev != 0 else 0
                        f_score += 1 if cr_curr > cr_prev else 0
                        
                        sh_curr = bs.loc['Ordinary Shares Number'].iloc[0] if 'Ordinary Shares Number' in bs.index else 0
                        sh_prev = bs.loc['Ordinary Shares Number'].iloc[1] if 'Ordinary Shares Number' in bs.index else 0
                        f_score += 1 if sh_curr <= sh_prev else 0
                        
                        gp_curr = is_.loc['Gross Profit'].iloc[0] if 'Gross Profit' in is_.index else 0
                        gp_prev = is_.loc['Gross Profit'].iloc[1] if 'Gross Profit' in is_.index else 0
                        rev_curr = is_.loc['Total Revenue'].iloc[0] if 'Total Revenue' in is_.index else 1
                        rev_prev = is_.loc['Total Revenue'].iloc[1] if 'Total Revenue' in is_.index else 1
                        gm_curr = gp_curr / rev_curr if rev_curr != 0 else 0
                        gm_prev = gp_prev / rev_prev if rev_prev != 0 else 0
                        f_score += 1 if gm_curr > gm_prev else 0
                        
                        at_curr = rev_curr / ta_curr if ta_curr != 0 else 0
                        at_prev = rev_prev / ta_prev if ta_prev != 0 else 0
                        f_score += 1 if at_curr > at_prev else 0
                    except:
                        pass
                
                # M-Score
                m_score = f_score # Placeholder for Beneish
                
                # Magic EY
                try:
                    ebit = is_.loc['EBIT'].iloc[0] if 'EBIT' in is_.index and not is_.empty else inf.get('ebitda', 0)
                    ev = inf.get('enterpriseValue', 0)
                    magic_ey = (ebit / ev) if ev and ev > 0 else 0
                except:
                    magic_ey = 0
                
                # DCF e Graham
                val_dcf = calculate_dcf_value(ticker_data)
                eps_g, bvps = inf.get('trailingEps', 0), inf.get('bookValue', 0)
                graham = np.sqrt(22.5 * eps_g * bvps) if eps_g is not None and bvps is not None and eps_g > 0 and bvps > 0 else 0
                margin_graham = ((graham - c_price)/graham*100) if graham > 0 else -100
                
                # Nuove Metriche per Rating
                ev_ebitda = inf.get('enterpriseToEbitda', 0) or 0
                ev_val = float(ev_ebitda)
                
                # ROIC/WACC Proxy
                ebit_val = is_.loc['EBIT'].iloc[0] if 'EBIT' in is_.index and not is_.empty else inf.get('ebitda', 0)
                ta = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index and not bs.empty else 1
                cl = bs.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in bs.index and not bs.empty else 0
                invested_capital = ta - cl
                roic = (ebit_val / invested_capital) if invested_capital > 0 else inf.get('returnOnAssets', 0)
                wacc = 0.04 + (inf.get('beta', 1.0) * 0.055)
                roic_wacc_spread = roic - wacc
                
                # EPS Growth
                tr_eps = inf.get('trailingEps', 1)
                eps_growth = ((inf.get('forwardEps', 0) - tr_eps) / tr_eps) * 100 if tr_eps > 0 else 0

                # Calcolo Score Finale
                s_fin = 0
                quote_type = inf.get('quoteType', 'EQUITY').upper()
                
                if quote_type in ['ETF', 'MUTUALFUND', 'INDEX']:
                    fund_metrics = calc_fund_metrics_v3(t_code, ticker_data)
                    if fund_metrics:
                        try:
                            score = compute_fund_score(fund_metrics)
                            s_fin = score / 10.0
                            st.session_state.fund_metrics = fund_metrics
                        except:
                            s_fin = 1.0
                            st.session_state.fund_metrics = None
                    else:
                        s_fin = 1.0  # Dati insufficienti
                        st.session_state.fund_metrics = None
                else:
                    st.session_state.fund_metrics = None
                    if z_val > 2.99: s_fin += 1.5
                    elif z_val > 1.81: s_fin += 0.75
                    if f_score >= 7: s_fin += 1.0
                    elif f_score >= 5: s_fin += 0.5
                    if roic > wacc: s_fin += 1.5
                    if roic > 0.15: s_fin += 1.0
                    if magic_ey > 0.10: s_fin += 0.5
                    if val_dcf and val_dcf > c_price: s_fin += 1.0
                    if margin_graham > 20: s_fin += 1.0
                    if 0 < ev_val < 12: s_fin += 1.0
                    elif 12 <= ev_val < 16: s_fin += 0.5
                    if eps_growth > 10: s_fin += 1.0
                    elif eps_growth > 0: s_fin += 0.5
                
                final_rating = max(1.0, min(10.0, s_fin))
                st.session_state.score = final_rating
            except Exception as e:
                final_rating = 1.0
                st.session_state.score = 1.0
                st.session_state.fund_metrics = None

            if quote_type in ['ETF', 'MUTUALFUND', 'INDEX'] and st.session_state.fund_metrics is None:
                st.title(f"🏢 {inf.get('shortName', t_code)} | ⚪ Dati Insufficienti")
            else:
                if quote_type in ['ETF', 'MUTUALFUND', 'INDEX']:
                    r_emj = "🟢" if s_fin >= 7.5 else "🟡" if s_fin >= 5.5 else "🔴"
                    st.title(f"🏢 {inf.get('shortName', t_code)} | {r_emj} Quant Score: {int(s_fin*10)}/100")
                else:
                    r_emj = "🟢" if final_rating >= 7.5 else "🟡" if final_rating >= 5.5 else "🔴"
                    st.title(f"🏢 {inf.get('shortName', t_code)} | {r_emj} Score: {final_rating:.1f}/10")
            st.markdown(f"**Analisi Quantitativa Istituzionale - Financial Deep Analysis**")
            
            if quote_type not in ['ETF', 'MUTUALFUND', 'INDEX']:
                # --- SEMAFORI ISTITUZIONALI (VISUAL IMPACT) ---
                st.subheader("🚀 Analisi Rapida Wall Street")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    color = "#2ecc71" if margin_graham > 20 else ("#f1c40f" if margin_graham > 0 else "#e74c3c")
                    txt = "SOTTOVALUTATA" if margin_graham > 20 else ("EQUA" if margin_graham > 0 else "SOPRAVVALUTATA")
                    st.markdown(f"<div class='metric-card'>Valutazione Graham<br><span class='status-pill' style='background:{color}; color:black;'>{txt}</span><br><small>Safety: {margin_graham:.1f}%</small></div>", unsafe_allow_html=True)
                    
                    if val_dcf:
                        upside_dcf = ((val_dcf - c_price) / c_price) * 100
                        color_dcf = "#2ecc71" if val_dcf > c_price else "#e74c3c"
                        st.markdown(f"""
                            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {color_dcf}; margin-top:10px;">
                                <p style="color: #888; margin: 0; font-size: 0.8em;">VALUTAZIONE INTRINSECA DCF</p>
                                <h3 style="margin: 0; color: white;">${val_dcf:.2f}</h3>
                                <p style="color: {color_dcf}; margin: 0; font-weight: bold;">{upside_dcf:+.2f}% vs Mercato</p>
                            </div>
                        """, unsafe_allow_html=True)

                with col_b:
                    color = "#2ecc71" if z_val > 2.6 else ("#f1c40f" if z_val > 1.1 else "#e74c3c")
                    txt = "SOLIDA" if z_val > 2.6 else "ALLERTA"
                    st.markdown(f"<div class='metric-card'>Rischio Fallimento<br><span class='status-pill' style='background:{color}; color:black;'>{txt}</span><br><small>Z-Score: {z_val:.2f}</small></div>", unsafe_allow_html=True)

                with col_c:
                    roe = inf.get('returnOnEquity', 0)
                    color = "#2ecc71" if roe > 0.15 else "#e74c3c"
                    txt = "ALTA REDDITIVITÀ" if roe > 0.15 else "BASSA EFFICIENZA"
                    st.markdown(f"<div class='metric-card'>Performance Capitale<br><span class='status-pill' style='background:{color}; color:black;'>{txt}</span><br><small>ROE: {roe*100:.1f}%</small></div>", unsafe_allow_html=True)

                st.write("---")
                st.subheader("🏛️ Analisi Avanzata & Solvibilità Istituzionale")
                
                try:
                    # --- RENDER SEMAFORICO ---
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        p_color = "normal" if f_score >= 7 else "off" if f_score >= 5 else "inverse"
                        p_label = "🟢 STRONG" if f_score >= 7 else "🟡 NEUTRAL" if f_score >= 5 else "🔴 WEAK"
                        st.metric("Piotroski Score", f"{f_score}/9", delta=p_label, delta_color=p_color)
                    
                    with c2:
                        ey = magic_ey / 100
                        ey_color = "normal" if ey > 0.10 else "off" if ey > 0.05 else "inverse"
                        ey_label = "🟢 UNDERVALUED" if ey > 0.10 else "🟡 FAIR" if ey > 0.05 else "🔴 OVERVALUED"
                        st.metric("Magic Yield (EY)", f"{ey:.2%}", delta=ey_label, delta_color=ey_color)
                    
                    with c3:
                        m_color = "normal" if f_score > 5 else "inverse"
                        m_label = "🟢 SAFE" if f_score > 5 else "🔴 RISK"
                        st.metric("Beneish M-Score", m_label, delta="Bilanci OK" if f_score > 5 else "Controllare", delta_color=m_color)
                    
                    with c4:
                        try:
                            peg_raw = inf.get('pegRatio')
                            peg_val = float(peg_raw) if peg_raw is not None else 2.0
                            peg_color = "normal" if peg_val < 1.0 else "off" if peg_val < 2.0 else "inverse"
                            peg_label = "🟢 CHEAP" if peg_val < 1.0 else "🟡 FAIR" if peg_val < 2.0 else "🔴 EXPENSIVE"
                        except:
                            peg_val, peg_label, peg_color = "N/D", "⚪ N/D", "off"
                        st.metric("PEG Ratio", f"{peg_val}", delta=peg_label, delta_color=peg_color)
                    
                    # SECONDA RIGA METRICHE ISTITUZIONALI
                    st.markdown("<br>", unsafe_allow_html=True)
                    c5, c6, c7, c8 = st.columns(4)
                    
                    with c5:
                        ev_val = float(ev_ebitda) if ev_ebitda is not None else 0.0
                        ev_color = "normal" if 0 < ev_val < 12 else "off" if ev_val < 16 else "inverse"
                        ev_label = "🟢 DISCOUNT" if 0 < ev_val < 12 else "🟡 FAIR" if ev_val < 16 else "🔴 PREMIUM"
                        st.metric("EV / EBITDA", f"{ev_val:.1f}x" if ev_val > 0 else "N/D", delta=ev_label, delta_color=ev_color)
                    
                    with c6:
                        roic_color = "normal" if roic > 0.10 else "inverse"
                        st.metric("ROIC (Ret. on Capital)", f"{roic:.2%}", delta="🟢 Eccellente" if roic > 0.15 else "🔴 Da migliorare", delta_color=roic_color)
                    
                    with c7:
                        spread_color = "normal" if roic_wacc_spread > 0 else "inverse"
                        spread_label = "🟢 VALUE CREATOR" if roic_wacc_spread > 0 else "🔴 VALUE DESTROYER"
                        st.metric("ROIC - WACC Spread", f"{roic_wacc_spread*100:+.1f}%", delta=spread_label, delta_color=spread_color)
                    
                    with c8:
                        eps_color = "normal" if eps_growth > 0 else "inverse"
                        eps_label = "🟢 ESPANSIONE" if eps_growth > 0 else "🔴 CONTRAZIONE"
                        st.metric("Stima Crescita EPS", f"{eps_growth:+.1f}%" if tr_eps > 0 else "N/D", delta=eps_label, delta_color=eps_color)
                    
                    st.write("")
                except Exception as e:
                    st.warning("Dati insufficienti per il calcolo dell'Analisi Avanzata.")
            else:
                # Se è un fondo, mostriamo il Semaforo Quantitativo
                st.write("---")
                st.subheader("🏛️ Analisi Quantitativa Fondi & ETF")
                m = st.session_state.get('fund_metrics')
                if m:
                    score = compute_fund_score(m)
                    
                    st.markdown(f"### 🛡️ Valutazione Quantitativa Professionale: {score}/100")
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#2ecc71" if score >= 70 else "#f1c40f" if score >= 40 else "#e74c3c"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(231,76,60,0.3)"},
                                {'range': [40, 70], 'color': "rgba(241,196,15,0.3)"},
                                {'range': [70, 100], 'color': "rgba(46,204,113,0.3)"}],
                            }
                    ))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
                    
                    col_gauge, col_spacer = st.columns([1, 2])
                    with col_gauge:
                        st.plotly_chart(fig, use_container_width=True)

                    r1 = st.columns(4)
                    with r1[0]: draw_metric_badge("SHARPE", f"{m['Sharpe']:.2f}" if pd.notna(m['Sharpe']) else "N/D", get_metric_color(m.get('Sharpe', 0), 'sharpe'), "EFFICIENZA")
                    with r1[1]: draw_metric_badge("MAX DRAWDOWN", f"{m['Max_DD']*100:.1f}%" if pd.notna(m['Max_DD']) else "N/D", get_metric_color(m.get('Max_DD', 0), 'drawdown'), "PROTEZIONE")
                    with r1[2]: draw_metric_badge("ALPHA", f"{m['Alpha']*100:.1f}%" if pd.notna(m['Alpha']) else "N/D", get_metric_color(m.get('Alpha', 0), 'alpha'), "EXTRA REND")
                    with r1[3]: draw_metric_badge("CAGR 5Y", f"{m['CAGR']*100:.1f}%" if pd.notna(m['CAGR']) else "N/D", get_metric_color(m.get('CAGR', 0), 'cagr'), "CRESCITA")

                    r2 = st.columns(3)
                    with r2[0]: draw_metric_badge("VOLATILITÀ", f"{m['Vol']*100:.1f}%" if pd.notna(m['Vol']) else "N/D", get_metric_color(m.get('Vol', 0), 'vol'), "STABILITÀ")
                    with r2[1]: draw_metric_badge("BETA", f"{m['Beta']:.2f}" if pd.notna(m['Beta']) else "N/D", get_metric_color(m.get('Beta', 0), 'beta'), "REATTIVITÀ")
                    with r2[2]: draw_metric_badge("TRACK ERROR", f"{m['TE']*100:.2f}%" if pd.notna(m['TE']) else "N/D", get_metric_color(m.get('TE', 0), 'te'), "QUALITÀ")
                else:
                    st.warning("Dati storici insufficienti per calcolare le metriche del fondo.")

            # --- TABS DATI PROFONDI ---
            t1, t2, t3, t4 = st.tabs(["📊 BILANCI DETTAGLIATI", "📈 GRAFICO INTERATTIVO", "🐋 FLUSSI WHALES", "📰 NEWS"])

            with t1:
                st.markdown("### 📊 Financial Intelligence Terminal")
                
                # 1. Layout Superiore: Switch Periodo e Valuta
                col_head1, col_head2 = st.columns([1, 1])
                with col_head1:
                    period_choice = st.toggle("📅 Visualizza Trimestrali (Latest Q)", value=False)
                    prefix = "q_" if period_choice else ""
                
                with col_head2:
                    currency_symbol = data['info'].get('currency', 'USD')
                    st.write(f"**Valuta:** {currency_symbol}")

                # 2. Menu a Tendina (Selectbox) per i Rendiconti
                statement_type = st.selectbox(
                    "Seleziona Rendiconto Finanziario:",
                    ["Conto Economico (Income Statement)", "Stato Patrimoniale (Balance Sheet)", "Rendiconto Finanziario (Cash Flow)"]
                )

                # Mappatura dati
                mapping = {
                    "Conto Economico (Income Statement)": f"{prefix}fin",
                    "Stato Patrimoniale (Balance Sheet)": f"{prefix}bs",
                    "Rendiconto Finanziario (Cash Flow)": f"{prefix}cf"
                }
                
                raw_data = data.get(mapping[statement_type])

                # 3. Formattazione Professionale (B/M)
                def format_finance(n):
                    try:
                        if pd.isna(n) or n == 0: return "-"
                        abs_n = abs(float(n))
                        sign = "-" if n < 0 else ""
                        if abs_n >= 1e9: return f"{sign}{currency_symbol} {abs_n/1e9:.2f}B"
                        if abs_n >= 1e6: return f"{sign}{currency_symbol} {abs_n/1e6:.1f}M"
                        return f"{sign}{currency_symbol} {abs_n:,.0f}"
                    except:
                        return str(n)

                if raw_data is not None and not raw_data.empty:
                    df_to_show = raw_data.copy()
                    # Applichiamo la formattazione a tutte le colonne (che sono date)
                    if hasattr(df_to_show, 'map'):
                        formatted_df = df_to_show.map(format_finance)
                    else:
                        formatted_df = df_to_show.applymap(format_finance)
                    
                    st.dataframe(
                        formatted_df,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.warning("Dati non disponibili per questa selezione.")
                    
                # --- START INSTITUTIONAL TREND ANALYSIS ---
                st.markdown("---")
                st.markdown("### 📈 Institutional Trend, Forward & Capital Analysis")
                
                fin_data = data.get(f"{prefix}fin")
                bs_data = data.get(f"{prefix}bs")
                cf_data = data.get(f"{prefix}cf")
                info_data = data.get("info", {})

                if all(d is not None and not d.empty for d in [fin_data, bs_data, cf_data]):
                    try:
                        def get_row(df, keywords):
                            for k in df.index:
                                if any(word.lower() in str(k).lower() for word in keywords):
                                    return df.loc[k]
                            return pd.Series([0]*len(df.columns), index=df.columns)

                        # --- 1. ESTRAZIONE DATI STORICI (8 Metriche) ---
                        rev = get_row(fin_data, ['Total Revenue', 'Operating Revenue', 'Revenue'])
                        net_inc = get_row(fin_data, ['Net Income'])
                        ebit = get_row(fin_data, ['EBIT', 'Operating Income'])
                        ocf = get_row(cf_data, ['Operating Cash Flow'])
                        capex = get_row(cf_data, ['Capital Expenditure'])
                        int_expense = abs(get_row(fin_data, ['Interest Expense']))
                        fcf_series = ocf - abs(capex)
                        
                        rev_growth = ((rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100) if len(rev)>1 and rev.iloc[1]!=0 else 0
                        ni_growth = ((net_inc.iloc[0] - net_inc.iloc[1]) / abs(net_inc.iloc[1]) * 100) if len(net_inc)>1 and net_inc.iloc[1]!=0 else 0
                        fcf_margin = (fcf_series.iloc[0] / rev.iloc[0] * 100) if rev.iloc[0]!=0 else 0
                        capex_ocf = (abs(capex.iloc[0]) / ocf.iloc[0] * 100) if ocf.iloc[0]!=0 else 0
                        
                        pct_rev = rev.pct_change(periods=-1)
                        pct_ebit = ebit.pct_change(periods=-1)
                        op_leverage = (pct_ebit.iloc[0] / pct_rev.iloc[0]) if len(pct_rev)>0 and pct_rev.iloc[0]!=0 else 0
                        int_coverage = (ebit.iloc[0] / int_expense.iloc[0]) if int_expense.iloc[0]>0 else 999
                        earn_quality = (ocf.iloc[0] / net_inc.iloc[0]) if net_inc.iloc[0]!=0 else 0
                        fcf_growth = ((fcf_series.iloc[0] - fcf_series.iloc[1]) / abs(fcf_series.iloc[1]) * 100) if len(fcf_series)>1 and fcf_series.iloc[1]!=0 else 0

                        # --- 2. ESTRAZIONE NUOVI DATI FORWARD & MACRO (3 Metriche) ---
                        # A. Forward Consensus (Target Price Upside)
                        curr_price = info_data.get('currentPrice', info_data.get('previousClose', 1))
                        target_price = info_data.get('targetMeanPrice', curr_price)
                        upside_pct = ((target_price - curr_price) / curr_price * 100) if curr_price else 0
                        # Sparkline fittizia per Consensus: Prezzo attuale -> Target
                        cons_spark_data = pd.Series([curr_price * 0.9, curr_price, target_price])

                        # B. Dynamic WACC vs ROIC Spread
                        # Calcolo proxy WACC dinamico
                        beta = info_data.get('beta', 1.0)
                        rf_rate = DYNAMIC_R # Base assumption 4.5% 10Y yield -> Using dynamic IRX
                        cost_of_equity = rf_rate + beta * (0.10 - rf_rate)
                        tot_debt = get_row(bs_data, ['Total Debt']).iloc[0]
                        tot_eq = get_row(bs_data, ['Total Equity', 'Stockholders Equity']).iloc[0]
                        cost_of_debt = (int_expense.iloc[0] / tot_debt) if tot_debt > 0 else 0
                        tax_rate = 0.21
                        tot_cap = tot_debt + tot_eq
                        w_e = tot_eq / tot_cap if tot_cap > 0 else 1
                        w_d = tot_debt / tot_cap if tot_cap > 0 else 0
                        dyn_wacc = (w_e * cost_of_equity) + (w_d * cost_of_debt * (1 - tax_rate))
                        # Calcolo ROIC e Spread
                        roic_val = (net_inc.iloc[0] / tot_cap) if tot_cap > 0 else 0
                        wacc_spread = (roic_val - dyn_wacc) * 100
                        # Sparkline storico spread
                        hist_cap = get_row(bs_data, ['Total Debt']) + get_row(bs_data, ['Total Equity'])
                        wacc_spark_data = ((net_inc / hist_cap.replace(0, 1)) - dyn_wacc) * 100

                        # C. Relative Benchmarking (vs S&P 500)
                        stock_52w = info_data.get('52WeekChange', 0)
                        spy_52w = info_data.get('SandP52WeekChange', 0)
                        stock_52w = stock_52w * 100 if stock_52w else 0
                        spy_52w = spy_52w * 100 if spy_52w else 0
                        rel_outperf = stock_52w - spy_52w
                        rel_spark_data = pd.Series([spy_52w, stock_52w])

                        # --- RILEVAMENTO SETTORE FINANZIARIO ---
                        is_financial = info_data.get('sector') == 'Financial Services'
                        
                        if is_financial:
                            # Estrazione Dati per Banche/Holding
                            equity = get_row(bs_data, ['Total Stockholders Equity', 'Common Stock Equity'])
                            net_inc_series = get_row(fin_data, ['Net Income Common Stockholders', 'Net Income'])
                            total_assets = get_row(bs_data, ['Total Assets'])
                            goodwill = get_row(bs_data, ['Goodwill', 'Goodwill And Other Intangible Assets'])
                            intangibles = get_row(bs_data, ['Intangible Assets'])
                            op_exp = get_row(fin_data, ['Total Operating Expenses', 'Operating Expense'])
                            total_rev = get_row(fin_data, ['Total Revenue'])
                            
                            # Calcolo Metriche Finanziarie (8 Pilastri)
                            # 1. ROE
                            roe = (net_inc_series.iloc[0] / equity.iloc[0]) * 100 if not equity.empty and equity.iloc[0] != 0 else np.nan
                            # 2. ROTCE (Tangible Common Equity)
                            tce = equity.iloc[0] - goodwill.iloc[0] - intangibles.iloc[0] if not equity.empty else 0
                            rotce = (net_inc_series.iloc[0] / tce) * 100 if tce > 0 else np.nan
                            # 3. ROA
                            roa = (net_inc_series.iloc[0] / total_assets.iloc[0]) * 100 if not total_assets.empty and total_assets.iloc[0] != 0 else np.nan
                            # 4. P/B Ratio
                            pb_ratio = info_data.get('priceToBook', np.nan)
                            # 5. Efficiency Ratio (Target < 60%)
                            eff_ratio = (op_exp.iloc[0] / total_rev.iloc[0]) * 100 if not total_rev.empty and total_rev.iloc[0] != 0 else np.nan
                            # 6. Net Margin
                            net_margin_fin = (net_inc_series.iloc[0] / total_rev.iloc[0]) * 100 if not total_rev.empty and total_rev.iloc[0] != 0 else np.nan
                            # 7. Price to Tangible Book (P/TBV)
                            shares = info_data.get('sharesOutstanding', 1)
                            ptbv = (curr_price * shares) / tce if tce > 0 else np.nan
                            # 8. Asset Turnover (Efficienza Patrimoniale)
                            asset_turnover = total_rev.iloc[0] / total_assets.iloc[0] if not total_assets.empty and total_assets.iloc[0] != 0 else np.nan
                        
                            # Raccolta Metriche Disponibili per Score
                            fin_metrics = {
                                "ROE": (roe, 12, 8), "ROTCE": (rotce, 15, 10), "ROA": (roa, 1.2, 0.8),
                                "P/B": (pb_ratio, 1.2, 1.8, True), "Eff. Ratio": (eff_ratio, 60, 75, True),
                                "Net Margin": (net_margin_fin, 20, 10), "P/TBV": (ptbv, 1.5, 2.0, True),
                                "Asset Turn": (asset_turnover, 0.10, 0.05)
                            }

                        # --- 3. CALCOLO SCORE INTEGRATO (Max 100 Punti) ---
                        score = 0
                        fwd_score = 0
                        # Nuove Metriche Macro/Forward (Max 36 Punti: 12 punti ciascuna, Pesi Maggiori)
                        fwd_score += 12 if upside_pct > 15 else (6 if upside_pct > 0 else 0)
                        fwd_score += 12 if wacc_spread > 5 else (6 if wacc_spread > 0 else 0)
                        fwd_score += 12 if rel_outperf > 5 else (6 if rel_outperf > -5 else 0)

                        if is_financial:
                            valid_fin_metrics = {k: v for k, v in fin_metrics.items() if not np.isnan(v[0])}
                            fin_score_total = 0
                            for k, v in valid_fin_metrics.items():
                                val, g, y = v[0], v[1], v[2]
                                rev_pol = v[3] if len(v) > 3 else False
                                if not rev_pol:
                                    fin_score_total += 8 if val >= g else (4 if val >= y else 0)
                                else:
                                    fin_score_total += 8 if val <= g else (4 if val <= y else 0)
                            
                            max_possible = len(valid_fin_metrics) * 8
                            hist_score = (fin_score_total / max_possible * 64) if max_possible > 0 else 0
                        else:
                            hist_score = 0
                            hist_score += 8 if rev_growth > 10 else (4 if rev_growth > 0 else 0)
                            hist_score += 8 if ni_growth > 10 else (4 if ni_growth > 0 else 0)
                            hist_score += 8 if fcf_margin > 15 else (4 if fcf_margin > 5 else 0)
                            hist_score += 8 if capex_ocf < 40 else (4 if capex_ocf < 70 else 0)
                            hist_score += 8 if op_leverage > 1.5 else (4 if op_leverage > 1.0 else 0)
                            hist_score += 8 if int_coverage > 5 else (4 if int_coverage > 2 else 0)
                            hist_score += 8 if earn_quality > 1.0 else (4 if earn_quality > 0.7 else 0)
                            hist_score += 8 if fcf_growth > 10 else (4 if fcf_growth > 0 else 0)
                            
                        score = hist_score + fwd_score

                        if score >= 75: 
                            status_label, status_color = "ESPANSIONE / STRONG BUY 🚀", "#00FF00"
                        elif score >= 45: 
                            status_label, status_color = "EQUITY / HOLD ⚖️", "#FFFF00"
                        else: 
                            status_label, status_color = "SCARSITÀ / SELL ⚠️", "#FF0000"

                        # --- RENDERING TACHIMETRO ---
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number", value = score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Total Health Score: {status_label}", 'font': {'size': 20, 'color': status_color}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': status_color}, 'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 2, 'bordercolor': "gray",
                                'steps': [{'range': [0, 45], 'color': 'rgba(255, 0, 0, 0.3)'},
                                          {'range': [45, 75], 'color': 'rgba(255, 255, 0, 0.3)'},
                                          {'range': [75, 100], 'color': 'rgba(0, 255, 0, 0.3)'}],
                                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
                            }
                        ))
                        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

                        # --- FUNZIONI HELPER GRAFICI E SEMAFORI ---
                        def render_sparkline(data_series, color):
                            plot_data = data_series.fillna(0).iloc[::-1]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=plot_data.values, mode='lines+markers', line=dict(color=color, width=3), marker=dict(size=6), hoverinfo='skip'))
                            fig.update_layout(height=60, margin=dict(l=0, r=0, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                        def get_status(val, g, y, rev_pol=False):
                            if not rev_pol:
                                if val >= g: return "Eccellente 🟢", "normal"
                                if val >= y: return "Stabile 🟡", "off"
                                return "Debole 🔴", "inverse"
                            else:
                                if val <= g: return "Ottimale 🟢", "normal"
                                if val <= y: return "Attenzione 🟡", "off"
                                return "Critico 🔴", "inverse"

                        # --- DASHBOARD VISIVA ---
                        st.write("**Historical Fundamentals (Passato)**")
                        
                        if is_financial:
                            st.info("🏛️ Modulo Analisi Istituti Finanziari Attivo")
                            # Verifica dati mancanti
                            missing_fin = [k for k, v in fin_metrics.items() if np.isnan(v[0])]
                            if missing_fin:
                                st.warning(f"⚠️ Dati parziali. Metriche non disponibili: {', '.join(missing_fin)}.")

                            # Riga 1 Finanziaria
                            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                            with r1c1:
                                msg, col = get_status(roe, 12, 8)
                                st.metric("ROE", f"{roe:.1f}%", delta=msg, delta_color=col, help="Return on Equity: Redditività del capitale proprio.")
                                render_sparkline(net_inc_series / equity.replace(0,1), "#00FFCC")
                            with r1c2:
                                msg, col = get_status(rotce, 15, 10)
                                st.metric("ROTCE", f"{rotce:.1f}%", delta=msg, delta_color=col, help="Return on Tangible Common Equity: Redditività sul capitale tangibile.")
                                render_sparkline(net_inc_series / (equity - goodwill - intangibles).replace(0,1), "#00CCFF")
                            with r1c3:
                                msg, col = get_status(pb_ratio, 1.2, 1.8, True)
                                st.metric("P/B Ratio", f"{pb_ratio:.2f}x", delta=msg, delta_color=col, help="Rapporto Prezzo/Valore di Libro.")
                                # Trend P/B approssimato (Market Cap storica / Equity storica)
                                render_sparkline((curr_price / info_data.get('previousClose', curr_price)) * (equity.iloc[0] / equity.replace(0,1)), "#FFCC00")
                            with r1c4:
                                msg, col = get_status(eff_ratio, 60, 75, True)
                                st.metric("Eff. Ratio", f"{eff_ratio:.1f}%", delta=msg, delta_color=col, help="Efficiency Ratio: Costi operativi / Ricavi totali.")
                                render_sparkline(op_exp / total_rev.replace(0,1), "#FF6600")

                            # Riga 2 Finanziaria
                            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                            with r2c1:
                                msg, col = get_status(roa, 1.2, 0.8)
                                st.metric("ROA", f"{roa:.2f}%", delta=msg, delta_color=col, help="Return on Assets: Utile Netto / Totale Attivo.")
                                render_sparkline(net_inc_series / total_assets.replace(0,1), "#AAFF00")
                            with r2c2:
                                msg, col = get_status(net_margin_fin, 20, 10)
                                st.metric("Net Margin", f"{net_margin_fin:.1f}%", delta=msg, delta_color=col, help="Margine Netto Bancario.")
                                render_sparkline(net_inc_series / total_rev.replace(0,1), "#FF00FF")
                            with r2c3:
                                msg, col = get_status(ptbv, 1.5, 2.0, True)
                                st.metric("P/TBV", f"{ptbv:.2f}x", delta=msg, delta_color=col, help="Price to Tangible Book Value.")
                                render_sparkline((curr_price * info_data.get('sharesOutstanding', 1)) / (equity - goodwill - intangibles).replace(0,1), "#BBBBBB")
                            with r2c4:
                                msg, col = get_status(asset_turnover, 0.10, 0.05)
                                st.metric("Asset Turn.", f"{asset_turnover:.2f}", delta=msg, delta_color=col, help="Asset Turnover: Efficienza nell'uso degli asset.")
                                render_sparkline(total_rev / total_assets.replace(0,1), "#00FF00")
                        
                        else:
                            # Riga 1 Storica
                            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                            with r1c1:
                                msg, col = get_status(rev_growth, 10, 0)
                                st.metric("Revenue Growth", f"{rev_growth:.1f}%", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Misura l'espansione del fatturato. Fondamentale per capire se l'azienda guadagna quote di mercato.\n\nSOGLIE: >10% 🟢 | >0% 🟡 | <0% 🔴")
                                render_sparkline(rev, "#00FFCC")
                            with r1c2:
                                msg, col = get_status(ni_growth, 10, 0)
                                st.metric("Net Inc Growth", f"{ni_growth:.1f}%", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Misura la crescita dell'utile netto. Deve idealmente crescere in linea o più dei ricavi.\n\nSOGLIE: >10% 🟢 | >0% 🟡 | <0% 🔴")
                                render_sparkline(net_inc, "#00CCFF")
                            with r1c3:
                                msg, col = get_status(fcf_margin, 15, 5)
                                st.metric("FCF Margin", f"{fcf_margin:.1f}%", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Quanta cassa libera resta da ogni dollaro di fatturato. Indica la profittabilità reale (Cash Cow).\n\nSOGLIE: >15% 🟢 | >5% 🟡 | <5% 🔴")
                                render_sparkline(fcf_series / rev.replace(0,1), "#FFCC00")
                            with r1c4:
                                msg, col = get_status(capex_ocf, 40, 70, True)
                                st.metric("CapEx / OCF", f"{capex_ocf:.1f}%", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Indica quanto flusso operativo viene assorbito dagli investimenti fissi. Più è basso, più l'azienda è leggera e flessibile.\n\nSOGLIE: <40% 🟢 | <70% 🟡 | >70% 🔴")
                                render_sparkline(abs(capex)/ocf.replace(0,1), "#FF6600")
    
                            # Riga 2 Storica
                            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                            with r2c1:
                                msg, col = get_status(op_leverage, 1.5, 1.0)
                                st.metric("Op. Leverage", f"{op_leverage:.2f}x", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Capacità di generare più utile operativo a parità di crescita ricavi. Indica l'efficienza dei costi fissi e la scalabilità.\n\nSOGLIE: >1.5x 🟢 | >1.0x 🟡 | <1.0x 🔴")
                                render_sparkline(ebit/rev.replace(0,1), "#AAFF00")
                            with r2c2:
                                msg, col = get_status(int_coverage, 5.0, 2.0)
                                st.metric("Int Coverage", f"{int_coverage:.1f}x", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Capacità di rimborsare gli interessi sul debito tramite l'utile operativo. Indica la protezione contro il rischio di default.\n\nSOGLIE: >5x 🟢 | >2x 🟡 | <2x 🔴")
                                render_sparkline(ebit/int_expense.replace(0, 1), "#FF00FF")
                            with r2c3:
                                msg, col = get_status(earn_quality, 1.0, 0.7)
                                st.metric("Earn Quality", f"{earn_quality:.2f}", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: Rapporto tra Flusso di Cassa Operativo e Utile Netto. Se < 1, gli utili potrebbero essere solo 'contabili' e non monetari.\n\nSOGLIE: >1.0 🟢 | >0.7 🟡 | <0.7 🔴")
                                render_sparkline(ocf/net_inc.replace(0,1), "#BBBBBB")
                            with r2c4:
                                msg, col = get_status(fcf_growth, 10, 0)
                                st.metric("FCF Growth", f"{fcf_growth:.1f}%", delta=msg, delta_color=col, 
                                          help="SIGNIFICATO: La crescita del denaro contante libero. È il vero motore dietro il pagamento di dividendi e il riacquisto di azioni proprie.\n\nSOGLIE: >10% 🟢 | >0% 🟡 | <0% 🔴")
                                render_sparkline(fcf_series, "#00FF00")

                        st.markdown("---")
                        st.write("**Institutional Consensus & Macro (Futuro e Contesto)**")
                        
                        # Riga 3 Nuove Metriche (Consensus, WACC, Benchmarking)
                        r3c1, r3c2, r3c3 = st.columns(3)
                        with r3c1:
                            msg, col = get_status(upside_pct, 15, 0)
                            st.metric("Analyst Consensus Upside", f"{upside_pct:+.1f}%", delta=msg, delta_color=col,
                                      help="SIGNIFICATO: Differenza percentuale tra il prezzo attuale e il Target Price Medio degli analisti per i prossimi 12 mesi.\n\nSOGLIE: >15% 🟢 (Ottimistico) | >0% 🟡 (Neutro) | <0% 🔴 (Pessimistico)")
                            render_sparkline(cons_spark_data, "#00BFFF")
                            
                        with r3c2:
                            msg, col = get_status(wacc_spread, 5, 0)
                            st.metric("ROIC vs Dynamic WACC Spread", f"{wacc_spread:+.1f}%", delta=msg, delta_color=col,
                                      help="SIGNIFICATO: Calcola in tempo reale il Costo del Capitale (WACC) basato sul Beta attuale. Se lo spread col ROIC è positivo, l'azienda crea valore netto reale.\n\nSOGLIE: >5% 🟢 (Value Creator) | >0% 🟡 (Stabile) | <0% 🔴 (Value Destroyer)")
                            render_sparkline(wacc_spark_data, "#FF1493")
                            
                        with r3c3:
                            msg, col = get_status(rel_outperf, 5, -5)
                            st.metric("Relative Strength (vs S&P500)", f"{rel_outperf:+.1f}%", delta=msg, delta_color=col,
                                      help="SIGNIFICATO: Confronta la performance aziendale a 1 anno rispetto a quella dell'indice di mercato (S&P 500). Misura l'Alpha istituzionale.\n\nSOGLIE: >+5% 🟢 (Outperformer) | >-5% 🟡 (Market Perform) | <-5% 🔴 (Underperformer)")
                            render_sparkline(rel_spark_data, "#FFD700")

                        # Tabella Storica Puntuale
                        st.write(f"**Storico Analitico ({'Trimestrale' if period_choice else 'Annuale'}):**")
                        trend_df = pd.DataFrame({
                            'Ricavi': rev.apply(format_finance),
                            'EBIT': ebit.apply(format_finance),
                            'Utile Netto': net_inc.apply(format_finance),
                            'Free Cash Flow': fcf_series.apply(format_finance),
                            'CapEx': capex.apply(format_finance)
                        }).head(4)
                        st.dataframe(trend_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Errore nell'analisi istituzionale: {e}")
                # --- END INSTITUTIONAL TREND ANALYSIS ---
                    
            with t2:
                fig = go.Figure(data=[go.Candlestick(x=data["history"].index, open=data["history"]['Open'], high=data["history"]['High'], low=data["history"]['Low'], close=data["history"]['Close'])])
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with t3:
                st.header("🐋 Whales & Insider Intelligence")
                col_ins, col_fund = st.columns(2)

                with col_ins:
                    st.markdown("### 👔 Insider Intelligence (Management)")
                    if inf.get('quoteType', 'EQUITY').upper() in ['ETF', 'MUTUALFUND', 'INDEX']:
                        st.info("L'analisi Insider non è applicabile a Fondi, ETF o Indici.")
                    else:
                        try:
                            ins_df = ticker_data.insider_transactions
                            if ins_df is not None and not ins_df.empty:
                                df_i = ins_df.copy()
                                
                                # Parser Semantico colonna Text
                                def parser_op(row):
                                    t = str(row['Text']).lower() if 'Text' in row else ""
                                    if 'purchase' in t: return "🟢 ACQUISTO NETTO"
                                    if 'sale' in t: return "🔴 VENDITA"
                                    if 'grant' in t or 'award' in t: return "🎁 BONUS / GRANT"
                                    return "⚪ MOVIMENTO TECNICO"
                                
                                df_i['Operazione'] = df_i.apply(parser_op, axis=1)
                                
                                # Formattazione Professionale
                                if 'Start Date' in df_i.columns:
                                    df_i['Start Date'] = pd.to_datetime(df_i['Start Date']).dt.strftime('%d/%m/%Y')
                                if 'Shares' in df_i.columns:
                                    df_i['Shares'] = df_i['Shares'].apply(lambda x: f"{int(x):,}".replace(",", ".") if pd.notnull(x) else "0")
                                if 'Value' in df_i.columns:
                                    df_i['Value'] = df_i['Value'].apply(lambda x: f"$ {int(x):,}".replace(",", ".") if pd.notnull(x) else "N/D")
                                
                                # Selezione e Rinomina per l'utente
                                cols_in = {
                                    'Start Date': 'Data', 'Insider': 'Soggetto', 
                                    'Position': 'Ruolo', 'Operazione': 'Tipo', 
                                    'Shares': 'Azioni', 'Value': 'Controvalore ($)'
                                }
                                df_final_ins = df_i.rename(columns=cols_in)
                                st.dataframe(df_final_ins[[v for v in cols_in.values() if v in df_final_ins.columns]].head(20), use_container_width=True, hide_index=True)
                                st.caption("Dati ufficiali SEC Form 4 filtrati per rilevanza economica.")
                            else:
                                st.info("Nessun movimento Insider rilevante per questo titolo.")
                        except:
                            st.error("Errore nel caricamento dati Insider.")

                with col_fund:
                    st.markdown("### 🏛️ Top 10 Institutional Whales (Fondi)")
                    try:
                        holders = ticker_data.institutional_holders
                        if holders is not None and not holders.empty:
                            df_h = holders.copy()
                            # Formattazione Fondi
                            if 'Date Reported' in df_h.columns:
                                df_h['Date Reported'] = pd.to_datetime(df_h['Date Reported']).dt.strftime('%d/%m/%Y')
                            if 'Shares' in df_h.columns:
                                df_h['Shares'] = df_h['Shares'].apply(lambda x: f"{int(x):,}".replace(",", ".") if pd.notnull(x) else "0")
                            if 'Value' in df_h.columns:
                                df_h['Value'] = df_h['Value'].apply(lambda x: f"$ {int(x):,}".replace(",", ".") if pd.notnull(x) else "N/D")
                            if '% Out' in df_h.columns:
                                df_h['% Out'] = df_h['% Out'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "0%")

                            rename_f = {
                                'Holder': 'Fondo', 'Shares': 'Azioni', 
                                'Date Reported': 'Report', '% Out': '% Port.', 
                                'Value': 'Valore Totale'
                            }
                            st.dataframe(df_h.rename(columns=rename_f).head(10), use_container_width=True, hide_index=True)
                            st.caption("I fondi indicano la stabilità del titolo nel lungo periodo.")
                        else:
                            st.write("Dati istituzionali non disponibili.")
                    except:
                        st.write("Errore nel caricamento dei Fondi.")

            with t4:
                st.subheader("Bloomberg Live Feed & Sentiment Analysis")
                
                if data["news"]:
                    for n in data["news"]:
                        # --- ESTRAZIONE DEEP (Fix per nuovi formati Yahoo) ---
                        # Cerchiamo il titolo ovunque sia nascosto
                        title = n.get('title') or n.get('content', {}).get('title', 'Titolo non disponibile')
                        
                        # Cerchiamo il link (priorità a quello canonico)
                        link = n.get('link') or n.get('url')
                        if not link or link == '#':
                            link = n.get('content', {}).get('canonicalUrl', {}).get('url', '#')
                        
                        # Cerchiamo il publisher (Fonte)
                        publisher = n.get('publisher') or n.get('content', {}).get('provider', {}).get('displayName', 'Fonte Istituzionale')

                        # Estrazione "Pezzettino di descrizione" (Deep Fallback)
                        summary = n.get('summary')
                        if not summary:
                            summary = n.get('content', {}).get('summary')
                        if not summary:
                            summary = n.get('content', {}).get('description')
                        
                        # Se ancora vuoto, usiamo il publisher come descrizione minima
                        if not summary:
                            summary = f"Approfondimento disponibile su {publisher}."

                        # --- LOGICA VALUTAZIONE SEMAFORI ---
                        t_low = (str(title) + " " + str(summary)).lower()
                        pos_words = ['growth', 'buy', 'up', 'surge', 'profit', 'bull', 'beating', 'positive', 'upgrade', 'strong']
                        neg_words = ['drop', 'fall', 'sell', 'risk', 'bear', 'loss', 'debt', 'layoffs', 'miss', 'downgrade', 'weak']
                        
                        if any(w in t_low for w in pos_words):
                            sentiment_label, border_color = "🟢 POSITIVA (BULLISH)", "#2ecc71"
                        elif any(w in t_low for w in neg_words):
                            sentiment_label, border_color = "🔴 NEGATIVA (BEARISH)", "#e74c3c"
                        else:
                            sentiment_label, border_color = "⚪ NEUTRALE", "#95a5a6"

                        # --- RENDERIZZAZIONE GRAFICA ---
                        if title != 'Titolo non disponibile':
                            st.markdown(f"""
                            <div style="border-left: 5px solid {border_color}; padding: 15px; margin-bottom: 20px; background-color: rgba(255,255,255,0.05); border-radius: 0 10px 10px 0;">
                                <h4 style="margin: 0 0 10px 0;"><a href="{link}" target="_blank" style="text-decoration: none; color: #ecf0f1;">{title}</a></h4>
                                <p style="font-size: 0.95em; color: #bdc3c7; line-height: 1.4;">{summary[:250] if summary else 'Nessuna descrizione disponibile per questa news.'}...</p>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                                    <span style="font-weight: bold; color: {border_color}; font-size: 0.9em;">{sentiment_label}</span>
                                    <span style="font-size: 0.8em; color: #7f8c8d;">Fonte: {publisher}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("In attesa di nuovi aggiornamenti dal mercato...")

elif menu == "🔍 GLOBAL SCANNER (Alpha)":
    st.title("🏛️ Global Alpha Engine (Massive Database)")
    st.markdown("---")

    # --- CONFIGURAZIONE DATABASE ESTESO ---
    # --- MEGA DATABASE 1000+ TICKERS ---
    DB_TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "TXN", "QCOM", "INTC", "MU", "AMAT", "LRCX", "ADI", "PANW", "SNPS", "CDNS", "KLAC", "IBM", "NOW", "ACN", "CSCO", "INTU", "FIS", "FISV", "GPN", "PYPL", "SHOP", "PLTR", "ARM", "SMCI", "MCHP", "ON", "NXPI", "STX", "WDC", "HPE", "DELL", "ANET", "MSI", "TEL", "APH", "CDW", "KEYS", "TER", "ENPH", "SEDG", "FSLR", "ZS", "OKTA", "CRWD", "DDOG", "MDB", "NET", "TEAM", "TSM", "ASML", "SAP", "STM", "IFX.DE", "UBER", "PINS", "SNAP", "TWLO", "DOCU", "DDOG", "SPLK", "OKTA", "AKAM", "NET", "VRSN", "STNE", "NU", "SE", "MELI", "CPNG"]
    
    DB_FINANCE = ["JPM", "V", "MA", "BRK-B", "BAC", "WFC", "MS", "GS", "SCHW", "C", "BLK", "AXP", "BX", "PGR", "CB", "MMC", "MET", "TRV", "USB", "PNC", "TFC", "COF", "BK", "STT", "AMP", "BEN", "IVZ", "AJG", "AON", "RE", "WRB", "L", "GL", "AFL", "PRU", "PFG", "ALL", "HIG", "SBNY", "FITB", "HBAN", "KEY", "RF", "MTB", "HBAN", "SIVB", "UCG.MI", "ISP.MI", "MB.MI", "BAMI.MI", "BPE.MI", "PST.MI", "HSBC", "RY", "TD", "BNP.PA", "DBK.DE"]
    
    DB_HEALTH = ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "DHR", "SYK", "ELV", "ISRG", "BMY", "VRTX", "REGN", "ZTS", "HCA", "GILD", "BSX", "CVS", "CI", "BDX", "MCK", "COR", "HUM", "IQV", "EW", "MDT", "IDXX", "A", "RMD", "BAX", "ZBH", "ALGN", "VTRS", "WBA", "CNC", "MOH", "STE", "TFX", "HOLX", "MRNA", "BNTX", "PFE", "SNY", "NVS", "AZN", "OR.PA", "BAYN.DE"]
    
    DB_CONSUMER = ["WMT", "PG", "KO", "PEP", "COST", "NKE", "EL", "CL", "PM", "MO", "TGT", "HD", "LOW", "SBUX", "MDLZ", "TJX", "DG", "DLTR", "KR", "SYY", "STZ", "GIS", "K", "HSY", "CHD", "CLX", "KMB", "TSN", "ADM", "MAR", "HLT", "YUM", "CMG", "MCD", "DPZ", "DRI", "BKNG", "EXPE", "RCL", "CCL", "NCLH", "LVS", "WYNN", "MGM", "LVMH.PA", "RMS.PA", "MC.PA", "KER.PA", "OR.PA", "RACE.MI", "MONC.MI"]
    
    DB_ENERGY_IND = ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "HES", "HAL", "DVN", "FANG", "APA", "BA", "CAT", "DE", "HON", "GE", "RTX", "LMT", "GD", "NOC", "MMM", "UPS", "FDX", "UNP", "NSC", "CSX", "WM", "RSG", "URI", "PWR", "ETN", "EMR", "ITW", "PH", "AME", "DOV", "XYL", "TTE.PA", "ENI.MI", "BP", "SHEL", "SIE.DE", "AIR.PA", "STLAM.MI", "PRY.MI", "LDO.MI"]
    
    DB_OTHERS = ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "VICI", "O", "SBAC", "WELL", "AVB", "EQR", "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG", "WEC", "ES", "ETR", "FE", "LIN", "APD", "SHW", "CTVA", "NEM", "FCX", "VMC", "MLM", "DOW", "DD", "ALB", "T", "VZ", "TMUS", "CMCSA", "DIS", "CHTR", "NFLX", "WBD", "PARA", "ENEL.MI", "TRN.MI", "SRG.MI"]

    DB_USA = DB_TECH + DB_FINANCE + DB_HEALTH + DB_CONSUMER + DB_ENERGY_IND + DB_OTHERS
    DB_EU = ["MC.PA", "ASML", "OR.PA", "SAP", "RMS.PA", "NESN.SW", "TTE.PA", "SIE.DE", "IDNA.MI", "RACE.MI", "ENI.MI", "UCG.MI", "ISP.MI", "ENEL.MI", "STLAM.MI", "PRY.MI", "AZM.MI"]
    
    st.sidebar.subheader("⚙️ Filtri Scanner")
    db_scelta = st.sidebar.multiselect("Mercati da analizzare:", ["USA Standard", "Europa/Italia", "Custom List"], default=["USA Standard", "Europa/Italia"])
    
    custom_input = st.sidebar.text_area("Aggiungi Ticker personalizzati (es: F, GM, PFE):", help="Puoi incollare centinaia di ticker separati da virgola")

    full_db = []
    if "USA Standard" in db_scelta: full_db.extend(DB_USA)
    if "Europa/Italia" in db_scelta: full_db.extend(DB_EU)
    if "Custom List" in db_scelta and custom_input:
        full_db.extend([x.strip().upper() for x in custom_input.split(",") if x.strip()])
    
    st.info(f"🔍 Scanner pronto: **{len(full_db)}** aziende nel database di ricerca.")

    c1, c2, c3 = st.columns(3)
    with c1: min_m = st.number_input("Margine Sicurezza Min (%)", value=15)
    with c2: min_roe = st.slider("ROE Minimo (%)", 0, 50, 10)
    with c3: limit = st.number_input("Max Risultati", value=50)

    if st.button("ESEGUI SCANSIONE GLOBALE 🚀"):
        results = []
        prog = st.progress(0)
        status_msg = st.empty()
        
        for i, ticker in enumerate(full_db):
            status_msg.text(f"Analisi Quantitativa in corso: {ticker}...")
            try:
                t = yf.Ticker(ticker)
                inf = t.info
                
                curr = inf.get('financialCurrency', inf.get('currency', '$'))
                s_curr = "€" if curr == "EUR" else "$"
                price = inf.get('currentPrice', 0)
                
                eps = inf.get('trailingEps', 0)
                bvps = inf.get('bookValue', 0)
                fair_v = np.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0
                margin = ((fair_v - price)/fair_v*100) if fair_v > 0 else -100
                
                roe = inf.get('returnOnEquity', 0) * 100
                
                if margin >= min_m and roe >= min_roe:
                    results.append({
                        "Ticker": ticker,
                        "Prezzo": f"{s_curr}{price:.2f}",
                        "Fair Value": f"{s_curr}{fair_v:.2f}",
                        "Margine": round(margin, 2),
                        "ROE %": round(roe, 2),
                        "Settore": inf.get('sector', 'N/D')
                    })
            except:
                continue
            prog.progress((i + 1) / len(full_db))
            
        status_msg.success(f"Scansione completata. Trovate {len(results)} opportunità.")

        if results:
            df_res = pd.DataFrame(results).sort_values(by="Margine", ascending=False).head(limit)
            
            def color_margin(val):
                c = '#2ecc71' if val > 25 else '#f1c40f'
                return f'color: {c}; font-weight: bold'

            try:
                if hasattr(df_res.style, 'map'):
                    styled_df = df_res.style.map(color_margin, subset=['Margine'])
                else:
                    styled_df = df_res.style.applymap(color_margin, subset=['Margine'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            except:
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                
            # --- AGGIUNTA CHIRURGICA: MATRICE DI CORRELAZIONE ---
            if not df_res.empty and len(df_res) > 1:
                st.markdown("---")
                # Estraiamo i ticker dalla colonna 'Ticker' del risultato
                list_of_tickers = df_res['Ticker'].tolist()
                # Chiamata alla funzione di visualizzazione
                display_correlation_matrix(list_of_tickers)
            
            # --- FOCUS MIGLIORE OPPORTUNITÀ (Logica esistente) ---
            st.markdown("---")
            st.subheader("💡 Focus Migliore Opportunità")
            top_t = df_res.iloc[0]['Ticker']
            top_t = top_t.upper().strip()
            indices_fix = {"SPX": "^GSPC", "SP500": "^GSPC", "NASDAQ": "^IXIC", "NDX": "^IXIC", "DAX": "^GDAXI", "CAC": "^FCHI", "FTSEMIB": "FTSEMIB.MI"}
            if top_t in indices_fix:
                top_t = indices_fix[top_t]
            t_data = yf.Ticker(top_t)
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.write(f"**{top_t}** è attualmente l'azienda più sottovalutata nel database con i criteri scelti.")
                try:
                    if t_data.news:
                        st.write(f"*Ultima News:* {t_data.news[0].get('title', 'N/D')}")
                except:
                    pass
            with col_t2:
                st.info(f"Copia '{top_t}' e torna nella sezione TERMINALE per l'analisi tecnica completa.")
        else:
            st.warning("Nessuna azienda soddisfa i criteri impostati. Prova ad abbassare il Margine di Sicurezza o il ROE minimo.")

elif menu == "🕸️ Macro & Correlazione":
    display_macro_correlation_page()

elif menu == "🌍 WAR ROOM (Dashboard Globale)":
    display_macro_war_room()

elif menu == "📅 SEASONALITY & CALENDAR":
    display_seasonality_and_calendar()
