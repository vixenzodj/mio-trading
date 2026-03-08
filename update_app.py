import re
import sys

with open('app.py', 'r') as f:
    content = f.read()

# Replace local database sidebar
local_db_old = """st.sidebar.markdown("## 📁 DATABASE LOCALE")
uploaded_file = st.sidebar.file_uploader("Carica file CSV (Database Locale)", type=['csv'])
if uploaded_file is not None:
    file_path = os.path.join(LOCAL_DB_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File {uploaded_file.name} salvato permanentemente nel Database Locale.")
st.sidebar.markdown("---")"""

local_db_new = """st.sidebar.markdown("## 📁 DATABASE LOCALE")
uploaded_file = st.sidebar.file_uploader("Carica file CSV (Database Locale)", type=['csv'])
if uploaded_file is not None:
    file_path = os.path.join(LOCAL_DB_DIR, uploaded_file.name)
    if os.path.exists(file_path):
        st.sidebar.warning(f"Il file {uploaded_file.name} esiste già. Verrà sovrascritto.")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File {uploaded_file.name} salvato permanentemente nel Database Locale.")

if st.sidebar.button("🗑️ Pulisci Database Locale"):
    for f in os.listdir(LOCAL_DB_DIR):
        if f.endswith('.csv') or f.endswith('.CSV'):
            os.remove(os.path.join(LOCAL_DB_DIR, f))
    st.sidebar.success("Database Locale svuotato con successo!")
st.sidebar.markdown("---")"""

content = content.replace(local_db_old, local_db_new)

# Add process_dataframe and fetch_data_smart after fetch_alpaca_history
fetch_alpaca_end = """    if all_bars:
        df = pd.DataFrame(all_bars)
        df['t'] = pd.to_datetime(df['t'])
        df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
        
    return pd.DataFrame()"""

new_functions = """

def process_dataframe(df, start_date, end_date):
    if df.empty:
        return df
        
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
            
    # Force numeric on OHLC
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop rows where Close is NaN
    if 'Close' in df.columns:
        df.dropna(subset=['Close'], inplace=True)
        
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    
    # Filter by date
    df = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]
    
    if not df.empty and len(df) > 1:
        df.sort_values('datetime', inplace=True)
        diffs = df['datetime'].diff()
        max_gap = diffs.max()
        if pd.notnull(max_gap) and max_gap > pd.Timedelta(days=3):
            st.warning(f"⚠️ Attenzione: Rilevato un buco temporale nei dati di {max_gap.days} giorni.")
            
        df.set_index('datetime', drop=False, inplace=True)
        
    return df

def fetch_data_smart(ticker, timeframe, start_date, end_date):
    import io
    import requests
    
    df = pd.DataFrame()
    
    # Determine asset type
    is_forex = "=X" in ticker
    is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
    is_crypto = "-USD" in ticker
    is_stock = not (is_forex or is_index or is_crypto)
    
    days_requested = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # ENGINE 1: Alpaca (Primary for US Stocks/ETFs)
    if is_stock and not is_crypto:
        try:
            tf_alpaca = timeframe
            if timeframe == "1d" or timeframe == "1D": tf_alpaca = "1Day"
            elif timeframe == "1h" or timeframe == "1H": tf_alpaca = "1Hour"
            elif timeframe == "15m" or timeframe == "15Min": tf_alpaca = "15Min"
            elif timeframe == "5m" or timeframe == "5Min": tf_alpaca = "5Min"
            elif timeframe == "1m" or timeframe == "1Min": tf_alpaca = "1Min"
            
            df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            st.error(f"Alpaca fetch failed: {e}")
            
    clean_ticker = ticker.replace('=X', '').replace('^', '')

    # ENGINE 1: Massive Cloud
    if df.empty:
        try:
            # Try multiple filename variations
            possible_keys = [f"{clean_ticker}.csv", f"{clean_ticker}.CSV", f"{clean_ticker.lower()}.csv"]
            obj = None
            for key in possible_keys:
                try:
                    obj = s3_client.get_object(Bucket=MASSIVE_BUCKET, Key=key)
                    break
                except Exception as e:
                    continue
                    
            if obj:
                df_massive = pd.read_csv(io.BytesIO(obj['Body'].read()))
                df = process_dataframe(df_massive, start_date, end_date)
                if not df.empty:
                    st.success("✅ Dati recuperati dai server cloud Massive.")
            else:
                st.info(f"ℹ️ File non trovato nel Bucket Massive per {ticker}.")
        except Exception as e:
            st.error(f"❌ Errore di connessione a Massive S3: {e}")

    # ENGINE 2: Local Database
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
                df = process_dataframe(df_local, start_date, end_date)
                if not df.empty:
                    st.success("📂 Dati recuperati dal Database Locale.")
        except Exception as e:
            st.error(f"❌ Errore lettura Database Locale: {e}")

    # ENGINE 3: yfinance (Fallback)
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
                df = process_dataframe(df_yf, start_date, end_date)
                if not df.empty:
                    st.warning("⚠️ Dati presi da Yahoo Finance (Limiti applicati).")
        except Exception as e:
            st.error(f"❌ Errore Yahoo Finance: {e}")
            
    # ENGINE 4: Fatal Error
    if df.empty:
        st.error("❌ ERRORE CRITICO: Dati non trovati in nessun motore (Alpaca, Massive, Locale, Yahoo). Per favore, carica un file CSV manualmente usando l'apposito uploader per testare questo asset.")
        st.stop()
            
    return df
"""

content = content.replace(fetch_alpaca_end, fetch_alpaca_end + new_functions)

# Now we need to remove the old fetch_data_smart definitions.
# We will use regex to find and remove them.
# The pattern starts with "    def fetch_data_smart(ticker, timeframe, start_date, end_date):"
# and ends right before "    # Data Verification Step" or "    def run_custom_strategy"

import re

pattern1 = re.compile(r'    def fetch_data_smart\(ticker, timeframe, start_date, end_date\):.*?    # Data Verification Step', re.DOTALL)
content = pattern1.sub('    # Data Verification Step', content)

pattern2 = re.compile(r'    def fetch_data_smart\(ticker, timeframe, start_date, end_date\):.*?    def run_custom_strategy', re.DOTALL)
content = pattern2.sub('    def run_custom_strategy', content)

with open('app.py', 'w') as f:
    f.write(content)

print("Done")
