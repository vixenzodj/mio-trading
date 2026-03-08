import re

with open('app.py', 'r') as f:
    content = f.read()

def replace_fetch_data_smart(content):
    pattern = re.compile(r'    def fetch_data_smart\(ticker, timeframe, start_date, end_date\):.*?        return df', re.DOTALL)
    
    new_func = """    def fetch_data_smart(ticker, timeframe, start_date, end_date):
        import io
        import requests
        from botocore.exceptions import ClientError
        
        df = pd.DataFrame()
        
        # Determine asset type
        is_forex = "=X" in ticker
        is_index = ticker.startswith("^") or ticker in ["FTSEMIB.MI"]
        is_crypto = "-USD" in ticker
        is_stock = not (is_forex or is_index or is_crypto)
        
        days_requested = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        clean_ticker = ticker.replace('=X', '').replace('^', '')
        
        # ENGINE 1: Local Database
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

        # ENGINE 2: Massive Cloud (Download & Cache)
        if df.empty:
            try:
                st.info('Scaricamento e decompressione archivi Massive in corso...')
                prefix = 'global_forex' if is_forex else 'us_stocks_sip'
                
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=MASSIVE_BUCKET, Prefix=prefix)
                
                df_list = []
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            if key.endswith('.csv.gz'):
                                try:
                                    s3_obj = s3_client.get_object(Bucket=MASSIVE_BUCKET, Key=key)
                                    df_temp = pd.read_csv(io.BytesIO(s3_obj['Body'].read()), compression='gzip')
                                    if 'ticker' in df_temp.columns:
                                        df_temp = df_temp[df_temp['ticker'] == clean_ticker]
                                        if not df_temp.empty:
                                            df_list.append(df_temp)
                                except Exception as e:
                                    continue
                                        
                if df_list:
                    df_massive = pd.concat(df_list, ignore_index=True)
                    # Rename columns
                    rename_dict = {'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
                    df_massive = df_massive.rename(columns=rename_dict)
                    
                    # Convert window_start
                    if 'window_start' in df_massive.columns:
                        df_massive['datetime'] = pd.to_datetime(df_massive['window_start'], unit='ns')
                        
                    # Process and cache
                    df = process_dataframe(df_massive, start_date, end_date)
                    if not df.empty:
                        # Cache to local database
                        cache_path = os.path.join(LOCAL_DB_DIR, f"{clean_ticker}.csv")
                        df.to_csv(cache_path, index=True)
                        st.success("✅ Dati recuperati dai server cloud Massive e salvati in cache.")
                else:
                    st.info(f"ℹ️ Nessun dato trovato nel Bucket Massive per {ticker}.")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                    st.error(f"❌ Errore di autenticazione Massive S3: {e}")
                else:
                    st.error(f"❌ Errore S3: {e}")
            except Exception as e:
                st.error(f"❌ Errore di connessione a Massive S3: {e}")

        # ENGINE 3: Alpaca (Primary for US Stocks/ETFs)
        if df.empty and is_stock and not is_crypto:
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
                
        # ENGINE 4: yfinance (Fallback)
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
                    st.warning("⚠️ yfinance supporta solo 730 giorni per il timeframe 1h. Date troncate.")
                    
                df_yf = fetch_yahoo_history(ticker, tf_yf, actual_start.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                df = process_dataframe(df_yf, start_date, end_date)
                if not df.empty:
                    st.success("⚠️ Dati recuperati da Yahoo Finance (Fallback).")
            except Exception as e:
                st.error(f"❌ Errore Yahoo Finance: {e}")
                
        return df"""
    
    return pattern.sub(new_func, content)

new_content = replace_fetch_data_smart(content)

with open('app.py', 'w') as f:
    f.write(new_content)
