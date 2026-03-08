import re

with open('app.py', 'r') as f:
    content = f.read()

old_str = """        # ENGINE 1: Massive Cloud
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
                st.error(f"❌ Errore di connessione a Massive S3: {e}")"""

new_str = """        # ENGINE 1: Massive Cloud
        if df.empty:
            try:
                # Try multiple filename variations
                possible_keys = [f"{clean_ticker}.csv", f"{clean_ticker}.CSV", f"{clean_ticker.lower()}.csv"]
                obj = None
                for key in possible_keys:
                    try:
                        obj = s3_client.get_object(Bucket=MASSIVE_BUCKET, Key=key)
                        break
                    except ClientError as e:
                        error_code = e.response['Error']['Code']
                        if error_code == 'NoSuchKey':
                            continue
                        elif error_code in ['AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
                            st.error(f"❌ Errore di autenticazione Massive S3: {e}")
                            break
                        else:
                            st.error(f"❌ Errore S3: {e}")
                            break
                    except Exception as e:
                        st.error(f"❌ Errore imprevisto S3: {e}")
                        break
                        
                if obj:
                    df_massive = pd.read_csv(io.BytesIO(obj['Body'].read()))
                    df = process_dataframe(df_massive, start_date, end_date)
                    if not df.empty:
                        st.success("✅ Dati recuperati dai server cloud Massive.")
                else:
                    st.info(f"ℹ️ File non trovato nel Bucket Massive per {ticker}.")
            except Exception as e:
                st.error(f"❌ Errore di connessione a Massive S3: {e}")"""

content = content.replace(old_str, new_str)

with open('app.py', 'w') as f:
    f.write(content)
