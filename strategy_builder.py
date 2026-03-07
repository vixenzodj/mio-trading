
elif menu == "🛠️ STRATEGY BUILDER":
    st.title("🛠️ Strategy Builder (No-Code)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Impostazioni Base")
    
    # Time Filters UI
    start_time = st.sidebar.time_input("Start Trading Time", value=datetime.strptime("09:30", "%H:%M").time())
    end_time = st.sidebar.time_input("End Trading Time", value=datetime.strptime("16:00", "%H:%M").time())
    eod_close = st.sidebar.checkbox("Close all at EOD", value=True)
    
    # ORB UI
    orb_enabled = st.sidebar.checkbox("Enable Opening Range Breakout (ORB)", value=True)
    orb_duration = 15
    if orb_enabled:
        orb_duration = st.sidebar.selectbox("ORB Candle Duration (min)", [5, 15, 30])
        
    # Ticker and Date Range
    ticker = st.text_input("Ticker", value="SPY").upper()
    
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with c2: end_date = st.date_input("End Date", value=datetime.now())
    
    timeframe = st.selectbox("Timeframe", ["1Min", "5Min", "15Min", "1H", "1D"], index=1)
    initial_capital = st.number_input("Initial Capital ($)", value=10000)
    
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

    def calculate_advanced_metrics(trades_list):
        fallback = {'expectancy': 0, 'profit_factor': 0, 'max_drawdown': 0, 'win_rate': 0, 'total_profit_abs': 0, 'max_dd_abs': 0}
        if not trades_list:
            return fallback
            
        df = pd.DataFrame(trades_list)
        df.columns = [str(c).lower() for c in df.columns]
        
        if 'pnl' not in df.columns:
            return fallback
            
        exits = df[df['pnl'].notna()]
        if exits.empty:
            return fallback
            
        wins = exits[exits['pnl'] > 0]['pnl']
        losses = exits[exits['pnl'] < 0]['pnl']
        
        win_rate = len(wins) / len(exits)
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
        
        total_profit_abs = exits['pnl'].sum()
        
        bal_col = 'balance' if 'balance' in df.columns else None
        max_dd = 0
        max_dd_abs = 0
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
            
        sim_length = min(50, n_trades)
        
        random_indices = np.random.randint(0, n_trades, size=(simulations, sim_length))
        simulated_pnls = pnls[random_indices]
        
        equity_curves = np.cumsum(simulated_pnls, axis=1) + initial_capital
        
        starting_capital = np.full((simulations, 1), initial_capital)
        equity_curves = np.hstack((starting_capital, equity_curves))
        
        median_curve = np.median(equity_curves, axis=0)
        
        final_balances = equity_curves[:, -1]
        prob_profit = (np.sum(final_balances > initial_capital) / simulations) * 100
        
        ruin_threshold = initial_capital * 0.80
        ruined_simulations = np.any(equity_curves < ruin_threshold, axis=1)
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

    def fetch_data_smart(ticker, timeframe, start_date, end_date):
        df = pd.DataFrame()
        try:
            tf_alpaca = timeframe
            if timeframe == "1D": tf_alpaca = "1Day"
            elif timeframe == "1H": tf_alpaca = "1Hour"
            elif timeframe == "15Min": tf_alpaca = "15Min"
            elif timeframe == "5Min": tf_alpaca = "5Min"
            df = fetch_alpaca_history(ticker, tf_alpaca, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Alpaca fetch failed: {e}")

        if df.empty:
            try:
                tf_yf = "1d"
                if timeframe == "1D": tf_yf = "1d"
                elif timeframe == "1H": tf_yf = "1h"
                elif timeframe == "15Min": tf_yf = "15m"
                elif timeframe == "5Min": tf_yf = "5m"
                df = yf.download(ticker, start=start_date, end=end_date, interval=tf_yf, progress=False)
                if not df.empty:
                    df.reset_index(inplace=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    rename_map = {
                        'Date': 'datetime', 'Datetime': 'datetime',
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    }
                    df.rename(columns=rename_map, inplace=True)
                    if 'datetime' not in df.columns and df.index.name in ['Date', 'Datetime']:
                        df.reset_index(inplace=True)
                        df.rename(columns={df.index.name: 'datetime'}, inplace=True)
                    df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception as e:
                pass
                
        if not df.empty:
            cols = df.select_dtypes(include=['float64']).columns
            if not cols.empty:
                df[cols] = df[cols].astype('float32')
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            df.sort_values('datetime', inplace=True)
            df.ffill().bfill(inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    def run_custom_strategy(df, start_time, end_time, eod_close, orb_enabled, orb_duration):
        trades = []
        if df.empty:
            return trades
            
        df = df.copy()
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        in_position = False
        entry_price = 0
        entry_time = None
        position_type = None
        size = 1
        
        grouped = df.groupby('date')
        
        for date, group in grouped:
            group = group.sort_values('datetime').reset_index(drop=True)
            
            orb_high = None
            orb_low = None
            orb_end_time = None
            
            if orb_enabled:
                if not group.empty:
                    first_candle_time = group.iloc[0]['datetime']
                    orb_end_time = first_candle_time + pd.Timedelta(minutes=orb_duration)
                    
                    orb_data = group[group['datetime'] < orb_end_time]
                    if not orb_data.empty:
                        orb_high = orb_data['High'].max()
                        orb_low = orb_data['Low'].min()
            
            for idx, row in group.iterrows():
                current_time = row['time']
                current_datetime = row['datetime']
                
                is_trading_time = start_time <= current_time <= end_time
                
                if in_position:
                    exit_triggered = False
                    exit_type = ""
                    
                    if eod_close and current_time >= end_time:
                        exit_triggered = True
                        exit_type = "EOD"
                    elif not is_trading_time:
                        exit_triggered = True
                        exit_type = "Out of Time"
                    
                    if exit_triggered:
                        exit_price = row['Close']
                        pnl = (exit_price - entry_price) if position_type == 'LONG' else (entry_price - exit_price)
                        pnl *= size
                        
                        trades.append({
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': current_datetime,
                            'Exit Price': exit_price,
                            'pnl': pnl,
                            'Return %': (pnl / (entry_price * size)) * 100 if size > 0 else 0,
                            'Type': 'long' if position_type == 'LONG' else 'short',
                            'Status': exit_type,
                            'type': f'EXIT {exit_type}',
                            'time': current_datetime,
                            'price': exit_price,
                            'Logica': f"Exit: {exit_type}"
                        })
                        in_position = False
                
                if not in_position and is_trading_time:
                    if orb_enabled and orb_high is not None and orb_low is not None:
                        if current_datetime >= orb_end_time:
                            if row['High'] > orb_high:
                                in_position = True
                                position_type = 'LONG'
                                entry_price = max(row['Open'], orb_high)
                                entry_time = current_datetime
                            elif row['Low'] < orb_low:
                                in_position = True
                                position_type = 'SHORT'
                                entry_price = min(row['Open'], orb_low)
                                entry_time = current_datetime
                    elif not orb_enabled:
                        if current_time >= start_time:
                            in_position = True
                            position_type = 'LONG'
                            entry_price = row['Close']
                            entry_time = current_datetime
                            
        return trades

    if st.button("🚀 Esegui Strategia Custom"):
        with st.spinner("Fetching data and running strategy..."):
            df = fetch_data_smart(ticker, timeframe, start_date, end_date)
            if not df.empty:
                trades = run_custom_strategy(df, start_time, end_time, eod_close, orb_enabled, orb_duration)
                
                if trades:
                    friction_pct = 0.0
                    adjusted_trades, adjusted_equity = apply_friction_post_process(trades, initial_capital, friction_pct)
                    
                    st.subheader("📊 Risultati Strategia")
                    metrics = calculate_advanced_metrics(adjusted_trades)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    col2.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
                    col4.metric("Total Profit", f"${metrics['total_profit_abs']:.2f}")
                    
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
