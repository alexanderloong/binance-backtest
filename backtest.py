from binance.um_futures import UMFutures
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from config import CONFIG

def fetch_data(symbol=CONFIG['symbol'], timeframe=CONFIG['timeframe'], limit=CONFIG['limit']):
    """Fetch OHLCV data from Binance or local cache."""
    filename = CONFIG['data_file']
    
    # Check if local data exists and is sufficient
    if os.path.exists(filename):
        print(f"Loading data from local file: {filename}...", flush=True)
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if len(df) >= limit:
            return df

    try:
        print(f"Fetching data from Binance for {symbol}...", flush=True)
        # Initialize Binance UM Futures client
        client = UMFutures()
        
        # klines returns: [ [Open time, Open, High, Low, Close, Volume, Close time, ...] ]
        klines = client.klines(symbol=symbol, interval=timeframe, limit=limit)
        
        if not klines:
            print(f"Error: No data returned from Binance for {symbol}. Please check the symbol and timeframe.", flush=True)
            return None
            
        # Extract required columns
        data = []
        for k in klines:
            data.append([
                k[0], # timestamp
                float(k[1]), # open
                float(k[2]), # high
                float(k[3]), # low
                float(k[4]), # close
                float(k[5])  # volume
            ])
            
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Cache data to CSV
        df.to_csv(filename, index=False)
        print(f"Saved data to {filename}", flush=True)
        return df
    except Exception as e:
        print(f"Error fetching data from Binance Futures: {e}", flush=True)
        return None

def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candles."""
    ha_df = df.copy()
    
    # HA Close = (Open + High + Low + Close) / 4
    ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HA Open = (Prev HA Open + Prev HA Close) / 2
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_df['close'].iloc[i-1]) / 2
    ha_df['open'] = ha_open
    
    # HA High = Max(High, HA Open, HA Close)
    ha_df['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha_df['high'] = np.maximum(ha_df['high'], ha_df['open'])
    ha_df['high'] = np.maximum(ha_df['high'], ha_df['close'])
    
    # HA Low = Min(Low, HA Open, HA Close)
    ha_df['low'] = df[['low', 'open', 'close']].min(axis=1)
    ha_df['low'] = np.minimum(ha_df['low'], ha_df['open'])
    ha_df['low'] = np.minimum(ha_df['low'], ha_df['close'])
    
    return ha_df

def calculate_supertrend(df, period=CONFIG['st_period'], multiplier=CONFIG['st_multiplier']):
    """Calculate SuperTrend based on Heikin Ashi candles."""
    ha_df = calculate_heikin_ashi(df)
    
    # ATR calculation
    true_range = pd.DataFrame()
    true_range['h-l'] = ha_df['high'] - ha_df['low']
    true_range['h-pc'] = abs(ha_df['high'] - ha_df['close'].shift(1))
    true_range['l-pc'] = abs(ha_df['low'] - ha_df['close'].shift(1))
    tr = true_range.max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Bands calculation
    hl2 = (ha_df['high'] + ha_df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    direction = np.ones(len(df))
    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    
    for i in range(period, len(df)):
        # Calculate Final Bands to filter volatility
        if upper_band.iloc[i] < final_upper_band.iloc[i-1] or ha_df['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
            final_upper_band.iloc[i] = upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
            
        if lower_band.iloc[i] > final_lower_band.iloc[i-1] or ha_df['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
            final_lower_band.iloc[i] = lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Determine Trend Direction
        if direction[i-1] == 1:
            if ha_df['close'].iloc[i] < final_lower_band.iloc[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            if ha_df['close'].iloc[i] > final_upper_band.iloc[i]:
                direction[i] = 1
            else:
                direction[i] = -1

    df['supertrend_dir'] = direction
    return df

def calculate_adx(df, period=CONFIG['adx_period']):
    """Calculate ADX (Average Directional Index) using normal candles."""
    df = df.copy()
    
    # Calculate True Range
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    # Calculate +DM and -DM
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff(-1).shift(1) # diff(1) on low is low[i] - low[i-1]
    # Standard DM logic:
    df['plus_dm'] = np.where((df['high'].diff() > df['low'].shift(1) - df['low']) & (df['high'].diff() > 0), df['high'].diff(), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low'] > df['high'].diff()) & (df['low'].shift(1) - df['low'] > 0), df['low'].shift(1) - df['low'], 0)
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with alpha=1/period)
    def wilders_smoothing(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()
        
    df['tr_s'] = wilders_smoothing(df['tr'], period)
    df['plus_dm_s'] = wilders_smoothing(df['plus_dm'], period)
    df['minus_dm_s'] = wilders_smoothing(df['minus_dm'], period)
    
    # Calculate +DI and -DI
    df['plus_di'] = 100 * (df['plus_dm_s'] / df['tr_s'])
    df['minus_di'] = 100 * (df['minus_dm_s'] / df['tr_s'])
    
    # Calculate DX and ADX
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
    df['adx'] = wilders_smoothing(df['dx'], period)
    
    return df['adx']

def calculate_rsi(df, period=CONFIG['rsi_period']):
    """Calculate RSI (Relative Strength Index)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Wilder's Smoothing for RSI
    avg_gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(df):
    """Calculate VWAP (Volume Weighted Average Price) reset daily."""
    df = df.copy()
    # Reset VWAP every day
    df['date'] = df['timestamp'].dt.date
    
    # Cumulative (Price * Volume) and Cumulative Volume per day
    df['pv'] = df['close'] * df['volume']
    
    vwap = df.groupby('date').apply(
        lambda x: x['pv'].cumsum() / x['volume'].cumsum()
    )
    
    # Flatten the multi-index result if necessary
    if isinstance(vwap, pd.Series):
        return vwap.values
    else:
        return vwap.reset_index(level=0, drop=True).values

def run_backtest(df):
    """Main backtest loop with SuperTrend, ADX, VWAP and RSI Pullback."""
    initial_balance = CONFIG['initial_balance']
    fee_rate = CONFIG['fee_rate']
    risk_per_trade = CONFIG['risk_per_trade']
    leverage = CONFIG['leverage']
    adx_threshold = CONFIG['adx_threshold']
    
    # RSI Pullback Params
    rl_prev_min = CONFIG['rsi_long_prev_min']
    rl_now_min = CONFIG['rsi_long_now_min']
    rl_now_max = CONFIG['rsi_long_now_max']
    
    rs_prev_max = CONFIG['rsi_short_prev_max']
    rs_now_min = CONFIG['rsi_short_now_min']
    rs_now_max = CONFIG['rsi_short_now_max']
    
    df = calculate_supertrend(df)
    df['adx'] = calculate_adx(df)
    df['rsi'] = calculate_rsi(df)
    df['vwap'] = calculate_vwap(df)
    
    df['total_fees'] = 0.0
    df['current_balance'] = float(initial_balance)
    
    balance = initial_balance
    position = 0 # 0: No position, 1: Long, -1: Short
    entry_price = 0
    trade_results = []
    
    print(f"\n--- TRADE LOG (Risk {risk_per_trade*100}%, Lev {leverage}x, ADX > {adx_threshold}) ---", flush=True)
    print(f"{'Date':<12} | {'Action':<12} | {'Price':<12} | {'Fee (USDT)':<10} | {'Balance (USDT)':<15}", flush=True)
    print("-" * 90, flush=True)
    
    for i in range(len(df)):
        if i < 31: # Need i-2 so min index is 2, indicators need stabilization
            df.at[i, 'current_balance'] = initial_balance
            continue 

        current_dir = df.loc[i-1, 'supertrend_dir']
        current_adx = df.loc[i-1, 'adx']
        rsi_now = df.loc[i-1, 'rsi']
        rsi_prev = df.loc[i-2, 'rsi']
        current_close = df.loc[i-1, 'close']
        current_vwap = df.loc[i-1, 'vwap']
        
        # 1. EXIT LOGIC (Flipped Trend)
        if position == 1 and current_dir == -1:
            exit_price = df.loc[i, 'open']
            profit_pct = (exit_price / entry_price) - 1
            pnl = margin * leverage * profit_pct
            
            # Check for Liquidation
            if pnl <= -margin:
                pnl = -margin
                print(f"{df.loc[i, 'timestamp'].date()} | LIQUIDATED LONG | {exit_price:<12.2f} | 0.00       | {balance:<15.2f}", flush=True)
            
            fee_exit = (abs(margin * leverage) + pnl) * fee_rate
            balance = balance + margin + pnl - fee_exit
            df.at[i, 'total_fees'] += fee_exit
            trade_results.append(pnl - fee_exit)
            print(f"{df.loc[i, 'timestamp'].date()} | CLOSE LONG  | {exit_price:<12.2f} | {fee_exit:<10.2f} | {balance:<15.2f} (Ret: {profit_pct*leverage*100:.2f}%)", flush=True)
            position = 0

        elif position == -1 and current_dir == 1:
            exit_price = df.loc[i, 'open']
            profit_pct = (entry_price - exit_price) / entry_price
            pnl = margin * leverage * profit_pct
            
            # Check for Liquidation
            if pnl <= -margin:
                pnl = -margin
                print(f"{df.loc[i, 'timestamp'].date()} | LIQUIDATED SHORT| {exit_price:<12.2f} | 0.00       | {balance:<15.2f}", flush=True)

            fee_exit = (abs(margin * leverage) + pnl) * fee_rate
            balance = balance + margin + pnl - fee_exit
            df.at[i, 'total_fees'] += fee_exit
            trade_results.append(pnl - fee_exit)
            print(f"{df.loc[i, 'timestamp'].date()} | CLOSE SHORT | {exit_price:<12.2f} | {fee_exit:<10.2f} | {balance:<15.2f} (Ret: {profit_pct*leverage*100:.2f}%)", flush=True)
            position = 0

        # 2. ENTRY LOGIC (Current Condition)
        if position == 0:
            # Long Entry Logic
            is_uptrend = current_dir == 1
            has_momentum = current_adx > adx_threshold
            above_vwap = current_close > current_vwap
            is_pullback_l = (rsi_prev > rl_prev_min) and (rl_now_min <= rsi_now <= rl_now_max)
            
            if is_uptrend and has_momentum and above_vwap and is_pullback_l:
                entry_price = df.loc[i, 'open']
                margin = balance * risk_per_trade
                fee_entry = (margin * leverage) * fee_rate
                balance -= (margin + fee_entry)
                position = 1
                df.at[i, 'total_fees'] += fee_entry
                print(f"{df.loc[i, 'timestamp'].date()} | OPEN LONG   | {entry_price:<12.2f} | {fee_entry:<10.2f} | {balance:<15.2f} (Lev: {leverage}x, P>VWAP)")

            # Short Entry Logic
            is_downtrend = current_dir == -1
            below_vwap = current_close < current_vwap
            is_pullback_s = (rsi_prev < rs_prev_max) and (rs_now_min <= rsi_now <= rs_now_max)
            
            if is_downtrend and has_momentum and below_vwap and is_pullback_s:
                entry_price = df.loc[i, 'open']
                margin = balance * risk_per_trade
                fee_entry = (margin * leverage) * fee_rate
                balance -= (margin + fee_entry)
                position = -1
                df.at[i, 'total_fees'] += fee_entry
                print(f"{df.loc[i, 'timestamp'].date()} | OPEN SHORT  | {entry_price:<12.2f} | {fee_entry:<10.2f} | {balance:<15.2f} (Lev: {leverage}x, P<VWAP)")

        # Update Daily Equity (Balance + Margin + Unrealized PnL)
        if position == 1:
            profit_pct = (df.loc[i, 'close'] / entry_price) - 1
            pnl_unrealized = margin * leverage * profit_pct
            df.at[i, 'current_balance'] = balance + margin + pnl_unrealized
        elif position == -1:
            profit_pct = (entry_price - df.loc[i, 'close']) / entry_price
            pnl_unrealized = margin * leverage * profit_pct
            df.at[i, 'current_balance'] = balance + margin + pnl_unrealized
        else:
            df.at[i, 'current_balance'] = balance
    
    return df, trade_results

def calculate_max_drawdown(balance_series):
    """Calculate the Maximum Drawdown percentage."""
    peak = balance_series.expanding(min_periods=1).max()
    drawdown = (balance_series - peak) / peak
    return drawdown.min() * 100

def report(df, trade_results):
    """Print performance summary and plot equity curve."""
    final_balance = df['current_balance'].iloc[-1]
    total_fees_paid = df['total_fees'].sum()
    total_return = (final_balance - df['current_balance'].iloc[0]) / df['current_balance'].iloc[0] * 100
    max_dd = calculate_max_drawdown(df['current_balance'])
    
    total_trades = len(trade_results)
    wins = [r for r in trade_results if r > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n--- PERFORMANCE SUMMARY ({CONFIG['symbol']}) ---", flush=True)
    print(f"Period: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}", flush=True)
    print(f"Initial Balance: {df['current_balance'].iloc[0]:.2f} USDT", flush=True)
    print(f"Final Balance: {final_balance:.2f} USDT", flush=True)
    print(f"Total Fees: {total_fees_paid:.2f} USDT", flush=True)
    print(f"Total Return: {total_return:.2f}%", flush=True)
    print(f"Max Drawdown: {max_dd:.2f}%", flush=True)
    print(f"Total Trades: {total_trades}", flush=True)
    print(f"Win Rate: {win_rate:.2f}%", flush=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['current_balance'], label='Equity Curve')
    plt.title(f"Backtest {CONFIG['symbol']} - SuperTrend (Risk {CONFIG['risk_per_trade']*100}%)")
    plt.xlabel('Date')
    plt.ylabel('USDT')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    print("\nEquity curve chart updated in 'equity_curve.png'", flush=True)

if __name__ == "__main__":
    df = fetch_data()
    if df is not None and not df.empty:
        df, trade_results = run_backtest(df)
        # Start reporting from index 15 to show exactly 1000.00 as Initial Balance
        if len(df) > 15:
            report(df.iloc[15:], trade_results)
        else:
            print("Error: Not enough data points to run backtest (minimum 16 required).", flush=True)
    else:
        print("Error: Failed to fetch or load data.", flush=True)
