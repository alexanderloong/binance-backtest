import ccxt
import pandas as pd
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

    print(f"Fetching data from Binance for {symbol}...", flush=True)
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Cache data to CSV
    df.to_csv(filename, index=False)
    print(f"Saved data to {filename}", flush=True)
    return df

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

def run_backtest(df):
    """Main backtest loop for Long and Short positions."""
    initial_balance = CONFIG['initial_balance']
    fee_rate = CONFIG['fee_rate']
    risk_per_trade = CONFIG['risk_per_trade']
    
    df = calculate_supertrend(df)
    
    df['total_fees'] = 0.0
    df['current_balance'] = float(initial_balance)
    
    balance = initial_balance
    position = 0 # 0: No position, 1: Long, -1: Short
    entry_price = 0
    trade_results = [] # To store profit/loss of each closed trade
    
    print(f"\n--- TRADE LOG (20% RISK, LONG & SHORT) ---", flush=True)
    print(f"{'Date':<12} | {'Action':<12} | {'Price':<12} | {'Fee (USDT)':<10} | {'Balance (USDT)':<15}", flush=True)
    print("-" * 90, flush=True)
    
    for i in range(len(df)):
        if i < 16: # Buffer for SuperTrend stability
            df.at[i, 'current_balance'] = initial_balance
            continue 

        current_dir = df.loc[i-1, 'supertrend_dir']
        prev_dir = df.loc[i-2, 'supertrend_dir']
        
        # FLIP TO LONG
        if current_dir == 1 and prev_dir == -1:
            # 1. Close Short if exists
            if position == -1:
                exit_price = df.loc[i, 'open']
                profit_pct = (entry_price - exit_price) / entry_price
                pnl = trade_capital * (1 + profit_pct) - trade_capital
                fee_exit = (trade_capital + pnl) * fee_rate
                balance = balance + pnl - fee_exit
                df.at[i, 'total_fees'] += fee_exit
                trade_results.append(pnl - fee_exit)
                print(f"{df.loc[i, 'timestamp'].date()} | CLOSE SHORT | {exit_price:<12.2f} | {fee_exit:<10.2f} | {balance:<15.2f} (Ret: {profit_pct*100:.2f}%)", flush=True)

            # 2. Open Long
            entry_price = df.loc[i, 'open']
            trade_capital = balance * risk_per_trade
            fee_entry = trade_capital * fee_rate
            balance -= fee_entry
            position = 1
            print(f"{df.loc[i, 'timestamp'].date()} | OPEN LONG   | {entry_price:<12.2f} | {fee_entry:<10.2f} | {balance:<15.2f}", flush=True)
            df.at[i, 'total_fees'] += fee_entry

        # FLIP TO SHORT
        elif current_dir == -1 and prev_dir == 1:
            # 1. Close Long if exists
            if position == 1:
                exit_price = df.loc[i, 'open']
                profit_pct = (exit_price / entry_price) - 1
                pnl = trade_capital * (1 + profit_pct) - trade_capital
                fee_exit = (trade_capital + pnl) * fee_rate
                balance = balance + pnl - fee_exit
                df.at[i, 'total_fees'] += fee_exit
                trade_results.append(pnl - fee_exit)
                print(f"{df.loc[i, 'timestamp'].date()} | CLOSE LONG  | {exit_price:<12.2f} | {fee_exit:<10.2f} | {balance:<15.2f} (Ret: {profit_pct*100:.2f}%)", flush=True)

            # 2. Open Short
            entry_price = df.loc[i, 'open']
            trade_capital = balance * risk_per_trade
            fee_entry = trade_capital * fee_rate
            balance -= fee_entry
            position = -1
            print(f"{df.loc[i, 'timestamp'].date()} | OPEN SHORT  | {entry_price:<12.2f} | {fee_entry:<10.2f} | {balance:<15.2f}", flush=True)
            df.at[i, 'total_fees'] += fee_entry

        # Update Daily Equity (80% Cash + 20% Unrealized PnL)
        if position == 1:
            profit_pct = (df.loc[i, 'close'] / entry_price) - 1
            df.at[i, 'current_balance'] = balance + (trade_capital * profit_pct)
        elif position == -1:
            profit_pct = (entry_price - df.loc[i, 'close']) / entry_price
            df.at[i, 'current_balance'] = balance + (trade_capital * profit_pct)
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
    df, trade_results = run_backtest(df)
    # Start reporting from index 15 to show exactly 1000.00 as Initial Balance
    report(df.iloc[15:], trade_results)
