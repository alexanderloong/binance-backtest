import pandas as pd
import numpy as np
from backtest import fetch_data, run_backtest
from config import CONFIG
import copy

def optimize():
    df_original = fetch_data()
    if df_original is None or df_original.empty:
        print("Failed to fetch data")
        return

    best_balance = -1
    best_params = {}
    
    # Define ranges for optimization
    tp1_ratios = np.arange(1.0, 3.1, 0.2) # 1.0, 1.2, ..., 3.0
    tp1_shares = np.arange(0.1, 0.9, 0.1) # 0.1, 0.2, ..., 0.8

    results = []

    print(f"{'Ratio':<10} | {'Share':<10} | {'Final Balance':<15} | {'Win Rate':<10} | {'Trades'}")
    print("-" * 60)

    for ratio in tp1_ratios:
        for share in tp1_shares:
            # Update CONFIG temporarily
            CONFIG['tp1_ratio'] = round(ratio, 2)
            CONFIG['tp1_share'] = round(share, 2)
            
            # Run backtest
            df_result, trade_results = run_backtest(df_original.copy())
            
            final_balance = df_result['current_balance'].iloc[-1]
            total_trades = len(trade_results)
            wins = [r for r in trade_results if r > 0]
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            
            results.append({
                'ratio': ratio,
                'share': share,
                'balance': final_balance,
                'win_rate': win_rate,
                'trades': total_trades
            })
            
            print(f"{ratio:<10.2f} | {share:<10.2f} | {final_balance:<15.2f} | {win_rate:<10.2f}% | {total_trades}")
            
            if final_balance > best_balance:
                best_balance = final_balance
                best_params = {'ratio': ratio, 'share': share}

    print("\n" + "="*30)
    print("BEST PARAMETERS FOUND:")
    print(f"TP1 Ratio: {best_params['ratio']:.2f}")
    print(f"TP1 Share: {best_params['share']:.2f}")
    print(f"Final Balance: {best_balance:.2f} USDT")
    print("="*30)

if __name__ == "__main__":
    import builtins
    original_print = builtins.print
    
    # Simple trick: override print only during the loop
    def silent_print(*args, **kwargs):
        pass

    def run_opt():
        # Temporarily silence print
        builtins.print = silent_print
        try:
            # We need to print results, so we define a way to do that
            def log(*args, **kwargs):
                original_print(*args, **kwargs)
            
            # Re-implement optimize here or pass log to it
            df_original = fetch_data()
            if df_original is None or df_original.empty:
                log("Failed to fetch data")
                return

            # Pre-calculate indicators once
            from backtest import calculate_supertrend, calculate_adx, calculate_rsi, calculate_vwap
            df_original = calculate_supertrend(df_original)
            df_original['adx'] = calculate_adx(df_original)
            df_original['rsi'] = calculate_rsi(df_original)
            df_original['vwap'] = calculate_vwap(df_original)

            best_balance = -1
            best_params = {}
            
            # Define ranges for optimization
            tp1_ratios = np.arange(1.0, 5.1, 0.4) 
            tp1_shares = np.arange(0.1, 1.0, 0.1) 

            log(f"{'Ratio':<10} | {'Share':<10} | {'Final Balance':<15} | {'Win Rate':<10} | {'Trades'}")
            log("-" * 60)

            for ratio in tp1_ratios:
                for share in tp1_shares:
                    CONFIG['tp1_ratio'] = round(ratio, 2)
                    CONFIG['tp1_share'] = round(share, 2)
                    
                    df_result, trade_results = run_backtest(df_original.copy())
                    
                    final_balance = df_result['current_balance'].iloc[-1]
                    total_trades = len(trade_results)
                    wins = [r for r in trade_results if r > 0]
                    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
                    
                    log(f"{ratio:<10.2f} | {share:<10.2f} | {final_balance:<15.2f} | {win_rate:<10.2f}% | {total_trades}")
                    
                    if final_balance > best_balance:
                        best_balance = final_balance
                        best_params = {'ratio': ratio, 'share': share}

            log("\n" + "="*30)
            log("BEST PARAMETERS FOUND:")
            log(f"TP1 Ratio: {best_params['ratio']:.2f}")
            log(f"TP1 Share: {best_params['share']:.2f}")
            log(f"Final Balance: {best_balance:.2f} USDT")
            log("="*30)
        finally:
            builtins.print = original_print

    run_opt()
