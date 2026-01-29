# Backtest Configuration File
CONFIG = {
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "limit": 35000,           # Number of candles to fetch (e.g., 1500 candles ~ 15 days on 15m)
    "backtest_days": 365,   # Number of days to report in results
    
    # SuperTrend Indicator Settings
    "st_period": 15,
    "st_multiplier": 1.5,
    
    # ADX Indicator Settings
    "adx_period": 14,
    "adx_threshold": 20,
    
    # RSI Indicator Settings
    "rsi_period": 14,
    "rsi_long_prev_min": 50,
    "rsi_long_now_min": 40,
    "rsi_long_now_max": 50,
    "rsi_short_prev_max": 50,
    "rsi_short_now_min": 50,
    "rsi_short_now_max": 60,
    
    # Capital Management
    "initial_balance": 1000,
    "risk_per_trade": 0.2,  # 20% of balance per trade
    "leverage": 5,         # Leverage (1x to 125x)
    "fee_rate": 0.0005,     # 0.05% Binance Futures Taker fee
    
    # Partial Take Profit Settings
    "tp1_ratio": 1.7,       # Take Profit 1 at 1.5x Risk/Reward
    "tp1_share": 0.3,       # Close 50% of position at TP1
    
    # Data Storage
    "data_file": "market_data.csv"
}
