# Backtest Configuration File
CONFIG = {
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "limit": 1500,           # Number of candles to fetch (e.g., 1500 candles ~ 15 days on 15m)
    "backtest_days": 365,   # Number of days to report in results
    
    # SuperTrend Indicator Settings
    "st_period": 15,
    "st_multiplier": 1.5,
    
    # ADX Indicator Settings
    "adx_period": 14,
    "adx_threshold": 20,
    
    # RSI Indicator Settings
    "rsi_period": 14,
    "rsi_long_pullback": 50,  # Buy when RSI < 50 in uptrend
    "rsi_short_pullback": 50, # Sell when RSI > 50 in downtrend
    
    # Capital Management
    "initial_balance": 1000,
     "risk_per_trade": 0.2,  # 20% of balance per trade
    "fee_rate": 0.0005,     # 0.05% Binance Futures Taker fee
    
    # Data Storage
    "data_file": "market_data.csv"
}
