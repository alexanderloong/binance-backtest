# Backtest Configuration File
CONFIG = {
    "symbol": "BTC/USDT",
    "timeframe": "1d",
    "limit": 405,           # Number of candles to fetch (365 days + buffer)
    "backtest_days": 365,   # Number of days to report in results
    
    # SuperTrend Indicator Settings
    "st_period": 15,
    "st_multiplier": 1.5,
    
    # Capital Management
    "initial_balance": 1000,
    "risk_per_trade": 0.2,  # 20% of balance per trade
    "fee_rate": 0.001,      # 0.1% Binance fee
    
    # Data Storage
    "data_file": "market_data.csv"
}
