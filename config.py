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
    "adx_threshold": 25,
    
    # RSI Indicator Settings
    "rsi_period": 14,
    "rsi_long_prev_min": 55,
    "rsi_long_now_min": 45,
    "rsi_long_now_max": 50,
    "rsi_short_prev_max": 45,
    "rsi_short_now_min": 50,
    "rsi_short_now_max": 55,
    
    # Capital Management
    "initial_balance": 1000,
    "risk_per_trade": 0.05,  # 5% risk
    "leverage": 5,         
    "fee_rate": 0.0005,     
    
    # Partial Take Profit Settings
    "tp1_ratio": 4.75,       # Tăng mạnh TP1 để bắt các con sóng lớn (Home Run)
    "tp1_share": 0.8,       # Chốt 80% lợi nhuận khi đạt TP lớn
    
    # Data Storage
    "data_file": "market_data.csv"
}
