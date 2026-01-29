# Binance Futures Advanced Backtester 📈

A professional-grade backtesting tool for **Binance USDT-M Futures**, optimized for Trend Following strategies combined with Pullback entries.

## 🚀 Key Features

- **Multi-Indicator Analysis**: Combines SuperTrend (Heikin Ashi), ADX, VWAP, and RSI to effectively filter market noise.
- **Partial Take Profit (50/50 Split)**: Automatically closes 50% of the position when the target is reached and trails the remaining 50%.
- **Smart Risk Management**: 
  - Automatically moves Stop Loss to **Break-even** after the first target (TP1) is hit.
  - Calculated Stop Loss based on SuperTrend volatility.
  - Configurable leverage and risk per trade.
- **Real-time Data**: Fetches data directly from the Binance Futures API with a local caching mechanism for efficiency.
- **Performance Analytics**: Comprehensive reporting including Net Profit, Max Drawdown, Win Rate (Net of fees), and **Profit Factor**.

## 🧠 Strategy Logic

The system utilizes Heikin Ashi candles to smooth out price action for trend determination but executes entries and exits using real market prices (Standard Candles).

### 1. Entry Conditions
- **Long Position**:
  - SuperTrend (HA) is in an Uptrend.
  - ADX > 20 (Strong market momentum).
  - Close Price > VWAP (Price is above the volume-weighted average).
  - RSI Pullback: Previous RSI > 50 and current RSI has pulled back into the 40-50 zone.
- **Short Position**: Inverse of the above conditions.

### 2. Exit Strategy
- **Take Profit 1 (TP1)**: Closes **50% of the volume** when profit reaches the configured **Risk/Reward ratio** (e.g., 1.5x the initial risk).
- **Break-even (BE)**: Immediately after TP1 is hit, the Stop Loss for the remaining position is moved to the entry price.
- **Full Exit**: Triggered when the **SuperTrend flips** direction.

## 🛠 Installation & Usage

### Install Dependencies
```bash
pip install binance-futures-connector pandas matplotlib numpy
```

### How to Use
1.  Adjust trading parameters in `config.py` (Symbol, Leverage, Risk, etc.).
2.  Run the backtest: `python backtest.py`.
3.  Review results: Check the trade log in the terminal and the equity graph in `equity_curve.png`.

## ⚙️ Configuration (config.py)

| Parameter | Description |
| :--- | :--- |
| `symbol` | Trading pair (e.g., `BTCUSDT`) |
| `leverage` | Leverage (1x - 125x) |
| `risk_per_trade` | % of balance dedicated to each trade (0.25 = 25%) |
| `tp1_ratio` | Risk/Reward ratio for hitting the first profit target |
| `tp1_share` | Fraction of position to close at TP1 (0.5 = 50%) |
| `adx_threshold` | Minimum ADX value to confirm a trending market |

---
*Disclaimer: Backtest results are based on historical data and do not guarantee future performance. Always manage your risk carefully.*
