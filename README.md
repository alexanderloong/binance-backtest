# Binance SuperTrend Backtest

A professional backtesting tool for Binance Spot trading using the SuperTrend indicator combined with Heikin Ashi candles.

## 🚀 Features
- **Heikin Ashi Analysis**: Calculates signals using Heikin Ashi candles to filter noise, while executing trades using real market prices (Standard Candles).
- **Long & Short Support**: Supports both long and short positions.
- **Risk Management**: Configurable risk per trade (default 20% of balance).
- **Data Caching**: Automatically saves candle data to `market_data.csv` for faster subsequent runs.
- **Performance Metrics**: Reports net profit, trading fees, and **Max Drawdown**.
- **Equity Curve**: Automatically generates a visual equity growth chart.

## 🛠 Installation

Requires Python 3.8+.

```bash
pip install binance-futures-connector pandas matplotlib numpy
```

## 📈 Usage

1. **Configuration**: Adjust trading parameters in `config.py`.
2. **Run Backtest**:
   ```bash
   python backtest.py
   ```
3. **Results**:
   - Review the trade log displayed in the terminal.
   - Check the equity growth chart in `equity_curve.png`.

## ⚙️ Configuration (config.py)

| Parameter | Description |
| :--- | :--- |
| `symbol` | Trading pair (e.g., `BTC/USDT`) |
| `st_period` | SuperTrend period |
| `st_multiplier` | SuperTrend multiplier |
| `risk_per_trade` | % of balance used per trade (0.2 = 20%) |
| `fee_rate` | Trading fee (0.001 = 0.1%) |

## 📝 Note
Data is fetched directly from the Binance Spot API. `market_data.csv` is created after the first run. To refresh data, delete this file before running again.

---
*Disclaimer: Backtesting results are based on historical data and do not guarantee future profits.*
