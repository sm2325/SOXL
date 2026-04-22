# SOXL Buying Power Trading Bot

Automated intraday trading bot for SOXL (3× semiconductor ETF) using
TradingView buying power signals and UVXY volatility hedging, executed
via Alpaca paper/live trading API.

## Strategy

**BuyingPower** — rotates 99% of buying power between SOXL and UVXY
based on TradingView's multi-timeframe indicator consensus score (0–99).

| Score (5-min) | Action |
|---|---|
| ≥ 65 | All-in SOXL |
| 20–65 | Hold current / cash |
| < 25 → exit, < 20 | Rotate to UVXY |

Score formula: `BUY_indicators / (BUY + SELL + NEUTRAL) * 99`
fetched live via `tradingview-ta` at 5-min and 15-min timeframes.

**Backtest result (Jan 2025 – Mar 2026):**
- Strategy: +95.8% return, -43.8% max drawdown
- Buy-and-hold SOXL: +60.7% return, -76.5% max drawdown

## Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env
# edit .env with ALPACA_API_KEY, ALPACA_SECRET_KEY

# Run buying power strategy (default, paper trading)
python bot.py

# Run backtest
python backtest_buying_power.py --entry 65 --exit 25 --uvxy 20
```

## Routine: Hourly Market Check

To check the market and execute trades every hour during trading hours:

```bash
python bot.py
```

The bot checks signals every 15 minutes via APScheduler (configured in
`config.py → BuyingPowerConfig.check_interval_min`). The scheduler
runs Mon–Fri and respects the `market_open`/`market_close` window (ET).

For a one-shot hourly check (e.g. from a Claude Code routine):

```bash
python -c "
import sys; sys.path.insert(0, '.')
from broker import AlpacaBroker
from strategy import BuyingPowerStrategy
from bot import BuyingPowerBot
from config import CONFIG
broker = AlpacaBroker(CONFIG.alpaca)
strategy = BuyingPowerStrategy(
    entry_thresh=CONFIG.buying_power.entry_thresh,
    exit_thresh=CONFIG.buying_power.exit_thresh,
    uvxy_thresh=CONFIG.buying_power.uvxy_thresh,
)
bot = BuyingPowerBot(broker, strategy, CONFIG.buying_power)
bot.tick()
"
```

## Key Files

| File | Purpose |
|---|---|
| `bot.py` | Main entry point — `BuyingPowerBot` and legacy bots |
| `strategy/buying_power.py` | Signal logic: fetches TV scores, returns buy/sell/hold |
| `config.py` | All thresholds and Alpaca credentials config |
| `backtest_buying_power.py` | Historical backtest vs buy-and-hold |
| `broker/alpaca_broker.py` | Alpaca API wrapper |
| `data/tv_feed.py` | TradingView bar data + analysis feed |
| `data/SPYQQQSOXL_OptionUltraShortBuyingPower.csv` | Historical signal reference data |

## Environment Variables

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true          # set false for live trading
TV_USERNAME=               # optional, improves TV data
TV_PASSWORD=               # optional
```

## Signal Thresholds (config.py → BuyingPowerConfig)

```python
entry_thresh  = 65.0   # buy SOXL when score ≥ this
exit_thresh   = 25.0   # exit SOXL when score < this
uvxy_thresh   = 20.0   # rotate to UVXY when score < this
check_interval_min = 15
market_open   = "09:35"  # ET
market_close  = "15:45"  # ET
position_pct  = 0.99     # fraction of buying power to deploy
```
