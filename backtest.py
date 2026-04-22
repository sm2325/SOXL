"""
Historical backtest engine — daily bar mode.

Simulates the strategy on daily OHLCV data with next-day-open execution
(signal fires on day N close, trade executes at day N+1 open). This is
the most realistic simulation for an EOD signal approach.

Usage:
    python backtest.py
    python backtest.py --days 90
    python backtest.py --strategy ml --model models/model.pkl
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from strategy.ema_rsi import EmaRsiStrategy
from strategy.momentum import MomentumStrategy
from strategy.ml_strategy import MLStrategy
from risk import should_stop, calc_position_size
from config import CONFIG

SLIPPAGE_PCT = 0.0005   # 0.05% — realistic for liquid ETF at open


def download_bars(symbol: str, days: int = 90, period: str = None) -> pd.DataFrame:
    # Use explicit period string (e.g. "5y") for long windows, else build from days
    yf_period = period if period else (f"{days}d" if days <= 365 else f"{days // 365}y")
    df = yf.download(symbol, period=yf_period, interval="1d", progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def run_backtest(strategy, bars: pd.DataFrame, cfg=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Signal on close of day i → execute at open of day i+1.
    Stop loss checked against intraday low of execution day.
    """
    cfg = cfg or CONFIG.strategy
    starting_equity = 100_000.0
    cash = starting_equity
    position = None   # {qty, entry_price, stop_price}
    equity_curve = []
    trades = []

    for i in range(strategy.min_bars_required, len(bars) - 1):
        today = bars.index[i]
        tomorrow = bars.index[i + 1]

        exec_open = float(bars["open"].iloc[i + 1])
        exec_low  = float(bars["low"].iloc[i + 1])
        close_price = float(bars["close"].iloc[i + 1])

        # mark-to-market on today's close
        mark_price = float(bars["close"].iloc[i])
        equity = cash + (position["qty"] * mark_price if position else 0)
        equity_curve.append({"timestamp": today, "equity": equity})

        # stop loss: triggered if tomorrow's low breaches stop
        if position:
            stop = position.get("stop_price") or (
                position["entry_price"] * (1 - cfg.stop_loss_pct)
            )
            if exec_low <= stop:
                # execute at stop price (or open if gapped below)
                fill_price = min(exec_open, stop) * (1 - SLIPPAGE_PCT)
                cash += position["qty"] * fill_price
                trades.append({
                    "timestamp": tomorrow, "action": "sell",
                    "qty": position["qty"], "price": round(fill_price, 4),
                    "reason": "stop loss",
                })
                position = None
                continue

        # generate signal on today's close
        window = bars.iloc[: i + 1]
        signal = strategy.generate_signal(window)

        if signal.action == "buy" and position is None:
            fill = exec_open * (1 + SLIPPAGE_PCT)
            qty = calc_position_size(equity, fill, cfg.max_position_pct)
            cost = qty * fill
            if cost <= cash and qty >= 1:
                cash -= cost
                stop_price = signal.stop_price or round(fill * (1 - cfg.stop_loss_pct), 4)
                position = {"qty": qty, "entry_price": fill, "stop_price": stop_price}
                trades.append({
                    "timestamp": tomorrow, "action": "buy",
                    "qty": qty, "price": round(fill, 4), "reason": signal.reason,
                })

        elif signal.action == "sell" and position:
            fill = exec_open * (1 - SLIPPAGE_PCT)
            cash += position["qty"] * fill
            trades.append({
                "timestamp": tomorrow, "action": "sell",
                "qty": position["qty"], "price": round(fill, 4), "reason": signal.reason,
            })
            position = None

    # close any open position at last bar
    if position:
        last_price = float(bars["close"].iloc[-1])
        cash += position["qty"] * last_price
        trades.append({
            "timestamp": bars.index[-1], "action": "sell",
            "qty": position["qty"], "price": last_price, "reason": "backtest end",
        })

    final_equity = cash
    equity_curve.append({"timestamp": bars.index[-1], "equity": final_equity})

    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["timestamp", "action", "qty", "price", "reason"]
    )
    return equity_df, trade_df


def compute_metrics(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    bars: pd.DataFrame,
) -> dict:
    equity = equity_df["equity"]
    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min() * 100

    rets = equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(252) if rets.std() > 0 else 0

    # buy-and-hold benchmark
    bh_return = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0] * 100

    buy_trades = trade_df[trade_df["action"] == "buy"]
    sell_trades = trade_df[trade_df["action"] == "sell"]
    n_trades = min(len(buy_trades), len(sell_trades))

    wins = sum(
        s > b for b, s in zip(buy_trades["price"].values, sell_trades["price"].values)
    )
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    return {
        "total_return_pct":    round(total_return, 2),
        "buy_and_hold_pct":    round(bh_return, 2),
        "max_drawdown_pct":    round(max_dd, 2),
        "sharpe_ratio":        round(sharpe, 3),
        "n_trades":            n_trades,
        "win_rate_pct":        round(win_rate, 1),
        "start_equity":        round(equity.iloc[0], 2),
        "end_equity":          round(equity.iloc[-1], 2),
    }


def plot_results(
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    bars: pd.DataFrame,
    symbol: str,
    strategy_name: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.suptitle(f"{strategy_name} Strategy — {symbol} Daily Backtest", fontsize=14)

    # equity curve vs buy-and-hold
    ax = axes[0]
    ax.plot(equity_df.index, equity_df["equity"], color="steelblue", linewidth=1.4, label="Strategy")
    start_eq = equity_df["equity"].iloc[0]
    bh = (bars["close"] / bars["close"].iloc[0]) * start_eq
    ax.plot(bars.index, bh, color="gray", linewidth=1, linestyle="--", alpha=0.7, label="Buy & Hold")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.set_title("Equity Curve vs Buy & Hold")
    ax.legend()
    ax.grid(alpha=0.3)

    # SOXL price with trade markers
    ax2 = axes[1]
    ax2.plot(bars.index, bars["close"], color="black", linewidth=1, alpha=0.7)
    if not trade_df.empty:
        buys = trade_df[trade_df["action"] == "buy"]
        sells = trade_df[trade_df["action"] == "sell"]
        ax2.scatter(buys["timestamp"], buys["price"], marker="^", color="green", s=80, zorder=5, label="Buy")
        ax2.scatter(sells["timestamp"], sells["price"], marker="v", color="red", s=80, zorder=5, label="Sell")
    ax2.set_ylabel("Price ($)")
    ax2.set_title(f"{symbol} Price + Trade Entries/Exits")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # drawdown
    ax3 = axes[2]
    roll_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - roll_max) / roll_max * 100
    ax3.fill_between(equity_df.index, drawdown, 0, color="crimson", alpha=0.4)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_title("Drawdown")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path("results") / f"backtest_{strategy_name.lower().replace(' ', '_')}_{symbol}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Chart saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOXL Daily Backtest")
    parser.add_argument("--strategy", default="ema_rsi", choices=["ema_rsi", "momentum", "ml"])
    parser.add_argument("--model", default=None, help="Path to ML model")
    parser.add_argument("--days", type=int, default=90, help="Days of history (ignored if --period set)")
    parser.add_argument("--period", default=None, help="yfinance period string: 1y 2y 5y max")
    parser.add_argument("--symbol", default="SOXL")
    args = parser.parse_args()

    label = args.period if args.period else f"{args.days}d"
    print(f"Downloading {label} of {args.symbol} daily bars...")
    bars = download_bars(args.symbol, days=args.days, period=args.period)
    print(f"Loaded {len(bars)} bars: {bars.index[0].date()} → {bars.index[-1].date()}")

    if args.strategy == "ml":
        strat = MLStrategy(CONFIG.ml)
        if not strat.load_model(args.model):
            print("ERROR: ML model not found. Run: python -m ml.trainer --trades <file>")
            exit(1)
    elif args.strategy == "momentum":
        strat = MomentumStrategy(CONFIG.strategy)
    else:
        strat = EmaRsiStrategy(CONFIG.strategy)

    print(f"Running {strat.name} backtest on daily bars...")
    equity_df, trade_df = run_backtest(strat, bars, CONFIG.strategy)

    metrics = compute_metrics(equity_df, trade_df, bars)
    print("\n=== Backtest Results ===")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    safe_name = strat.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    out_csv = f"results/trades_{safe_name}_{args.symbol}.csv"
    trade_df.to_csv(out_csv, index=False)
    print(f"Trades saved → {out_csv}")

    plot_results(equity_df, trade_df, bars, args.symbol, safe_name)
