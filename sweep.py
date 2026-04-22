"""
Parameter sweep: momentum variants vs best EMA combos.

Usage:
    python sweep.py
    python sweep.py --days 90 --symbol SOXL
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategy.ema_rsi import EmaRsiStrategy
from strategy.momentum import MomentumStrategy
from backtest import run_backtest, compute_metrics, download_bars
from config import CONFIG, StrategyConfig

# EMA combos that scored best in previous sweep
EMA_COMBOS = [(9, 21), (10, 30), (5, 20)]

# Momentum variants: (short_window, medium_window, threshold, use_relative)
MOMENTUM_COMBOS = [
    (10, 40, 0.01, True),
    (20, 60, 0.02, True),
    (20, 60, 0.02, False),   # without SPY filter to see its contribution
    (15, 45, 0.015, True),
    (10, 60, 0.01, True),
]


def build_strategies():
    strategies = []

    for fast, slow in EMA_COMBOS:
        cfg = StrategyConfig(fast_ema=fast, slow_ema=slow, timeframe="1Day")
        strategies.append((f"EMA({fast},{slow})", EmaRsiStrategy(cfg), cfg))

    for short_w, med_w, thresh, rel in MOMENTUM_COMBOS:
        cfg = StrategyConfig(timeframe="1Day")
        rel_tag = "+SPY" if rel else "-SPY"
        label = f"Mom({short_w}/{med_w},{thresh*100:.0f}%{rel_tag})"
        strategies.append((label, MomentumStrategy(cfg, short_w, med_w, thresh, use_relative=rel), cfg))

    return strategies


def run_sweep(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bh_return = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0] * 100
    print(f"\nBuy & Hold benchmark: {bh_return:+.2f}%\n")

    header = f"{'Strategy':35s} {'Return%':>8} {'B&H%':>8} {'MaxDD%':>8} {'Sharpe':>8} {'Trades':>7} {'Win%':>6}"
    print(header)
    print("-" * len(header))

    results = []
    equity_curves = {}

    for label, strat, cfg in build_strategies():
        equity_df, trade_df = run_backtest(strat, bars, cfg)
        m = compute_metrics(equity_df, trade_df, bars)

        print(
            f"{label:35s} {m['total_return_pct']:>+8.2f} {m['buy_and_hold_pct']:>+8.2f}"
            f" {m['max_drawdown_pct']:>+8.2f} {m['sharpe_ratio']:>8.3f}"
            f" {m['n_trades']:>7} {m['win_rate_pct']:>5.1f}%"
        )

        results.append({"strategy": label, **m})
        equity_curves[label] = equity_df["equity"]

    df = pd.DataFrame(results).sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    return df, equity_curves, bh_return


def plot_sweep(equity_curves: dict, bars: pd.DataFrame, bh_return: float, symbol: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11))
    fig.suptitle(f"Momentum vs EMA — {symbol} 90-day Daily Backtest", fontsize=13)

    start_eq = 100_000.0
    bh = (bars["close"] / bars["close"].iloc[0]) * start_eq

    # split into EMA and momentum groups for visual clarity
    ema_curves = {k: v for k, v in equity_curves.items() if k.startswith("EMA")}
    mom_curves = {k: v for k, v in equity_curves.items() if k.startswith("Mom")}

    for ax, group, title in [
        (ax1, ema_curves, "EMA Crossover strategies"),
        (ax2, mom_curves, "Momentum strategies"),
    ]:
        ax.plot(bars.index, bh, color="black", linewidth=1.5, linestyle="--",
                alpha=0.5, label=f"Buy & Hold ({bh_return:+.1f}%)")
        colors = plt.cm.tab10.colors
        for i, (label, eq) in enumerate(group.items()):
            ret = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0] * 100
            ax.plot(eq.index, eq.values, linewidth=1.4, color=colors[i],
                    label=f"{label} ({ret:+.1f}%)")
        ax.set_ylabel("Equity ($)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path("results") / f"sweep_{symbol}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nChart saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--period", default=None, help="yfinance period: 1y 2y 5y max")
    parser.add_argument("--symbol", default="SOXL")
    args = parser.parse_args()

    label = args.period if args.period else f"{args.days}d"
    print(f"Downloading {label} of {args.symbol} daily bars...")
    bars = download_bars(args.symbol, days=args.days, period=args.period)
    print(f"Loaded {len(bars)} bars: {bars.index[0].date()} → {bars.index[-1].date()}")

    results_df, equity_curves, bh_return = run_sweep(bars)

    print("\n=== Ranked by Return ===")
    print(results_df[["strategy", "total_return_pct", "max_drawdown_pct",
                       "sharpe_ratio", "n_trades", "win_rate_pct"]].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest: {best['strategy']}  |  Return: {best['total_return_pct']:+.2f}%  "
          f"|  MaxDD: {best['max_drawdown_pct']:.2f}%  |  Sharpe: {best['sharpe_ratio']:.3f}")

    results_df.to_csv("results/sweep_results.csv", index=False)
    plot_sweep(equity_curves, bars, bh_return, f"{args.symbol}_{label}")
