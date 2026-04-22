"""
Backtest for the BuyingPower SOXL/UVXY rotation strategy.

Uses the actual TradingView buying power scores from the historical CSV
(data/SPYQQQSOXL_OptionUltraShortBuyingPower.csv) as ground truth signals,
then simulates execution on SOXL daily bars. Compares vs buy-and-hold SOXL.

Usage:
    python backtest_buying_power.py
    python backtest_buying_power.py --entry 65 --exit 30 --uvxy 25
    python backtest_buying_power.py --no-uvxy   # SOXL/cash only, no UVXY hedge
"""
import argparse
import math
from pathlib import Path

import pandas as pd
import numpy as np

CAPITAL = 100_000.0
SLIPPAGE = 0.0005   # 0.05% each side — realistic for liquid ETF


def load_data() -> pd.DataFrame:
    daily = pd.read_csv("data/SOXL_2yr_daily_tv.csv")
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily["date_only"] = daily["date"].dt.date
    daily = daily.sort_values("date").reset_index(drop=True)

    bp = pd.read_csv("data/SPYQQQSOXL_OptionUltraShortBuyingPower.csv")
    bp["date_only"] = pd.to_datetime(bp["日期"]).dt.date
    bp = bp.rename(columns={"SOXL 超短": "soxl_score", "UVXY 超短": "uvxy_score"})
    bp["soxl_score"] = pd.to_numeric(bp["soxl_score"], errors="coerce")
    bp["uvxy_score"] = pd.to_numeric(bp["uvxy_score"], errors="coerce")
    bp["uvxy_price"] = pd.to_numeric(bp["UVXY"], errors="coerce")

    df = daily.merge(
        bp[["date_only", "soxl_score", "uvxy_score", "uvxy_price"]],
        on="date_only", how="left",
    )
    return df


def run_backtest(
    df: pd.DataFrame,
    entry_thresh: float = 65.0,
    exit_thresh: float = 30.0,
    uvxy_thresh: float = 25.0,
    use_uvxy: bool = True,
) -> dict:
    """
    Signal fires on day i close → execute at day i+1 open (realistic).
    Returns performance dict with equity curve and trade log.
    """
    df_valid = df.dropna(subset=["soxl_score"]).copy().reset_index(drop=True)

    cash = CAPITAL
    soxl_qty = 0.0
    uvxy_qty = 0.0
    mode = "cash"

    equity_curve = []
    trades = []

    for i in range(len(df_valid) - 1):
        row = df_valid.iloc[i]
        nxt = df_valid.iloc[i + 1]

        sig = row["soxl_score"]
        uvxy_p = row["uvxy_price"] if pd.notna(row["uvxy_price"]) else 0.0
        uvxy_nxt = nxt["uvxy_price"] if pd.notna(nxt["uvxy_price"]) else None

        equity = cash + soxl_qty * row["close"] + uvxy_qty * uvxy_p
        equity_curve.append({"date": row["date_only"], "equity": equity, "mode": mode, "soxl_score": sig})

        # ── Exit SOXL ─────────────────────────────────────────────────────────
        if mode == "soxl" and sig < exit_thresh:
            sp = nxt["open"] * (1 - SLIPPAGE)
            cash = soxl_qty * sp
            trades.append(("SELL SOXL", str(row["date_only"]), sp, sig, cash))
            soxl_qty = 0.0
            mode = "cash"

        # ── Exit UVXY ─────────────────────────────────────────────────────────
        elif mode == "uvxy" and sig > exit_thresh and uvxy_nxt:
            sp = uvxy_nxt * (1 - SLIPPAGE)
            cash = uvxy_qty * sp
            trades.append(("SELL UVXY", str(row["date_only"]), sp, sig, cash))
            uvxy_qty = 0.0
            mode = "cash"

        # ── Enter SOXL ────────────────────────────────────────────────────────
        if mode == "cash" and sig >= entry_thresh:
            ep = nxt["open"] * (1 + SLIPPAGE)
            soxl_qty = cash / ep
            trades.append(("BUY  SOXL", str(row["date_only"]), ep, sig, cash))
            cash = 0.0
            mode = "soxl"

        # ── Enter UVXY ────────────────────────────────────────────────────────
        elif mode == "cash" and use_uvxy and sig < uvxy_thresh and uvxy_nxt:
            ep = uvxy_nxt * (1 + SLIPPAGE)
            uvxy_qty = cash / ep
            trades.append(("BUY  UVXY", str(row["date_only"]), ep, sig, cash))
            cash = 0.0
            mode = "uvxy"

    # Close final position at last close
    last = df_valid.iloc[-1]
    uvxy_p_last = last["uvxy_price"] if pd.notna(last["uvxy_price"]) else 0.0
    final_equity = cash + soxl_qty * last["close"] + uvxy_qty * uvxy_p_last

    eq_df = pd.DataFrame(equity_curve)
    roll_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
    max_dd = drawdown.min()

    # Buy-and-hold SOXL over same period
    bah_start = df_valid["close"].iloc[0]
    bah_end = df_valid["close"].iloc[-1]
    bah_final = CAPITAL * bah_end / bah_start
    bah_dd_series = (df_valid["close"] - df_valid["close"].cummax()) / df_valid["close"].cummax() * 100
    bah_max_dd = bah_dd_series.min()

    # Annualised return
    n_days = (df_valid["date_only"].iloc[-1] - df_valid["date_only"].iloc[0]).days
    years = n_days / 365.25
    strat_cagr = ((final_equity / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    bah_cagr = ((bah_final / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "start": str(df_valid["date_only"].iloc[0]),
        "end": str(df_valid["date_only"].iloc[-1]),
        "n_days": n_days,
        "strat_final": final_equity,
        "strat_return": (final_equity / CAPITAL - 1) * 100,
        "strat_cagr": strat_cagr,
        "strat_max_dd": max_dd,
        "bah_final": bah_final,
        "bah_return": (bah_final / CAPITAL - 1) * 100,
        "bah_cagr": bah_cagr,
        "bah_max_dd": bah_max_dd,
        "n_trades": len(trades),
        "trades": trades,
        "equity_curve": eq_df,
    }


def print_report(r: dict, label: str = ""):
    sep = "=" * 60
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    print(f"  Period: {r['start']} → {r['end']}  ({r['n_days']} days)")
    print(sep)
    print(f"  {'':30s}  {'Return':>8s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'Final $':>10s}")
    print(f"  {'Buy-and-hold SOXL':30s}  {r['bah_return']:>7.1f}%  {r['bah_cagr']:>6.1f}%  {r['bah_max_dd']:>6.1f}%  ${r['bah_final']:>9,.0f}")
    print(f"  {'Signal + UVXY hedge':30s}  {r['strat_return']:>7.1f}%  {r['strat_cagr']:>6.1f}%  {r['strat_max_dd']:>6.1f}%  ${r['strat_final']:>9,.0f}")
    beat = "✓ BEATS" if r["strat_return"] > r["bah_return"] else "✗ TRAILS"
    dd_better = "✓ BETTER" if r["strat_max_dd"] > r["bah_max_dd"] else "✗ WORSE"
    print(f"\n  Return:   {beat} buy-and-hold by {r['strat_return'] - r['bah_return']:+.1f}%")
    print(f"  DrawDown: {dd_better} than buy-and-hold by {r['strat_max_dd'] - r['bah_max_dd']:+.1f}%")
    print(f"  Trades:   {r['n_trades']}")

    print("\n  Last 10 trades:")
    for t in r["trades"][-10:]:
        action, date, price, sig, capital = t
        print(f"    {action}  {date}  ${price:7.2f}  score={sig:5.1f}  capital=${capital:>10,.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry",  type=float, default=65.0)
    parser.add_argument("--exit",   type=float, default=30.0)
    parser.add_argument("--uvxy",   type=float, default=25.0)
    parser.add_argument("--no-uvxy", action="store_true")
    args = parser.parse_args()

    df = load_data()

    r = run_backtest(
        df,
        entry_thresh=args.entry,
        exit_thresh=args.exit,
        uvxy_thresh=args.uvxy,
        use_uvxy=not args.no_uvxy,
    )

    label = f"BuyingPower(entry={args.entry}/exit={args.exit}/uvxy={args.uvxy})"
    if args.no_uvxy:
        label += "  [no UVXY hedge]"
    print_report(r, label)

    # Also show no-UVXY comparison
    if not args.no_uvxy:
        r2 = run_backtest(df, args.entry, args.exit, args.uvxy, use_uvxy=False)
        print_report(r2, "SOXL/cash only (no UVXY)")


if __name__ == "__main__":
    main()
