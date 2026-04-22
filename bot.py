"""
Live intraday trading bot — paper or live via Alpaca.

Usage:
    python bot.py                          # buying_power strategy (SOXL/UVXY rotation)
    python bot.py --strategy momentum      # legacy momentum strategy
    python bot.py --strategy ema_rsi
    python bot.py --strategy ml --model models/model.pkl
    python bot.py --live                   # live trading (requires ALPACA_PAPER=false in .env)

Buying Power strategy specifics:
  - Checks TradingView 5-min + 15-min consensus scores every 15 min
  - Deploys 99% of buying power into SOXL when score > entry_thresh
  - Rotates to UVXY when score < uvxy_thresh (extreme bearish)
  - Flattens to cash when score is in the neutral zone
"""
import argparse
import logging
from datetime import datetime, timezone, timedelta

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from broker import AlpacaBroker, Order
from strategy import EmaRsiStrategy, MomentumStrategy, MLStrategy, BuyingPowerStrategy
from risk import calc_position_size, should_stop, vix_filter_ok
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("soxl_bot")

ET = pytz.timezone("America/New_York")


# ── Legacy single-symbol bot ────────────────────────────────────────────────

class TradingBot:
    """Original single-symbol bot kept for momentum/ema_rsi/ml strategies."""

    def __init__(self, broker: AlpacaBroker, strategy, cfg=None):
        self.broker = broker
        self.strategy = strategy
        self.cfg = cfg or CONFIG.strategy
        self._active_stop: dict = {}
        self._symbol = self.cfg.symbol
        self._timeframe = self.cfg.timeframe

    def _get_recent_bars(self, lookback_bars: int = 100):
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=lookback_bars + 30)).isoformat()
        end = now.isoformat()
        return self.broker.get_bars(self._symbol, self._timeframe, start, end, limit=lookback_bars)

    def _is_trading_window(self) -> bool:
        now = datetime.now(ET)
        from datetime import time
        oh, om = map(int, self.cfg.market_open.split(":"))
        ch, cm = map(int, self.cfg.market_close.split(":"))
        return time(oh, om) <= now.time() <= time(ch, cm)

    def _flatten(self, reason: str = ""):
        pos = self.broker.get_position(self._symbol)
        if pos and pos.qty > 0:
            oid = self.broker.place_order(Order(self._symbol, pos.qty, "sell"))
            log.info(f"SELL {pos.qty} {self._symbol} @ market | {reason} | order={oid}")
            self._active_stop.pop(self._symbol, None)

    def tick(self):
        if not self.broker.is_market_open():
            log.debug("Market closed, skipping tick")
            return

        vix_ok, vix_val = vix_filter_ok(self.cfg.vix_threshold)
        if not vix_ok:
            log.warning(f"VIX={vix_val:.1f} > threshold — flattening")
            self._flatten("VIX too high")
            return

        bars = self._get_recent_bars(self.strategy.min_bars_required + 10)
        if bars.empty:
            log.warning("No bars returned")
            return

        price = float(bars["close"].iloc[-1])
        pos = self.broker.get_position(self._symbol)

        if pos:
            stop = self._active_stop.get(self._symbol)
            if should_stop(pos.avg_entry_price, price, self.cfg.stop_loss_pct, stop):
                log.warning(f"Stop triggered @ {price:.2f} (entry={pos.avg_entry_price:.2f})")
                self._flatten("stop loss")
                return

        signal = self.strategy.generate_signal(bars)
        log.info(f"Signal: {signal.action.upper():4s} | {signal.reason} | price={price:.2f}")

        if signal.action == "buy" and pos is None:
            acct = self.broker.get_account()
            qty = calc_position_size(acct.equity, price, self.cfg.max_position_pct)
            if qty < 1:
                log.warning("Position size too small, skipping")
                return
            oid = self.broker.place_order(Order(self._symbol, qty, "buy"))
            if signal.stop_price:
                self._active_stop[self._symbol] = signal.stop_price
            log.info(f"BUY  {qty} {self._symbol} @ market | conf={signal.confidence:.2f} | order={oid}")

        elif signal.action == "sell" and pos:
            self._flatten(signal.reason)


# ── Buying Power rotation bot ────────────────────────────────────────────────

class BuyingPowerBot:
    """
    Full-capital SOXL / UVXY rotation bot.

    Uses TradingView 5-min buying power scores. Deploys 99% of buying power
    into SOXL (bullish) or UVXY (extreme bearish) and holds cash otherwise.
    Fractional shares are used so the full buying power is always deployed.
    """

    def __init__(self, broker: AlpacaBroker, strategy: BuyingPowerStrategy, cfg=None):
        self.broker = broker
        self.strategy = strategy
        self.cfg = cfg or CONFIG.buying_power

    def _is_trading_window(self) -> bool:
        now = datetime.now(ET)
        from datetime import time
        oh, om = map(int, self.cfg.market_open.split(":"))
        ch, cm = map(int, self.cfg.market_close.split(":"))
        return time(oh, om) <= now.time() <= time(ch, cm)

    def _current_mode(self) -> str:
        """Returns 'soxl', 'uvxy', or 'cash'."""
        if self.broker.get_position(self.cfg.symbol):
            return "soxl"
        if self.broker.get_position(self.cfg.hedge_symbol):
            return "uvxy"
        return "cash"

    def _close_position(self, symbol: str, reason: str):
        pos = self.broker.get_position(symbol)
        if pos and pos.qty > 0:
            oid = self.broker.place_order(Order(symbol, pos.qty, "sell"))
            log.info(f"SELL {pos.qty:.4f} {symbol} @ market | {reason} | order={oid}")

    def _buy_all(self, symbol: str, price: float, reason: str):
        acct = self.broker.get_account()
        # Use buying_power (already accounts for margin / existing positions)
        capital = acct.buying_power * self.cfg.position_pct
        qty = round(capital / price, 4)  # fractional qty
        if qty <= 0:
            log.warning(f"Buying power too low to buy {symbol}")
            return
        oid = self.broker.place_order(Order(symbol, qty, "buy"))
        log.info(f"BUY  {qty:.4f} {symbol} @ ~{price:.2f} | capital=${capital:,.0f} | {reason} | order={oid}")

    def tick(self):
        if not self.broker.is_market_open():
            log.debug("Market closed, skipping tick")
            return

        if not self._is_trading_window():
            now_et = datetime.now(ET).strftime("%H:%M")
            log.debug(f"Outside trading window ({now_et} ET), skipping")
            return

        # Generate signal (fetches live TV scores internally)
        signal = self.strategy.generate_signal(bars=None)
        log.info(f"Signal: {signal.action.upper():4s} target={signal.target_symbol} | {signal.reason}")

        mode = self._current_mode()

        # ── Sell logic ────────────────────────────────────────────────────────
        if signal.action == "sell":
            if mode == "soxl":
                self._close_position(self.cfg.symbol, signal.reason)
            elif mode == "uvxy":
                self._close_position(self.cfg.hedge_symbol, signal.reason)
            return

        # ── Buy logic ─────────────────────────────────────────────────────────
        if signal.action == "buy":
            target = signal.target_symbol  # 'SOXL' or 'UVXY'

            # Already in the right asset — do nothing
            if (target == "SOXL" and mode == "soxl") or \
               (target == "UVXY" and mode == "uvxy"):
                log.debug(f"Already in {target}, holding")
                return

            # Wrong asset or cash — rotate
            if mode == "soxl":
                self._close_position(self.cfg.symbol, f"rotate to {target}")
            elif mode == "uvxy":
                self._close_position(self.cfg.hedge_symbol, f"rotate to {target}")

            # Fetch current price for sizing
            try:
                sym = self.cfg.symbol if target == "SOXL" else self.cfg.hedge_symbol
                price = self.broker.get_latest_price(sym)
                self._buy_all(sym, price, signal.reason)
            except Exception as e:
                log.error(f"Failed to buy {target}: {e}")

    def flatten_all(self, reason: str = "manual shutdown"):
        """Close all positions."""
        for sym in [self.cfg.symbol, self.cfg.hedge_symbol]:
            self._close_position(sym, reason)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOXL Trading Bot")
    parser.add_argument(
        "--strategy", default="buying_power",
        choices=["buying_power", "ema_rsi", "momentum", "ml"],
        help="Trading strategy (default: buying_power)",
    )
    parser.add_argument("--model", default=None, help="ML model path")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    args = parser.parse_args()

    if args.live:
        CONFIG.alpaca.paper = False
        log.warning("=== LIVE TRADING MODE — real money at risk ===")

    broker = AlpacaBroker(CONFIG.alpaca)
    cfg_bp = CONFIG.buying_power

    if args.strategy == "buying_power":
        strategy = BuyingPowerStrategy(
            entry_thresh=cfg_bp.entry_thresh,
            exit_thresh=cfg_bp.exit_thresh,
            uvxy_thresh=cfg_bp.uvxy_thresh,
            confirm_timeframe=cfg_bp.confirm_timeframe,
            confirm_min=cfg_bp.confirm_min,
        )
        bot = BuyingPowerBot(broker, strategy, cfg_bp)
        log.info(
            f"BuyingPowerBot started | {cfg_bp.symbol}/{cfg_bp.hedge_symbol} | "
            f"entry={cfg_bp.entry_thresh} exit={cfg_bp.exit_thresh} uvxy={cfg_bp.uvxy_thresh} | "
            f"paper={CONFIG.alpaca.paper}"
        )

        scheduler = BlockingScheduler(timezone=ET)
        scheduler.add_job(
            bot.tick, "cron",
            day_of_week="mon-fri",
            minute=f"*/{cfg_bp.check_interval_min}",
        )
        try:
            bot.tick()
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Bot stopped")
            bot.flatten_all("manual shutdown")

    else:
        # Legacy strategies
        cfg = CONFIG.strategy
        if args.strategy == "ml":
            strategy = MLStrategy(CONFIG.ml)
            if not strategy.load_model(args.model):
                log.error("ML model not loaded.")
                return
        elif args.strategy == "momentum":
            strategy = MomentumStrategy(cfg)
        else:
            strategy = EmaRsiStrategy(cfg)

        bot = TradingBot(broker, strategy, cfg)
        log.info(f"Bot started | strategy={strategy.name} | paper={CONFIG.alpaca.paper}")

        scheduler = BlockingScheduler(timezone=ET)
        scheduler.add_job(bot.tick, "cron", day_of_week="mon-fri", hour=15, minute=50)
        try:
            bot.tick()
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Bot stopped")
            bot._flatten("manual shutdown")


if __name__ == "__main__":
    main()
