from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StrategyConfig:
    symbol: str = "SOXL"
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_entry_min: float = 45.0
    rsi_entry_max: float = 72.0
    rsi_exit: float = 80.0
    stop_loss_pct: float = 0.025       # 2.5% hard stop
    max_position_pct: float = 0.10     # 10% of portfolio per trade
    vix_threshold: float = 30.0        # skip trading when VIX > this
    market_open: str = "10:00"         # ET
    market_close: str = "15:30"        # ET, flatten before close
    timeframe: str = "1Day"


@dataclass
class BuyingPowerConfig:
    # Primary asset
    symbol: str = "SOXL"
    symbol_exchange: str = "AMEX"
    # Hedge asset (inverse volatility)
    hedge_symbol: str = "UVXY"
    hedge_exchange: str = "CBOE"
    # Signal thresholds (0–99 scale, matching the buying power CSV)
    entry_thresh: float = 65.0         # SOXL score above → all-in SOXL
    exit_thresh: float = 25.0          # SOXL score below → exit SOXL (tight: avoids shallow pullback exits)
    uvxy_thresh: float = 20.0          # SOXL score below → rotate to UVXY (only extreme bearish)
    confirm_timeframe: str = "15Min"   # secondary TV timeframe for entry confirmation
    confirm_min: float = 45.0          # secondary score must exceed this
    # Execution
    check_interval_min: int = 15       # signal check frequency (minutes)
    market_open: str = "09:35"         # ET — first check after open settle
    market_close: str = "15:45"        # ET — flatten before close
    # Position sizing: use nearly all buying power (fractional shares supported)
    position_pct: float = 0.99         # 99% of available buying power


@dataclass
class AlpacaConfig:
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    paper: bool = field(default_factory=lambda: os.getenv("ALPACA_PAPER", "true").lower() == "true")

    @property
    def base_url(self) -> str:
        return (
            "https://paper-api.alpaca.markets"
            if self.paper
            else "https://api.alpaca.markets"
        )


@dataclass
class MLConfig:
    model_type: str = "random_forest"  # random_forest | xgboost | lstm
    model_path: str = "models/"
    feature_window: int = 20           # bars of lookback for feature extraction
    min_training_samples: int = 50     # minimum trades needed before training


@dataclass
class Config:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    buying_power: BuyingPowerConfig = field(default_factory=BuyingPowerConfig)


CONFIG = Config()
