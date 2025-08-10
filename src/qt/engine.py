from dataclasses import dataclass
from typing import Dict, Any, Protocol
import pandas as pd

class Strategy(Protocol):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame: ...

class Broker(Protocol):
    def execute(self, signals: pd.DataFrame) -> pd.DataFrame: ...

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    stats: Dict[str, Any]

class Backtester:
    def __init__(self, strategy: Strategy, broker: Broker):
        self.strategy = strategy
        self.broker = broker

    def run(self, df: pd.DataFrame) -> BacktestResult:
        signals = self.strategy.generate_signals(df)
        trades  = self.broker.execute(signals)
        equity  = trades["equity"].astype(float)
        stats   = {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
            "n_trades": int(trades.get("trade_id", pd.Series(dtype=int)).nunique()) if "trade_id" in trades else 0,
        }
        return BacktestResult(equity_curve=equity, stats=stats)
