import pandas as pd

class BuyTheDipStrategy:
    def __init__(self, window:int=20):
        self.window = window
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ma = df["close"].rolling(self.window, min_periods=self.window).mean()
        df["signal"] = 0
        df.loc[df["close"] < ma, "signal"] = 1
        df.loc[df["close"] > ma, "signal"] = -1
        return df[["signal"]]

class NaiveBroker:
    def execute(self, signals: pd.DataFrame) -> pd.DataFrame:
        eq = (signals["signal"].shift().fillna(0).cumsum() + 100.0)
        out = signals.copy()
        out["equity"] = eq
        return out
