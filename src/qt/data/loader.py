 
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import pandas as pd

_CANON = ["timestamp","open","high","low","close","volume"]

_COL_ALIASES = {
    "timestamp": {"timestamp","time","date","datetime","dt"},
    "open": {"open","o","op","open_price"},
    "high": {"high","h","hi","high_price"},
    "low": {"low","l","lo","low_price"},
    "close": {"close","c","cl","close_price","price"},
    "volume": {"volume","v","vol","qty","quantity","tick volume","tick_volume","tickvol"},

}

def _rename_to_canon(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for canon, aliases in _COL_ALIASES.items():
        for a in aliases:
            if a in cols_lower:
                mapping[cols_lower[a]] = canon
                break
    out = df.rename(columns=mapping)
    missing = [c for c in _CANON if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return out[_CANON]

def _parse_timestamp(ts: pd.Series, tz: Optional[str]) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="raise")
    if tz and tz.upper() != "UTC":
        # interpret as tz then convert to UTC
        s = s.tz_convert("UTC") if s.dt.tz is not None else s.dt.tz_localize(tz).dt.tz_convert("UTC")
    return s

def read_ohlcv(path: str, fmt: Optional[str]=None, tz: str="UTC") -> pd.DataFrame:
    if fmt is None:
        fmt = "parquet" if path.lower().endswith((".parquet",".pq",".parq")) else "csv"
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")

    df = _rename_to_canon(df)
    df["timestamp"] = _parse_timestamp(df["timestamp"], tz)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    # coerce numeric
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])  # allow volume to be NaN
    return df.reset_index(drop=True)

def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, dict]:
    issues = {}
    if not all(c in df.columns for c in _CANON):
        issues["columns"] = "missing required columns"
    from pandas.api.types import is_datetime64_any_dtype
    if not is_datetime64_any_dtype(df["timestamp"]) or df["timestamp"].dt.tz is None:
        issues["timestamp"] = "timestamp must be timezone-aware"
    if not df["timestamp"].is_monotonic_increasing:
        issues["order"] = "timestamps not sorted"
    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes:
        issues["duplicates"] = int(n_dupes)
    bad_prices = (df[["open","high","low","close"]] < 0).any().any()
    if bad_prices:
        issues["negatives"] = True
    range_viol = (df["high"] < df["low"]).sum()
    if range_viol:
        issues["high_low"] = int(range_viol)
    ok = len(issues) == 0
    return ok, issues

def resample_ohlcv(df: pd.DataFrame, rule: str, how: str="ohlc", volume: str="sum") -> pd.DataFrame:
    if how != "ohlc":
        raise ValueError("Only 'ohlc' resampling is supported currently.")
    ohlc = df.set_index("timestamp")[["open","high","low","close"]].resample(rule).agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    )
    vol = df.set_index("timestamp")[["volume"]].resample(rule).agg({"volume": volume})
    out = pd.concat([ohlc, vol], axis=1).dropna(subset=["open","high","low","close"], how="any")
    out = out.reset_index()
    out = out[ _CANON ]
    return out

def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)
