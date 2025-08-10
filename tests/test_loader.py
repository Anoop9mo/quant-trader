import pandas as pd
from io import StringIO
from qt.data import read_ohlcv, validate_ohlcv, resample_ohlcv

CSV = """timestamp,open,high,low,close,volume
2024-01-01 09:00:00,100,101,99,100.5,10
2024-01-01 09:01:00,100.5,101.5,100.4,101,12
2024-01-01 09:02:00,101,101.2,100.8,100.9,8
"""

def test_read_and_validate_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV)
    df = read_ohlcv(str(path), fmt="csv", tz="UTC")
    ok, issues = validate_ohlcv(df)
    assert ok, issues
    assert list(df.columns) == ["timestamp","open","high","low","close","volume"]
    assert df["timestamp"].dt.tz is not None

def test_resample():
    df = pd.read_csv(StringIO(CSV))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp","open","high","low","close","volume"]]
    out = resample_ohlcv(df, "3min")
    assert len(out) == 1
    assert out.iloc[0]["open"] == 100
    assert out.iloc[0]["close"] == 100.9
