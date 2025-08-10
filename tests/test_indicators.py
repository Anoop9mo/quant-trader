import pandas as pd
import numpy as np
from qt.features.indicators import ema, atr, rsi

def test_ema_basic():
    s = pd.Series([1,2,3,4,5], dtype=float)
    out = ema(s, period=3)
    assert len(out) == 5
    assert out.iloc[-1] > out.iloc[0]

def test_atr_nonnegative():
    df = pd.DataFrame({
        "high":[10,11,12,13,12,14],
        "low":[9,9.5,10,11,10.5,12],
        "close":[9.5,10.5,11.5,12,11.7,13]
    })
    a = atr(df, period=3)
    assert (a >= 0).all()
    assert len(a) == len(df)

def test_rsi_range():
    s = pd.Series([1,2,3,4,5,4,3,2,1,2,3,4,5], dtype=float)
    r = rsi(s, period=5)
    assert ((r >= 0) & (r <= 100)).all()
    assert not np.isnan(r.iloc[-1])
