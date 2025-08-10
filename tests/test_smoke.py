import pandas as pd
from qt.engine import Backtester
from qt.utils.toy import BuyTheDipStrategy, NaiveBroker

def test_smoke_backtest():
    df = pd.DataFrame({"close":[100,101,99,98,102,101,103,97,96,105]})
    bt = Backtester(BuyTheDipStrategy(window=3), NaiveBroker())
    res = bt.run(df)
    assert hasattr(res, "equity_curve")
    assert isinstance(res.stats, dict)
