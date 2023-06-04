import pytest
from jaxdlm.TimeFrame import TSDateTime
from datetime import datetime
import pandas as pd


@pytest.mark.parametrize('min_ts_datetime, max_ts_datetime, ts_periods, max_ts_length, use_dates, expected', [
    [None, None, None, 10, False, list(range(10))],  # Max length
    [datetime(2023, 1, 1), datetime(2023, 12, 31), 12, None, True,
     pd.date_range(start=datetime(2023, 1, 1), end=datetime(2023, 12, 31), periods=12).to_pydatetime().tolist()],
    ["2023-01-01", "2023-12-31", 12, None, True,
     pd.date_range(start="2023-01-01", end="2023-12-31", periods=12).to_pydatetime().tolist()],
])
def test_TSDateTime(min_ts_datetime, max_ts_datetime, ts_periods, max_ts_length, use_dates, expected):
    obj = TSDateTime(min_ts_datetime=min_ts_datetime,
                     max_ts_datetime=max_ts_datetime,
                     ts_periods=ts_periods,
                     max_ts_length=max_ts_length)

    print(f"expected = {expected} \n")
    print(f"obj.ts_datetime = {obj.ts_datetime} \n")
    print("="*100)
    assert obj.ts_datetime == expected
    assert obj.min_ts_datetime == min_ts_datetime
    assert obj.max_ts_datetime == max_ts_datetime
    assert obj.ts_periods == ts_periods
    assert obj.max_ts_length == max_ts_length
    assert obj.use_dates == use_dates
