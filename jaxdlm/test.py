import jax.numpy as jnp
from datetime import datetime, timedelta
import pandas as pd
import warnings

from TimeFrame import TimeFrame

DATE_FORMATS = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%Y %H:%M:%S', '%Y-%m', '%Y']


def parse_time_string(t):
    # Try to parse the time string using a list of known formats.
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError(f"No known format to parse date string: {t}")


full_ts_datetime_range = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
full_ts_datetime_range = [parse_time_string(t) if isinstance(t, str) else t for t in full_ts_datetime_range]
long_data = pd.read_csv('data/long_data.csv')
print(long_data)
tf = TimeFrame.from_pandas(ts_df=long_data,
                           datetime_col="Datetime",
                           series_values_col='Values',
                           full_ts_datetime_range=full_ts_datetime_range,
                           features_df=None,
                           data_frequency=timedelta(days=1),
                           ts_max_length=None,
                           series_labels_col='Timeseries',
                           feature_names=['X1', 'X2', 'X3'],
                           is_wide=False)
