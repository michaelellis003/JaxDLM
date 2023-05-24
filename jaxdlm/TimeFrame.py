import jax.numpy as jnp
from datetime import datetime, timedelta
import pandas as pd
import warnings

# Specify a constant for the formats to be used across the module.
DATE_FORMATS = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%Y %H:%M:%S', '%Y-%m', '%Y']


def parse_time_string(t):
    # Try to parse the time string using a list of known formats.
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError(f"No known format to parse date string: {t}")


def _align_and_pad_long_pandas(df, datetime_col, full_ts_datetime_range, series_names_col, all_series_names):
    # Now, set 'Datetime' as the index
    df.set_index([series_names_col, datetime_col], inplace=True)

    # Create a multi-index with all combinations of series and dates
    new_index = pd.MultiIndex.from_product([all_series_names, full_ts_datetime_range],
                                           names=[series_names_col, datetime_col])

    aligned_df = df.reindex(new_index)
    aligned_df = aligned_df.reset_index()

    # Sort the DataFrame by Timeseries and Datetime
    aligned_df = aligned_df.sort_values([series_names_col, datetime_col])

    return aligned_df


def _align_and_pad_wide_pandas():
    raise NotImplementedError


def _check_dataframe(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataframe is missing required column: '{col}'")


def _from_long(df, datetime_col, series_values_col, full_ts_datetime_range, series_names_col=None,
               regressor_names=None):
    if regressor_names is None:
        regressor_names = [col for col in df.columns if (col != datetime_col and col != series_names_col and
                                                         col != series_values_col
                                                         )]
        if regressor_names is not None:
            if series_names_col is not None:
                warnings.warn(f"'regressor_names' is not specified. All columns other than {datetime_col}, "
                              f"{series_values_col}, and {series_names_col} will be treated as regressors.")
            else:
                warnings.warn(f"'regressor_names' is not specified. All columns other than {datetime_col}, "
                              f"{series_values_col}, and {series_names_col} will be treated as regressors.")
    _check_dataframe(df, [datetime_col, series_values_col] + regressor_names)

    if series_names_col is None:
        series_names_col = 'series_labels'
        df[series_names_col] = 1
        all_series_names = [1]
    else:
        all_series_names = list(set(df[series_names_col].unique()))

    aligned_df = _align_and_pad_long_pandas(df, datetime_col, full_ts_datetime_range, series_names_col,
                                            all_series_names)

    series_list = []
    regressors_dict = {}
    for name in all_series_names:
        series_df = aligned_df[aligned_df[series_names_col] == name]
        series_list.append(series_df[series_values_col].tolist())

        if regressor_names is not None:
            regressors_dict[name] = jnp.array(aligned_df[regressor_names].values.tolist())

    return series_list, regressors_dict


def from_wide():
    raise NotImplementedError


def _check_intervals(dates):
    # calculate differences between each pair of dates
    differences = list(map(lambda x: x[1] - x[0], zip(dates[:-1], dates[1:])))

    # check if all differences are the same
    return len(set(differences)) == 1


def _validate_full_ts_datetime_range(full_ts_datetime_range, ts_max_length):
    if full_ts_datetime_range is None:
        if ts_max_length is None:
            raise ValueError("Either 'full_ts_datetime_range' or 'ts_max_length' must be specified.")

        full_ts_datetime_range = list(range(ts_max_length))
        data_frequency = None
    else:  # full_ts_datetime_range is not None
        full_ts_datetime_range = [parse_time_string(t) if isinstance(t, str) else t for t in full_ts_datetime_range]

        if ts_max_length is not None:
            warnings.warn("Both 'full_ts_datetime_range' is specified. 'ts_max_length' will be ignored.", UserWarning)

        # Check that the interval between dates matches the data_frequency
        date_differences_equal = _check_intervals(full_ts_datetime_range)
        if not date_differences_equal:
            raise ValueError(
                "The interval between datetime objects in full_ts_datetime_range does not match the "
                "specified data_frequency at index {i}.")
        else:
            data_frequency = full_ts_datetime_range[-1] - full_ts_datetime_range[-2]

    return full_ts_datetime_range, data_frequency


class TimeFrame:
    def __init__(self, series, features=None, full_ts_datetime_range=None, data_frequency=None, ts_max_length=None,
                 validate_full_ts_datetime_range=True
                 ):
        self.series = series
        self.features = features or {}

        if validate_full_ts_datetime_range:
            self.full_ts_datetime_range, self.data_frequency = _validate_full_ts_datetime_range(full_ts_datetime_range,
                                                                                                ts_max_length)
        else:
            self.full_ts_datetime_range = full_ts_datetime_range
            self.data_frequency = data_frequency

        self.ts_max_length = ts_max_length

    @classmethod
    def from_pandas(cls, ts_df, datetime_col, series_values_col, full_ts_datetime_range, features_df=None,
                    data_frequency=None, ts_max_length=None, series_labels_col=None, feature_names=None, is_wide=False):
        if not is_wide and features_df is not None:
            raise ValueError('If you have a "long" dataframe then the time series data and features data should be in '
                             'one dataframe')

        full_ts_datetime_range, data_frequency = _validate_full_ts_datetime_range(full_ts_datetime_range,
                                                                                  ts_max_length)

        ts_df[datetime_col] = [parse_time_string(t) if isinstance(t, str) else t
                               for t in ts_df[datetime_col].tolist()]

        if features_df is not None:
            features_df[datetime_col] = [parse_time_string(t) if isinstance(t, str) else t
                                         for t in features_df[datetime_col].tolist()]

        if is_wide:
            timeseries_data = None
            features_data = None
            raise NotImplementedError
        else:  # long
            timeseries_data, features_data = _from_long(ts_df, datetime_col, series_values_col, full_ts_datetime_range,
                                                        series_labels_col, feature_names)

        return cls(series=timeseries_data, features=features_data, full_ts_datetime_range=full_ts_datetime_range,
                   data_frequency=data_frequency, ts_max_length=ts_max_length, validate_full_ts_datetime_range=False
                   )

    def add_time_series(self, series_label, series):
        raise NotImplementedError

    def remove_time_series(self, series_label, series):
        raise NotImplementedError

    def get_time_series(self, series_label):
        raise NotImplementedError

    def get_time_series_names(self):
        raise NotImplementedError

    def get_regressors(self, series_label):
        raise NotImplementedError

    def get_regressors_names(self, series_label):
        raise NotImplementedError
