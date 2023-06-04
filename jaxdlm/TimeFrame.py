import jax.numpy as jnp
from typing import Union, Optional, List
from jaxtyping import Array, Float, Int
from datetime import datetime
import pandas as pd
import warnings


class TSDateTime:
    def __init__(self,
                 min_ts_datetime: Optional[Union[datetime, str]] = None,
                 max_ts_datetime: Optional[Union[datetime, str]] = None,
                 ts_periods: Optional[int] = None,
                 max_ts_length: Optional[int] = None,
                 use_dates: Optional[bool] = None
                 ):

        self.min_ts_datetime = min_ts_datetime
        self.max_ts_datetime = max_ts_datetime
        self.ts_periods = ts_periods
        self.max_ts_length = max_ts_length
        self.use_dates = use_dates
        self.ts_datetime = None

        if all(arg is None for arg in [min_ts_datetime, max_ts_datetime, ts_periods]) and max_ts_length is not None:
            self.ts_datetime = list(range(max_ts_length))
        else:
            self.ts_datetime = pd.date_range(start=self.min_ts_datetime,
                                             end=self.max_ts_datetime,
                                             periods=ts_periods)


class TimeSeries:
    def __init__(self,
                 values: Union[Float[Array, "num_series ts_length"], Int[Array, "num_series ts_length"]],
                 ts_length: Optional[int] = None,
                 num_series: Optional[int] = None,
                 names: Optional[Union[List[str], List[int]]] = None
                 ):
        self.values = values
        self.ts_length = ts_length if ts_length is not None else self.values.shape[1]
        self.num_series = num_series if num_series is not None else self.values.shape[0]
        self.names = names if names is not None else list(range(self.num_series))


def _align_and_pad_long_pandas(pandas_df: pd.DataFrame,
                               ts_datetime_col: str,
                               ts_datetime: TSDateTime,
                               series_names_col: str,
                               all_series_names: List[str]):
    # Now, set 'Datetime' as the index
    pandas_df = pandas_df.sort_values([series_names_col, ts_datetime_col])
    pandas_df.set_index([series_names_col, ts_datetime_col], inplace=True)

    full_ts_datetime_range = ts_datetime.ts_datetime

    # Create a multi-index with all combinations of series and dates
    new_index = pd.MultiIndex.from_product([all_series_names, full_ts_datetime_range],
                                           names=[series_names_col, ts_datetime_col])

    aligned_df = pandas_df.reindex(new_index)
    aligned_df = aligned_df.reset_index()

    # Sort the DataFrame by Timeseries and Datetime
    aligned_df = aligned_df.sort_values([series_names_col, ts_datetime_col])

    return aligned_df


def _get_datetime_col_metadata(df: pd.DataFrame,
                               ts_datetime_col: str,
                               series_names_col: Optional[str] = None,
                               min_ts_datetime: Optional[datetime] = None,
                               max_ts_datetime: Optional[datetime] = None
                               ):
    if pd.api.types.is_string_dtype(df[ts_datetime_col]):
        warnings.warn("'ts_datetime_col' has a string dtype. Attempting to convert it to datetime object",
                      Warning)
        df[ts_datetime_col] = pd.to_datetime(df[ts_datetime_col])

    if pd.api.types.is_datetime64_dtype(df[ts_datetime_col]):
        min_ts_datetime_list = [df[ts_datetime_col].min(), min_ts_datetime]
        max_ts_datetime_list = [df[ts_datetime_col].max(), max_ts_datetime]

        min_ts_datetime = min(value for value in min_ts_datetime_list if value is not None)
        max_ts_datetime = min(value for value in max_ts_datetime_list if value is not None)

        # find minimum interval of data
        if series_names_col is not None:
            df['difference'] = df.groupby(series_names_col)[ts_datetime_col].diff()
        else:
            df['difference'] = df[ts_datetime_col].diff()

        # Find the minimum difference
        min_difference = df['difference'].min()
        # Calculate number of periods
        ts_periods = int((max_ts_datetime - min_ts_datetime) / min_difference) + 1  # +1 to include the end date

        max_ts_length = None

        use_dates = True
    elif pd.api.types.is_integer_dtype(df[ts_datetime_col]):
        min_ts_datetime = None
        max_ts_datetime = None
        ts_periods = None

        max_ts_length = df.groupby(series_names_col).size().tolist().max()

        use_dates = False
    else:
        raise ValueError(f"dtype {df[ts_datetime_col].dtype} for {ts_datetime_col} not supported. dtype must "
                         f"be an integer, string, or datetime.")

    return min_ts_datetime, max_ts_datetime, ts_periods, max_ts_length, use_dates


class TimeFrame:
    def __init__(self,
                 observations: TimeSeries,
                 ts_datetime: TSDateTime,
                 features: Optional[TimeSeries] = None,
                 ):
        self.observations = observations
        self.ts_datetime = ts_datetime
        self.features = features

    @classmethod
    def from_wide_pandas(cls,
                         target_df: pd.DataFrame,
                         features_df: Optional[pd.DataFrame] = None,
                         ts_datetime_col: Optional[str] = 'index',
                         min_ts_datetime: Optional[Union[datetime, str]] = None,
                         max_ts_datetime: Optional[Union[datetime, str]] = None,
                         fill_missing_ts_datetime: Optional[bool] = True
                         ):

        target_df = target_df.reset_index()
        features_df = features_df.reset_index() if features_df is not None else None

        all_series_names = [col for col in target_df.columns if col != ts_datetime_col]
        if features_df is None:
            all_feats_names = None
        else:
            all_feats_names = [col for col in features_df.columns if col != ts_datetime_col]

        target_ts_datetime_metadata = _get_datetime_col_metadata(target_df, ts_datetime_col, min_ts_datetime,
                                                                 max_ts_datetime)
        target_min_ts_datetime = target_ts_datetime_metadata[0]
        target_max_ts_datetime = target_ts_datetime_metadata[1]
        target_ts_periods = target_ts_datetime_metadata[2]
        target_max_ts_length = target_ts_datetime_metadata[3]
        target_use_dates = target_ts_datetime_metadata[4]

        target_ts_datetime = TSDateTime(target_min_ts_datetime, target_max_ts_datetime, target_ts_periods,
                                        target_max_ts_length, target_use_dates)

        if features_df is not None:
            feats_ts_datetime_metadata = _get_datetime_col_metadata(features_df, ts_datetime_col, min_ts_datetime,
                                                                    max_ts_datetime)
            feats_min_ts_datetime = feats_ts_datetime_metadata[0]
            feats_max_ts_datetime = feats_ts_datetime_metadata[1]
            feats_ts_periods = feats_ts_datetime_metadata[2]
            feats_max_ts_length = feats_ts_datetime_metadata[3]
            feats_use_dates = feats_ts_datetime_metadata[4]

            feats_ts_datetime = TSDateTime(feats_min_ts_datetime, feats_max_ts_datetime, feats_ts_periods,
                                           feats_max_ts_length, feats_use_dates)
        else:
            feats_ts_datetime = None

        if fill_missing_ts_datetime:
            if all_feats_names is not None:
                warnings.warn(
                    "Your features dataframe should not have missing values. If they do have missing values you "
                    "should move them to the target dataframe", Warning)

            target_df = target_df.sort_values([ts_datetime_col])

            # Create a multi-index with all combinations of series and dates
            new_index = target_ts_datetime.ts_datetime

            aligned_target_df = target_df.reindex(new_index)
            aligned_target_df = aligned_target_df.reset_index()

            # Sort the DataFrame by Timeseries and Datetime
            aligned_target_df = aligned_target_df.sort_values([ts_datetime_col])
        else:
            aligned_target_df = target_df

        if features_df is not None:
            if aligned_target_df.shape[0] != features_df.shape[0]:
                raise ValueError(f"The target_df and features_df do not have the same number of observations. "
                                 f"The shape of target_df is {target_df.shape} and the shape of features_df "
                                 f"is {features_df.shape}. This could be caused by features_df having missing values. "
                                 f"Your features dataframe should not have missing values. If they do have missing "
                                 f"values you should move them to the target dataframe")

            if target_ts_datetime.ts_datetime != feats_ts_datetime.ts_datetime:
                raise ValueError(f"The target_df and features_df do not have the same dates"
                                 f"This could be caused by features_df having missing values. Your features dataframe "
                                 f"should not have missing values. If they do have missing values you should move "
                                 f"them to the target dataframe")

        aligned_target_df = aligned_target_df.drop(columns=ts_datetime_col)
        series_list = aligned_target_df.values.tolist()
        series_array = jnp.array(series_list)
        target_ts = TimeSeries(series_array, names=all_series_names)

        if features_df is not None:
            features_df = features_df.drop(columns=ts_datetime_col)
            features_list = features_df.values.tolist()
            features_array = jnp.array(features_list)
        else:
            features_array = None

        features_ts = TimeSeries(features_array, names=all_feats_names)


        return cls(observations=target_ts,
                   ts_datetime=target_ts_datetime,
                   features=features_ts
                   )

    @classmethod
    def from_long_pandas(cls,
                         pandas_df: pd.DataFrame,
                         target_col: str,
                         ts_datetime_col: Optional[str] = 'index',
                         feature_cols: Optional[List[str]] = None,
                         series_names_col: Optional[str] = None,
                         min_ts_datetime: Optional[Union[datetime, str]] = None,
                         max_ts_datetime: Optional[Union[datetime, str]] = None,
                         fill_missing_ts_datetime: Optional[bool] = True
                         ):

        pandas_df = pandas_df.reset_index()

        if series_names_col is None:
            warnings.warn("'series_names_col' is not specified. Assuming only one time series present in pandas "
                          "dataframe", Warning)

            series_names_col = 'series_labels'
            pandas_df[series_names_col] = 0
            all_series_names = [0]
        else:
            all_series_names = list(set(pandas_df[series_names_col].unique()))

        min_ts_datetime, max_ts_datetime, ts_periods, max_ts_length, use_dates = \
            _get_datetime_col_metadata(pandas_df, ts_datetime_col, min_ts_datetime, max_ts_datetime)

        ts_datetime = TSDateTime(min_ts_datetime, max_ts_datetime, ts_periods, max_ts_length, use_dates)

        if feature_cols is None:
            feature_cols = [col for col in pandas_df.columns if (col != target_col and col != series_names_col)]

        if fill_missing_ts_datetime:
            if feature_cols is not None:
                warnings.warn(
                    "Your features columns should not have missing values. If they do have missing values you "
                    "should treat them as target variables", Warning)

            aligned_df = _align_and_pad_long_pandas(pandas_df,
                                                    ts_datetime_col,
                                                    ts_datetime,
                                                    series_names_col,
                                                    all_series_names)
        else:
            aligned_df = pandas_df

        series_list = []
        features_list = []
        for name in all_series_names:
            series_df = aligned_df[aligned_df[series_names_col] == name]
            series_list.append(series_df[target_col].tolist())

            if feature_cols is not None:
                features_list.append(series_df[feature_cols].tolist())

        series_array = jnp.array(series_list)
        observations_ts = TimeSeries(series_array, names=all_series_names)

        if feature_cols is not None:
            features_array = jnp.array(features_list)
            features_ts = TimeSeries(features_array, names=feature_cols)
        else:
            features_ts = None

        return cls(observations=observations_ts,
                   ts_datetime=ts_datetime,
                   features=features_ts
                   )
