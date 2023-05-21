import jax.numpy as jnp
from datetime import datetime, timedelta
import warnings


class TimeSeries:
    def __init__(self, series_name, time_series):
        self.series_name = series_name
        self.time_series = jnp.array(time_series)


class Regressors:
    def __init__(self, series_name, regressors, regressors_names=None):
        self.series_name = series_name
        if regressors is None:
            if regressors_names is not None:
                warnings.warn('regressors is None and regressors_names is not None. Ignoring regressors_names.')
            self.regressors = None
            self.regressor_names = None
        else:
            self.regressors = jnp.array(regressors)
            self.regressor_names = regressors_names if regressors_names is not None else [f"{series_name}_X_{i}" for i
                                                                                          in range(regressors.shape[1])]


class TimeFrame:
    def __init__(self):
        self.time_series = {}
        self.regressors = {}
        self.ts_datetime = []
        self.auto_ts_id = 0

    def _align_and_pad_time_series(self, time_series, time):
        # Create a dictionary where keys are dates and values are observations
        obs_dict = {date: obs for date, obs in zip(time, time_series)}

        # Create the full date range
        full_date_range = self.ts_datetime

        # Create a new list of observations that will be aligned with the full date range
        aligned_observations = []
        for date in full_date_range:
            if date in obs_dict:
                aligned_observations.append(obs_dict[date])
            else:
                aligned_observations.append(jnp.nan)
                warnings.warn(f"Missing observation for {date}, imputing with NaN.")

        return jnp.array(aligned_observations)

    def _align_and_pad_regressors(self, regressors, time):
        # Create a dictionary where keys are dates and values are regressors
        reg_dict = {date: reg for date, reg in zip(time, regressors)}

        # Create the full date range
        full_date_range = self.ts_datetime

        # Create new lists of regressors that will be aligned with the full date range
        aligned_regressors = []
        for date in full_date_range:
            if date in reg_dict:
                aligned_regressors.append(reg_dict[date])
            else:
                # Impute with jnp.nan for each regressor
                aligned_regressors.append([jnp.nan for _ in range(len(regressors[0]))])
                warnings.warn(f"Missing regressor data for {date}, imputing with NaN.")

        return jnp.array(aligned_regressors)

    def add_time_series(self, time_series, max_ts_length, ts_datetime=None, data_frequency=None, series_name=None,
                        regressors=None,
                        regressors_names=None):
        if data_frequency is not None and not isinstance(data_frequency, timedelta):
            raise ValueError('data_frequency must be a datetime.timedelta object')

        if series_name in self.time_series:
            raise ValueError(f"Time series with name {series_name} already exists")
        elif series_name is None:
            series_name = self.auto_ts_id
            self.auto_ts_id += 1

        if ts_datetime is None and not ts_datetime:
            warnings.warn("Time is not provided, defaulting to sequence of integers.")
            self.ts_datetime = list(range(max_ts_length - len(time_series), max_ts_length))
        else:
            self.ts_datetime = time = [parse_time_string(t) if isinstance(t, str) else t for t in ts_datetime]

            if data_frequency is None:
                data_frequency = ts_datetime[-1] - ts_datetime[-2]
                warnings.warn(f"data_frequency is not provided, defaulting to inferred frequency of {data_frequency}.")

            if self.ts_datetime is None:
                self.ts_datetime = [ts_datetime[-1] - data_frequency * i for i in range(max_ts_length)][::-1]

        # Create the new time series and regressors
        new_time_series = TimeSeries(series_name, time_series)
        new_regressors = Regressors(series_name, regressors, regressors_names) if regressors is not None else None

        # Adjust the length of the new time series and time to match the maximum length
        new_time_series.time_series = self._align_and_pad_time_series(new_time_series.time_series, self.ts_datetime)
        if regressors is None:
            new_regressors.regressors = None
        else:
            new_regressors.regressors = self._align_and_pad_regressors(new_regressors.regressors, self.ts_datetime)

        self.time_series[series_name] = TimeSeries(series_name, time_series)
        self.regressors[series_name] = Regressors(series_name, regressors, regressors_names)

    def get_time_series(self, series_name):
        return self.time_series.get(series_name, None)

    def get_regressors(self, series_name):
        return self.regressors.get(series_name, None)


def parse_time_string(t):
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%Y %H:%M:%S', '%Y-%m', '%Y']
    for fmt in formats:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError(f"No known format to parse date string: {t}")
