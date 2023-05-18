import warnings
from jaxdlm.utils.observation_vector_utils import construct_observation_vector
from jaxdlm.utils.state_matrix_utils import construct_state_matrix


class DynamicLinearModel:
    def __init__(self,
                 trend_order=None,
                 seasonal_periods=None,
                 num_harmonics=None,
                 seasonal_representation='fourier'
                 ):

        self.trend_order = trend_order
        if isinstance(seasonal_periods, int):
            self.seasonal_periods = [seasonal_periods]
        else:
            self.seasonal_periods = seasonal_periods
        if isinstance(num_harmonics, int):
            self.num_harmonics = [num_harmonics]
        else:
            self.num_harmonics = num_harmonics
        self.seasonal_representation = seasonal_representation
        self.__validate_inputs()

        if (self.trend_order is None or self.trend_order == 0) and self.seasonal_periods is None:
            warnings.warn(f"obs_vector and state_matrix are being set to None. All arguments either 0 or None."
                          f"trend_order = {trend_order}, "
                          f"seasonal_periods = {seasonal_periods},"
                          f"num_harmonics = {num_harmonics},"
                          f"seasonal_representation = {seasonal_representation}")

            self.obs_vector = None
            self.state_matrix = None
        else:
            self.obs_vector = construct_observation_vector(self.trend_order,
                                                           self.seasonal_periods,
                                                           self.num_harmonics,
                                                           self.seasonal_representation)

            self.state_matrix = construct_state_matrix(self.trend_order,
                                                       self.seasonal_periods,
                                                       self.num_harmonics,
                                                       self.seasonal_representation)

    def __validate_inputs(self):
        def is_int_and_gt(val, min_val, arg_name):
            """Check if value is an integer and greater than min_val."""
            if val is not None:
                if not isinstance(val, int):
                    raise TypeError(f"The {arg_name} argument must be an integer greater than or equal to {min_val}.")
                if val < min_val:
                    raise ValueError(f"The {arg_name} argument must be an integer greater than or equal to {min_val}.")

        # Check trend_order
        is_int_and_gt(self.trend_order, 0, 'trend_order')

        # Check seasonal_representation
        if self.seasonal_representation not in ["seasonal_factor", "fourier", None]:
            raise ValueError("seasonal_representation argument must be either 'seasonal_factor' or 'fourier' or None.")

        # Check seasonal_periods
        if self.seasonal_representation == 'seasonal_factor':
            if self.seasonal_periods is not None:
                for sp in self.seasonal_periods:
                    is_int_and_gt(sp, 2, 'seasonal_periods')

        # Check num_harmonics
        if self.seasonal_representation == 'fourier':
            if self.num_harmonics is not None:
                for nh in self.num_harmonics:
                    is_int_and_gt(nh, 1, 'num_harmonics')

            if self.seasonal_periods is not None and self.num_harmonics is not None:
                if len(self.seasonal_periods) != len(self.num_harmonics):
                    raise ValueError("seasonal_periods and num_harmonics must have the same length.")

        if self.num_harmonics is not None and self.seasonal_representation == "seasonal_factor":
            warnings.warn("The num_harmonics argument will be ignored when using the 'seasonal_factor' representation.")

        # Check relationship between num_harmonics and seasonal_periods
        if self.seasonal_periods is not None and self.num_harmonics is not None and self.seasonal_representation == "fourier":
            for sp in self.seasonal_periods:
                for nh in self.num_harmonics:
                    if sp % 2 == 0 and nh > sp / 2:
                        raise ValueError(f"When seasonal_periods is even, num_harmonics can be at most seasonal_periods/2. "
                                         f"seasonal_periods = {sp}, num_harmonics = {nh}")
                    elif sp % 2 == 1 and nh > (sp - 1) / 2:
                        raise ValueError(f"When seasonal_periods is odd, num_harmonics can be at most (seasonal_periods - 1)/2. "
                                         f"seasonal_periods = {sp}, num_harmonics = {nh}")