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
        self.seasonal_periods = seasonal_periods
        self.num_harmonics = num_harmonics
        self.seasonal_representation = seasonal_representation
        self.__validate_inputs()

        self.obs_vector = construct_observation_vector(self.trend_order,
                                                       self.seasonal_periods,
                                                       self.num_harmonics,
                                                       self.seasonal_representation)

        self.state_matrix = construct_state_matrix(self.trend_order,
                                                   self.seasonal_periods,
                                                   self.num_harmonics,
                                                   self.seasonal_representation)

    def __validate_inputs(self):
        if self.trend_order is not None and not isinstance(self.trend_order, int):
            raise TypeError("The trend_order argument must be an integer greater than or equal to 0.")

        if self.trend_order is not None and self.trend_order < 0:
            raise ValueError("The trend_order argument must be an integer greater than or equal to 0.")

        if self.seasonal_representation not in ["seasonal_factor", "fourier", None]:
            raise ValueError("seasonal_representation argument must be either 'seasonal_factor' or 'fourier' or None.")

        if self.seasonal_representation == 'seasonal_factor' and not isinstance(self.seasonal_periods, int):
            raise TypeError("The seasonal_periods argument must be an integer greater than or equal to 2 when using "
                            "seasonal_representation == 'seasonal_factor'.")

        if self.seasonal_periods is not None and self.seasonal_periods < 2:
            raise ValueError("The seasonal_periods argument must be greater than or equal to 2.")

        if self.num_harmonics is not None and not isinstance(self.num_harmonics, int):
            raise ValueError("The num_harmonics argument must be an integer greater than or equal to 1.")

        if self.num_harmonics is not None and self.num_harmonics < 1:
            raise ValueError("The num_harmonics argument must be an integer greater than or equal to 1.")

        if self.num_harmonics is not None and self.seasonal_representation == "seasonal_factor":
            warnings.warn("The num_harmonics argument will be ignored when using the 'seasonal_factor' representation.")

        if self.seasonal_periods is not None and self.num_harmonics is not None \
                and self.seasonal_representation == "fourier":

            if self.seasonal_periods % 2 == 0 and self.num_harmonics > self.seasonal_periods / 2:
                raise ValueError("When seasonal_periods is even, num_harmonics can be at most seasonal_periods/2.")
            elif self.seasonal_periods % 2 == 1 and self.num_harmonics > (self.seasonal_periods - 1) / 2:
                raise ValueError("When seasonal_periods is odd, num_harmonics can be at most (seasonal_periods - 1)/2.")