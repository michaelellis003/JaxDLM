from jaxdlm.utils import state_matrix_utils, observation_vector_utils

trend_order = None
seasonal_periods = 4
num_harmonics = None
seasonal_representation = 'seasonal_factor'
state_matrix = state_matrix_utils.construct_state_matrix(trend_order,
                                                         seasonal_periods,
                                                         num_harmonics,
                                                         seasonal_representation)

print(state_matrix)
print(state_matrix.shape)