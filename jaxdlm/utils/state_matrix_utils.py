import warnings
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

def construct_state_matrix(trend_order, seasonal_periods, num_harmonics, seasonal_representation):
    if trend_order == 0 or trend_order is None:
        state_trend_matrix = None
    else:
        state_trend_matrix = __trend_matrix(trend_order)

    if seasonal_periods is None:
        state_seasonal_matrix = None
    else:
        if seasonal_representation == 'seasonal_factor':
            state_seasonal_matrix = __seasonal_factor_matrix(seasonal_periods)
        else: # fourier
            state_seasonal_matrix = _seasonal_fourier_matrix(seasonal_periods, num_harmonics)

    # Filter out None matrices and construct the block diagonal state matrix
    state_matrices = [state_trend_matrix, state_seasonal_matrix]
    state_matrices = [m for m in state_matrices if m is not None]
    state_matrix = block_diag(*state_matrices)

    return state_matrix


def __trend_matrix(trend_order):
    state_trend_matrix = jnp.eye(trend_order)

    if trend_order != 1:
        state_trend_matrix = state_trend_matrix.at[jnp.arange(trend_order - 1), jnp.arange(1, trend_order)].set(1.0)

    return state_trend_matrix


def __seasonal_factor_matrix(seasonal_periods):
    seasonal_factor_matrix_dim = seasonal_periods - 1

    below_diagonal = jnp.eye(seasonal_factor_matrix_dim, k=-1)

    # Create a matrix of -1s for the top row
    top_row = -jnp.ones((1, seasonal_factor_matrix_dim))

    # Create a matrix of zeros for the remaining rows
    remaining_rows = jnp.zeros((seasonal_factor_matrix_dim - 1, seasonal_factor_matrix_dim))

    # Stack the top row and remaining rows vertically to create the top part of the state matrix
    top_part = jnp.vstack((top_row, remaining_rows))

    # Add the below-diagonal ones to the top part to create the final state matrix
    state_seasonal_factor_matrix = top_part + below_diagonal

    return state_seasonal_factor_matrix

def _seasonal_fourier_matrix(seasonal_periods, num_harmonics):
    last_h_nyquist_frequency = False
    if seasonal_periods % 2 == 0 and num_harmonics == seasonal_periods/2:
        # if seasonal_periods is even and num_harmonics is exactly half of num_harmonics then we ignore the last
        # dimension
        last_h_nyquist_frequency = True

    # For each harmonic, construct the Hj matrix and add it to the state matrix
    h_list = []
    for j in range(num_harmonics):
        if j+1 == num_harmonics and last_h_nyquist_frequency:
            h_j = jnp.array([[-1.0]])
        else:
            wj = 2 * jnp.pi * (j + 1) / seasonal_periods
            h_j = jnp.array([[jnp.cos(wj), jnp.sin(wj)], [-jnp.sin(wj), jnp.cos(wj)]])

        h_list.append(h_j)

    state_seasonal_fourier_matrix = block_diag(*h_list)

    return state_seasonal_fourier_matrix
