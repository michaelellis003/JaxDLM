import warnings
import jax.numpy as jnp


def construct_observation_vector(trend_order, seasonal_periods, num_harmonics, seasonal_representation):
    if trend_order == 0 or trend_order is None:
        obs_trend_vector = None
    else:
        obs_trend_vector = __trend_vector(trend_order)

    obs_seasonal_vectors = []
    if seasonal_periods is not None:
        for i in range(len(seasonal_periods)):
            if seasonal_representation == 'seasonal_factor':
                obs_seasonal_vectors.append(__seasonal_factor_vector(seasonal_periods[i]))
            else:  # fourier
                obs_seasonal_vectors.append(__seasonal_fourier_vector(seasonal_periods[i], num_harmonics[i]))

    # Filter out None matrices
    obs_vectors = [obs_trend_vector] + obs_seasonal_vectors
    obs_vectors = [m for m in obs_vectors if m is not None]

    obs_vector = jnp.concatenate(obs_vectors)

    return obs_vector


def __trend_vector(trend_order):
    if trend_order == 1:
        obs_trend_vector = jnp.array([1.0])
    else:
        obs_trend_vector = jnp.zeros(trend_order)
        obs_trend_vector = obs_trend_vector.at[0].set(1.0)

    return obs_trend_vector


def __seasonal_factor_vector(seasonal_periods):
    obs_seasonal_factor_vector = jnp.zeros(seasonal_periods - 1)
    obs_seasonal_factor_vector = obs_seasonal_factor_vector.at[0].set(1.0)

    return obs_seasonal_factor_vector


def __seasonal_fourier_vector(seasonal_periods, num_harmonics):
    if seasonal_periods % 2 == 0 and num_harmonics == seasonal_periods/2:
        # if seasonal_periods is even and num_harmonics is exactly half of num_harmonics then we ignore the last
        # dimension
        vector_len = (num_harmonics * 2) - 1
    else:
        vector_len = (num_harmonics * 2)

    # Create an array that alternates between 1 and 0
    obs_seasonal_fourier_vector = jnp.arange(2 * vector_len) % 2

    return obs_seasonal_fourier_vector
