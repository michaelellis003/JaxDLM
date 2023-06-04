import pytest
from jaxdlm.DLM import DynamicLinearModel
from jaxdlm.utils import observation_vector_utils
from jaxdlm.utils import state_matrix_utils
import jax.numpy as jnp
from jax.scipy.linalg import block_diag


@pytest.mark.parametrize("trend_order, seasonal_periods, num_harmonics, seasonal_representation", [
    (2, None, None, None),
    (0, None, None, None),
    (None, None, None, None),
    (None, [12], [3], 'fourier'),
    (None, [12, 7], [3, 2], 'fourier'),
    (None, [12], None, 'seasonal_factor'),
    (None, [7], None, 'seasonal_factor'),
    (2, [12], [3], 'fourier'),
    (1, [12, 7], [4, 2], 'fourier'),
    (1, [4], None, 'seasonal_factor'),
    (1, [12, 4], None, 'seasonal_factor')
])
def test_init(trend_order, seasonal_periods, num_harmonics, seasonal_representation):
    # Test case: Normal initialization
    model = DynamicLinearModel(trend_order=trend_order,
                               seasonal_periods=seasonal_periods,
                               num_harmonics=num_harmonics,
                               seasonal_representation=seasonal_representation)

    assert model.trend_order == trend_order
    assert model.seasonal_periods == seasonal_periods
    assert model.num_harmonics == num_harmonics
    assert model.seasonal_representation == seasonal_representation


@pytest.mark.parametrize("trend_order, seasonal_periods, num_harmonics, seasonal_representation, exc", [
    (-1, None, None, None, ValueError),
    (1.5, None, None, None, TypeError),
    ('abc', None, None, None, TypeError),
    (None, None, None, 'wrong', ValueError),
    (None, -1, None, 'seasonal_factor', ValueError),
    (None, 1.5, None, 'seasonal_factor', TypeError),
    (None, 'abc', None, 'seasonal_factor', TypeError),
    (None, [-1, 12], None, 'seasonal_factor', ValueError),
    (None, [1.5, 7], None, 'seasonal_factor', TypeError),
    (None, ['abc', 4], None, 'seasonal_factor', TypeError),
    (None, [12, -1], None, 'seasonal_factor', ValueError),
    (None, [7, 1.5], None, 'seasonal_factor', TypeError),
    (None, [4, 'abc'], None, 'seasonal_factor', TypeError),
    (None, -1, None, 'fourier', ValueError),
    (None, 1.5, None, 'fourier', TypeError),
    (None, 'abc', None, 'fourier', TypeError),
    (None, [-1, 12], None, 'fourier', ValueError),
    (None, [1.5, 7], None, 'fourier', TypeError),
    (None, ['abc', 4], None, 'fourier', TypeError),
    (None, [12, -1], None, 'fourier', ValueError),
    (None, [7, 1.5], None, 'fourier', TypeError),
    (None, [4, 'abc'], None, 'fourier', TypeError),
    (None, 12, -1, 'fourier', ValueError),
    (None, 12, 1.5, 'fourier', TypeError),
    (None, 12, 'abc', 'fourier', TypeError),
    (None, [12, 7], [-1, 2], 'fourier', ValueError),
    (None, [12, 7], [1.5, 2], 'fourier', TypeError),
    (None, [12, 7], ['abc', 2], 'fourier', TypeError),
    (None, [12, 7], [2, -1], 'fourier', ValueError),
    (None, [12, 7], [2, 1.5], 'fourier', TypeError),
    (None, [12, 7], [2, 'abc'], 'fourier', TypeError),
    (None, [12, 7], [1, 2, 3], 'fourier', ValueError),
    (None, [12, 7, 4], [1, 2], 'fourier', ValueError),
    (None, 12, 7, 'fourier', ValueError),
    (None, [12, 7], [7, 3], 'fourier', ValueError),
    (None, [12, 7], [3, 4], 'fourier', ValueError)
])
def test__validate_inputs(trend_order, seasonal_periods, num_harmonics, seasonal_representation, exc):
    with pytest.raises(exc):
        DynamicLinearModel(trend_order=trend_order,
                           seasonal_periods=seasonal_periods,
                           num_harmonics=num_harmonics,
                           seasonal_representation=seasonal_representation
                           )


@pytest.mark.parametrize("trend_order, trend_order_vector", [
    (1, jnp.array([1.0])),
    (2, jnp.array([1.0, 0.0])),
    (4, jnp.array([1.0, 0.0, 0.0, 0.0]))
])
def test_trend_vector(trend_order, trend_order_vector):
    assert jnp.array_equal(observation_vector_utils.__trend_vector(trend_order), trend_order_vector)


@pytest.mark.parametrize("seasonal_periods, seasonal_factor_vector", [
    (4, jnp.array([1.0, 0.0, 0.0])),
    (7, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    (12, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
])
def test_seasonal_factor_vector(seasonal_periods, seasonal_factor_vector):
    assert jnp.array_equal(observation_vector_utils.__seasonal_factor_vector(seasonal_periods), seasonal_factor_vector)


@pytest.mark.parametrize("seasonal_periods, num_harmonics, seasonal_fourier_vector", [
    (4, 1, jnp.array([1.0, 0.0])),
    (4, 2, jnp.array([1.0, 0.0, 1.0])),
    (7, 3, jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
    (12, 4, jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
    (12, 6, jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
])
def test_seasonal_fourier_vector(seasonal_periods, num_harmonics, seasonal_fourier_vector):
    assert jnp.array_equal(observation_vector_utils.__seasonal_fourier_vector(seasonal_periods, num_harmonics),
                           seasonal_fourier_vector)


@pytest.mark.parametrize("trend_order, seasonal_periods, num_harmonics, seasonal_representation, observation_vector", [
    (2, None, None, None, jnp.array([1.0, 0.0])),
    (None, [7], None, 'seasonal_factor', jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    (None, [7, 4], None, 'seasonal_factor', jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
    (None, [4], [2], 'fourier', jnp.array([1.0, 0.0, 1.0])),
    (None, [12, 4], [4, 2], 'fourier', jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
    (2, [7], None, 'seasonal_factor', jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    (2, [7, 4], None, 'seasonal_factor', jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
    (2, [12], [4], 'fourier', jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
    (2, [12], [6], 'fourier', jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
    (2, [12, 4], [4, 2], 'fourier', jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
])
def test_construct_observation_vector(trend_order, seasonal_periods, num_harmonics, seasonal_representation,
                                      observation_vector):
    assert jnp.array_equal(observation_vector_utils.construct_observation_vector(trend_order, seasonal_periods,
                                                                                 num_harmonics,
                                                                                 seasonal_representation),
                           observation_vector)


@pytest.mark.parametrize("trend_order, trend_order_matrix", [
    (1, jnp.array([[1.0]])),
    (2, jnp.array([[1.0, 1.0],
                   [0.0, 1.0]
                   ])),
    (4, jnp.array([[1.0, 1.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0]]))
])
def test_trend_matrix(trend_order, trend_order_matrix):
    assert jnp.array_equal(state_matrix_utils.__trend_matrix(trend_order), trend_order_matrix)


@pytest.mark.parametrize("seasonal_periods, seasonal_factor_order_matrix", [
    (2, jnp.array([[-1]])),
    (4, jnp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0]]))
])
def test_seasonal_factor_matrix(seasonal_periods, seasonal_factor_order_matrix):
    assert jnp.array_equal(state_matrix_utils.__seasonal_factor_matrix(seasonal_periods), seasonal_factor_order_matrix)


@pytest.mark.parametrize("seasonal_periods, num_harmonics, seasonal_fourier_matrix", [
    (4, 2, jnp.array([[jnp.cos(2 * jnp.pi / 4), jnp.sin(2 * jnp.pi / 4), 0],
                      [-jnp.sin(2 * jnp.pi / 4), jnp.cos(2 * jnp.pi / 4), 0],
                      [0, 0, -1]
                      ])
     ),
    (7, 2, block_diag(jnp.array([[jnp.cos(2 * jnp.pi / 7), jnp.sin(2 * jnp.pi / 7)],
                                 [-jnp.sin(2 * jnp.pi / 7), jnp.cos(2 * jnp.pi / 7)]]),
                      jnp.array([[jnp.cos(2 * jnp.pi * 2 / 7), jnp.sin(2 * jnp.pi * 2 / 7)],
                                 [-jnp.sin(2 * jnp.pi * 2 / 7), jnp.cos(2 * jnp.pi * 2 / 7)]])
                      )
     ),
    (12, 2, block_diag(jnp.array([[jnp.cos(2 * jnp.pi / 12), jnp.sin(2 * jnp.pi / 12)],
                                  [-jnp.sin(2 * jnp.pi / 12), jnp.cos(2 * jnp.pi / 12)]]),
                       jnp.array([[jnp.cos(2 * jnp.pi * 2 / 12), jnp.sin(2 * jnp.pi * 2 / 12)],
                                  [-jnp.sin(2 * jnp.pi * 2 / 12), jnp.cos(2 * jnp.pi * 2 / 12)]])
                       )
     )
])
def test__seasonal_fourier_matrix(seasonal_periods, num_harmonics, seasonal_fourier_matrix):
    assert jnp.array_equal(state_matrix_utils.__seasonal_fourier_matrix(seasonal_periods, num_harmonics),
                           seasonal_fourier_matrix)


@pytest.mark.parametrize("trend_order, seasonal_periods, num_harmonics, seasonal_representation, state_matrix", [
    (None, [4], None, 'seasonal_factor', state_matrix_utils.__seasonal_factor_matrix(4)),
    (None, [7, 4], None, 'seasonal_factor', block_diag(state_matrix_utils.__seasonal_factor_matrix(7),
                                                       state_matrix_utils.__seasonal_factor_matrix(4)
                                                       ),
     ),
    (None, [12, 4], [3, 2], 'fourier', block_diag(state_matrix_utils.__seasonal_fourier_matrix(12, 3),
                                                  state_matrix_utils.__seasonal_fourier_matrix(4, 2)
                                                  )
     ),
    (2, None, None, None, state_matrix_utils.__trend_matrix(2)),
    (2, [4], None, 'seasonal_factor', block_diag(state_matrix_utils.__trend_matrix(2),
                                                 state_matrix_utils.__seasonal_factor_matrix(4))
     ),
    (2, [7, 4], None, 'seasonal_factor', block_diag(state_matrix_utils.__trend_matrix(2),
                                                    state_matrix_utils.__seasonal_factor_matrix(7),
                                                    state_matrix_utils.__seasonal_factor_matrix(4),
                                                    )
     ),
    (2, [12], [3], 'fourier', block_diag(state_matrix_utils.__trend_matrix(2),
                                         state_matrix_utils.__seasonal_fourier_matrix(12, 3))
     ),
    (2, [12, 7], [3, 2], 'fourier', block_diag(state_matrix_utils.__trend_matrix(2),
                                               state_matrix_utils.__seasonal_fourier_matrix(12, 3),
                                               state_matrix_utils.__seasonal_fourier_matrix(7, 2)
                                               )
     )
])
def test_construct_state_matrix(trend_order, seasonal_periods, num_harmonics, seasonal_representation, state_matrix):
    assert jnp.array_equal(state_matrix_utils.construct_state_matrix(trend_order, seasonal_periods, num_harmonics,
                                                                     seasonal_representation),
                           state_matrix)
