import pytest
from jaxdlm.dlm import DynamicLinearModel
import jax.numpy as jnp


########## TREND ONLY MODEL TESTS ##########
def test_init_with_negative_trend_order():
    with pytest.raises(ValueError):
        DynamicLinearModel(trend_order=-1)


def test_init_with_non_integer_trend_order():
    with pytest.raises(TypeError):
        DynamicLinearModel(trend_order=1.5)


def test_init_with_local_level_trend_order():
    model = DynamicLinearModel(trend_order=1)
    assert model.trend_order == 1

    assert jnp.array_equal(model.obs_vector, jnp.array([1.0]))
    assert jnp.array_equal(model.state_matrix, jnp.array([[1.0]]))


def test_init_with_local_linear_trend_order():
    model = DynamicLinearModel(trend_order=2)
    assert model.trend_order == 2
    assert jnp.array_equal(model.obs_vector, jnp.array([1.0, 0.0]))
    assert jnp.array_equal(model.state_matrix, jnp.array([[1.0, 1.0], [0.0, 1.0]]))


def test_init_with_large_trend_order():
    model = DynamicLinearModel(trend_order=4)
    assert model.trend_order == 4

    assert jnp.array_equal(model.obs_vector, jnp.array([1.0, 0.0, 0.0, 0.0]))
    assert jnp.array_equal(model.state_matrix, jnp.array([[1.0, 1.0, 0.0, 0.0],
                                                          [0.0, 1.0, 1.0, 0.0],
                                                          [0.0, 0.0, 1.0, 1.0],
                                                          [0.0, 0.0, 0.0, 1.0],
                                                          ]))


def test_init_with_zero_order():
    model = DynamicLinearModel(trend_order=0)
    assert model.trend_order == 0
    assert model.obs_vector is None
    assert model.state_matrix is None


########## SEASONAL FACTOR ONLY MODEL TESTS ##########
def test_init_with_wrong_seasonal_representation():
    with pytest.raises(ValueError):
        DynamicLinearModel(seasonal_representation='wrong')


def test_init_with_negative_seasonal_periods_seasonal_factor():
    with pytest.raises(ValueError):
        DynamicLinearModel(seasonal_periods=-1, seasonal_representation='seasonal_factor')


def test_init_with_non_integer_seasonal_periods_seasonal_factor():
    with pytest.raises(TypeError):
        DynamicLinearModel(seasonal_periods=1.5, seasonal_representation='seasonal_factor')


def test_init_with_quarterly_seasonal_periods_seasonal_factor():
    model = DynamicLinearModel(seasonal_periods=4, seasonal_representation='seasonal_factor')
    assert model.seasonal_periods == 4

    print(model.state_matrix)
    assert jnp.array_equal(model.obs_vector, jnp.array([1.0, 0.0, 0.0]))
    assert jnp.array_equal(model.state_matrix, jnp.array([[-1, -1, -1],
                                                          [1, 0, 0],
                                                          [0, 1, 0]]
                                                         )
                           )


def test_init_with_weekly_seasonal_periods_seasonal_factor():
    model = DynamicLinearModel(seasonal_periods=7, seasonal_representation='seasonal_factor')
    assert model.seasonal_periods == 7

    assert jnp.array_equal(model.obs_vector, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert jnp.array_equal(model.state_matrix, jnp.array([[-1, -1, -1, -1, -1, -1],
                                                          [1, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 1, 0, 0],
                                                          [0, 0, 0, 0, 1, 0]
                                                          ]
                                                         )
                           )


def test_init_with_yearly_seasonal_periods_seasonal_factor():
    model = DynamicLinearModel(seasonal_periods=12, seasonal_representation='seasonal_factor')
    assert model.seasonal_periods == 12

    assert jnp.array_equal(model.obs_vector, jnp.array([1.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0,
                                                        0.0, 0.0
                                                        ]
                                                       )
                           )
    assert jnp.array_equal(model.state_matrix, jnp.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                                                          ]
                                                         )
                           )

########## SEASONAL FACTOR ONLY MODEL TESTS ##########
def test_init_with_non_integer_seasonal_periods_seasonal_factor():
    with pytest.raises(TypeError):
        DynamicLinearModel(seasonal_periods=1.5, seasonal_representation='seasonal_factor')