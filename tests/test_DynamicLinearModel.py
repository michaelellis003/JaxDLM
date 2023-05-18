import pytest
from jaxdlm.dlm import DynamicLinearModel
import jax.numpy as jnp


########## TREND ONLY MODEL TESTS ##########
@pytest.mark.parametrize("trend_order, exc", [
    (-1, ValueError),
    (1.5, TypeError),
    ('abc', TypeError)
])
def test_init_with_wrong_trend_order(trend_order, exc):
    with pytest.raises(exc):
        DynamicLinearModel(trend_order=trend_order)


@pytest.mark.parametrize("trend_order, obs_vector, state_matrix", [
    (1, jnp.array([1.0]), jnp.array([[1.0]])),
    (2, jnp.array([1.0, 0.0]), jnp.array([[1.0, 1.0], [0.0, 1.0]])),
    (4, jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([[1.0, 1.0, 0.0, 0.0],
                                                    [0.0, 1.0, 1.0, 0.0],
                                                    [0.0, 0.0, 1.0, 1.0],
                                                    [0.0, 0.0, 0.0, 1.0]]))
])
def test_init_with_valid_trend_order(trend_order, obs_vector, state_matrix):
    model = DynamicLinearModel(trend_order=trend_order)
    print(model.obs_vector)
    print(model.state_matrix)
    assert model.trend_order == trend_order
    assert jnp.array_equal(model.obs_vector, obs_vector)
    assert jnp.array_equal(model.state_matrix, state_matrix)


########## SEASONAL FACTOR ONLY MODEL TESTS ##########
def test_init_with_wrong_seasonal_representation():
    with pytest.raises(ValueError):
        DynamicLinearModel(seasonal_representation='wrong')


@pytest.mark.parametrize("seasonal_periods, exc", [
    (-1, ValueError),
    (1.5, TypeError),
    ('abc', TypeError)
])
def test_init_with_wrong_seasonal_periods_seasonal_factor(seasonal_periods, exc):
    with pytest.raises(exc):
        DynamicLinearModel(seasonal_periods=seasonal_periods, seasonal_representation='seasonal_factor')


@pytest.mark.parametrize("seasonal_periods, obs_vector, state_matrix", [
    # include cases for each seasonal period as per your requirement
    (4, jnp.array([1.0, 0.0, 0.0]), jnp.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0]]))
])
def test_init_with_valid_seasonal_periods_seasonal_factor(seasonal_periods, obs_vector, state_matrix):
    model = DynamicLinearModel(seasonal_periods=seasonal_periods, seasonal_representation='seasonal_factor')
    assert model.seasonal_periods == seasonal_periods
    assert jnp.array_equal(model.obs_vector, obs_vector)
    assert jnp.array_equal(model.state_matrix, state_matrix)


########## TREND AND SEASONAL FACTOR MODEL TESTS ##########
@pytest.mark.parametrize("trend_order, seasonal_periods, obs_vector, state_matrix", [
    (1, 4, jnp.array([1.0, 1.0, 0.0, 0.0]), jnp.array([[1, 0, 0, 0], [0, -1, -1, -1], [0, 1, 0, 0], [0, 0, 1, 0]]))
])
def test_init_with_trend_order_and_seasonal_periods(trend_order, seasonal_periods, obs_vector, state_matrix):
    model = DynamicLinearModel(trend_order=trend_order, seasonal_periods=seasonal_periods, seasonal_representation='seasonal_factor')
    assert model.trend_order == trend_order
    assert model.seasonal_periods == seasonal_periods
    assert jnp.array_equal(model.obs_vector, obs_vector)
    assert jnp.array_equal(model.state_matrix, state_matrix)

@pytest.mark.parametrize("trend_order, seasonal_periods, obs_vector, state_matrix", [
    (0, None, None, None),
    (None, None, None, None)
])
def test_init_with_none_trend_order_seasonal_periods(trend_order, seasonal_periods, obs_vector, state_matrix):
    with pytest.warns(UserWarning):
        model = DynamicLinearModel(trend_order=trend_order,
                                   seasonal_periods=seasonal_periods)

        assert model.trend_order == trend_order
        assert model.seasonal_periods == seasonal_periods
        assert model.obs_vector == obs_vector
        assert model.state_matrix == state_matrix