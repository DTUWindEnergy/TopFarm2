import warnings
import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

from topfarm.constraint_components.load import (
    SurrogateModel,
    compute_error,
    predict_output,
    predict_gradient)


def test_SurrogateModel():
    """
    Test SurrogateModel class.
    """

    obj = SurrogateModel(model=MLPRegressor(),
                         input_scaler=MinMaxScaler(),
                         output_scaler=StandardScaler(),
                         input_channel_names=['x0', 'x1'],
                         output_channel_name='y')

    assert obj is not None


def test_compute_error():
    """
    Basic test of compute_error().
    """

    data_1 = np.array([2.0, 2.0])
    data_2 = np.array([1.0, 3.0])

    abs_err, rel_err = compute_error(data_1, data_2)

    assert np.array_equal(abs_err, np.array([-1.0, 1.0]))
    assert np.array_equal(rel_err, np.array([-0.5, 0.5]))


def test_predict_output_1():
    """
    Basic test of predict_output().
    """

    prng = RandomState(seed=0)
    input = prng.rand(5, 2)

    def model(input):
        return (input[:, 0] * input[:, 1]).reshape(-1, 1)

    output_ok = np.array(model(input))
    output_try, _ = predict_output(model, input)

    assert np.array_equal(output_ok, output_try)


def test_predict_output_2():
    """
    Test of predict_output() when input is a dict.
    """

    input = {'x0': [1.0, 2.0, 3.0],
             'x1': [4.0, 5.0, 6.0]}

    def model(input):
        return (input[:, 0] * input[:, 1]).reshape(-1, 1)

    output_ok = np.array(model(np.array([input['x0'], input['x1']]).transpose()))
    output_try, _ = predict_output(model, input, model_in_keys=list(input))

    assert np.array_equal(output_ok, output_try)


def test_predict_output_with_input_scaler():
    """
    Test with an input scaler.
    """

    prng = RandomState(seed=0)
    input = -100.0 + 200.0 * prng.rand(5, 2)
    input_scaler = MinMaxScaler()
    input_scaled = input_scaler.fit_transform(input)

    def model(input):
        return (input[:, 0] * input[:, 1]).reshape(-1, 1)

    output_ok = model(input_scaled)
    output_try, _ = predict_output(model, input, input_scaler=input_scaler)

    assert np.array_equal(output_ok, output_try)


def test_predict_output_with_output_scaler():
    """
    Test with an output scaler.
    """

    prng = RandomState(seed=0)
    input = prng.rand(5, 2)

    def model(input):
        return (- 100.0 + 200.0 * input[:, 0] * input[:, 1]).reshape(-1, 1)

    output_ok = model(input)
    output_scaler_mm = MinMaxScaler()
    output_scaler_std = StandardScaler()
    output_scaler_mm.fit(output_ok)
    output_scaler_std.fit(output_ok)

    def model_scaled_mm(input):
        output = model(input)
        return output_scaler_mm.transform(output)

    def model_scaled_std(input):
        output = model(input)
        return output_scaler_std.transform(output)

    output_try_mm, _ = predict_output(model_scaled_mm, input,
                                      output_scaler=output_scaler_mm)
    output_try_std, _ = predict_output(model_scaled_std, input,
                                       output_scaler=output_scaler_std)
    output_try_mm = output_try_mm
    output_try_std = output_try_std

    assert np.allclose(output_ok, output_try_mm)
    assert np.allclose(output_ok, output_try_std)


def test_wrong_input_type():
    """
    Test of predict_output() and predict_gradient() when input has the wrong
    type.
    """

    input = pd.DataFrame()

    def model():
        pass

    try:
        predict_output(model, input)
        assert False
    except TypeError:
        assert True

    try:
        predict_gradient(model, input)
        assert False
    except TypeError:
        assert True


def test_boundary():
    """
    Test of predict_output() when there is a boundary.
    """

    prng = RandomState(seed=0)
    input = -1 + 2 * prng.rand(10, 1)
    input[2] = 10
    input[4] = -5

    def model(input):
        return 2 * input

    def bound_fun(x):
        return np.abs(x) <= 1

    with warnings.catch_warnings(record=True) as w:
        _, extrapolation_sample = predict_output(
            model, input, boundary=bound_fun)
        assert np.array_equal(extrapolation_sample, [2, 4])
        assert 'evaluated outside' in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        _, extrapolation_sample = predict_gradient(
            model, input, boundary=bound_fun)
        assert np.array_equal(extrapolation_sample, [2, 4])
        assert 'evaluated outside' in str(w[-1].message)


def test_wrong_model_type():
    """
    Test of predict_output() and predict_gradient() when the model type is
    unknown.
    """

    input = np.random.rand(5, 1)
    input_scaler = MinMaxScaler()
    input_scaler.fit(input)

    model = MLPClassifier()

    try:
        predict_output(model, input)
        assert False
    except KeyError:
        assert True

    try:
        predict_gradient(model, input)
        assert False
    except KeyError:
        assert True

    try:
        predict_gradient(model, input, input_scaler=input_scaler)
        assert False
    except KeyError:
        assert True


def test_predict_gradient_1():
    """
    Basic test of predict_gradient
    """

    input = {'x0': [1.0, 2.0, 3.0],
             'x1': [4.0, 5.0, 6.0]}

    def fun(input):
        x0, x1 = input.T
        return x0 ** 2 * np.sin(x1)

    def Dfun_Dx0(input):
        x0, x1 = input.T
        return 2 * x0 * np.sin(x1)

    def Dfun_Dx1(input):
        x0, x1 = input.T
        return x0 ** 2 * np.cos(x1)

    def Dfun(input):
        x0, x1 = input.T
        return np.column_stack((2 * x0 * np.sin(x1), x0 ** 2 * np.cos(x1)))

    grad_x0, _ = predict_output(Dfun_Dx0, input, model_in_keys=list(input))
    grad_x1, _ = predict_output(Dfun_Dx1, input, model_in_keys=list(input))
    grad_ok = np.column_stack((grad_x0, grad_x1))

    grad_test, _ = predict_gradient(Dfun, input, model_in_keys=['x0', 'x1'])

    assert np.array_equal(grad_ok, grad_test)


def test_predict_gradient_with_input_scaler():
    """
    Test predict_gradient with an input scaler.
    """

    prng = RandomState(seed=0)
    input = -100.0 + 200.0 * prng.rand(5, 2)
    input_scaler_mm = MinMaxScaler()
    input_scaled_mm = input_scaler_mm.fit_transform(input)
    input_scaler_std = StandardScaler()
    input_scaled_std = input_scaler_std.fit_transform(input)

    def fun(input):
        x0, x1 = input.T
        return x0 ** 2 * np.sin(x1)

    def Dfun_Dx0(input):
        x0, x1 = input.T
        return 2 * x0 * np.sin(x1)

    def Dfun_Dx1(input):
        x0, x1 = input.T
        return x0 ** 2 * np.cos(x1)

    def Dfun(input):
        x0, x1 = input.T
        return np.column_stack((2 * x0 * np.sin(x1), x0 ** 2 * np.cos(x1)))

    # Test input_scaler_mm
    grad_x0, _ = predict_output(Dfun_Dx0, input_scaled_mm)
    grad_x1, _ = predict_output(Dfun_Dx1, input_scaled_mm)
    grad_ok = np.column_stack((grad_x0, grad_x1)) * input_scaler_mm.scale_

    grad_test, _ = predict_gradient(Dfun, input, input_scaler=input_scaler_mm)

    assert np.array_equal(grad_ok, grad_test)

    # Test input_scaler_std
    grad_x0, _ = predict_output(Dfun_Dx0, input_scaled_std)
    grad_x1, _ = predict_output(Dfun_Dx1, input_scaled_std)
    grad_ok = np.column_stack((grad_x0, grad_x1)) / input_scaler_std.scale_

    grad_test, _ = predict_gradient(Dfun, input, input_scaler=input_scaler_std)

    assert np.array_equal(grad_ok, grad_test)


def test_predict_gradient_with_output_scaler():
    """
    Test predict_gradient with an output scaler.
    """

    prng = RandomState(seed=0)
    input = prng.rand(5, 2)

    def fun(input):
        x0, x1 = input.T
        return (x0 ** 2 * np.sin(x1)).reshape(-1, 1)

    output_ok = fun(input)
    output_scaler_mm = MinMaxScaler()
    output_scaler_std = StandardScaler()
    output_scaler_mm.fit(output_ok)
    output_scaler_std.fit(output_ok)

    def Dfun_Dx0(input):
        x0, x1 = input.T
        return 2 * x0 * np.sin(x1)

    def Dfun_Dx1(input):
        x0, x1 = input.T
        return x0 ** 2 * np.cos(x1)

    def Dfun(input):
        x0, x1 = input.T
        return np.column_stack((2 * x0 * np.sin(x1), x0 ** 2 * np.cos(x1)))

    # Test input_scaler_mm
    grad_x0, _ = predict_output(Dfun_Dx0, input)
    grad_x1, _ = predict_output(Dfun_Dx1, input)
    grad_ok = np.column_stack((grad_x0, grad_x1)) / output_scaler_mm.scale_

    grad_test, _ = predict_gradient(Dfun, input, output_scaler=output_scaler_mm)

    assert np.array_equal(grad_ok, grad_test)

    # Test input_scaler_std
    grad_x0, _ = predict_output(Dfun_Dx0, input)
    grad_x1, _ = predict_output(Dfun_Dx1, input)
    grad_ok = np.column_stack((grad_x0, grad_x1)) * output_scaler_std.scale_

    grad_test, _ = predict_gradient(Dfun, input, output_scaler=output_scaler_std)

    assert np.array_equal(grad_ok, grad_test)


def test_MLPRegressor():
    """
    Test predict_output() when model is a MLPRegressor.
    """

    prng = RandomState(seed=0)
    input = prng.rand(100, 1)
    output = 2.0 * input
    model = MLPRegressor(
        hidden_layer_sizes=(1),
        solver='lbfgs',
        alpha=1e-4,
        max_iter=3000,
        activation='identity',
        learning_rate='adaptive')
    model.fit(input, output.ravel())

    predicted_output, _ = predict_output(model, np.array(input))

    assert np.allclose(output, predicted_output, atol=1.e-3)

#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots()
#    ax.scatter(input, output)
#    ax.scatter(input, predicted_output)
