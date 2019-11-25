# %% Import section.

"""
This module contains classes and functions needed to form the loads constraint.
"""

import warnings
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import topfarm


# %% Generic stuff.

def compute_error(res_1, res_2):
    """
    Compute the error between two set of data.

    Parameters
    ----------
    res_1 : numpy array or pandas DataFrame
        The reference results.
    res_2 : numpy array or pandas DataFrame
        The new results.

    Returns
    -------
    abs_err : numpy array or pandas DataFrame
        The absolute error between the results.
    rel_err : numpy array or pandas DataFrame
        The relative error between the results.

    """
    abs_err = res_2 - res_1
    rel_err = res_2 / res_1 - 1.0
    return abs_err, rel_err


# Dictionary used by predict_output() to call the correct function for
# each model type.
_switch_output = {}

# Dictionary used by predict_gradient() to call the correct function for
# each model type.
_switch_gradient = {}


# %% Add python functions to the switch dictionaries.

def _predict_output_python_function(model, input):
    """
    Predict output function for python functions.
    """
    output = model(input)
    return output


def _predict_gradient_python_function(model, input):
    """
    Predict gradient function for python functions.
    """
    output = model(input)
    return output


_switch_output[type(lambda: None)] = _predict_output_python_function
_switch_gradient[type(lambda: None)] = _predict_gradient_python_function


# %% Add scikit-learn neural networks to the switch dictionaries.

def _predict_output_scikit_MLPRegressor(model, input):
    """
    Predict output function for scikit-learn MLPRegressor objects.
    """
    output = model.predict(input)
    # Ensure that output is 2D even when there is only one output channel.
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    return output


_switch_output[MLPRegressor] = _predict_output_scikit_MLPRegressor


# %% Add plugins to the switch dictionaries.

if 'openturnsloads' in topfarm.plugins:

    from openturnsloads.load import update_switch
    _switch_output_update, _switch_gradient_update = update_switch()
    _switch_output.update(_switch_output_update)
    _switch_gradient.update(_switch_gradient_update)


if 'tensorflowloads' in topfarm.plugins:

    from tensorflowloads.load import update_switch
    _switch_output_update, _switch_gradient_update = update_switch()
    _switch_output.update(_switch_output_update)
    _switch_gradient.update(_switch_gradient_update)

if 'wind2loads' in topfarm.plugins:

    from wind2loads.load import update_switch
    _switch_output_update, _switch_gradient_update = update_switch()
    _switch_output.update(_switch_output_update)
    _switch_gradient.update(_switch_gradient_update)


# %% Functions for evaluating surrogate models.

unknown_model_exception = KeyError(
    'Unknown model type. Please extend _switch_output '
    'with the appropriate predict function.')


def predict_output(model,
                   input,
                   model_in_keys=None,
                   input_scaler=None,
                   output_scaler=None,
                   boundary=None):
    """
    Predict the response of a model.

    Parameters
    ----------
    model : python function
            scikit-learn MLPRegressor
        The model to be evaluated, which can be a Multiple Input Multiple Output.
        Support for additional model types is provided through the Loads cutting
        edge plugins. model must return a 2D array, where each row is a
        different sample, and each column a different output.

    input : numpy.ndarray
            dict
        Input on which evaluate the model. If it is a 2D array, then it is
        assumed that each row is a different sample, and each column a
        different input variable. If instead it is a dictionary, then its
        keys are compared against the ones in model_in_keys, and a 2D
        array is assembled afterwards.

    model_in_keys : list
        List of input variables of the surrogate model. This variable is
        used only if input is a dictionary.

    input_scaler : sklearn.preprocessing.StandardScaler
                   sklearn.preprocessing.MinMaxScaler
        The model might have been constructed on a scaled input. This
        parameter ensures that input is scaled before evaluating the model.

    output_scaler : sklearn.preprocessing.StandardScaler
                    sklearn.preprocessing.MinMaxScaler
        The model might have been constructed on a scaled output. This
        parameter ensures that the predicted output is scaled back to its
        original dimensions.

    boundary : function
        A function that takes as input a 2D numpy array and retuns a list,
        with True if a point lies within the boudary, and False otherwise. A
        warning is issued if a point lies outside of the boundary.

    Returns
    -------
    output : numpy.ndarray
        Model output, optionally scaled through output_scaler.
        2D array, where each row is a different sample, and each column a
        different output.

    extrapolation_sample : list
        Identifiers of the points outside of the boundary.


    Raises
    ------
    KeyError : if the model is not supported.
    TypeError : if input is not a dictionary or a numpy array.
    Warning: if some points are outside of the boundary.

    """

    # Form the input array.
    if type(input) is dict:
        input_array = np.column_stack(
            [input[key] for key in model_in_keys])
    elif type(input) is np.ndarray:
        input_array = input
    else:
        raise TypeError(
            'Parameter input must be a dictionary or a numpy array.')

    # Check for extrapolation.
    if boundary is not None:
        extrapolation_sample = np.where(~boundary(input_array))[0].tolist()
    else:
        extrapolation_sample = []
    if extrapolation_sample:
        warnings.warn('Surrogate evaluated outside of input domain. See extrapolation_sample.')

    # If possible scale the input.
    if input_scaler is not None:
        input_scaler.copy = True
        input_scaled = input_scaler.transform(input_array)
    else:
        input_scaled = input_array

    # Predict the output by means of a switch on the model type.
    try:
        output = np.array(
            _switch_output[type(model)](model, input_scaled))
    except KeyError:
        # Unknown model.
        raise unknown_model_exception

    # If possible scale back the output.
    if output_scaler is not None:
        output = output_scaler.inverse_transform(output)

    return output, extrapolation_sample


def predict_gradient(model,
                     input,
                     model_in_keys=None,
                     input_scaler=None,
                     output_scaler=None,
                     boundary=None):
    """
    Predict the gradient of a model.

    Parameters
    ----------
    model : python function
            scikit-learn MLPRegressor
        The model to be evaluated, which must be a Multiple Input Single Output.
        Support for additional model types is provided through the Loads cutting
        edge plugins.

    input : numpy.ndarray
            dict
        Input on which evaluate the model. If it is a 2D array, then it is
        assumed that each row is a different sample, and each column a
        different input variable. If instead it is a dictionary, then its
        keys are compared against the ones in model_in_keys, and a 2D
        array is assembled afterwards.

    model_in_keys : list
        List of input variables of the surrogate model. This variable is
        used only if input is a dictionary.

    input_scaler : sklearn.preprocessing.StandardScaler
                   sklearn.preprocessing.MinMaxScaler
        The model might have been constructed on a scaled input.
        The derivative is computed as
            y' = f'(x)
               = ft'(xt) * g'(x)
        with f the unscaled model, ft() the scaled one, x the unscaled input,
        xt the scaled input and g() the input_scaler.

    output_scaler : sklearn.preprocessing.StandardScaler
                    sklearn.preprocessing.MinMaxScaler
        The model might have been constructed on a scaled output. This
        parameter ensures that the predicted gradient is scaled back to its
        original dimensions. The derivative is computed as
            y' = f'(x)
               = h^{-1}'(yt) * ft'(x)
        with h() the output_scaler, yt the scaled output, ft() the scaled
        model and x the unscaled input. If there are both input and output
        scalares, the formula becomes
            y' = h^{-1}'(yt) * ft'(xt) * g'(x)

    Returns
    -------
    gradient : numpy.ndarray
        Model gradient, optionally scaled through output_scaler.
        2D array, where each row is associated to a different sample.

    extrapolation_sample : list
        Identifiers of the points outside of the boundary.

    Raises
    ------
    KeyError : if the model is not supported.
    TypeError : if input is not a dictionary or a numpy array.
    Warning: if some points are outside of the boundary.

    """

    # This function can be extended to the Multiple Output case,
    # but we must be careful with the shape of the output:
    # - always 2D?
    # - 2D or 3D depending on the case?

    # Form the input array.
    if type(input) is dict:
        input_array = np.column_stack(
            [input[key] for key in model_in_keys])
    elif type(input) is np.ndarray:
        input_array = input
    else:
        raise TypeError(
            'Parameter input must be a dictionary or a numpy array.')

    # Check for extrapolation.
    if boundary is not None:
        extrapolation_sample = np.where(~boundary(input_array))[0].tolist()
    else:
        extrapolation_sample = []
    if extrapolation_sample:
        warnings.warn('Surrogate evaluated outside of input domain. See extrapolation_sample.')

    # If possible scale the input.
    if input_scaler is not None:
        input_scaler.copy = True
        input_scaled = input_scaler.transform(input_array)
        try:
            gradient = np.array(
                _switch_gradient[type(model)](model, input_scaled))
            if type(input_scaler) is MinMaxScaler:
                gradient *= input_scaler.scale_
            elif ((type(input_scaler) is StandardScaler) and
                  (input_scaler.scale_ is not None)):
                gradient /= input_scaler.scale_
        except KeyError:
            raise unknown_model_exception
    else:  # The input is not scaled.
        try:
            gradient = np.array(
                _switch_gradient[type(model)](model, input_array))
        except KeyError:
            raise unknown_model_exception

    # If possible scale back the gradient.
    if output_scaler is not None:
        if type(output_scaler) is MinMaxScaler:
            gradient /= output_scaler.scale_
        elif ((type(output_scaler) is StandardScaler) and
              (output_scaler.scale_ is not None)):
            gradient *= output_scaler.scale_

    return gradient, extrapolation_sample


# %% Classes needed for the load constraint.

class SurrogateModel():
    """
    Class to store variables related to a surrogate model.
    """

    def __init__(self,
                 model=None,
                 input_scaler=None,
                 output_scaler=None,
                 input_channel_names=None,
                 output_channel_name=None,
                 ):
        """
        Initialize SurrogateModel.

        Parameters
        ----------
        model : any surrogate model supported by predict_output().
            The surrogate model.

        input_scaler : sklearn.preprocessing.StandardScaler
                       sklearn.preprocessing.MinMaxScaler
            The scaler used to transform the input.

        output_scaler : sklearn.preprocessing.StandardScaler
                       sklearn.preprocessing.MinMaxScaler
            The scaler used to anti-transform the output.

        input_channel_names : list of strings
            The name of the input channels.

        output_channel_name : str
            The name of the input channel.
        """
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.input_channel_names = input_channel_names
        self.output_channel_name = output_channel_name
