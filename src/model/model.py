"""
The `model.py` module contains functions for creating a periodic spline
transformer and for evaluating a model using a sliding window approach
on validation data.

Functions:

periodic_spline_transformer(period, n_splines=None, degree=3)
    Creates a SplineTransformer object that encodes time-related features
    using spline transformations.

Parameters:
    period (float): The time period.
    n_splines (int, optional): The number of splines to use. Defaults to None.
    degree (int, optional): The degree of the polynomial used for the spline
                            basis. Defaults to 3.

Returns:
    SplineTransformer: The SplineTransformer object.

run_evaluation(train, valid, model, series_column,
               n_tests, n_lag, n_steps, exog)
    Runs an evaluation on a given model using a sliding window approach on
    the validation data.

Parameters:
    train (pandas.DataFrame): The training data used to fit the model.
    valid (pandas.DataFrame): The validation data used to evaluate the model.
    model: The trained model object used to make predictions.
    series_column (str): The name of the column in the dataframes that contains
                         the time series values.
    n_tests (int): The number of test windows to slide over the validation
                   data.
    n_lag (int): The size of the window used for making predictions.
    n_steps (int): The number of steps to forecast for each window.
    exog (list): A list of column names in the dataframes that contain any
                 exogenous variables to use in the model.

Returns:
    tuple: A tuple containing a pandas Series object of the absolute errors
           for all predictions made,
           and a list of all the predicted values for each test window.
"""

from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd


def periodic_spline_transformer(period, n_splines=None, degree=3):
    """
    Makes spline transformations encoding of the periodic time-related features

    period - time period
    n_slines - number of splines
    degree - the polynomial degree of the spline basis

    """
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


def run_evaluation(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model,
    series_column: str,
    n_tests: int,
    n_lag: int,
    n_steps: int,
    exog: list,
) -> tuple[pd.Series, list]:
    """
    Runs an evaluation on a given model using a sliding window approach on
    the validation data.

    Args:
        train (pd.DataFrame): The training data used to fit the model.
        valid (pd.DataFrame): The validation data used to evaluate the model.
        model: The trained model object used to make predictions.
        series_column (str): The name of the column in the dataframes that
                             contains the time series values.
        n_tests (int): The number of test windows to slide over the validation
                       data.
        n_lag (int): The size of the window used for making predictions.
        n_steps (int): The number of steps to forecast for each window.
        exog (list): A list of column names in the dataframes that contain any
                     exogenous variables to use in the model.

    Returns:
        tuple: A tuple containing a pandas Series object of the absolute errors
               for all predictions made, and a list of all the predicted values
               for each test window.
    """

    test_df = pd.concat([train[-n_lag:], valid])
    aes = []
    predictions = []

    for i in range(n_tests):
        # define indexes to get window for predictions
        # and following observations
        window_idx = i
        observation_idx = i + n_lag

        # slice window and observations from valid dataset
        window = test_df[window_idx : window_idx + n_lag][series_column]
        observation = test_df[observation_idx : observation_idx + n_steps][
            series_column
        ]

        # forecast on train data
        forecast = model.predict(
            steps=n_steps, last_window=window, exog=test_df[exog]
        )

        # keep predictions and absolute errors
        predictions.append(forecast)
        aes.append(np.abs(forecast.values - observation.values))

    aes = pd.Series(np.concatenate(aes))

    return aes, predictions
