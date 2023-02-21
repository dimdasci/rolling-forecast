"""
This module provides a command-line interface for making rolling 6-months
revenue forecast.

    Usage:
        forecast.py [OPTIONS]

    Options:
    -m, --month INTEGER RANGE       start month for forecast  [1<=x<=12;
                                    required]
    -y, --year INTEGER RANGE        start year for forecast  [2023<=x<=2025;
                                    required]
    -r, --revenue <FLOAT FLOAT>...  actual revenue for two preceding months
                                    [required]
    -ht, --hotel [City Hotel|Resort Hotel]
                                    name of the hotel for forecast  [required]
    --help                          Show this message and exit.

    Returns:
        None.


    Example:

    To make a forecast for Resort hotel starting in February 2023,
    and with actual revenue in December 2022 of $180,000 and
    January 2023 of $200,000, run the command

    >>> python src/cli/forecast.py \
    ... -m 2 -y 2023 \
    ... -r 180000 200000 \
    ... -ht 'Resort Hotel'
    INFO:root:Requested rolling revenue forecast for Resort Hotel starting from 02-2023
    INFO:root:Parameters check passed successfully
    INFO:root:Model and transformer was loaded successfully
    INFO:root:Resort Hotel rolling 6-month revenue forecast
    INFO:root:Actual   2022-12-31   180000.00
    INFO:root:Actual   2023-01-31   200000.00
    INFO:root:Forecast 2023-02-28   192170.55
    INFO:root:Forecast 2023-03-31   245650.90
    INFO:root:Forecast 2023-04-30   329826.22
    INFO:root:Forecast 2023-05-31   454316.10
    INFO:root:Forecast 2023-06-30   529041.16
    INFO:root:Forecast 2023-07-31   548446.39
"""

import click
from src.utils.utils import load_params, get_abs_path, load_pickle
from skforecast.utils import load_forecaster
import logging
import pandas as pd
from pandas.tseries.offsets import DateOffset


@click.command()
@click.option(
    "-m",
    "--month",
    type=click.IntRange(1, 12),
    required=True,
    help="start month for forecast",
)
@click.option(
    "-y",
    "--year",
    type=click.IntRange(2023, 2025),
    required=True,
    help="start year for forecast",
)
@click.option(
    "-r",
    "--revenue",
    type=(float, float),
    required=True,
    help="actual revenue for two preceding months",
)
@click.option(
    "-ht",
    "--hotel",
    required=True,
    type=click.Choice(
        ["City Hotel", "Resort Hotel"],
        case_sensitive=True,
    ),
    help="name of the hotel for forecast",
)
def main(
    month: int, year: int, revenue: tuple[float, float], hotel: str
) -> None:
    """
    Loads saved model and transformer, prepares predictors and makes forecast.

    Params:
        month (int): start month for forecast
        year (int):  start year for forecast
        revenue (tuple[float, float]): actual revenue for two preceding months
        hotel (str): name of the hotel for forecast

    Returns:
        None.

    Forecast will be logged in a standard log stream.

    The function performs the following steps:
    1. Reads configuration from params.yaml
    2. Checks if all required parameters are specified
    3. Loads the model and the spline transformer
    4. Builds predictors
    5. Makes forecast
    """

    logging.info(
        f"Requested rolling revenue forecast for {hotel}"
        f" starting from {month:02d}-{year}"
    )

    # load parameters from params.yaml
    try:
        params = load_params()
    except Exception as e:
        logging.error(f"Can't load param.yaml. {e}")
        return

    # check if all required parameters are specified
    for param_name in ["models_path", "exog_columns", "n_steps", "n_lags"]:
        if param_name not in params:
            logging.error(f"{param_name} parameter is missing in params.yaml")
            return

    if params["n_lags"] != 2:
        logging.error(
            f"Command expects model trained with n_lags=2, "
            f"but {params['n_lags']} was given"
        )

    logging.info("Parameters check passed successfully")

    # load the model and the spline transformer
    model_path = get_abs_path(
        rel_path=params["models_path"],
        filename=f"{hotel.replace(' ', '')}.pkl",
    )
    transformer_path = get_abs_path(
        rel_path=params["models_path"], filename="spline_transformer.pkl"
    )

    try:
        model = load_forecaster(model_path, verbose=False)
    except Exception as e:
        logging.error(f"Can't load model. {e}")
        return

    try:
        transformer = load_pickle(transformer_path)
    except Exception as e:
        logging.error(f"Can't load transformer. {e}")
        return

    logging.info("Model and transformer was loaded successfully")

    # For prediction we need 2 actual observations as we fit model
    # with two lags as predictors.
    # Also we need to transform forecast month numbers
    # to splines as they are used as predictors too.

    # calculate datetime ranges for forecast period and
    # n_lags window before it
    start_ts = pd.Timestamp(f"{year}-{month:02d}-01 00:00:00")
    prev_ts = start_ts - DateOffset(months=params["n_lags"])
    forecast_index = pd.date_range(
        start=start_ts, periods=params["n_steps"], freq="M"
    )
    last_window_index = pd.date_range(
        start=prev_ts, periods=params["n_lags"], freq="M"
    )

    # create last window for prediction
    last_window = pd.DataFrame(
        {hotel: list(revenue), "month_of_the_year": last_window_index.month},
        index=last_window_index,
    )

    # create forecast window to fill with exogenous predictors
    foreacast_window = pd.DataFrame(
        {"month_of_the_year": forecast_index.month},
        index=forecast_index,
    )

    # transform months number to splines
    exogs = pd.DataFrame(
        transformer.transform(foreacast_window[["month_of_the_year"]]),
        index=forecast_index,
        columns=params["exog_columns"],
    )

    # make prediction
    forecast = model.predict(
        steps=params["n_steps"],
        last_window=last_window[hotel],
        exog=exogs,
    )

    logging.info(f"{hotel} rolling {params['n_steps']}-month revenue forecast")
    for i, row in last_window.iterrows():
        logging.info(f"Actual   {i.date()}\t{row[hotel]:.2f}")
    for i, revenue_forecast in forecast.items():
        logging.info(f"Forecast {i.date()}\t{revenue_forecast:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
