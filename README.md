# Hotels Rolling revenue forecasting

The project presents multi-step time series forecasting models for two different hotels located in different locations.

The time series for each hotel are treated as independent and two different models have been developed using the direct multi-step forecasting strategy.

The models show the following performance on the validation dataset:
- MAE 141.54K and 95% CI 21.51K-315.84K for City Hotel, and
- MAE 114.23k and 95% confidence interval 5.39-466.84k for the resort hotel.

The Inference CLI command can be run at any time of the year and provides a forecast for the next 6 months, taking into account the actual income for the previous two months and the name of the hotel for the forecast.

## Usage

### Project setup

Clone the repository and go into it.

Create and activate virtual environment.

Run `make install` to create directories and install required packages.

### Research with Jupiter Notebbok

Run `make run` and open provided URL in browser. 

### Forecasting

    python src/cli/forecast.py [OPTIONS]

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

The command outputs the forecast to the standard log stream.
    
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



## Project structure

- models — contains fitted models and spline transformer
- notebooks 
  - rolling-forecast.ipynb — Jupyter notebook with data analysis, model selection and evaluation 
  - data - dataset that includes booking transactions from 2 different hotels located in 2 different locations
- src
  - cli — provides a command-line interface for making rolling 6-months revenue forecast
  - data — contains functions for working with booking transactions and calculate revenue
  - eda — contains functions for exploratory data analysis of time-series data
  - model — contains functions for creating a periodic spline transformer and for evaluating a model
  - utils — collection of functions for common file I/O and logging tasks
- Makefile — contains commands to install and run project
- params.yaml — contains configuration parameters, saved during model evaluation stage 
- requirements.txt — list of project dependencies
