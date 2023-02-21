"""
The `eda.py` module contains several functions for exploratory data analysis of
time-series data. The functions include:

    - draw_hist_and_boxplot:  plots a histogram and boxplot of a column
                              in a DataFrame.
    - draw_num_transactions:  plots a line plot of the number of transactions
                              per month, grouped by index and column.
    - plotMovingAverage:      plots a moving average chart with optional
                              confidence intervals and anomalies.
    - window_statisctics:     calculates statistics for a time window in
                              a series.
    - draw_window_statistics: plots the statistics calculated by
                              window_statistics.

Each function takes a set of arguments as described in their respective
docstrings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def draw_hist_and_boxplot(
    data: pd.DataFrame, column: str, title: str, vline: int = None
) -> None:
    """
    Draw a histogram and boxplot of a column in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to plot.
        column (str): The name of the column to plot.
        title (str): The title to display above the plots.
        vline (int, optional): The x-coordinate of a vertical line to draw
                               on the histogram, by default None.

    Returns:
        None: The function plots the histogram and boxplot using matplotlib
              and does not return any values.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
    data[column].hist(bins=60, ax=ax1)
    data.boxplot(ax=ax2, column=column, vert=False)
    ax2.set_xlabel(column)
    if vline:
        ax1.axvline(vline, color="red")
    ax1.title.set_text(title)
    plt.show()


def draw_num_transactions(
    data: pd.DataFrame, index: str, column: str, title: str
) -> None:
    """
    Draw a line plot of the number of transactions per month, grouped
    by index and column.

    Args:
        data (pd.DataFrame): The DataFrame containing the transaction data.
        index (str): The name of the column to group the data by on the x-axis.
        column (str): The name of the column to group the data by on the
                      legend.
        title (str): The title to display above the plot.

    Returns:
        None: The function plots the line plot using matplotlib and does not
              return any values.

    Create a pivot table of the transaction data, grouped by index and column.
    Resample the pivot table by month and sum the counts of transactions.
    Plot the resulting pivot table as a line plot with markers.

    """
    pd.pivot_table(
        data, index=index, columns=column, values="hotel", aggfunc="count"
    ).resample("M").sum().plot(
        figsize=(10, 4),
        marker="o",
        markersize=9,
        lw=3,
        color=["C1", "C2", "C3"],
    )
    plt.ylabel("Number of transactions")
    plt.title(title.upper())
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plotMovingAverage(
    series: pd.DataFrame,
    window: int,
    plot_intervals: bool = False,
    scale: float = 1.96,
    plot_anomalies: bool = False,
    ylabel: str = "",
) -> None:
    """
    Plots moveing average chart

    Args:
    - series (pd.DataFrame): dataframe with timeseries
    - window (int): rolling window size
    - plot_intervals (bool): show confidence intervals
    - scale (float): scale factor for confidence intervals
    - plot_anomalies (bool): show anomalies
    - ylabel (str): ylabel of plot

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", lw=2, label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(
            upper_bond, "r--", alpha=0.25, label="Upper Bond / Lower Bond"
        )
        plt.plot(lower_bond, "r--", alpha=0.25)

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(
                index=series.index, columns=series.columns
            )
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=5)

    plt.plot(series[window - 1 :], alpha=0.75, lw=0.5, label="Actual values")
    plt.legend(loc="upper left")
    plt.ylabel(ylabel)
    plt.grid(True)


def window_statisctics(ts, rule, date_part_func):
    """
    Calculates statistics for a time window in a series

    ts - dataframe with timeseries
    rule - the offset string or object representing target conversion
    date_part_func - function that extract a part of a date

    return dataframe with statisitics
    """
    column = ts.columns.to_list()[0]
    _df = (
        ts.resample(rule)
        .sum()
        .assign(date_part=date_part_func)
        .pivot_table(
            index="date_part", values=column, aggfunc=[np.mean, np.std]
        )
    )
    _df.columns = ["mean", "std"]
    return _df


def draw_window_statistics(ts, rule, date_part_func, ax, xlabel, title):
    """
    Calculates statistics for a time window in a series

    ts - dataframe with timeseries
    rule - the offset string or object representing target conversion
    date_part_func - function that extract a part of a date
    ax - pyplot axes to draw
    xabel - x ax label
    title - chart title

    return dataframe with statisitics
    """
    _df = window_statisctics(ts, rule, date_part_func)
    ax.plot(_df.index, _df["mean"], lw=3, label="mean")
    ax.fill_between(
        _df.index,
        _df["mean"] - _df["std"],
        _df["mean"] + _df["std"],
        alpha=0.1,
        label="std",
    )
    ax.set_title(title.upper())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Revenue")
    ax.legend().set_visible(True)


def draw_train_test_predict(
    ax: plt.axes,
    train: pd.Series,
    test: pd.Series,
    predict: list[pd.Series] = None,
    title: str = "",
) -> None:
    """
    Plots the train, test and (optionally) prediction time series data.

    Args:
    - ax: matplotlib axes object to plot on
    - train: pandas Series with the train data
    - test: pandas Series with the test data
    - predict: pandas Series with the predicted data (optional)
    - title: title for the plot (optional)

    Returns:
    - None
    """

    chart_title = title.upper()
    train.plot(ax=ax, color="C2", lw=3, label="train")
    test.plot(ax=ax, color="C1", lw=3, label="test")
    if predict is not None:
        label = "prediction"
        for p in predict:
            p.plot(ax=ax, color="C1", lw=2, ls="--", label=label)
            label = "_nolegend_"
        if len(predict) == 1:
            mae = mean_absolute_error(test[: predict[0].shape[0]], predict[0])
            chart_title += f"\n MAE: {mae:.2f}"
    ax.title.set_text(chart_title)
    ax.set_ylabel("Monthly Revenue")
    ax.legend(loc="upper left")
    ax.grid(visible=True)
