"""
The `data.py` module contains functions for working with pandas DataFrame
objects that contain booking data. These functions can be used to construct
a pandas.Timestamp object representing a specific date, and to calculate
the daily revenue stream for a given period of time based on the booking data.

Functions:
- build_datetime(year: int, month: str, day: int) -> pd.Timestamp:
    Constructs a pandas.Timestamp object representing a specific date.

- calculate_daily_revenue_stream(data: pd.DataFrame,
                                 n_days: int,
                                 start_date: pd.Timestamp) -> np.ndarray:
    Calculates the daily revenue stream for a given period of time based on
    the booking data.
"""

import pandas as pd
import numpy as np


def build_datetime(year: int, month: str, day: int) -> pd.Timestamp:
    """
    Constructs a pandas.Timestamp object representing a specific date.

    Args:
        year (int): The year of the date.
        month (str): The month of the date, as a string in title case
                     (e.g. "January").
        day (int): The day of the month.

    Returns:
        pd.Timestamp: A pandas.Timestamp object representing the
                      specified date.
    """
    return pd.to_datetime(
        str(year) + "-" + month + "-" + str(day), format="%Y-%B-%d"
    )


def calculate_daily_revenue_stream(
    data: pd.DataFrame, n_days: int, start_date: pd.Timestamp
) -> np.ndarray:
    """
    Calculate the daily revenue stream for a given period of time.

    Args:
        data (pd.DataFrame): The DataFrame containing the booking data.
        n_days (int): The number of days to analyze for the revenue stream.
        start_date (pd.Timestamp): The starting date of the period to analyze.

    Returns:
        np.ndarray: A 1D numpy array of length n_days, representing the
                    daily revenue stream.
    """

    # Initialize an array of zeros to hold the daily revenue stream.
    revenue_stream = np.zeros(n_days)

    # Loop over the bookings in the DataFrame.
    for i, row in data.iterrows():
        # If the booking was canceled and was non-refundable,
        # add the revenue to the reservation_status_date.
        if row.is_canceled == 1 and row.deposit_type == "Non Refund":
            idx = (row.reservation_status_date - start_date).days
            if 0 <= idx < n_days:
                revenue = row.adr * row.total_nights
                revenue_stream[idx] += revenue

        # If the booking was not canceled, add the revenue evenly
        # over the length of stay.
        elif row.is_canceled == 0:
            idx = (row.arrival_date - start_date).days
            for step in range(row.total_nights):
                if idx + step > n_days - 1:
                    # the staying goes after end of anylised period
                    break
                revenue_stream[idx + step] += row.adr

    return revenue_stream
