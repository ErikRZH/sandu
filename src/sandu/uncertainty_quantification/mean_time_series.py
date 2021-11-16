import pandas as pd


def get_mean(df_in: pd.DataFrame, time_name: str, quantity_name: str) -> pd.DataFrame:
    """Obtain the mean time series from a pandas dataframe of timeseries

    Args:
        df_in: input dataframe with all time series. Columns: "run_name","time_name","quantity_name",...
        time_name: Name of column containing time values.
        quantity_name: Name of column containing the model run indices.

    Returns:
        df_mean: Dataframe containing the mean time series
    """
    df_mean = df_in.groupby(time_name, as_index=False)[quantity_name].mean()
    return df_mean
