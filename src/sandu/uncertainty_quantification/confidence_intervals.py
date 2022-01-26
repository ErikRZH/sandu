import matplotlib.pyplot as plt
import pandas as pd


def get_confidence_intervals(df_in: pd.DataFrame, time_name_in: str, quantity_name_in: str) -> pd.DataFrame:
    """Returns confidence interval statistics for each timestep.
    These are: lower/upper 1.5*IQR, lower/upper quartile, mean, median

    Args:
        df_in: input dataframe with all time series. Columns: "run_name","time_name","quantity_name",...
        time_name_in: Name of column containing time values.
        quantity_name_in: Name of column containing the model output to be averaged over.

    Returns:
        df_out: Dataframe with columns: [time_name_in, "lower_IQR", "upper_IQR", "lower_quartile", "upper_quartile",
        "mean", "median"] and one row for each time step.
    """
    unique_times = df_in[time_name_in].unique()
    unique_times = unique_times.tolist()
    unique_times.sort()
    data = []
    for time in unique_times:
        df_temp = df_in.loc[df_in[time_name_in] == time]
        values = df_temp[quantity_name_in]
        B = plt.cbook.boxplot_stats(values)[0]
        lower_whisker = B["whislo"]
        upper_whisker = B["whishi"]
        lower_box = B["q1"]
        upper_box = B["q3"]
        mean = B["mean"]
        median = B["med"]
        data.append([time, lower_whisker, upper_whisker, lower_box, upper_box, mean, median])
    df_out = pd.DataFrame(data,
                          columns=[time_name_in, "lower_IQR", "upper_IQR", "lower_quartile", "upper_quartile", "mean",
                                   "median"])
    return df_out
