import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import collections


def get_time_series_dataset(df: pd.DataFrame, quantity_mean_in: str) -> np.ndarray:
    """Converts entries in quantity_mean column of the dataframe to a time series dataset for ts-learn.

    Args:
        df: Dataframe of model data, each row is a different sample.
            Columns are: parameters, output mean, output variance.
        quantity_mean_in: Name of the column containing the mean of the output quantity.

    Returns:
        time_series: Numpy array of the time series dataset format ts-learn takes as input.
    """

    nr_of_runs = df.shape[0]
    entry = df[quantity_mean_in][0]
    nr_of_time_samples = len(entry) if isinstance(entry, collections.Sized) else 1
    # make time series dataset
    time_series = np.zeros((nr_of_runs, nr_of_time_samples, 1))
    for idx in range(nr_of_runs):
        time_series[idx, :, 0] = df[quantity_mean_in][idx][:] if isinstance(entry, collections.Sized) else \
            df[quantity_mean_in][idx]
    return time_series


def get_kmeans_clusters(df: pd.DataFrame, quantity_mean_in: str, n_clusters: int, metric: str, verbose=False,
                        random_state=42) -> np.ndarray:
    """Computes cluster labels for time series stored in a column of a pandas dataframe, using kmeans clustering.

    Args:
        df: Dataframe of model data, each row is a different sample.
            Columns are: parameters, output mean, output variance.
        quantity_mean_in: Name of the column containing the mean of the output quantity.
        n_clusters: Number of clusters to find.
        metric: Metric to use, choices correspond to those for TimeSeriesKMeans, "euclidean", "dtw", "softdtw".
        verbose: If true, print information about inertia while learning.  Default: False
        random_state: Seed used to initialise cluster centres. Default: 42

    Returns:
        clusters: Numpy array containing the index of the cluster assigned to each parameter combination
        The order of the indices match the order of the rows in the dataframe.
    """
    time_series = get_time_series_dataset(df, quantity_mean_in)
    clusters = TimeSeriesKMeans(n_clusters=n_clusters, verbose=verbose, random_state=random_state,
                                metric=metric).fit_predict(time_series)
    return clusters
