from sandu.data_types import SensitivityInput
import json
import numpy as np
import matplotlib.pyplot as plt
import sandu.sensitivity_analysis.cluster_based as cb
import collections

"""
Clusters output time series, given as lists in Dataframe and plots the clusters and parameters ranges for each cluster.
Cluster-based analysis helps identify the sensitivity of the model to the different parameters.
In addition, the visualisation reveals the correspondence between parameter ranges and model behaviour.

Change sensitivity_input_list.json to sensitivity_input.json for clustering of scalar outputs instead of time series.
"""
k = 4
my_metric = 'euclidean'

# Load Sensitivity Input Object
with open("sensitivity_input_list.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: SensitivityInput(**d))
df = x.df()
# Compute Cluster Labels
clusters = cb.get_kmeans_clusters(df, x.quantity_mean, k, my_metric, verbose=True)
df["cluster"] = clusters

# Plot Results
colours = ["#E69F00", "#009E73", "#DB4D56", "#0072B2", "#D55E00", "#F0E442", "#56B4E9"]

fig, axs = plt.subplots(2, figsize=(10.5, 6), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(hspace=0.3, top=0.95)
nr_of_runs = df.shape[0]
nr_of_time_samples = len(df[x.quantity_mean][0]) if isinstance(df[x.quantity_mean][0], collections.Sized) else 1
time_series = True if nr_of_time_samples > 1 else False
# All Lines
g = df.groupby("cluster")
marker = "None" if time_series else "_"
ms = 0 if time_series else 1000
for i in range(nr_of_runs):
    model_run = df[x.quantity_mean][i]
    cluster_index = df["cluster"][i]
    axs[0].plot(range(nr_of_time_samples), model_run, alpha=0.15, color=colours[cluster_index], marker=marker, ms=ms)

# Mean Lines
for cluster_index in range(k):
    df = g.get_group(cluster_index)
    mean = np.mean(df[x.quantity_mean].tolist(), axis=0)
    axs[0].plot(range(nr_of_time_samples), mean, alpha=1, color=colours[cluster_index], linewidth=2,
                label="Cluster " + str(cluster_index), marker=marker, ms=ms)
axs[0].set_xlabel("Day", fontsize=13)
y_label = "Daily Cases" if time_series else "Total Cases"
axs[0].set_ylabel(y_label, fontsize=13)
axs[0].legend(fontsize=13, markerscale=0)

# Parameter Ranges Bar Chart
# Get non-constant parameters and bounds
parameters = [x.parameters[i] for i in range(len(x.parameters)) if len(x.bounds[i]) > 1]
bounds = [x.bounds[i] for i in range(len(x.parameters)) if len(x.bounds[i]) > 1]

nr_of_parameters = len(parameters)
width = 1 / k - 0.05
r = np.arange(nr_of_parameters)
param_range = [bounds[i][1] - bounds[i][0] for i in range(nr_of_parameters)]

for cluster_index in range(k):
    df = g.get_group(cluster_index)
    # Compute quantities to plot the range of parameters within a cluster
    param_min = [df[i].min() for i in parameters]
    param_min_normalised = [(param_min[i] - bounds[i][0]) / param_range[i] for i in range(nr_of_parameters)]
    param_diff = [df[i].max() - df[i].min() for i in parameters]
    param_diff_normalised = [param_diff[i] / param_range[i] for i in range(nr_of_parameters)]
    axs[1].bar([x + width * cluster_index for x in r], param_diff_normalised, width=width, bottom=param_min_normalised,
               color=colours[cluster_index])
axs[1].set_xticks([x + width * k / 2 for x in r], labels=parameters, fontsize=10)
axs[1].set_ylim([0, 1])
axs[1].set_ylabel("Normalised Range", fontsize=13)
axs[1].set_xlabel("Parameters", fontsize=13)
plt.show()
