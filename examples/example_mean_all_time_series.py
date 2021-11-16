import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sandu.uncertainty_quantification import mean_time_series
from sandu.data_types import UncertaintyInput

"""
Computes the mean time series from the ensemble of time series in uncertainty_input.json. 
The mean time series is then plotted over all the individual time series.
The data is retrieved from an UncertaintyInput object stored using JSON.

To do the same calculation using the data in "uncertainty_example_data.csv", compare with "example_sobol.py".
"""
with open("uncertainty_input.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: UncertaintyInput(**d))

df_mean = mean_time_series.get_mean(x.df(), x.time_name, x.quantity_name)
df_all = x.df()

plt.figure(figsize=(10.5, 6))
for i in df_all[x.run_name].unique():
    model_run = df_all.loc[df_all[x.run_name] == i]  # get one model run
    plt.plot(model_run[x.time_name], model_run[x.quantity_name], alpha=0.01, color="#A6CCD8")
mean_line, = plt.plot(df_mean[x.time_name], df_mean[x.quantity_name], color="#D72B1E", label='Mean Time Series')
plt.title("All Time Series and Mean", fontsize=15)
plt.xlabel("Day", fontsize=15)
plt.ylabel("Daily Cases", fontsize=15)
all_legend = mpatches.Patch(color='#A9D0D9', label='All Time Series')
plt.legend(handles=[all_legend, mean_line], loc='upper right')
plt.ylim(0)
plt.xlim([0, 200])
plt.show()
