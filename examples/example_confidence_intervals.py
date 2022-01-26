import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sandu.data_types import UncertaintyInput
from sandu.uncertainty_quantification.confidence_intervals import get_confidence_intervals

"""
Computes confidence interval statistics for each time step independently for an ensemble of time series. 
The upper and lower quartiles, and 1.5 times the interquartile range are plotted over all the individual time series.
The data is retrieved from an UncertaintyInput object stored using JSON.

To do the same calculation using the data in "uncertainty_example_data.csv", compare with "example_sobol.py".
"""


with open("uncertainty_input.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: UncertaintyInput(**d))

df_all = x.df()

df_confidence = get_confidence_intervals(x.df(), x.time_name, x.quantity_name)

plt.figure(figsize=(10.5, 6))
for i in df_all[x.run_name].unique():
    model_run = df_all.loc[df_all[x.run_name] == i]  # get one model run
    plt.plot(model_run[x.time_name], model_run[x.quantity_name], alpha=0.01, color="#A6CCD8")
mean_line, = plt.plot(df_confidence[x.time_name], df_confidence["mean"], color="#D72B1E", label='Mean Time Series')
quartiles, = plt.plot(df_confidence[x.time_name], df_confidence["upper_quartile"], color="#A62FFA",
                      label="Upper and Lower Quartiles")
plt.plot(df_confidence[x.time_name], df_confidence["lower_quartile"], color="#A62FFA")
IQR, = plt.plot(df_confidence[x.time_name], df_confidence["upper_IQR"], color="#E072CB", label="1.5*Interquartile Range")
plt.plot(df_confidence[x.time_name], df_confidence["lower_IQR"], color="#E072CB")
plt.title("All Time Series and Confidence Intervals", fontsize=15)
plt.xlabel("Day", fontsize=15)
plt.ylabel("Daily Cases", fontsize=15)
all_legend = mpatches.Patch(color='#A9D0D9', label='All Time Series')
plt.legend(handles=[all_legend, mean_line], loc='upper right')
plt.legend()
plt.ylim(0)
plt.xlim([0, 200])
plt.show()
