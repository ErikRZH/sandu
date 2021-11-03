import pandas as pd
import matplotlib.pyplot as plt

from sandu.sensitivity_analysis import sobol
"""
Calcualates, prints and plots the Sobol sensitivity indices from the the data in parameters_output.csv.
"""

df = pd.read_csv("parameters_output.csv")

parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
              "T_hos", "K", "inf_asym"]  # Names of columns containing model parameters, which can include constants.

bounds = [[0, 1], [0, 1], [1, 80], [0, 1], [0, 1], [0, 1], [1], [1e-9, 1e-3], [0.1, 14], [0, 1], [0.1, 21], [1, 28],
          [0.1, 14], [1, 35], [10000], [0, 1]]  # Bounds on model parameter values, single entries are constants.

quantity_mean = "total_deaths_mean"  # Name of the column containing the mean of the output quantity.

quantity_varaince = "total_deaths_variance"  # Name of the column containing the variance of the output quantity.

N = 2 ** 12 # Note: Must be a power of 2.  (N*(2D+2) parameter value samples are used in the Sobol analysis.

Si_df = sobol.get_indices(df, parameters, bounds, quantity_mean, quantity_varaince, N)  # Perform analysis
Si_df.index.name = "Parameter"

print("The first and total order Sobol sensitivity indices, and their 95% confidence intervals, are: ")
print(Si_df)

labels = Si_df.index.values
S1 = Si_df ["S1"]
S_interaction = Si_df["ST"]-Si_df["S1"]
width = 0.95
fig, ax = plt.subplots(figsize=(10.5, 6))
ax.bar(labels, S1, width, label='Main effect',color='blue', edgecolor = "black")
ax.bar(labels, S_interaction, width, bottom=S1, label='Interaction', color='red', edgecolor="black")
ax.set_xlabel('Parameters', fontsize=15)
ax.set_title('Sobol Sensitivity of Model Parameters', fontsize=15)
ax.set_ylim([0, 1])
ax.legend(fontsize=15)
plt.show()









