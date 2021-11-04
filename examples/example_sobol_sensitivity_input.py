import json
import matplotlib.pyplot as plt

from sandu.data_types import SensitivityInput
from sandu.sensitivity_analysis import sobol

"""
Example of Sobol sensitivity analysis using data from a SensitivityInput object.
Calculates prints and plots the Sobol sensitivity indices from the the data in example_sensitivity_input.json.
"""

with open("sensitivity_input.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: SensitivityInput(**d))

N = 2 ** 12 # Note: Must be a power of 2.  (N*(2D+2) parameter value samples are used in the Sobol analysis.

Si_df = sobol.get_indices(x.df(), x.parameters, x.bounds, x.quantity_mean, x.quantity_variance, N)  # Perform analysis

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









