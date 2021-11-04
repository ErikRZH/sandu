import pandas as pd
from sandu.data_types import SensitivityInput
import json

"""
Creates a SensitivityInput object by specifying the necessary metadata associated with parameters-output.csv.
Saves the SensitivityInput object as a JSON file (new_sensitivity_input.json).
Then imports the Object again from the JSON file and prints the data of the object.
"""

parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec", "T_sym",
              "T_hos", "K", "inf_asym"]  # Names of columns containing model parameters, which can include constants.

bounds = [[0, 1], [0, 1], [1, 80], [0, 1], [0, 1], [0, 1], [1], [1e-9, 1e-3], [0.1, 14], [0, 1], [0.1, 21], [1, 28],
          [0.1, 14], [1, 35], [10000], [0, 1]]  # Bounds on model parameter values, single entries are constants.

quantity_mean = "total_deaths_mean"  # Name of the column containing the mean of the output quantity.

quantity_variance = "total_deaths_variance"  # Name of the column containing the variance of the output quantity.

df = pd.read_csv("parameters_output.csv")

new_sensitivity_input = SensitivityInput(df.to_json(index=False, orient="split"), parameters, bounds, quantity_mean, quantity_variance)

with open("new_sensitivity_input.json", "w", encoding="utf-8") as f:
    json.dump(new_sensitivity_input.__dict__, f, ensure_ascii=False, indent=4)

with open("new_sensitivity_input.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: SensitivityInput(**d))

print(x.df())
