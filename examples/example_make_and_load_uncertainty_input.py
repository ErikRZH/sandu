import pandas as pd
from sandu.data_types import UncertaintyInput
import json

"""
Creates a UncertaintyInput object by specifying the necessary metadata associated with uncertainty_example_data.csv.
Saves the UncertaintyInput object as a JSON file (new_uncertainty_input.json).
Then imports the Object from the JSON file and prints the data of the object.
"""
iter_name = "iter"  # name of column containing variable identifying different runs
quantity_name = " inc_case"
time_name = " day"

df = pd.read_csv("uncertainty_example_data.csv")

new_uncertainty_input = UncertaintyInput(df.to_json(index=False, orient="split"), quantity_name, time_name, iter_name)

with open("new_uncertainty_input.json", "w", encoding="utf-8") as f:
    json.dump(new_uncertainty_input.__dict__, f, ensure_ascii=False, indent=4)

with open("new_uncertainty_input.json", "r") as read_file:
    x = json.load(read_file, object_hook=lambda d: UncertaintyInput(**d))

print(x.df())
