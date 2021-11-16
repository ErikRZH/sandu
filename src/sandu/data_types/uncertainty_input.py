import pandas as pd


class UncertaintyInput:
    def __init__(self, data: dict, quantity_name: str, time_name: str, run_name: str):
        """Class for uncertainty quantification input data.

        Args:
            data: Parameters and outputs for multiple model runs in JSON format (orient='split').
                The columns are: "run_name","time_name","quantity_name",...
            quantity_name: Name of column containing the output quantitiy of interest
            time_name: Name of column containing time values.
            run_name: Name of column containing the model run indices.
        """
        self.data = data
        self.quantity_name = quantity_name
        self.time_name = time_name
        self.run_name = run_name

    def df(self) -> pd.DataFrame:
        """"Returns self.data as a pandas dataframe.

        This is a method so it is not included in the __dict__ of UncertaintyInput objects.
        """
        return pd.read_json(self.data, orient='split')
