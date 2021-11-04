import pandas as pd


class SensitivityInput:
    def __init__(self, data: dict, parameters: list, bounds: list, quantity_mean: str, quantity_variance: str):
        """Class for sensitivity analysis input data

        Args:
            data: Parameters and outputs for multiple model runs in JSON format.
                The columns are columns: "self.parameters, self.quantity_mean, self.quantity_variance"
            parameters: Names of model parameters, which can include constants, as ["param_1","param_2","param_x",...]
            bounds: Bounds of model parameters as [[param1_lower_bound,param1_upper_bound],[...],[param_x_fixed],..].
                For parameters with a fixed value provide one entry with the fixed value.
            quantity_mean: Name of the column containing the mean of the output quantity.
            quantity_variance: Name of the column containing the variance of the output quantity.
        """
        self.parameters = parameters
        self.bounds = bounds
        self.quantity_mean = quantity_mean
        self.quantity_variance = quantity_variance
        self.data = data

    def df(self) -> pd.DataFrame:
        """"Returns self.data as a pandas dataframe
        """
        return pd.read_json(self.data, orient='split')
