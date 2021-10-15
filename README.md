# sandu ![](https://img.shields.io/pypi/v/sandu) ![](https://img.shields.io/badge/python-%3E%3D3.6-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*High Level Sensitivity and Uncertainty (SandU) analysis tools for python.*



## *Sandu aims to provide high level functionality for sensitivity and uncertainty analysis.*

Sandu was developed to provide Sensitivity and Uncertainty analysis functionality for the [rampvis-api](https://github.com/ScottishCovidResponse/rampvis-api).

## Installation

The package is named `sandu` and listed on [PyPI](https://pypi.org/project/sandu/). You can use the pip to install:

*Unix/MacOS*
```bash
python3 -m pip install sandu
```
*Windows*
```bash
py -m pip install sandu
```
## Description

The motivation is to provide easy to use, end to end, sensitivity analysis and uncertainty quantification functionality. Thereby lowering the barrier of entry for this type of analysis in python. Sandu was developed to analyse agent based models but may be applied more generally to any model or experimental data.

**Illustration of the package's raison d'être.**   
If you want to implement [*Sobol sensitivity analysis*](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis), using sandu, no direct integration with a model is needed, just a table containing a selection of the models parameter values and associated outputs. A [*Gaussian process emulator*](https://en.wikipedia.org/wiki/Gaussian_process_emulator) is trained on the input-output data, which then acts as a surrogate model. By sampling from the gaussian process emulator the Sobol sensitivity indices can be effectively estimated with fewer expensive model runs required. This process usually would involve different libraries and a substantial amount of code but using sandu to perform this analysis one simply provides a df` is the [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with parameter-output data from the model  the names of the relevant columns in the data frame, and N which determines the number of samples drawn from the surrogate model and runs:

```
import pandas as pd
from sandu.sensitivity_analysis import sobol

Si_df = sobol.get_indices(df, parameters, bounds, quantity_mean, quantity_varaince, N)
```
Where `Si_df` is a pandas dataframe with the first and total sensitivity indicies. This is shown in detail, using example data, in `/examples/example_sobol.py`.

## Contents
1. **Sensitivity Analysis Algorithms**
    1. **Sobol Sensitivity Analysis**
2. **Gaussian Process Emulator**


# Examples
## (1.i) Computing Sobol indices
Running  `/examples/example_sobol.py` analyses the parameter sensitivities  from `parameters_output.csv` and produces a plot which should appear as below.

![alt text](images/example_sobol.png)

## (2) Training and evaluating a Gaussian Process Emulator. 
Running `/example/example_gaussian_process_emulator.py` trains a model on `parameters_output.csv` and plot the models test set performance. It should produce a plot as below.

![alt text](images/example_gaussian_process_emulator.png)

# Credits
* [SALib](http://salib.github.io/SALib/)
* [scikit-learn](https://scikit-learn.org/stable/)
