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
To implement [*Sobol sensitivity analysis*](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis), using sandu, no direct integration with a model is needed, just a table containing a selection of the models parameter values and associated outputs. A [*Gaussian process emulator*](https://en.wikipedia.org/wiki/Gaussian_process_emulator) is trained on the input-output data, which then acts as a surrogate model. By sampling from the gaussian process emulator the Sobol sensitivity indices can be effectively estimated with fewer expensive model runs required. This process usually would involve different libraries and a substantial amount of code. However, when using sandu to perform this analysis one simply provides: a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with parameter-output data from the model, the names of the relevant columns in the data frame, and N which determines the number of samples drawn from the surrogate model. One then runs:

```
import pandas as pd
from sandu.sensitivity_analysis import sobol

Si_df = sobol.get_indices(df, parameters, bounds, quantity_mean, quantity_varaince, N)
```
Where `Si_df` is a pandas dataframe with the first and total sensitivity indicies. This is shown in detail, using example data, in `/examples/example_sobol.py`.

**Data Types**
A sensitivity input class is included, this allows the user to bundle data needed for sensitivity analysis into objects and saved as JSON files. While the use of sensitivity input objects is voluntary, they are included to ease the integration of the sensitivity analysis algorithms in a data processing pipeline. The advantages of using sensitivity input objects are illustrated in the Sobol analysis examples (1.i).
## Contents

1. **Sensitivity Analysis Algorithms**
    1. **Sobol Sensitivity Analysis**
2. **Data Types**
    1. **Sensitivity Input Data Object**
3. **Gaussian Process Emulator**


# Examples

## (1.i) Computing Sobol indices
Two examples of computing Sobol indices from the same data are included, (a) one where the data is read from a CSV and additional parameters supplied by hand and (b) one using a sensitivity input object to streamline the process.

(a) Running  `/examples/example_sobol.py` analyses the parameter sensitivities from `parameters_output.csv` and produces a plot which should appear as below.

![alt text](images/example_sobol.png)

(b) The same analysis is performed in `/examples/example_sobol_sensitivity_input.py` but using a sensitivity input object `sensitivity_input.json`, thus removing the need to specify the parameter names, bounds, etc. manually.
## (2.i) Creating and Saving a SensitivityInput Object
A sensitivity input object is an object which contains all the information needed to perform sensitivity analysis. It is not necessary to use sensitivity input objects as the two examples of calculating Sobol sensitivity show.
This means that in addition to the parameter-output data, a sensitivity input object contains the names and bounds of the parameters of the model in question and the name of the model output and output variance.
Sensitivity input objects can be stored using JSON and allows all the input data needed for sensitivity analysis to be stored in one place. 
An example of creating, saving and loading a sensitivity input object is found in `/examples/example_make_and_load_sensitivity_input.py`, which creates a sensitivity input object from `parameters_output.csv` and saves it to `new_sensitivity_input.json`.

## (3) Training and evaluating a Gaussian Process Emulator. 
Running `/example/example_gaussian_process_emulator.py` trains a model on `parameters_output.csv` and plot the models test set performance. It should produce a plot as below.

![alt text](images/example_gaussian_process_emulator.png)

# Credits

* [SALib](http://salib.github.io/SALib/)
* [scikit-learn](https://scikit-learn.org/stable/)
