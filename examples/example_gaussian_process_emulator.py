import pandas as pd

import sandu.gaussian_process_emulator as gpe

#Runs the gaussian process emulator example, training and evaluating a model on the included data.

df = pd.read_csv("parameters_output.csv")

parameters = ["p_inf", "p_hcw", "c_hcw", "d", "q", "p_s", "rrd", "lambda", "T_lat", "juvp_s", "T_inf", "T_rec",
              "T_sym",
              "T_hos",
              "K", "inf_asym"]
quantity_mean = "total_deaths_mean"
quantity_variance = "total_deaths_variance"
testsetsize = 20

# Randomly select a few samples to form a training set and a test set
df_tr, df_te = gpe.split_training_test_set(df, testsetsize)

# Training set
X_tr, y_tr, alpha = gpe.form_training_set(df_tr, parameters, quantity_mean, quantity_variance)

# Test set
X_te, y_te = gpe.form_test_set(df_te, parameters, quantity_mean)

# Train model on training set
my_model, my_scaler = gpe.train_GP_emulator(X_tr, y_tr, alpha)

# Evaluate results on test set
gpe.evaluate_GP_emulator(X_te, y_te, my_model, my_scaler)
