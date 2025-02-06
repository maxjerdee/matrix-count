# Script to use matrix_count in order to evaluate the performance of the symmetric binary estimates as the size n and number of edges m vary.

import matrix_count
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import ast

filename = "test_margins_binary.csv"

# Read csv
df = pd.read_csv(filename)

# Function to calculate true log count for a row
def calculate_true_log_count(i, row):
    # if np.isnan(row["true_log_count"]) and row["m"] <= 400:
    if row["m"] <= 400:
        print(np.array(ast.literal_eval(row["margin"])))
        true_log_count, true_log_count_err = matrix_count.count_log_symmetric_matrices(np.array(ast.literal_eval(row["margin"])), binary_matrix=True)
        df.at[i, "true_log_count"] = true_log_count
        df.at[i, "true_log_count_err"] = true_log_count_err
        print(true_log_count, true_log_count_err)

        # Save the updated dataframe
        df.to_csv(filename, index=False)
        print(f"{filename} updated.")

# Parallel processing


results = Parallel(n_jobs=1)(delayed(calculate_true_log_count)(i, row) for i, row in df.iterrows())

