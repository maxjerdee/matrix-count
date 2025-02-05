# Script to use matrix_count in order to evaluate the performance of the symmetric multigraph estimates as the size n and number of edges m vary.

import matrix_count

import pandas as pd
import numpy as np
import ast

filename = "test_margins_multigraph.csv"

# Read csv
df = pd.read_csv(filename)

# Calculate the true log count for each margin
for i, row in df.iterrows():
    # Check if the true log count has already been calculated (not a number)
    if np.isnan(row["true_log_count"]) and row["m"] <= 400:
        true_log_count, true_log_count_err = matrix_count.count_log_symmetric_matrices(np.array(ast.literal_eval(row["margin"])), binary_matrix=False)
        df.at[i, "true_log_count"] = true_log_count
        df.at[i, "true_log_count_err"] = true_log_count_err

        # Update the dataframe
        df.to_csv(filename, index=False)