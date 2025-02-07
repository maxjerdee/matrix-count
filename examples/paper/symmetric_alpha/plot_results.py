# Plot the results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
df_binary = pd.read_csv("test_margins_binary.csv")
df_multigraph = pd.read_csv("test_margins_multigraph.csv")

print(df_binary)
print(df_multigraph)
