# Aggregrate the results of the SIS and store the linear time and SIS estimates and error

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd

# overflow protected log_sum_exp
def log_sum_exp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

margins_folder = "../data/test_margins"
input_folder = "../outputs/test_margins"
output_csv = "test_margins.csv"

# margins_folder = "../data/test_margins"
# input_folder = "../outputs/test_margins_2"
# output_csv = "test_margins_2.csv"

# margins_folder = "../data/real_degrees/margins" # To obtain m and n
# input_folder = "../outputs/real_degrees" # To obtain the SIS samples (and the linear time estimate)
# output_csv = "real_degrees.csv" # To store the estimate, information, and error

# filename,m,n,estimate,value,error,num_samples
# dict to store the results
results = {"filename":[],"m":[], "n":[], "estimate":[], "value":[], "error":[], "num_samples":[]}

files = listdir(input_folder)
files.sort()
# Load the samples from the SIS
for filename in files:
  with open(f"{input_folder}/{filename}") as f:
    SIS_samples = []
    for line in f.readlines():
      if line[:12] == "log_Omega = ":
        linear_estimate = float(line[12:])
      else:
        SIS_samples.append(float(line))
    if len(SIS_samples) > 1:
      SIS_samples = np.array(SIS_samples)
      SIS_samples.sort()
      # print(SIS_samples)
      # print(log_sum_exp(-SIS_samples))
      # Read the actual margin from the margins file to get m and n
      with open(f"{margins_folder}/{filename}") as f:
        margin = list(map(int,f.readline().split()))
        m = sum(margin)/2
        n = len(margin)
      
      results["filename"].append(filename)
      results["m"].append(m)
      results["n"].append(n)
      results["estimate"].append(linear_estimate)
      minus_log_mean = -(log_sum_exp(-SIS_samples)-np.log(len(SIS_samples)))
      # print(log_sum_exp(-SIS_samples),minus_log_mean,linear_estimate)
      results["value"].append(minus_log_mean)
      results["error"].append(-1) # TODO: compute the approximate error of the SIS estimate
      results["num_samples"].append(len(SIS_samples))

      # print(filename)
      # minus_log_mean = -(log_sum_exp(-SIS_samples)-np.log(len(SIS_samples)))
      # minus_log_mean_x2 = -(log_sum_exp(-2*SIS_samples)-np.log(len(SIS_samples)))
      # print(f"minus_log_mean: {minus_log_mean}, 2*minus_log_mean: {2*minus_log_mean}, minus_log_mean_x2: {minus_log_mean_x2}")
      # # var_x = 2*mean - mean_x2 - np.log(len(SIS_samples))/2 # log((E (x^2) - (Ex)^2)/sqrt(n))
      # minus_log_var = 2*minus_log_mean - np.log(np.exp(-minus_log_mean_x2 + 2*minus_log_mean) - 1)
      # print(f"minus_log_var: {minus_log_var}")
      # print(SIS_samples[:3],minus_log_mean)
      # print(linear_estimate)
      # print(len(SIS_samples))
      # print(-2*(log_sum_exp(-SIS_samples)-np.log(len(SIS_samples))))
      # print(-(log_sum_exp(-2*SIS_samples)-np.log(len(SIS_samples))))

# Write the results to a csv
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
