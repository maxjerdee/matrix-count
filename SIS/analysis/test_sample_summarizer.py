# Summarize the samples of a single file

import numpy as np

# overflow protected log_sum_exp
def log_sum_exp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))


filename = "../out.txt"

with open(filename) as f:
    SIS_samples = []
    for line in f.readlines():
      if line[:12] == "log_Omega = ":
        linear_estimate = float(line[12:])
      else:
        SIS_samples.append(float(line))
    if len(SIS_samples) > 1:
      SIS_samples = np.array(SIS_samples)
      SIS_samples.sort()
      print(np.mean(np.exp(SIS_samples)))
      minus_log_mean = -(log_sum_exp(-SIS_samples)-np.log(len(SIS_samples)))
      print(minus_log_mean, linear_estimate)