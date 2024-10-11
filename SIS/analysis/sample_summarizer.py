# Aggregrate the results of the SIS and store the linear time and SIS estimates and error

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
from math import lgamma

# Linear time estimates
def log_binom(n,m):
    return lgamma(n+1) - lgamma(m+1) - lgamma(n-m+1)

def alpha_2(m,n):
    numerator = -((1 + n) * (2 + n)) + 2 * m * (2 + n * (2 + n))
    denominator = (-2 + n) * (1 + n) + 2 * m * (2 + n)
    result = numerator / denominator
    return result

# Return the two alphas so that the mixed distribution matches the covariances
def alpha_3_mixed(m,n):
    term1 = (1 + n) ** 2 * (128 + n * (160 + n * (224 + n * (136 + n * (79 + 8 * n * (3 + n))))))
    term2 = 2 * m * (1 + n) * (256 + n * (576 + n * (768 + n * (729 + n * (439 + 16 * n * (14 + n * (4 + n)))))))
    term3 = 4 * m ** 2 * (128 + n * (416 + n * (672 + n * (753 + 8 * n * (73 + n * (42 + n * (18 + n * (5 + n))))))))
    sqrt_term = np.sqrt((-1 + 2 * m - n) * n * (2 * m + n + n ** 2) * (term1 - term2 + term3))
    
    numerator = 32 + 2 * m * (1 + n) * (-64 + n * (-96 + n * (-81 - 19 * n + 4 * n ** 3))) + 4 * m ** 2 * (32 + n * (72 + n * (85 + 4 * n * (13 + n * (5 + n))))) + n * (88 + n * (92 - n * (-39 + n * (10 + n * (17 + 4 * n))))) + sqrt_term
    numerator_alt = 32 + 2 * m * (1 + n) * (-64 + n * (-96 + n * (-81 - 19 * n + 4 * n ** 3))) + 4 * m ** 2 * (32 + n * (72 + n * (85 + 4 * n * (13 + n * (5 + n))))) + n * (88 + n * (92 - n * (-39 + n * (10 + n * (17 + 4 * n))))) - sqrt_term

    denominator = 8 * m ** 2 * (4 + n * (2 + n)) * (4 + n * (5 + 2 * n)) + 2 * (1 + n) ** 2 * (16 + (-2 + n) * n * (2 + 3 * n ** 2)) + 4 * m * (1 + n) * (-32 + n * (-24 + (-1 + n) * n * (8 + 3 * n)))

    alpha1A, alpha1B = numerator / denominator, numerator_alt / denominator

    term1 = (1 + n) ** 2 * (128 + n * (160 + n * (224 + n * (136 + n * (79 + 8 * n * (3 + n))))))
    term2 = 2 * m * (1 + n) * (256 + n * (576 + n * (768 + n * (729 + n * (439 + 16 * n * (14 + n * (4 + n)))))))
    term3 = 4 * m ** 2 * (128 + n * (416 + n * (672 + n * (753 + 8 * n * (73 + n * (42 + n * (18 + n * (5 + n))))))))
    sqrt_term = np.sqrt((-1 + 2 * m - n) * n * (2 * m + n + n ** 2) * (term1 - term2 + term3))
    
    numerator = 16 + 2 * m * (1 + n) * (-32 + n * (-48 + n * (-45 - 5 * n + 2 * n ** 3))) + n * (44 + n * (46 - n * (5 + 2 * n) * (-3 + n * (4 + n)))) + 4 * m ** 2 * (16 + n * (36 + n * (47 + 2 * n * (13 + n * (5 + n))))) - sqrt_term
    numerator_alt = 16 + 2 * m * (1 + n) * (-32 + n * (-48 + n * (-45 - 5 * n + 2 * n ** 3))) + n * (44 + n * (46 - n * (5 + 2 * n) * (-3 + n * (4 + n)))) + 4 * m ** 2 * (16 + n * (36 + n * (47 + 2 * n * (13 + n * (5 + n))))) + sqrt_term

    denominator = 2 * (1 + n) ** 2 * (8 + n * (-2 + n + 3 * (-1 + n) * n ** 2)) + 16 * m * (-4 + n * (-7 - 4 * n + n ** 3)) + 8 * m ** 2 * (8 + n * (14 + n * (11 + n * (3 + n))))

    alpha2A, alpha2B = numerator / denominator, numerator_alt / denominator

    return alpha1A, alpha1B, alpha2A, alpha2B

# Return the two alphas so that the mixture matches the 
def alpha_3(m,n):

    common_numerator = (
        8 + 2 * m * (1 + n) * (-16 + n * (-24 + n * (-21 - 4 * n + n**3)))
        - n * (-22 + n * (-23 + n * (-9 + n * (1 + n) * (4 + n))))
        + 4 * m**2 * (8 + n * (18 + n * (22 + n * (13 + n * (5 + n)))))
    )

    sqrt_term = np.sqrt(
        (-1 + 2 * m - n) * n * (2 * m + n + n**2)
        * (-((1 + n) * (4 + n**2)) + 2 * m * (4 + n * (4 + n * (2 + n))))
        * (-((1 + n) * (4 + n * (5 + n * (6 + n * (3 + n))))) 
           + 2 * m * (4 + n * (9 + n * (10 + n * (8 + n * (3 + n))))))
    )

    denominator = (
        2 * m * (1 + n) * (-16 + (-2 + n) * n * (2 + n) * (3 + n))
        + (1 + n)**2 * (8 + n * (-2 + n - 3 * n**2 + 2 * n**3))
        + 4 * m**2 * (8 + n * (2 + n) * (7 + n * (2 + n)))
    )

    alpha_plus = (common_numerator + sqrt_term) / denominator
    alpha_minus = (common_numerator - sqrt_term) / denominator

    return alpha_plus, alpha_minus

def log_Omega_2(ks):
    n = len(ks)
    m = sum(ks)/2
    alpha = alpha_2(m,n)
    result = log_binom(m + n*(n+1)/2 - 1, n*(n+1)/2 - 1) 
    log_P = - log_binom(2*m + n*alpha - 1, n*alpha - 1)
    for k in ks:
        log_P += log_binom(k + alpha - 1, alpha - 1)
    result += log_P
    # print(log_P, alpha)
    # print(ks, result, alpha)
    return result

# print(log_Omega_2([]))

# overflow protected log_sum_exp
def log_sum_exp(x):
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

def log_Omega_3(ks):
    n = len(ks)
    m = sum(ks)/2
    alpha_plus, alpha_minus = alpha_3(m,n)
    result = log_binom(m + n*(n+1)/2 - 1, n*(n+1)/2 - 1) 
    log_P_plus = - log_binom(2*m + n*alpha_plus - 1, n*alpha_plus - 1)
    for k in ks:
        log_P_plus += log_binom(k + alpha_plus - 1, alpha_plus - 1)
    log_P_minus = - log_binom(2*m + n*alpha_minus - 1, n*alpha_minus - 1)
    for k in ks:
        log_P_minus += log_binom(k + alpha_minus - 1, alpha_minus - 1)
    # print(log_P_plus, log_P_minus, alpha_plus, alpha_minus)
    result += log_sum_exp([log_P_plus, log_P_minus])-np.log(2)
    return result

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
results = {"filename":[],"m":[], "n":[], "estimate":[], "estimate_3":[], "value":[], "error":[], "num_samples":[]}

files = listdir(input_folder)
files.sort()
# Load the samples from the SIS
for filename in files:
  # print(filename)
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
      linear_estimate = log_Omega_2(margin)
      results["estimate"].append(linear_estimate)
      # if filename[:3] == "9-1":
      minus_log_mean = log_sum_exp(SIS_samples)-np.log(len(SIS_samples))
      linear_estimate_3 = log_Omega_3(margin)
      # print(linear_estimate,linear_estimate_3,minus_log_mean)
      results["estimate_3"].append(linear_estimate_3)
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
