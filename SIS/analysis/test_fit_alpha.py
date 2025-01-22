# Test the algorithm for finding the MLE for alpha for the Dirichlet Multinomial distribution

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
from math import lgamma
from scipy.special import gammaln
from scipy.optimize import minimize

def generate_margin_alpha(m, K, alpha, shift):
    if 2*m < K:
        raise ValueError("m must be greater than or equal to K")
    ps = np.random.dirichlet(alpha * np.ones(K))
    #print(f"ps is {ps}, corresponding K is {K}, alpha is {alpha}")
    if(shift):
        # Sample from a multinomial and shift by 1
        ks = np.random.multinomial(2*m - K, ps) + 1
    else:
        # Sample from a multinomial and remove 0 entries
        ks = np.random.multinomial(2*m, ps)
        ks = ks[ks != 0]
    # Sort in ascending order (makes the SIS more efficient)
    ks = np.sort(ks)
    return ks

# Define the negative log-likelihood function for the symmetric Dirichlet-Multinomial
def neg_dirichlet_multinomial_loglik(alpha, x):
    K = len(x)  # Number of categories
    n = np.sum(x)  # Total count
    alpha = float(alpha)  # Ensure alpha is a scalar
    
    # Log-likelihood components
    log_lik = (
        gammaln(K * alpha) 
        + gammaln(n + 1) 
        - gammaln(n + K * alpha) 
        + np.sum(gammaln(x + alpha)) 
        - K * gammaln(alpha)
        - np.sum(gammaln(x + 1))
    )
    return -log_lik  # Return negative log-likelihood for maximization

# Find MLE for alpha
def fit_alpha(margin_seq, shift):
    if(shift):
        margin_seq_shifted = [val - 1 for val in margin_seq] # subtract 1 from each entry
        
        # Initial guess for alpha
        initial_alpha = 0.5

        # Minimize the negative log-likelihood to find the MLE of alpha
        result = minimize(
            neg_dirichlet_multinomial_loglik,
            initial_alpha,
            args=(np.array(margin_seq_shifted),),
            method="L-BFGS-B",
            bounds=[(1e-3, None)],  # Ensure alpha stays positive
        )

        # Extract the MLE for alpha
        fitted_alpha = result.x[0]
        return fitted_alpha

alpha_vals = 10.0 ** np.linspace(0, -3, num=10, endpoint=False)

#ks = generate_margin_alpha(441, 10, alpha_vals[1], True)
ks = [114, 114, 185, 187, 282]
fitted_alpha = fit_alpha(ks, True)

print(alpha_vals[1])
print(f"Ks is: {ks}")
print(f"fitted alpha is: {fitted_alpha}")