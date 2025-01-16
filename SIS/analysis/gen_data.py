<<<<<<< HEAD
# Generate test margins for SIS

# These will be sampled uniformly at random from the set of all possible margins
# for a given matrix size which are positive vectors of sum 2m and length K

# Import packages
import numpy as np
import graph_tool.all as gt
import math\
    
# Helper functions
## We can sample as a Dirichlet-Multinomial distribution (note that these are basically degree sequences)
def generate_margin(m, K):
    if 2*m < K:
        raise ValueError("m must be greater than or equal to K")
    ps = np.random.dirichlet(np.ones(K))
    # Sample from a multinomial
    ks = np.random.multinomial(2*m - K, ps) + 1
    # Sort in ascending order (makes the SIS more efficient)
    ks = np.sort(ks)
    return ks

## We can sample as a Dirichlet-Multinomial distribution for different prespecified parameter alpha adjusting the heterogeneity (note that these are basically degree sequences)
## We try two different ways to handle the situations where the margin contains entries being zero
def generate_margin_alpha(m, K, alpha, shift):
    if 2*m < K:
        raise ValueError("m must be greater than or equal to K")
    
    ps = np.random.dirichlet(alpha * np.ones(K))
    
    if(shift):
        # Sample from a multinomial and shift by 1
        ks = np.random.multinomial(2 * m - K, ps) + 1
    else:
        # Sample from a multinomial and remove 0 entries
        ks = np.random.multinomial(2 * m, ps)
        ks = ks[ks != 0]
    
    # Sort in ascending order (makes the SIS more efficient)
    ks = np.sort(ks)
    return ks

# Main program

# Define range of alpha
alpha_vals = 10.0 ** np.linspace(0, -3, num=10, endpoint=False) ## 10 ** (-3) is too small

# Synthetic
## Generally we are more interested in the constant degree regime anyway (m/K fixed)
m_vals = 100 * 2 ** np.array([2, 3, 4, 5]) # 200, 400, 800, 1600, 3200, 6400
K_vals = 10 * 2 ** np.array([1, 2, 3]) # 20, 40, 80, 160, 320, 640

# Numebr of margins to generate for each combination
num_trials = 10
for m in m_vals:
    for K in K_vals:
        for i, alpha in enumerate(alpha_vals):
            if 2 * m >= K:
                for v in range(num_trials):
                    margin = generate_margin_alpha(m, K, alpha, True)
                    out_filename = f"../../data/more_verbose_test_margins/margin_{str(m).zfill(4)}_{str(K).zfill(3)}_{str(i).zfill(1)}_{str(v).zfill(1)}.txt"
                    with open(out_filename, "w") as f:
                        f.write(f"{' '.join(map(str, margin))}")






# # Real data (to make it compatible; fix m)
# ## Load the network
# g = gt.collection.ns["polbooks"]

# ## Print basic information about the network
# n = g.num_vertices()
# m = g.num_edges()

# print(f"Number of nodes is {n}, number of edges is {m}")

# # Generate samples
# num_trials = 10

# for K in range(2, 10): # TODO: change range of number of groups; for polbooks network, 10 is reasonably large
#     for i, alpha in enumerate(alpha_vals):
#         if 2 * m >= K:
#             for v in range(num_trials):
#                 margin = generate_margin_alpha(m, K, alpha, True)
#                 out_filename = f"../../data/verbose_test_margins/margin_{str(i).zfill(1)}_{str(K).zfill(2)}_{str(v).zfill(1)}.txt" ## TODO: change path
#                 with open(out_filename, "w") as f:
#                     f.write(f"{' '.join(map(str, margin))}")
                    
=======
# Generate test margins for SIS

# These will be sampled uniformly at random from the set of all possible margins
# for a given matrix size which are positive vectors of sum 2m and length n

import numpy as np

# Generally we are particularly interested in the constant degree regime anyway (m/n fixed)
m_vals = 100*2**np.array([1, 2, 3, 4, 5, 6]) # 200, 400, 800, 1600, 3200, 6400
n_vals = 10*2**np.array([1, 2, 3, 4, 5, 6]) # 20, 40, 80, 160, 320, 640

# We can sample as a Dirichlet-Multinomial distribution (note that these are basically degree sequences)
def generate_margin(m, n):
    if 2*m < n:
        raise ValueError("m must be greater than or equal to n")
    ps = np.random.dirichlet(np.ones(n))
    # Sample from a multinomial
    ks = np.random.multinomial(2*m - n, ps) + 1
    # Sort in ascending order (makes the SIS more efficient)
    ks = np.sort(ks)
    return ks

# Numebr of margins to generate for each combination
num_trials = 10
for m in m_vals:
    for n in n_vals:
        if 2*m >= n:
            for v in range(num_trials):
                margin = generate_margin(m, n)
                out_filename = f"../data/test_margins/margin_{str(m).zfill(4)}_{str(n).zfill(3)}_{str(v).zfill(1)}.txt"
                with open(out_filename, "w") as f:
                    f.write(f"{' '.join(map(str, margin))}")
                
>>>>>>> f25b5b1caa0b2f45b3e011f11766d9572cbc9330
