# Generate test margins for SIS

# These will be sampled uniformly at random from the set of all possible margins
# for a given matrix size which are positive vectors of sum 2m and length n

import numpy as np

# Generally we are more interested in the constant degree regime anyway (m/n fixed)
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
                