# We would like to compare our method against that of 
# "Efficient and Exact Sampling of Simple Graphs with
# Given Arbitrary Degree Sequence"

# Apparently the weights that result from our algorithm are much more tightly concentrated
# owing to our adaptive estimate, so the convergence for evaluating expectations
# should be much faster (although I believe still not particularly competitive with MCMC)

import matrix_count
import matplotlib.pyplot as plt

# Distribute with a power law
test_margin = []
for i in range(100):
    test_margin.append(int(30*((i+40)/50)**-2))

if sum(test_margin) % 2 == 1:
    test_margin[-1] += 1

print(test_margin)

num_samples = 1000

entropies = []

for sample_num in range(num_samples):
    sample, entropy = matrix_count.sample_symmetric_matrix(test_margin, binary_matrix=False)
    entropies.append(entropy)
    print(sample_num,entropy)

binary_entropies = []

for sample_num in range(num_samples):
    sample, entropy = matrix_count.sample_symmetric_matrix(test_margin, binary_matrix=True)
    binary_entropies.append(entropy)
    print(sample_num,entropy)

plt.hist(entropies, bins=50, density=True, alpha=0.5, label="Non-binary")
plt.hist(binary_entropies, bins=50, density=True, alpha=0.5, label="Binary")
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.legend()
plt.title("Entropy of sampled matrices")
plt.savefig("sample_binary.png")