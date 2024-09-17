
import matplotlib.pyplot as plt
import numpy as np

# Each entropy reading is on a new line
basename = "example2"
sample_filename = f"../outputs/{basename}.txt"

entropies = []
with open(sample_filename, "r") as f:
    for line in f:
        entropies.append(float(line))

# Plot the log density of the entropy values
plt.hist(entropies, bins=50, density=True, log=True)
plt.xlabel("Entropy")
plt.ylabel("Density")
plt.title("Entropy Distribution")

plt.savefig(f"../figures/{basename}.png")
