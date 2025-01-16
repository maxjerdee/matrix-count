# Sample partitions from the real world network using graph tool

# Import packages
import numpy as np
import graph_tool.all as gt
import math
import os

# Helper functions
## Function: Convert state to partition
def state_to_partition(g, state):
    colors = []
    for vertex in g.get_vertices():
        colors.append(state.get_blocks()[vertex])
    
    # Preprocess the group partiton recovered
    element_to_dense = {element: index for index, element in enumerate(sorted(set(colors)))}
    dense_sequence = [element_to_dense[element] for element in colors]
    return dense_sequence

# Main program

## Import networks; 
# g = gt.collection.ns["polbooks"] ## TODO: change to import different networks
# output_dir = 'polbooks_test_margins' ## TODO: change directory accordingly

g = gt.collection.ns["facebook_friends"]
output_dir = 'facebook_friends_test_margins'

## Run sequence of MCMC (merge-split) and record the partition after each sweep
state = gt.BlockState(g)
num_sweeps = 10000 ## TODO: might change the number of sweeps
partitions = []

for _ in range(num_sweeps):
    state.multiflip_mcmc_sweep(niter = 1)
    partitions.append(state_to_partition(g, state))

## Only record each partition once
unique_partitions = set(tuple(partition) for partition in partitions)
print(f"Number of unique partitions is: {len(unique_partitions)}")

for i, partition in enumerate(unique_partitions):
    ## Record degree sequence
    community_degree_sums = {}
    for node, community in enumerate(partition):
        degree = g.vertex(node).out_degree()
        if community not in community_degree_sums:
            community_degree_sums[community] = 0
        community_degree_sums[community] += degree
    
    ## Sort in ascending order
    sorted_community_sums = sorted(community_degree_sums.values())
    file_name = f"../../data/{output_dir}/partition_{i + 1}.txt"
    with open(file_name, 'w') as f:
        f.write(" ".join(map(str, sorted_community_sums)))
        