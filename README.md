# matrix_count

### Estimating nonnegative integer matrix counting problems

##### Maximilian Jerdee

We provide estimates for the following quantities and variants:

- $\Omega_S(\mathbf{r})$: count of nonnegative integer symmetric matrices with a given row sum and assorted estimates from the paper.
    - $\Omega_S^{(2)}(\mathbf{r})$: count of such matrices with even diagonal entries.
    - $\Omega_S(\mathbf{r},d), \Omega_S^{(2)}(\mathbf{r},d)$: counts of such matrices with diagonal entries that sum to $d$.
- $\Omega(\mathbf{r},\mathbf{c})$: count of integer symmetric matrices with given row and column sums as described in Jerdee, Kirkley, Newman (2022) https://arxiv.org/abs/2209.14869

We also give estimates for a generalized counting problem where the matrices are weighted by a Dirichlet-multinomial factor on their entries. 
- $\Omega_S(\mathbf{r};\alpha)$: sum over all such matrices with a Dirichlet-multinomial weighting of $$\Omega_S(\mathbf{r};\alpha) = \sum_{X \in A_S(\mathbf{r})} \prod_{r \leq s}\frac{\Gamma(X_{rs} + \alpha)}{\Gamma(\alpha)\Gamma(X_{rs} + 1)}.$$
    Note that $\alpha = 1$ reduces to the earlier uniform cases. We may 

The `\estimates` directory contains python implementations of linear time and maximum entropy estimates of these quantities. 

The `\SIS` directory applies the linear time estimates of these quantities to numerically approximate the quantities with sequential importance sampling implemented in c++. In principle these algorithms should converge to the true results although for large cases it make take a prohibitively long time to do so. 

Run the python script `run_examples.py` for examples of using these implementations on the test cases in the `\data` directory. To use the sequential importance sampling code it must first be compiled by running `make` in the `\SIS` directory. 
