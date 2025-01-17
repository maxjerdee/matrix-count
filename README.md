# matrix-count

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/matrix-count/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/matrix-count/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/matrix-count
[conda-link]:               https://github.com/conda-forge/matrix-count-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/maxjerdee/matrix-count/discussions
[pypi-link]:                https://pypi.org/project/matrix-count/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/matrix-count
[pypi-version]:             https://img.shields.io/pypi/v/matrix-count
[rtd-badge]:                https://readthedocs.org/projects/matrix-count/badge/?version=latest
[rtd-link]:                 https://matrix-count.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

# matrix-count

### Estimating nonnegative integer matrix counting problems

##### Maximilian Jerdee

We provide estimates for a variety of counting problems defined over non-negative integer matrices. For example, we estimate

- $\Omega^S(\mathbf{r})$: count of nonnegative integer symmetric matrices with a given row sum $\mathbf{r}$ and assorted estimates from an upcoming paper.
    - $\Omega_{2}^S(\mathbf{r})$: count of such matrices with even diagonal entries.
    - $\Omega^S(\mathbf{r},d), \Omega_{2}^S(\mathbf{r},d)$: counts of such matrices with diagonal entries that sum to $d$.
- $\Omega(\mathbf{r},\mathbf{c})$: count of integer symmetric matrices with given row sums $\mathbf{r}$ and column sums $\mathbf{c}$ as described in Jerdee, Kirkley, Newman (2022) https://arxiv.org/abs/2209.14869

We also give estimates for a generalized counting problem where the matrices are weighted by a Dirichlet-multinomial factor on their entries. For example, we may more generally define $\Omega^S(\mathbf{r}|\alpha)$ as the sum over the set $A^S(\mathbf{r})$ of all symmetric matrices with sum $\mathbf{r}$ with a Dirichlet-multinomial weighting on the matrices so that $$\Omega^S(\mathbf{r}|\alpha) = \sum_{X \in A^S(\mathbf{r})} \prod_{r \leq s}\frac{\Gamma(X_{rs} + \alpha)}{\Gamma(\alpha)\Gamma(X_{rs} + 1)}.$$
Note that $\alpha = 1$ reduces to the earlier uniform counting of cases. This more general estimate acts as the partition function of a generalized random multigraph model. 

We may also estimate the number of symmetric non-negative integer matrices with a particular margin and given "block" sums $\mathbf{M}$.

The above quantities may also be counted only among matrices with entries of zeros and ones, indicated with a subscript of "0." Under the adjacency matrix interpretation, this is then a restriction to simple graphs. 

The `\estimates` directory contains python implementations of linear time and maximum entropy estimates of these quantities. 

The `\SIS` directory applies the linear time estimates of these quantities to numerically approximate the quantities with sequential importance sampling implemented in c++. In principle these algorithms should converge to the true results although it make take a prohibitively long time to do so for large cases. 

Run the python script `run_examples.py` for examples of using these implementations on the test cases in the `\data` directory. To use the sequential importance sampling code it must first be compiled by running `make` in the `\SIS` directory. 
