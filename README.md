# matrix-count

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][[pypi-link](https://pypi.org/project/pairwise-ranking/)] [![PyPI platforms][pypi-platforms]][[pypi-link](https://pypi.org/project/pairwise-ranking/)]

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

We provide analytic estimates and sampling-based algorithms for a variety of
counting problems defined over non-negative integer matrices.

For example, we may count the number of non-negative symmetric integer matrices
with even diagonal entries and a given row sum. This is the number of
(multi-)graphs with a given degree sequence. We can also estimate the number of
such matrices under combinations of the further conditions:

- Fixed total sum of diagonal entries.
- Fixed sum of entries in blocks of matrix. (Not yet implemented)

We also include methods for estimating the number of non-negative integer
matrices with a given row sum and column sum as described in Jerdee, Kirkley,
Newman (2022) https://arxiv.org/abs/2209.14869. (Not yet implemented)

These problems can also be generalized as sums over matrices $A$ weighted by a
Dirichlet-multinomial factor on their entries

$$w(A) = \prod_{i < j}\binom{A_{ij} + \alpha - 1}{\alpha - 1} \prod_i \binom{A_{ii}/2 + \alpha - 1}{\alpha - 1}.$$

Note that $\alpha = 1$ corresponds to the uniform count. This more general
estimate acts as the partition function of a generalized random multigraph
model.

## Installation

`matrix-count` may be installed through pip:

```bash
pip install matrix-count
```

## Typical usage

Once installed, the package can be imported as

```python
import matrix_count
```

Note that this is not `import matrix-count`.

The package can then be used to evaluate rapid analytic estimates of these
counting problems, to sample from the space of such matrices, and to converge to
the exact count of matrices.

```python
# Margin of a 8x8 symmetric non-negative integer matrix with even diagonal entries
margin = [10, 9, 8, 7, 6, 5, 4, 3]

# Estimate the logarithm of the number of symmetric matrices with given margin sum
# (number of multigraphs with given degree sequence)
estimate = matrix_count.estimate_log_symmetric_matrices(margin, verbose=True, alpha=1)
print("Estimated log count of symmetric matrices:", estimate)

num_samples = 1000
for t in range(num_samples):
    sample, entropy = matrix_count.sample_symmetric_matrix(margin)

# Count the number of such matrices
count, count_err = matrix_count.count_log_symmetric_matrices(
    margin, verbose=True, alpha=1
)
print("Log count of symmetric matrices:", count, "+/-", count_err)
```

Further usage examples can be found in the `examples` directory of the
repository.
