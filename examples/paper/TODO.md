# TODO

## matrix_count

Overall purpose:
Consider rebranding the package as margin-matrices, since this may be more descriptive of the package's purpose. 
matrix-count may be a bit overbroad, since we are really interested in counting and sampling matrices with given margins. 
Although, perhaps we could include some other scenarios where margins are not constrained, although I would expect that many of those combinatorial
problems are analytically tractable.

Python package to count and sample integer matrices with given margins. 

Tagline: "Count and sample integer matrices with given margins"

### Code
#### Documentation/package publishing
When locally building docs with nox -s "docs" get certain warnings
/mnt/c/Users/mjerdee/Documents/GitHub/package-development/matrix-count/docs/modules.md:: WARNING: py:class reference target not found: ArrayLike [ref.class] 
and links are not created to these types

Include the LaTeX expressions and code examples inside the documentation.

#### Python code
##### Features to add:

Get the test coverage up to 100%, we can check which of the lines are not covered by the tests from pages like:
https://app.codecov.io/github/maxjerdee/matrix-count/blob/main/src%2Fmatrix_count%2Festimate.py

Example of how we can test logging, also wondering if verbose should just set the logging level to print and we always write to logs?:

```python
LOGGER = logging.getLogger(__name__)

def test_func(caplog):
    caplog.set_level(logging.INFO)
    run_function()
    assert 'Something bad happened!' in caplog.text
``` 

Check whether the pseudo DM estimate is useful in practice, decide whether it should be the default. 

Should mention in the logger for the binary case when there are no solutions, and return the appropriate value
Ex: 13,  5, 14,  5,  5, 15,  9,  3,  1,  6,  2,  7,  2,  6,  6,  1

Include warning when the SIS estimate of the count has really not converged at all (when +/- is about 1)

##### Bugs: 

Seems like you can get a nan result if sample is given a matrix with a row sum of 0. 
Example:
```python
matrix_count.sample_symmetric_matrix([4, 3, 3, 2, 2, 1, 1, 0], binary_matrix=True, seed=0)
```

Getting nan result although the margin passes Erdos-Gallai, this might be because we are not sorting..
```python
matrix_count.sample_symmetric_matrix([2, 16, 14, 30, 9, 24, 11, 25, 12, 4, 12, 8, 11, 11, 24, 1, 8, 10, 13, 16, 20, 9, 13, 12, 8, 19, 22, 7, 19, 1, 6, 3], binary_matrix=True, seed=0)
```

Errors in the inference of expectations over the ensemble seem to be overly optimistic. Figure out why this is the case.

#### SIS code (c++)
Generally it would be nice to figure out a way to quickly debug/develop the c++ code in place without needing to copy into another folder and include a main function. [I actually deleted the SIS testing stuff anyway, so will need to come up with some new way to perform the tests.]

Add ability to pass a timeout to the sampler so that we can exit if taking a single sample is taking too long. 

Improve the performance of the SIS code:
- Check if rearranging the degrees in the SIS algorithm can improve the performance for the multigraph case. (I'm guessing that largest to smallest might yield the best performance of the estimate, although smallest to largest is more computationally efficient). 

Compute the complexity of the algorithm and include in the paper. Also run some tests showing that this is indeed the rough scaling. (speed.py)

For the binary matrix case, the efficiency can be improved by better checks of the Erdos-Gallai condition. 
- Adjusted this to not rerun the Erdos-Gallai condition each time, only recalculate in full for each row. This can be made more efficient still O(n^3) -> O(m^2) if we globally define the EG values and only locally update as we move from row to row. 

- Implement a sparse formatting for the matrices. 

#### MCMC code (c++)
Include code which performs configuration model rewiring of the graphs. This will allow us to sample from the space of graphs with a given degree sequence and compare how these local moves perform compared to the SIS algorithm for purposes of computing expectations of functions over the space of graphs.
- Currently the code appears to be converging to the wrong number? Need to check if the configuration model MCMC is actually evenly exploring the possibilities. 

- Add functionality to get MCMC samples from the specified ensemble using rewiring techniques. The SIS can be a useful way to initially seed this process.


### Paper
In writing the paper use this package in order to generate the figures. 

Note that we have had to do something fairly non-trivial and interesting to make the sampling work for the binary_matrix case, so the main body of the paper should include both of these cases. I think that the paper should really be framed as the minimum possible background required to numerically converge to the true number of multigraphs and graphs of a given degree sequence. This new method may also be a more efficient way to sample from the asymmetric binary matrices. 

Paper outline:
1. Introduction
- Show what the problem is, go over some of the literature that has considered the problem. 
2. Estimation
- Moment matching technique in this case. Demonstrate that this is possible, also give the estimate for the case of fixed diagonal sum (since this will be needed)
3. Sampling and Counting
- Show how this estimate can be used in order to sample and count matrices.
4. Estimate performance
- Using the sampling method to calculate the true numbers, show the typical performance of the main estimate over various ranges. 
- Changes in size and density.
- Note the weakness at approximating counts for highly heterogeneous margins. 
5. Further estimates
- Note that in at least the unconstrained setting we may also obtain a 3rd order estimate which performs slightly better. (Should have some more detail on what exactly goes wrong when trying to push this to 4th order)

Appendix:
- More detailed derivations of a variety of different cases

TODO:
- Verify that we cannot use a pseudo-DM estimate for the binary case to compute the one-point marginals. 

## Future work

This will then be necessary for three papers on microcanonical network models.
1. Degree-corrected random multigraphs and the resulting SBMs (block sums) [For this the performance of the block sum estimate is crucial]
2. Alternative formulation of the microcanonical DCSBM 

Generally the idea is to show that having some control over this type of problem allows for the creation of a wider variety of microcanonical models of discrete data. 


Goals:
- Get this paper out the door in the next few weeks (which seems like it might be kind of ridiculous)
- Work on the microcanonical DCSBM paper at the same time. 
- Similarly package the alternative microcanonical DCSBM paper as a c++/python package for usability. 
Steps for this paper:
    - Ensure that the way that we shield the estimate from particularly unwidely margins is valid.. (Xiaoran)
    - Compare the inference and the (exact) description lengths obtained by each model. 
    - Write it up with Xiaoran
    - Ask Mark for comments and submit. 

- Think a bit about whether it is possible to use similar direct sampling techniques in order to estimate more generic partition functions or to compute Bayesian evidences.

- This would look like taking a sample and using an approximation of the remaining partition function to inform the next sample taken. 