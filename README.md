# Barrier Sampling

This package implements Nested Sampling with Barriers as introduced in my thesis (which is included in the GitHub repository).

It lets you do Bayesian statistical inference on generative models.
See also the original Nested Sampling paper by Skilling \[1\].

## Usage

To use this algorithm, you need to implement:
- the model's loglikelihood and prior PDF
- a function generating samples from the prior
- (if using HMC for LRPS) the model's loglikelihood and prior log-PDF in jax
 
See the example folder for usage samples, and take a look at the docstrings in `barriersampler.py` for more detailed information.

## Likelihood-Restricted Prior Sampling

As a modified version of Nested Sampling \[1\], this algorithm needs to do likelihood-restricted prior sampling.
For LRPS, two methods have been implemented:
- the Metropolis algorithm \[2\]
- Hamiltonian Monte Carlo \[3\]

If you want to implement your own method of LRPS, I suggest forking this repository, adding a new method to the `BarrierSampler` class, and registering it in `get_sampling_method`.

## References

- \[1\] John Skilling. "Nested sampling for general Bayesian computation." Bayesian Analysis 1 (4) 833 - 859, 2006. https://doi.org/10.1214/06-BA127
- \[2\] Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, Edward Teller. "Equation of State Calculations by Fast Computing Machines." The Journal of Chemical Physics 21 (6) 1087–1092, 1953. https://pubs.aip.org/aip/jcp/article-abstract/21/6/1087/202680/Equation-of-State-Calculations-by-Fast-Computing
- \[3\] Simon Duane, A.D. Kennedy, Brian J. Pendleton, Duncan Roweth. "Hybrid Monte Carlo." Physics Letters B 195 (2) 216-222, 1987. https://doi.org/10.1016/0370-2693(87)91197-X