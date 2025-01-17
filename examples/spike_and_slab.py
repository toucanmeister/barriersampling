import numpy as np
from barriersampling import BarrierSampler
from scipy.stats import multivariate_normal
import jax
from jax.scipy.stats import multivariate_normal as jax_multivariate_normal
import jax.numpy as jnp
import math

## Running Nested Sampling with Barries on a spike-and-slab example

## Setting up a general spike-and-slab loglikelihood
spike_lo = -0.5
spike_hi = 0.5
spike_weights = np.array([100, 1])
def spike_loglikelihood(x, dim, modes, u, v):
    like = spike_weights[0] * multivariate_normal.pdf(x, mean=modes[0], cov=np.eye(dim)*u) + spike_weights[1] * multivariate_normal.pdf(x, mean=modes[1], cov=np.eye(dim)*v)
    if like == 0: # zero values would cause problems, so move to the next-highest possible float
        like = math.ulp(0.0)
    return np.log(like)
def spike_loglikelihood_jax(x, dim, modes, u, v):
    like = spike_weights[0] * jax_multivariate_normal.pdf(x, mean=modes[0], cov=jnp.eye(dim)*u) + spike_weights[1] * jax_multivariate_normal.pdf(x, mean=modes[1], cov=jnp.eye(dim)*v)
    return jnp.log(like)

# Using the above definition to make a 2D spike-and-slab loglikelihood
spike_sigmas = np.array([0.01, 0.1])
spike_2d_modes = [np.zeros(2), np.zeros(2)]
spike_2d_modes_jax = [jnp.zeros(2), jnp.zeros(2)]
def spike_2d_loglikelihood(x):
    return spike_loglikelihood(x, 2, spike_2d_modes, spike_sigmas[0], spike_sigmas[1])
def spike_2d_loglikelihood_jax(x):
    return spike_loglikelihood_jax(x, 2, spike_2d_modes_jax, spike_sigmas[0], spike_sigmas[1])

# Defining a uniform prior
def prior_d(theta, d, lo, hi):
    theta = np.atleast_2d(theta)
    return np.where(((theta >= lo) & (theta <= hi)).all(axis=1), 1 / ((hi-lo)**d), 0)
    

if __name__ == '__main__':
    dim = 2 
    loglikelihood = spike_2d_loglikelihood
    jax_loglikelihood = spike_2d_loglikelihood_jax
    modes = spike_2d_modes
    sigmas = spike_sigmas
    lo = spike_lo
    hi = spike_hi
    
    prior = lambda x: prior_d(x, dim, lo, hi) # prior pdf
    gen_prior = lambda n: np.random.uniform(low=lo, high=hi, size=(n,dim)) # prior sampler
    jax_logprior = lambda theta: jax.scipy.stats.uniform.logpdf(theta, loc=lo, scale=hi-lo).sum() # logprior in jax
    
    bs = BarrierSampler(loglikelihood, prior, gen_prior, 50) # create the BarrierSampler
    Z = bs.go(stop_threshold=1e-6, sampling_method='lrps_hmc', hmc_jax_loglike=jax_loglikelihood, hmc_jax_logprior=jax_logprior) # run the sampler using HMC for likelihood-restricted prior sampling (LRPS)
    print(f'Log evidence estimate: {np.log(Z)}')
    
    samples = bs.get_samples() # get as many posterior samples as the effective sample size allows