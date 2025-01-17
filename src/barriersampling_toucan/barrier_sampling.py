import numpy as np
from scipy.special import gamma, gammaincc
from scipy.stats import multivariate_normal
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class BarrierSampler:
    def __init__(self, loglikelihood, prior, gen_prior_samples, num_live_points, t=1, qmax=2):
        """
        Creates an instance of BarrierSampler.
        Arguments:
        - loglikelihood: The loglikelihood function of the model.
        - prior: The prior probability distribution function of the model.
        - gen_prior_samples: A function that takes an integer n and return n samples from the prior.
        - num_live_points: The number of live points to use.
        - t: The shape parameter of the log barrier term.
        - qmax: The domain parameter of the log barrier term.
        """
        self.theta_loglikelihood = loglikelihood
        self.theta_prior = prior
        self.theta_gen_prior = gen_prior_samples
        self.num_live_points = num_live_points
        self.t = t
        self.qmax = qmax
        self.total_proposals = 0 # total number of LRPS proposals
        self.accepted_proposals = 0 # number of accepted LRPS proposals
        self.jump_distances = [] # distances from the start of each Markov Chain to the end (in LRPS)

    def get_sampling_method(self, string):
        d = {'lrps_metropolis': self.lrps_theta_metropolis,
             'lrps_hmc': self.lrps_theta_hmc}
        return d[string]

    def get_samples(self):
        """
        After self.run has finished, this function returns samples from the posterior.
        It returns as many samples as the effective sample size allows.
        """
        weights = np.exp(np.array(self.dead_points_loglikelihood)) * np.array(self.dead_points_volume)
        effective_sample_size = int(len(weights) / (1 + (weights / weights.mean() - 1)**2).mean())
        thetas = np.array([t for (t,q) in self.dead_points])
        return thetas[np.random.choice(np.arange(len(thetas)), size=effective_sample_size, replace=True, p=weights/sum(weights)),:]

    def q_cdf(self, q):
        if q <= 1:
            return 0
        if q >= self.qmax:
            return 1
        cdf = lambda q: np.log(q)**(1/self.t)
        return cdf(q) / cdf(self.qmax)

    def q_cdf_vectorized(self, q):
        out = np.empty(q.shape)
        out[q <= 1] = 0
        out[q >= self.qmax] = 1
        cdf = lambda q: np.log(q)**(1/self.t)
        out[(q > 1) & (q < self.qmax)] = cdf(q[(q > 1) & (q < self.qmax)]) / cdf(self.qmax)
        return out

    def log_q_cdf_vectorized(self, k):
        out = np.empty(k.shape)
        out[k <= 0] = 0
        out[k >= np.log(self.qmax)] = 1
        cdf = lambda k: k**(1/self.t)
        out[(k > 0) & (k < np.log(self.qmax))] = cdf(k[(k > 0) & (k < np.log(self.qmax))]) / cdf(np.log(self.qmax))
        return out

    def q_cdf_jax(self, q):
        return jax.lax.cond(q <= 1.0,
                        lambda q: 0.0,
                        lambda q: jax.lax.cond(q >= self.qmax,
                                     lambda q: 1.0,
                                     lambda q: jnp.log(q)**(1/self.t) / jnp.log(self.qmax)**(1/self.t),
                                     q),
                        q)

    def log_q_cdf_jax(self, k):
        return jax.lax.cond(k <= 0,
                    lambda k: 0.0,
                    lambda k: jax.lax.cond(k >= np.log(self.qmax),
                                 lambda k: 1.0,
                                 lambda k: k**(1/self.t) / np.log(self.qmax)**(1/self.t),
                                 k),
                    k)
                
    def q_gen_prior(self, n):
        cdf_inv = lambda p: np.exp((p * np.log(self.qmax)**(1/self.t))**self.t)
        ps = np.random.uniform(low=0, high=1, size=n)
        qs = cdf_inv(ps)
        return qs

    def q_loglikelihood(self, q):
        return np.log(1/q)

    def loglikelihood(self, points):
        return [self.theta_loglikelihood(theta) + self.q_loglikelihood(q) for (theta, q) in points]

    def lrps_q(self, theta, loglikelihood_min):
        ratio = np.exp(self.theta_loglikelihood(theta) - loglikelihood_min)
        qs_sampled = 0
        qs_accepted = 0
        while True:
            q_proposals = self.q_gen_prior(100000)
            qs_sampled += 100000
            if (q_proposals < ratio).any():
                qs_accepted += (q_proposals < ratio).sum()
                i = np.where(q_proposals < ratio)[0][0]
                return q_proposals[i]

    def lrps(self, loglikelihood_min):
        theta = self.lrps_theta(loglikelihood_min)
        q = self.lrps_q(theta, loglikelihood_min)
        return theta, q

    def q_evidence(self):
        return (gamma(1/self.t) - gamma(1/self.t)*gammaincc(1/self.t, np.log(self.qmax))) / (self.t * np.log(self.qmax)**(1/self.t))
    
    def go(self, stop_threshold=0.01, num_iter=None, sampling_method='lrps_metropolis', metropolis_starting_step_size=1, metropolis_num_steps=50, hmc_jax_loglike=None, hmc_jax_logprior=None, hmc_path_len=3, hmc_step_size=0.5, hmc_chain_len=5, hmc_adaptive_step_size = True, vis_iter=None, vis_lo=-5, vis_hi=5):
        """
        Run Nested Sampling with Barriers.
        Arguments:
        -- stop_threshold: Stopping criterion. If the quotient of estimated remaining evidence and the accumulated evidence is below this value, the algorithm stops.
        -- num_iter: Number of iterations to run at most.
        -- sampling_method: One of the strings in the dictionary in get_sampling_method.
        -- metropolis_starting_step_size: If using Metropolis for LRPS, this defines the starting step size.
        -- metropolis_num_steps: If using Metropolis for LRPS, this defines the number of steps per Markov chain for one sample.
        -- hmc_jax_loglike: REQUIRED IF USING HMC FOR LRPS. The loglikelihood function of the model, defined using jax.numpy.
        -- hmc_jax_logprior: REQUIRED IF USING HMC FOR LRPS. The prior probability function of the model, defined using jax.numpy.
        -- hmc_path_len: If using HMC for LRPS, this defines the length of the integration path of each proposal.
        -- hmc_step_size: If using HMC for LRPS, this defines the length of the integration steps. Each path contains int(hmc_path_len / hmc_step_size) - 1) steps.
        -- hmc_chain_len: If using HMC for LRPS, this defines the number of paths integrated for one sample.
        -- hmc_adaptive_step_size: If using HMC for LRPS, this defines whether the step size should be automatically adapted in each iteration. If true, the path length, and step size are both multiplied by 0.99 if a proposal was rejected and 1.01 if a proposal was accepted.
        -- vis_iter: If this is set to an integer n, a visualization of the likelihood contour and the proposals of the nth iteration is generated.
        -- vis_lo: If vis_iter is set to an integer and this is set to a float, this defines the lower limits of the visualized area.
        -- vis_lo: If vis_iter is set to an integer and this is set to a float, this defines the upper limits of the visualized area.
        """
        thetas = self.theta_gen_prior(self.num_live_points)
        qs = self.q_gen_prior(self.num_live_points)
        self.live_points = [(theta, q) for (theta, q) in zip(thetas, qs)]
        loglikelihoods = self.loglikelihood(self.live_points)
        volume = 1
        self.dead_points = []
        self.dead_points_loglikelihood = []
        self.dead_points_volume = []
        keep_going = True
        evidence = 0
        evidence_remaining_approx = 1
        self.vis_iter = vis_iter
        self.vis_lo = vis_lo
        self.vis_hi = vis_hi
        self.iteration = 0
        self.lrps_theta = self.get_sampling_method(sampling_method)
        if sampling_method =='lrps_metropolis':
            self.metropolis_step_size = metropolis_starting_step_size
            self.metropolis_num_steps = metropolis_num_steps
        if sampling_method == 'lrps_hmc':
            if not hmc_jax_loglike:
                print('Have to specify loglikelihood in jax using argument hmc_jax_loglike for lrps_hmc.')
                return 0
            self.hmc_path_len = hmc_path_len
            self.hmc_step_size = hmc_step_size
            self.hmc_chain_len= hmc_chain_len
            self.hmc_neg_logprob = lambda theta, min_loglike: -hmc_jax_logprior(theta) - jnp.log(self.log_q_cdf_jax(hmc_jax_loglike(theta) - min_loglike))
            self.hmc_neg_logprob = jax.jit(self.hmc_neg_logprob)
            self.hmc_neg_logprob_grad = jax.jit(jax.grad(self.hmc_neg_logprob))
            self.hmc_momentum_logpdf = jax.jit(lambda x: jax.scipy.stats.norm.logpdf(x, loc=0, scale=1))
            self.hmc_adaptive_step_size = hmc_adaptive_step_size
            self.path_lengths = []
            self.hmc_loglike_grad = jax.jit(jax.grad(hmc_jax_loglike))
            
        while keep_going:
            # find the current likelihood threshold
            loglikelihood_minimum = min(loglikelihoods)
            loglikelihood_minimum_idx = np.argmin(loglikelihoods)
            if self.vis_iter == self.iteration: # only for 2D problems
                x = np.linspace(vis_lo, vis_hi, 500)
                X, Y = np.meshgrid(x, x)
                points = np.vstack((X.flatten(), Y.flatten())).transpose()
                Z = self.theta_loglikelihood(points).reshape((500, 500))
                plt.contour(X, Y, Z, levels=[loglikelihood_minimum], colors=['black'], alpha=0.8)

            # shrink volume
            volume_shell_fraction = np.random.beta(1, self.num_live_points)
            volume_shell = volume * volume_shell_fraction
            volume = volume - volume_shell
            
            # eject this point
            self.dead_points.append(self.live_points[loglikelihood_minimum_idx])
            self.dead_points_loglikelihood.append(np.copy(loglikelihood_minimum))
            self.dead_points_volume.append(volume_shell)
            
            replacement_theta, replacement_q = self.lrps(loglikelihood_minimum)
            self.live_points[loglikelihood_minimum_idx] = replacement_theta, replacement_q
            loglikelihoods[loglikelihood_minimum_idx] = self.theta_loglikelihood(replacement_theta) + self.q_loglikelihood(replacement_q)
            
            # approximate remaining evidence
            evidence += volume_shell * np.exp(loglikelihood_minimum) # update estimate of evidence
            evidence_remaining_approx = volume * np.exp(max(loglikelihoods)) # update estimate of remaining evidence

            if num_iter:
                if self.iteration >= num_iter:
                    keep_going = False
            else:
                if evidence != 0 and np.exp(np.log(evidence_remaining_approx) - np.log(evidence)) < stop_threshold:
                    keep_going = False
            #if self.iteration % 10000 == 0: print(f'Iteration: {self.iteration}  Z: {evidence/self.q_evidence}')
            self.iteration += 1
        return evidence / self.q_evidence()

    def lrps_theta_metropolis(self, loglikelihood_minimum):
        """
        Generate a sample from the likelihood-restricted prior of the model parameters using Metropolis.
        """
        def p(x):
            loglike = self.theta_loglikelihood(x)
            if loglike < loglikelihood_minimum:
                return 0
            likelihood_ratio = np.exp(loglike - loglikelihood_minimum)
            return self.theta_prior(x)*self.q_cdf(likelihood_ratio)
        start_index = np.random.choice(self.num_live_points)
        previous = self.live_points[start_index][0]
        steps = [previous]
        for i in range(self.metropolis_num_steps):
            self.total_proposals += 1
            proposal = previous + np.random.randn(previous.shape[0])*self.metropolis_step_size
            if p(proposal) == 0: # never go outside the likelihood contour
                if i < self.metropolis_num_steps // 2: self.metropolis_step_size *= 0.99
                continue
            elif i < self.metropolis_num_steps // 2: self.metropolis_step_size *= 1.01
            r = p(proposal) / p(previous)
            u = np.random.uniform(0,1)
            if r > u:
                self.accepted_proposals += 1
                steps.append(proposal)
                previous = proposal
        if self.iteration == self.vis_iter:
            steps = np.array(steps)
            plt.scatter(steps[0,0], steps[0,1], color='black')
            plt.scatter(steps[1:,0], steps[1:,1], marker='x', color='blue', alpha=0.5)
            plt.plot(steps[:,0], steps[:,1], color='blue', alpha=0.3)
        self.jump_distances.append(np.linalg.norm(self.live_points[start_index][0] - previous))
        return previous

    def lrps_theta_hmc(self, loglikelihood_minimum):
        """
        Generate a sample from the likelihood-restricted prior of the model parameters using HMC.
        """
        neg_logprob = lambda theta: self.hmc_neg_logprob(theta, loglikelihood_minimum)
        if self.vis_iter == self.iteration:
            x = np.linspace(self.vis_lo, self.vis_hi, 500)
            X, Y = np.meshgrid(x, x)
            points = np.vstack((X.flatten(), Y.flatten())).transpose()
            Z = np.array([neg_logprob(p) for p in points]).reshape((500, 500))
            plt.imshow(Z, alpha=0.5, extent=(self.vis_lo, self.vis_hi, self.vis_lo, self.vis_hi), origin='lower')
        neg_logprob_grad = lambda theta: self.hmc_neg_logprob_grad(theta, loglikelihood_minimum)
        previous = self.live_points[np.random.choice(self.num_live_points)][0]
        proposal = self.hmc_run(neg_logprob, neg_logprob_grad, self.hmc_chain_len, previous, loglikelihood_minimum)
        self.jump_distances.append(np.linalg.norm(previous - proposal))
        return proposal

    def hmc_run(self, neg_logprob, neg_logprob_grad, n_samples, initial_position, loglikelihood_minimum):
        """
        Runs HMC to generate a sample from the likelihood-restricted prior.
        """
        dVdq = neg_logprob_grad
        samples = [initial_position]
        if self.vis_iter == self.iteration:
            self.hmc_step_size = self.hmc_path_len/50
            plt.scatter(samples[0][0], samples[0][1], color='black')
        # Saving time by drawing all required momenta in one go.
        # If initial_position is a 10d vector and n_samples is 100, we want
        # 100 x 10 momentum draws, then iterate over the rows.
        size = (n_samples,) + initial_position.shape[:1]
        for p0 in np.random.standard_normal(size=size):
            self.total_proposals += 1
            # Integrate over our path to get a new position and momentum
            if self.vis_iter == self.iteration: 
                q_new, p_new, steps = self.hmc_leapfrog_vis(dVdq, samples[-1], p0, loglikelihood_minimum)
            else:
                q_new, p_new = self.hmc_leapfrog(dVdq, samples[-1], p0, loglikelihood_minimum)
            if q_new == None:
                samples.append(np.copy(samples[-1]))
                if self.hmc_adaptive_step_size:
                    self.hmc_path_len *= 0.99
                    self.hmc_step_size *= 0.99
                if self.vis_iter == self.iteration:
                    plt.plot(steps[:,0], steps[:,1], color='red', alpha=0.7)
                continue
            # Check Metropolis acceptance criterion
            start_log_p = neg_logprob(samples[-1]) - np.sum(self.hmc_momentum_logpdf(p0))
            new_log_p = neg_logprob(q_new) - np.sum(self.hmc_momentum_logpdf(p_new))
            if self.theta_loglikelihood(q_new) > loglikelihood_minimum and np.log(np.random.uniform(0,1)) < start_log_p - new_log_p:
                self.accepted_proposals += 1
                samples.append(q_new)
                if self.hmc_adaptive_step_size:
                    self.hmc_path_len *= 1.01
                    self.hmc_step_size *= 1.01
                if self.vis_iter == self.iteration: 
                    plt.scatter(q_new[0], q_new[1], marker='x', color='black')
                    plt.plot(steps[:,0], steps[:,1], color='black', alpha=0.7, linewidth=1)
            else:
                samples.append(np.copy(samples[-1]))
                if self.hmc_adaptive_step_size:
                    self.hmc_path_len *= 0.99
                    self.hmc_step_size *= 0.99
                if self.vis_iter == self.iteration:
                    plt.plot(steps[:,0], steps[:,1], color='red', alpha=0.7, linewidth=1)
        self.path_lengths.append(self.hmc_path_len)
        return samples[-1]

    def hmc_leapfrog(self, dVdq, q, p, loglikelihood_minimum):
        """
        The leapfrog integrator for HMC. If the particle moves outside the likelihood contour, this returns None.
        """
        q, p = np.copy(q), np.copy(p)
        p -= self.hmc_step_size * dVdq(q) / 2  # half step
        for _ in range(int(self.hmc_path_len / self.hmc_step_size) - 1):
            q += self.hmc_step_size * p  # whole step
            if self.theta_loglikelihood(q) > loglikelihood_minimum:
                p -= self.hmc_step_size * dVdq(q)  # whole step
            else:
                return None, None # if we move outside the likelihood contour, return None and reject this sample
        q += self.hmc_step_size * p  # whole step
        p -= self.hmc_step_size * dVdq(q) / 2  # half step
        # momentum flip at end
        return q, -p

    def hmc_leapfrog_vis(self, dVdq, q, p, loglikelihood_minimum):
        """
        Version of the leapfrog integrator that includes visualization.
        """
        q, p = np.copy(q), np.copy(p)
        steps = [q]
        p -= self.hmc_step_size * dVdq(q) / 2  # half step
        for _ in range(int(self.hmc_path_len / self.hmc_step_size) - 1):
            old_q = q
            q += self.hmc_step_size * p  # whole step
            if self.theta_loglikelihood(q) > loglikelihood_minimum:
                p -= self.hmc_step_size * dVdq(q)  # whole step
                steps.append(q)
            else:
                return None, None, np.array(steps) # if we move outside the likelihood contour, return None and reject this sample
        q += self.hmc_step_size * p  # whole step
        p -= self.hmc_step_size * dVdq(q) / 2  # half step
        steps.append(q)
        # momentum flip at end
        return q, -p, np.array(steps)