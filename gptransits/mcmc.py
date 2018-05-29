import numpy as np
import emcee
import timeit
import logging
import sys

# def mcmc(data, gp, nwalkers=20, iterations=2000, burnin=500):
def run_mcmc(model, settings):

	init_time_mcmc = timeit.default_timer()
	
	# Draw samples from the prior distributions to have initial values for all the walkers
	init_params = model.gp.gp_model.prior_sample(settings.nwalkers) # Should be replaced with model.prior_sample(nwalkers) after MeanModel is implemented
	ndim = init_params.shape[1]

	# Instanciate the sampler. Parameters for likelihhood calculation are included in model object
	sampler = emcee.EnsembleSampler(settings.nwalkers, ndim, model.log_likelihood)

	# Burn-in
	logging.info('Runnning Burn-in ...')
	progress_bool = not bool(logging.getLogger().getEffectiveLevel()-20)
	burnin_params, _, _ = sampler.run_mcmc(init_params, settings.burnin, progress=progress_bool)
	sampler.reset()

	# Main run
	logging.info('Running MCMC ...')
	results, _, _ = sampler.run_mcmc(burnin_params, settings.iterations, progress=progress_bool)

	time_mcmc = timeit.default_timer() - init_time_mcmc
	logging.info("MCMC execution time: {:.4f} usec\n".format(time_mcmc))
	
	# Get the samples from the MCMC run and calculate the median and 1-sigma uncertainties
	samples = sampler.flatchain
	results = np.percentile(samples.T, [16,50,84], axis=1)

	return samples, results