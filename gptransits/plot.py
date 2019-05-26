import sys

import numpy as np
import matplotlib.pyplot as plt
import corner
import celerite
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel

from .convergence import gelman_rubin

days_to_microsec = (24*3600) / 1e6
font = 15

# ----------------------------- PLOTTING ----------------------------------
def gelman_rubin_plot(chain, steps=100, pnames=None):
	R_hat = np.zeros([steps, chain.shape[2]])
	V = np.zeros([steps, chain.shape[2]])
	W = np.zeros([steps, chain.shape[2]])
	
	for s in range(steps):
		R_hat[s], V[s], W[s] = gelman_rubin(chain[:,0:int((s+1)/steps*chain.shape[1]),:])
	
	nrows = int((chain.shape[2]+1)/2)
	fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows*3))
	plt.subplots_adjust(bottom=0.1, top=0.95, left=0.08, right=0.9, wspace=0.4, hspace=0.4)
	ax = ax.flatten()

	x = np.array(range(1,steps+1))/steps * chain.shape[1]
	for p in range(chain.shape[2]):
		if pnames is not None:
			ax[p].set_title(f"Parameter - {pnames[p]}")
		else:
			ax[p].set_title(f"Parameter - {p+1}")

		ax[p].plot(x, R_hat[:,p], color='tab:red', lw=2, alpha=0.7, label=r"$\hat{R}$")
		ax[p].set_xlabel("Iterations")
		ax[p].set_ylabel(r"$\hat{R}$", color='tab:red')
		ax[p].legend(loc='upper left')

		ax2 = ax[p].twinx()
		ax2.plot(x, V[:,p], label=r"$\hat{V}$", color='tab:blue', lw=2, ls='--', alpha=0.6)
		ax2.plot(x, W[:,p], label="W", color='tab:blue', lw=2, ls=':', alpha=0.6)
		ax2.set_ylabel("Variance Estimates", color='tab:blue')
		ax2.legend()

	return fig

def parameter_hist(chain, params, bins=50, pnames=None):
	
	w = chain.reshape([chain.shape[0]*chain.shape[1], chain.shape[2]])

	nrows = int(np.ceil(chain.shape[2] / 2))
	fig, ax = plt.subplots(ncols=2, nrows=nrows, figsize=(14,nrows*4))
	fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)
	ax = ax.ravel()

	for i in range(chain.shape[2]):
		if pnames is None:
			ax[i].set_title(f"Parameter - {i+1}")
		else:
			ax[i].set_title(f"Parameter - {pnames[i]}")
		ax[i].hist(w[:,i], bins=bins, density=True, fill=False, hatch='\\', alpha=0.7)
		
		if "median" in params:
			ax[i].axvline(params["median"][i], color="r", alpha=0.8, label="Median")
			if "hpd_down" in params and "hpd_up" in params:
				ax[i].axvline(params["hpd_up"][i], color="r", ls="--", alpha=0.8, label="HPD+")
				ax[i].axvline(params["hpd_down"][i], color="r", ls="--", alpha=0.8, label="HPD-")
		if "mapv" in params:
			ax[i].axvline(params["mapv"][i], color="b", alpha=0.8, label="MAP")
		if "modes" in params:
			ax[i].axvline(params["modes"][i], color="g", alpha=0.8, label="Mode")
		ax[i].legend()
	
	return fig

def trace_plot(chain, posterior, pnames=None, downsample=10):
	
	assert posterior.shape[1] == chain.shape[0]

	nrows = chain.shape[2]+1
	fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(10,nrows*3))
	fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)

	# Plot the ln-probabilty for each of the walkers
	# assert(chain.shape[1] % downsample == 0)
	x = np.linspace(1, chain.shape[1], num=int(np.ceil(chain.shape[1]/downsample)))
	ax[0].set_title("Ln-Posterior")
	for w in range(posterior.shape[1]):
		ax[0].plot(x, posterior[::downsample,w], alpha=0.5)

	for i in range(1, len(ax)):
		if pnames is None:
			ax[i].set_title(f"Parameter - {i}")
		else:
			ax[i].set_title(f"Parameter - {pnames[i-1]}")
		for w in range(chain.shape[0]):
			ax[i].plot(x, chain[w,::downsample,i-1], alpha=0.5)
	
	return fig

def posterior_hist(posterior):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

	ax.set_title("Ln-Posterior Histogram")
	ax.hist(posterior.reshape(posterior.shape[0]*posterior.shape[1]), bins=200, density=True) 
	ax.set_xlabel("Ln-Posterior")
	ax.set_ylabel("Probability")

	return fig

def corner_plot(chain, truths=None, pnames=None, downsample=2):
	sampled_chain = chain[:,::downsample,:]
	samples = sampled_chain.reshape(sampled_chain.shape[0]*sampled_chain.shape[1], sampled_chain.shape[2])
	fig = corner.corner(samples, quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt='.3f', truths=truths, labels=pnames, bins=50, plot_contours=True)

	return fig

def gp_plot(gp_model, mean_model, params, time, flux, flux_err=None, zoom=0.1, offset=0.0, oversample=5):

	cadence = time[1]-time[0]
	x = np.linspace(time[0], time[-1], num=int((time[-1]-time[0])/cadence*oversample)) # Timespan / cadence * 5
	gp_ndim = gp_model.get_parameters_names().size

	# Setup the transit model if there is a transit. Get both the normal sampled and oversampled mean.
	if mean_model is not None:
		mean_model.init_model(time, time[1]-time[0], 1)
		mean = mean_model.get_value(params["median"][gp_ndim:], time)
		mean_model.init_model(x, time[1]-time[0], 1)
		overmean = mean_model.get_value(params["median"][gp_ndim:], x)
	
	# Setup the gp from the gp_model
	kernel = gp_model.get_kernel(params["median"][:gp_ndim])
	gp = celerite.GP(kernel)
	if flux_err is None:
		gp.compute(time*days_to_microsec)
	else:
		gp.compute(time*days_to_microsec, yerr=flux_err)

	# Calculate conditional predictive distribution of the model in upper plot. TODO: Add mean model to subtract here
	if mean_model is not None:
		mu, var = gp.predict(flux-mean, x*days_to_microsec, return_var=True)
		model_lc = mu + overmean
	else:
		mu, var = gp.predict(flux, x*days_to_microsec, return_var=True)
		model_lc = mu
	std = np.sqrt(var)
	std = np.nan_to_num(std)
	
	font=23
	# Overall figure
	fig, ax = plt.subplots(figsize=(14, 10))
	fig.subplots_adjust(left=.12, bottom=.10, right=.95, top=.92)

	# Plot initial data with errors in both subplots
	ax.errorbar(time, flux, yerr=flux_err, fmt=".k", capsize=0, label="Data points", markersize="5", elinewidth=1.2)

	# Plot the GP predictive distribution
	ax.plot(x, model_lc, color="#ff7f0e", linewidth=1, label= 'GP')
	ax.fill_between(x, model_lc+std, model_lc-std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"1$\sigma$", linewidth=1.2)
	# ax.fill_between(x, model_lc+2*std, model_lc-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"2$\sigma$", linewidth=1.2)
	# ax.fill_between(x, model_lc+3*std, model_lc-3*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"3$\sigma$", linewidth=1.2)
	
	if mean_model is not None:
		ax.plot(x, overmean, color="blue", linewidth=1, label='Transit')

	# ax.set_title("GP", fontsize=font)
	ax.set_xlabel('Time (days)', fontsize=font)
	ax.set_ylabel('Flux (ppm)', fontsize=font)
	ax.legend(loc='upper left', fontsize=font-4)
	ax.tick_params(labelsize=font-1)

	# Zoomed in figure
	fig_zoom, ax_zoom = plt.subplots(figsize=(14, 10))
	fig_zoom.subplots_adjust(left=.12, bottom=.10, right=.95, top=.92)

	if (zoom + offset) > 1.0:
		print("Zoom + Offset percentages are larger than 1.0")
		sys.exit()
	
	low = np.where(time >= (((time[-1]-time[0]) * offset) + time[0]) )[0][0]
	up = np.where(time >= (((time[-1]-time[0]) * (offset + zoom)) + time[0]))[0][0]
	olow = np.where(x >= (((time[-1]-time[0]) * offset) + time[0]))[0][0]
	oup = np.where(x >= (((time[-1]-time[0]) * (offset + zoom)) + time[0]))[0][0]

	time_zoom = time[low:up]
	flux_zoom = flux[low:up]
	if flux_err is None:
		flux_err_zoom = None
	else:
		flux_err_zoom = flux_err[low:up]
	x_zoom = x[olow:oup]
	mu_zoom = mu[olow:oup]
	std_zoom = std[olow:oup]
	model_lc_zoom = model_lc[olow:oup]
	if mean_model is not None:
		overmean_zoom = overmean[olow:oup]

	# Plot initial data with errors in both subplots
	ax_zoom.errorbar(time_zoom, flux_zoom, yerr=flux_err_zoom, fmt=".k", capsize=0, label="Data points", markersize="8", elinewidth=1)

	# Plot the GP predictive distribution
	ax_zoom.plot(x_zoom, model_lc_zoom, color="#ff7f0e", label= 'GP', linewidth=2)
	ax_zoom.fill_between(x_zoom, model_lc_zoom+std_zoom, model_lc_zoom-std_zoom, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"1$\sigma$", linewidth=1.5)
	# ax_zoom.fill_between(x_zoom, model_lc_zoom+2*std_zoom, model_lc_zoom-2*std_zoom, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"2$\sigma$", linewidth=1.2)
	# ax_zoom.fill_between(x_zoom, model_lc_zoom+3*std_zoom, model_lc_zoom-3*std_zoom, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"3$\sigma$", linewidth=1.2)

	if mean_model is not None:
		ax_zoom.plot(x_zoom, overmean_zoom, color="blue", linewidth=2, label='Transit')
	
	# ax_zoom.set_title("GP Zoom-in", fontsize=font)
	ax_zoom.set_xlabel('Time (days)', fontsize=font)
	ax_zoom.set_ylabel('Flux (ppm)', fontsize=font)
	ax_zoom.legend(loc='upper left', fontsize=font-4)
	ax_zoom.tick_params(labelsize=font-1)


	return fig, fig_zoom

def psd_plot(gp_model, params, time, flux, include_data=True, parseval_norm=True):
	
	fig, ax = plt.subplots(figsize=(14, 10))
	fig.subplots_adjust(left=.10, bottom=.10, right=.95, top=.95)

	names = gp_model.get_component_names()
	gp_ndim = gp_model.get_parameters_names().size
	freq, psd_vector = gp_model.get_psd(params["median"][:gp_ndim], time)
	assert names.size == psd_vector.shape[0]

	# nobump_power = np.zeros(freq.size)
	full_psd = np.zeros(freq.size)
	for i in range(len(names)):
		full_psd += psd_vector[i]
		ax.loglog(freq, psd_vector[i], ls="--", alpha=0.8, lw=2, label=names[i])

	# ax.loglog(freq, nobump_power, ls='--', color='r', label='Model without gaussian', lw=2, alpha=0.8)
	ax.loglog(freq, full_psd, ls="-", color='k', label='Power Spectrum', lw=2)
	
	ax.set_title("Power Spectrum", fontsize=font)
	ax.set_xlabel(r'Frequency ($\mu$Hz)', fontsize=font)
	ax.set_ylabel(r'PSD (ppm$^2$/$\mu$Hz)', fontsize=font)
	ax.tick_params(labelsize=font)
	
	if include_data:
		# Psd from data
		freq2, power = LombScargle(time*days_to_microsec, flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)

		# Parseval Normalization (Enrico)
		if parseval_norm:
			resolution = freq2[1]-freq2[0]
			variance = np.var(flux)
			power = (power * variance / sum(power)) / resolution

		# Celerite Normalization
		else:
			power = power / model.time.size

		ax.loglog(freq2, power, color='k', alpha=0.3, lw=1, label='Data')
		ax.loglog(freq2, convolve(power, Box1DKernel(10)), color='k', alpha=0.5, lw=2, label='Smoothed Data')
	
	ax.set_xlim([1,300])
	ax.set_ylim([1e-1, 5e4])
	ax.legend(loc="lower left", fontsize=font)

	return fig
