import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel
from scipy.signal import medfilt

def plot_corner(model, samples, settings):
	labels = model.gp.gp_model.get_parameters_latex()
	params = model.gp.gp_model.get_parameters()
	corner_plot = corner.corner(samples, labels=labels, quantiles=[0.5], show_titles=True, title_fmt='.3f', truths=params)
	return corner_plot

def plot_gp(model, data, settings):

	gp_plot, ax = plt.subplots(num=1, figsize=(14, 7))
	
	# Plot initial data with errors in both subplots
	try:
		err = model.error
	except AttributeError:
		err = None
	ax.errorbar(model.time/(24*3600), model.flux, yerr=err, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
	x = np.linspace(model.time[0], model.time[-1], num=2*model.time.size)

	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = model.gp.predict(model.flux, x/1e6)
	std = np.sqrt(np.diag(cov))
	std = np.nan_to_num(std)
	
	ax.plot(x/(24*3600), mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.5)
	ax.fill_between(x/(24*3600), mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma', linewidth=0.6)
	#ax.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
	#ax.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
	
	ax.set_xlabel('Time [days]',fontsize="medium")
	ax.set_ylabel('Flux[ppm]',fontsize="medium")
	ax.tick_params(axis='both', which='major', labelsize="medium")
	ax.legend(loc='upper left', fontsize="medium")

	return gp_plot

def plot_psd(model, data, settings, include_data=True, parseval_norm=False):

	ls_dict = {"Granulation": "--", "OscillationBump": "-.", "WhiteNoise": ":"}
	alpha_dict = {"Granulation": 0.8, "OscillationBump": 0.8, "WhiteNoise": 0.6}
	label_dict = {"Granulation": "Granulation", "OscillationBump": "Gaussian envelope", "WhiteNoise": "White noise"}

	freq, power_list = model.gp.gp_model.get_psd(model.time)
	nobump_power = np.zeros(freq.size)
	full_power = np.zeros(freq.size)
	
	psd_plot, ax = plt.subplots(figsize=(14, 7))
	for name, power in power_list:
		# TODO: Need alternative to fix white noise psd
		if name == "WhiteNoise":
			power += 0.0
		if name != "OscillationBump":
			nobump_power += power
		full_power += power

		ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], label=label_dict[name])
	
	ax.loglog(freq, nobump_power, ls='--', color='r', label='Model without gaussian')
	ax.loglog(freq, full_power, ls='-', color='k', label='Full Model')
	
	# ax.set_title('KIC012008916')
	ax.set_xlabel(r'Frequency [$\mu$Hz]',fontsize="large")
	ax.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]',fontsize="large")
	ax.tick_params(labelsize="large")
	
	if include_data:
		# Psd from data
		freq2, power = LombScargle(model.time/1e6, model.flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)

		if parseval_norm:
			# Parseval Normalization (Enrico)
			resolution = freq2[1]-freq2[0]
			variance = np.var(model.flux)
			power = (power * variance / sum(power)) / resolution

		else:
			# Celerite Normalization
			power = power / model.time.size

		ax.loglog(freq2, power, 5, color='k', alpha=0.4)
		# ax.loglog(freq2, medfilt(power, 5), color='k', alpha=0.4)
		ax.loglog(freq2, convolve(power, Box1DKernel(10)), color='k', alpha=0.6)
	
	ax.set_xlim([1,300])
	ax.set_ylim([1e-1, 1e4])
	ax.legend(fontsize="large", loc="lower left")

	return psd_plot