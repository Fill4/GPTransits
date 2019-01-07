import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.stats import LombScargle
from astropy.convolution import convolve, Box1DKernel
from scipy.signal import medfilt

def plot_corner(model, samples, settings):
	labels = model.get_parameters_latex()
	params = model.get_parameters()
	corner_plot = corner.corner(samples, labels=labels, quantiles=[0.5], show_titles=True, title_fmt='.3f', truths=params)
	return corner_plot

def plot_gp(model, settings):

	gp_plot, ax = plt.subplots(figsize=(14, 8))
	gp_plot.subplots_adjust(left=.15, bottom=.15, right=.97, top=.95)
	
	# Plot initial data with errors in both subplots
	try:
		err = model.error
	except AttributeError:
		err = None
	ax.errorbar(model.time/(24*3600), model.flux, yerr=err, fmt=".k", capsize=0, label= 'Data points', markersize='5', elinewidth=1)
	x = np.linspace(model.time[0], model.time[-1], num=(model.time[-1] - model.time[0]) / 60 / 30 *5) # Timespan(seconds) * to_min / cadence_in_min * 5

	# Plot conditional predictive distribution of the model in upper plot
	mu, var = model.gp.predict(model.flux - model.mean_model.eval(), x/1e6, return_var=True)
	std = np.sqrt(var)
	std = np.nan_to_num(std)
	
	# mean_model_lc = model.mean_model.eval()
	mean_model_lc = np.zeros_like(mu)

	ax.plot(x/(24*3600), mu + mean_model_lc, color="#ff7f0e", label= 'Model', linewidth=1)
	ax.fill_between(x/(24*3600), mu+mean_model_lc+std, mu+mean_model_lc-std, color="#ff7f0e", alpha=0.5, edgecolor="none", label= r"1$\sigma$", linewidth=1.2)
	# ax.fill_between(x/(24*3600), mu+mean_model_lc+2*std, mu+mean_model_lc-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"2$\sigma$", linewidth=1.2)
	# ax.fill_between(x/(24*3600), mu+mean_model_lc+3*std, mu+mean_model_lc-3*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= r"3$\sigma$", linewidth=1.2)

	ax.plot(x/(24*3600), mean_model_lc, color="red", label= 'Transit', linewidth=1)
	ax.plot(x/(24*3600), mu, color="blue", label= 'GP', linewidth=1)
	

	ax.set_xlabel('Time (days)', fontsize=settings.fontsize)
	ax.set_ylabel('Flux (ppm)', fontsize=settings.fontsize)
	ax.tick_params(labelsize=settings.fontsize)
	ax.legend(loc='upper left', fontsize=settings.fontsize)




	# Extra zoomed plot
	zoom_plot, ax2 = plt.subplots(figsize=(14, 8))
	zoom_plot.subplots_adjust(left=.15, bottom=.15, right=.97, top=.95)

	# 3-day time index
	index = np.argmax(model.time/(24*3600) > 3)

	if err is not None:
		err = err[:index]

	ax2.errorbar(model.time[:index]/(24*3600), model.flux[:index], yerr=err, fmt=".k", capsize=0, label= 'Flux', markersize='5', elinewidth=1)
	
	ax2.plot(x[:720]/(24*3600), mu[:720], color="#ff7f0e", label= 'Mean', linewidth=1)
	ax2.fill_between(x[:720]/(24*3600), mu[:720]+std[:720], mu[:720]-std[:720], color="#ff7f0e", alpha=0.5, edgecolor="none", label= r"1$\sigma$", linewidth=1.2)
	# ax2.fill_between(x[:720]/(24*3600), mu[:720]+2*std[:720], mu[:720]-2*std[:720], color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"2$\sigma$", linewidth=1.2)
	# ax2.fill_between(x[:720]/(24*3600), mu[:720]+3*std[:720], mu[:720]-3*std[:720], color="#ff7f0e", alpha=0.3, edgecolor="none", label= r"3$\sigma$", linewidth=1.2)

	ax2.set_xlabel('Time (days)', fontsize=settings.fontsize)
	ax2.set_ylabel('Flux (ppm)', fontsize=settings.fontsize)
	ax2.tick_params(labelsize=settings.fontsize)
	ax2.legend(loc='lower left', fontsize=settings.fontsize)



	return gp_plot, zoom_plot

def plot_psd(model, settings, include_data=True, parseval_norm=False):

	ls_dict = {"Granulation": "--", "OscillationBump": "-", "WhiteNoise": ":"}
	alpha_dict = {"Granulation": 0.8, "OscillationBump": 0.8, "WhiteNoise": 0.6}
	label_dict = {"Granulation": "Granulation", "OscillationBump": "Oscillations envelope", "WhiteNoise": "White noise"}

	freq, power_list = model.gp.gp_model.get_psd(model.time)
	nobump_power = np.zeros(freq.size)
	full_power = np.zeros(freq.size)
	
	psd_plot, ax = plt.subplots(figsize=(14, 8))
	psd_plot.subplots_adjust(left=.15, bottom=.15, right=.97, top=.95)

	unique_labels = []
	first_gran = False
	for name, power in power_list:
		if name != "OscillationBump":
			nobump_power += power
		full_power += power

		# if label_dict[name] not in unique_labels:
		# 	unique_labels.append(label_dict[name])
		# 	ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], lw=2, label=label_dict[name])
		# else:
		# 	ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], lw=2)
		
		if name == 'Granulation':
			if not first_gran:
				ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], lw=2, label=label_dict[name]+' 1')
				first_gran = True
			else:
				ax.loglog(freq, power, ls='-.', color='b', alpha=alpha_dict[name], lw=2, label=label_dict[name]+' 2')
		else:
			ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], lw=2, label=label_dict[name])

	ax.loglog(freq, nobump_power, ls='--', color='r', label='Model without gaussian', lw=2, alpha=0.8)
	ax.loglog(freq, full_power, ls='-', color='r', label='Full Model', lw=2)
	
	# ax.set_title('KIC012008916')
	ax.set_xlabel(r'Frequency ($\mu$Hz)',fontsize=settings.fontsize)
	ax.set_ylabel(r'PSD (ppm$^2$/$\mu$Hz)',fontsize=settings.fontsize)
	ax.tick_params(labelsize=settings.fontsize)
	
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

		ax.loglog(freq2, power, color='k', alpha=0.3, lw=1, label='Data')
		# ax.loglog(freq2, medfilt(power, 5), color='k', alpha=0.4)
		ax.loglog(freq2, convolve(power, Box1DKernel(10)), color='k', alpha=0.5, lw=2, label='Smooth Data')
	
	ax.set_xlim([1,300])
	ax.set_ylim([1e-1, 1e4])
	ax.legend(fontsize=settings.fontsize-4, loc="lower left")

	return psd_plot