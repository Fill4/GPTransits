#!/usr/bin/env python3

import os, sys
import pickle
import subprocess as sp
from pathlib import Path
from shutil import copyfile
import importlib

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .gp import GPModel
from .transit import BatmanModel
from .convergence import geweke, gelman_rubin, gelman_brooks
from .stats import mapv, mode, hpd
from .plot import *

code_path = "/mnt/c/work/astro/gptransits-tests/_run"
work_path = "/mnt/c/work/astro/gp-paper"

"""
-------------------------- CHAIN ANALYSIS --------------------------
DEPRECATED: Moved model analysis to the Model class
"""
"""
def analise_all():
	for mission in ["tess-sim", "kepler"]:	
		for model in [1,2]:
			for star in os.listdir(f"{work_path}/{mission}"):
				for run in os.listdir(f"{work_path}/{mission}/{star}"):
					analyse(mission, star, run, model)

def analysis(lc_file, config_file):
	# Setup folder names and load chain and posterior from the pickle files
	print(f"{f'Analysis - Mission: {mission}, Star: {star}, Run: {run}, Model: {model}':80s}", sep=' ', end='', flush=True)
	# star_home = f"{work_path}/{mission}/{star}/{run}/model{model}"
	star_home = f"{code_path}/../stars/{star}_{run}"

	output_folder = f"{star_home}/output"
	if not os.path.isdir(f"{star_home}/figures"):
		os.mkdir(f"{star_home}/figures")
	figure_folder = f"{star_home}/figures"

	# Read in all data
	with open(f"{output_folder}/chain.pk", "rb") as f:
		chain = pickle.load(f)
	with open(f"{output_folder}/lnprobability.pk", "rb") as p:
		posterior = pickle.load(p)
	filepath = f"{star_home}/{star}_{run}.lc"
	try:
		time, flux, flux_err = np.loadtxt(filepath, unpack=True)
	except ValueError:
		time, flux = np.loadtxt(filepath, unpack=True)
		flux_err = None
	
	# Get time in days and flux in ppm
	time = time.copy() # Stay the same
	# time = time.copy() / (24.*3600.) # time to days
	flux = (flux.copy() - 1.0) * 1e6
	# flux = flux.copy()

	# Get config from file
	config_path = Path(f"{star_home}/config_model{model}.py")
	try:
		spec = importlib.util.spec_from_file_location("config", config_path)
		config = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(config)
	except Exception:
		sys.exit(f"Couldn't import config.py from the folder: {work_path}/config_model{model}")
	sys.modules["config"] = config


	gp_model = GPModel(config.gp)
	gp_names = gp_model.get_parameters_latex()
	if hasattr(config, "transit"):
		mean_model = BatmanModel(config.transit["name"], config.transit["params"])
		# mean_model.init_model(time, time[1]-time[0], 2)
		transit_names = mean_model.get_parameters_latex()
		names = np.hstack([gp_names, transit_names])
	else:
		mean_model = None
		names = gp_names

	output = f"{star:>13s} {run:>3s} {model:>3d}"
	print("[DONE]")
	plt.close("all")

	# Run burn-in diagnostics (Geweke)
	print(f"{f'Calculating Geweke diagnostic...':80s}", sep=' ', end='', flush=True)
	starts, zscores = geweke(chain)
	chains_over = np.max(zscores, axis=2) > 2
	num_chains_over = np.sum(chains_over, axis=1)
	start_index = np.where(num_chains_over == np.min(num_chains_over))[0][0]
	if start_index == 0:
		start_index = 1
	
	walkers_mask = ~chains_over[start_index]
	geweke_flag = False
	if np.sum(chains_over[start_index])/chain.shape[0] > 0.4:
		walkers_mask[:] = True
		geweke_flag = True
	num_walkers = np.sum(walkers_mask)

	reduced_chain = chain[walkers_mask,starts[start_index]:,:]
	reduced_posterior = posterior[starts[start_index]:,walkers_mask]
	
	output = f"{output} {str(geweke_flag):>7s} {num_walkers:>4d} {starts[start_index]/chain.shape[1]:>5.2f}"
	print("[DONE]")

	# Cut chains with lower posterior
	print(f"{f'Removing lower posterior chains...':80s}", sep=' ', end='', flush=True)
	posterior_mask = (np.median(reduced_posterior, axis=0) - np.median(reduced_posterior)) > -np.std(reduced_posterior)

	reduced_chain = reduced_chain[posterior_mask,:,:]
	reduced_posterior = reduced_posterior[:,posterior_mask]
	print("[DONE]")

	# Check consistency between chains
	print(f"{f'Calculating Gelman-Rubin diagnostic...':80s}", sep=' ', end='', flush=True)
	r_hat, _, _ = gelman_rubin(reduced_chain)
	r_mvar = gelman_brooks(reduced_chain)
	print("[DONE]")

	print(f"{f'Plotting Gelman-Rubin analysis...':80s}", sep=' ', end='', flush=True)
	gelman_fig = gelman_rubin_plot(reduced_chain, pnames=names)
	gelman_fig.savefig(f"{figure_folder}/gelman_plot.pdf")
	print("[DONE]")

	print(f"{f'Calculating Gelman-Brooks diagnostic...':80s}", sep=' ', end='', flush=True)
	r_hat_str = "".join([" {:>9.5f}".format(ri) for ri in r_hat])
	output = f"{output} {r_mvar.max():>9.5f}{r_hat_str}"
	print("[DONE]")


	print(f"{f'Calculating parameter statistics...':80s}", sep=' ', end='', flush=True)
	params = {}
	samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
	params["median"] = np.median(samples, axis=0)
	params["hpd_down"], params["hpd_up"] = hpd(reduced_chain, level=0.683)
	hpd_99_down, hpd_99_up = hpd(reduced_chain, level=0.99)
	params["hpd_99_interval"] = (hpd_99_up - hpd_99_down)
	params["mapv"] = mapv(reduced_chain, reduced_posterior)
	params["modes"] = mode(chain)

	results_str = "".join([f"{params['mapv'][i]:>15.10f} {params['modes'][i]:>15.10f} {params['median'][i]:>15.10f} {params['hpd_down'][i]:>15.10f} {params['hpd_up'][i]:>15.10f} {params['hpd_99_interval'][i]:>15.10f}" for i in range(params['median'].size)])
	output = f"{output}{results_str}\n"
	print("[DONE]")

	# Get the median and 68% intervals for each of the parameters
	# print(f"{f'Calculating medians and stds...':80s}", sep=' ', end='', flush=True)
	# samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
	# percentiles = np.percentile(samples.T, [50,16,84], axis=1)
	# median = percentiles[0]
	# lower = median - percentiles[1]
	# upper = percentiles[2] - median

	# results_str = "".join([f" {median[i]:>15.10f} {lower[i]:>15.10f} {upper[i]:>15.10f}" for i in range(median.size)])
	# output = f"{output}{results_str}\n"
	# print("[DONE]")

	print(f"{f'Plotting parameter histograms...':80s}", sep=' ', end='', flush=True)
	parameter_fig = parameter_hist(chain, params, pnames=names)
	parameter_fig.savefig(f"{figure_folder}/parameter_hist.pdf")
	print("[DONE]")

	print(f"{f'Plotting corner...':80s}", sep=' ', end='', flush=True)
	corner_fig = corner_plot(reduced_chain, pnames=names, downsample=5)
	corner_fig.savefig(f"{figure_folder}/corner_plot.pdf")
	print("[DONE]")

	print(f"{f'Plotting posterior histogram...':80s}", sep=' ', end='', flush=True)
	posterior_fig = posterior_hist(reduced_posterior)
	posterior_fig.savefig(f"{figure_folder}/posterior_hist.pdf")
	print("[DONE]")

	print(f"{f'Plotting traces...':80s}", sep=' ', end='', flush=True)
	trace_fig = trace_plot(chain, posterior, pnames=names, downsample=10)
	trace_fig.savefig(f"{figure_folder}/trace_plot.pdf")
	print("[DONE]")


	# Plot the GP dist, and PSD of the distributions
	print(f"{f'Plotting GP...':80s}", sep=' ', end='', flush=True)
	gp_fig, gp_zoom_fig = gp_plot(gp_model, mean_model, params, time, flux, flux_err)
	gp_fig.savefig(f"{figure_folder}/gp_plot.pdf")
	gp_zoom_fig.savefig(f"{figure_folder}/gp_zoom_plot.pdf")
	print("[DONE]")

	print(f"{f'Plotting PSD...':80s}", sep=' ', end='', flush=True)
	psd_fig = psd_plot(gp_model, params, time, flux, include_data=True, parseval_norm=True)
	psd_fig.savefig(f"{figure_folder}/psd_plot.pdf")
	print("[DONE]")

	# Output to file
	# print(f"{f'Writing output to file...':80s}", sep=' ', end='', flush=True)
	# with open(f"{work_path}/results/{mission}_model{model}_runs_test.txt", "a+") as o:
	# 	o.write(output)
	# print("[DONE]")
	# print(f"{'-'*86}")

	plt.close("all")
"""
"""
--------------------------------------------------------------------
"""

"""
---------------------------- JOIN RESULTS --------------------------

- Fix jitter in the kepler diamonds comparison
- 2nd granulation prior - add flag (do for all parameters)
- flag any prior flag in the countour of the scatter circles (do "..")

Models look for:
sim-2460682/1/model2 - ver
sim-1541647/2/model2 - ver
kic-009475697/9/model1
"""

# Join the results
def join_results():
	"""
	0 - Star ID
	1 - Run
	2 - Model
	3 - Geweke flag
	4 - Geweke num walkers
	5 - Geweke burn-in percentage
	6 - Gelman-Brooks
	Model 1
	7-12 - Gelman-Rubin
	13-48 (6*6) - Params (MAP, mode, median, hpd_down, hpd_up, hpd_99_interval)
	Model 2
	7-14 - Gelman-Rubin
	15-62 (8*6) - Params (MAP, mode, median, hpd_down, hpd_up, hpd_99_interval)
	
	Selection of values
	- (Ignored) Remove flagged by geweke
	- If at least 5 runs have Gelman-Brooks < 1.1
		- Threshold = 1.1
	- Else:
		- Threshold = 5th lowest Gelman-Brooks value of the 10 runs
	- Select all runs with Gelman-Brooks < threshold
	- Output: 
		- (Ignored) Geweke Flag
		- Gelman-Brooks Threshold
		- Highest Gelman-Rubin from all selected runs
		- Number of selected runs
		- Median of the chosen runs (sort runs by median, mode and mapv and then choose the middle one for each)
		- Take the hpd_down and hpd_up of final run and add it with the (std) of the remaining runs.
	- Out string: star_id, multi_threshold, uni_threshold, num_runs, (mapv, mode, med, down, up)*num_param 
	"""
	for mission in ["tess-sim", "kepler"]:	
		for model in [1,2]:
			
			filename = f"{work_path}/results/{mission}_model{model}_runs_test.txt"
			data = np.genfromtxt(filename, dtype="str", unpack=True)
			stars = np.unique(data[0])
			priors = {
				"tess-sim_model1": [1990, 19, 140, 390, 190, 400],
				"tess-sim_model2": [1490, 17, 120, 390, 60, 390, 220, 400],
				"kepler_model1": [1490, 13.8, 140, 390, 190, 200],
				"kepler_model2": [1490, 17, 120, 390, 60, 390, 220, 400]
			}
			
			for star in stars:
				d = data[:,data[0]==star]
				
				gelman_multi = d[6].astype(float)
				gelman_uni = d[7:11+model*2].astype(float)
				# If we can get away with threshold 1.1 we do
				if np.sum(gelman_multi <= 1.1) >= 5:
					multi_threshold = 1.1
				# Otherwise we get the lowest threshold that guarantees us 5 runs
				else:
					multi_threshold = np.sort(gelman_multi)[4]

				gelman_mask = gelman_multi <= multi_threshold
				uni_threshold = np.max(gelman_uni[:,gelman_mask])
				num_runs = np.sum(gelman_mask)

				output = f"{star:>13s} {multi_threshold:>6.3f} {uni_threshold:>6.3f} {num_runs:>3.0f}"

				npar = 4 + model*2
				init = 11 + model*2
				for i in range(npar):
					res = d[init+(i*6):init+((i+1)*6), gelman_mask].astype(float)
					args = np.argsort(res[:3], axis=1)[:,int((gelman_mask.sum()-1)/2)]
					mapv = res[0,args[0]]

					mode = res[1,args[1]]
					mode_down = np.sqrt( (mode - res[3,args[1]])**2 + np.std(np.delete(res[1], args[1]))**2 )
					mode_up = np.sqrt( (res[4,args[1]] - mode)**2 + np.std(np.delete(res[1], args[1]))**2 )
					
					med = res[2,args[2]]
					med_down = np.sqrt( (med - res[3,args[2]])**2 + np.std(np.delete(res[2], args[2]))**2 )
					med_up = np.sqrt( (res[4,args[2]] - med)**2 + np.std(np.delete(res[2], args[2]))**2 )

					prior_mode_flag = int(res[5,args[1]] > (priors[f"{mission}_model{model}"][i]*0.9))
					prior_med_flag = int(res[5,args[2]] > (priors[f"{mission}_model{model}"][i]*0.9))
					prior_flag = int(np.any(res[5] > (priors[f"{mission}_model{model}"][i]*0.9)))
					
					res_str = f" {mapv:>15.10f} {mode:>15.10f} {mode_down:>15.10f} {mode_up:>15.10f} {med:>15.10f} {med_down:>15.10f} {med_up:>15.10f} {prior_mode_flag:>3d} {prior_med_flag:>3d} {prior_flag:>3d}"
					output = f"{output}{res_str}"
				
				output = f"{output}\n"
				with open(f"{work_path}/results/{mission}_model{model}_results_test.txt", "a+") as o:
					o.write(output)

"""
--------------------------------------------------------------------
"""

"""
-------------------------- PLOT COMPARISON --------------------------
"""

def plot_comparison():
	for mission in ["tess-sim", "kepler"]:	
	# for mission in ["kepler"]:
		for model in [1,2]:
			# Setup the gp_model
			sys.path.append(work_path)
			try:
				config = importlib.import_module(f"config_model{model}")
			except:
				sys.exit(f"Couldn't import config.py from the folder: {work_path}")
			gp_model = GPModel(config.gp)
			names = gp_model.get_parameters_latex()
			units = gp_model.get_parameters_units()

			# Read the compiled results data
			filename = f"{work_path}/results/{mission}_model{model}_results_test.txt"
			data = np.genfromtxt(filename, dtype="str", unpack=True)

			if mission == "tess-sim":
				truth = np.loadtxt(f"{work_path}/results/truth/tess-sim_model{model}_truth.txt", unpack=True)
				star_data = np.genfromtxt(f"{work_path}/results/truth/tess-sim_star_data.txt", names=True)
				map_fig, mode_fig, med_fig = plot_comparison_tess_sim(data, truth, star_data, pnames=names, punits=units, fsize=14.5)
			elif mission == "kepler":
				truth = np.loadtxt(f"{work_path}/results/truth/diamonds_model{model}.txt", unpack=True)
				star_data = np.genfromtxt(f"{work_path}/results/truth/diamonds_star_data.txt", names=True)
				map_fig, mode_fig, med_fig = plot_comparison_diamonds(data, truth, star_data, pnames=names, punits=units, fsize=14.5)
			else:
				sys.exit(f"Unrecognized mission: {mission}")
			
			print(f"{f'Saving figures {mission}_model{model}...':80s}", sep=' ', end='', flush=True)
			med_fig.savefig(f"{work_path}/results/figures/{mission}_model{model}_med_comparison.pdf")
			map_fig.savefig(f"{work_path}/results/figures/{mission}_model{model}_map_comparison.pdf")
			mode_fig.savefig(f"{work_path}/results/figures/{mission}_model{model}_mode_comparison.pdf")
			print("[DONE]")
			plt.close("all")


def plot_comparison_tess_sim(data, truth, star_data, pnames=None, punits=None, fsize=12):
	# Prepare comparison plots for the Median, MAP and Mode
	nrows = int((pnames.size-2)/2)
	map_fig, map_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	mode_fig, mode_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	med_fig, med_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	
	fig_list = [map_fig, mode_fig, med_fig]
	axes_list = [map_ax.ravel(), mode_ax.ravel(), med_ax.ravel()]
	for fig in fig_list:
		fig.subplots_adjust(left=.10, bottom=.15, right=.90, top=.95, hspace=0.30, wspace=0.30)
	
	if pnames.size == 6:
		plot_order = [3,4,2,5]
	elif pnames.size == 8:
		plot_order = [3,4,5,6,2,7]
	else:
		sys.exit(f"pnames has size: {pnames.size}")

	# Remove the star_id, gelman_threshold and the P_g and Q values from the results
	res = data[24:].astype(float)
	# Remove ony the star_id. Truth from simulations has no P_g or Q in the data
	t = truth[1:]
	# Go through the map, mode and median plots
	for a, axes in enumerate(axes_list):
		# Go through each of the parameters in the results
		for p, i in enumerate(plot_order):
			ax = axes[p]
			res_param = res[(i-2)*10:((i-1)*10)].copy()
			
			x = t[i-2]
			if a == 0:
				y = (res_param[0] - x) / x * 100
				yerr = None
				self_mask = np.full(x.size, False)
			elif a == 1:
				y = (res_param[1] - x) / x * 100
				yerr_m = res_param[2] / x * 100
				yerr_p = res_param[3] / x * 100
				yerr = (yerr_p, yerr_m)
				self_mask = res_param[7].astype(bool)
			elif a == 2:
				y = (res_param[4] - x) / x * 100
				yerr_m = res_param[5] / x * 100
				yerr_p = res_param[6] / x * 100
				yerr = (yerr_p, yerr_m)
				self_mask = res_param[8].astype(bool)
			# any_mask = np.bitwise_xor(res_param[9].astype(bool), self_mask)
			# none_mask = ~np.logical_or(self_mask, any_mask)
			self_mask = res_param[9].astype(bool)
			none_mask = ~self_mask


			plot = ax.scatter(x[self_mask], y[self_mask], zorder=5, c=star_data["Logg"][self_mask], cmap="autumn", edgecolor='black', linestyle=':')
			# plot = ax.scatter(x[any_mask], y[any_mask], zorder=5, c=star_data["Logg"][any_mask], cmap="autumn", edgecolor='black', linestyle='--')
			plot = ax.scatter(x[none_mask], y[none_mask], zorder=5, c=star_data["Logg"][none_mask], cmap="autumn", edgecolor='black', linestyle='-')
			ax.errorbar(x, y, yerr=yerr, fmt="none", marker=None, elinewidth=0.5, color=(0,0,0,1.0), capsize=2)

			if i == pnames.size-1:
				bias = np.median(y[1:])
				sigma = np.std(y[1:])
			else:
				bias = np.median(y)
				sigma = np.std(y)
			bias_label = f"Bias = {bias:.2f} %"
			sigma_label = r'1$\sigma$ = {:.2f} %'.format(sigma)
			
			ax.axhline(bias, color='black', label=bias_label, lw=1)
			# ax.fill_between(x, plot_y.mean()+plot_y.std(), plot_y.mean()-plot_y.std(), color="blue", alpha=0.2, label= r"1$\sigma$", facecolor=None)
			ax.axhline(bias + sigma, color='black', lw=1, ls='--', label=sigma_label)
			ax.axhline(bias - sigma, color='black', lw=1, ls='--')

			# Zero line
			ax.axhline(0, color='red', lw=1, ls='--')

			if pnames is None:
				ax.set_title(f"Parameter - {i-1}", fontsize=fsize)
			else:
				ax.set_title(f"{pnames[i]}", fontsize=fsize)
			ax.set_xlabel(r'Truth [{:}]'.format(punits[i]), fontsize=fsize-2)
			ax.set_ylabel(r'$\Delta${:}/{:} [%]'.format(pnames[i], pnames[i]), fontsize=fsize-2)
			ax.tick_params(axis='both', which='major', direction='in', labelsize=fsize-2)
			ax.legend(fontsize=fsize-3)
		
		cbar_ax = fig_list[a].add_axes([0.25, 0.10 - np.ceil((pnames.size-2)/2)*0.01, 0.50, 0.01])
		cbar_ax.tick_params(direction='in', labelsize=fsize-2)
		cbar = fig_list[a].colorbar(plot, cax=cbar_ax, orientation='horizontal')
		cbar.set_label(r'log($g$)', fontsize=fsize)

	# plt.show()
	# sys.exit()

	return fig_list

# Same as plot_comparison_tess-sim but with some special handling of values for diamonds data
def plot_comparison_diamonds(data, truth, star_data, pnames=None, punits=None, fsize=12):
	# Prepare comparison plots for the Median, MAP and Mode
	nrows = int((pnames.size-2)/2)
	map_fig, map_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	mode_fig, mode_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	med_fig, med_ax = plt.subplots(ncols=2, nrows=nrows, figsize=(10, nrows*4))
	
	fig_list = [map_fig, mode_fig, med_fig]
	axes_list = [map_ax.ravel(), mode_ax.ravel(), med_ax.ravel()]
	for fig in fig_list:
		fig.subplots_adjust(left=.10, bottom=.15, right=.90, top=.95, hspace=0.30, wspace=0.30)
	
	if pnames.size == 6:
		plot_order = [3,4,2,5]
	elif pnames.size == 8:
		plot_order = [3,4,5,6,2,7]
	else:
		sys.exit(f"pnames has size: {pnames.size}")

	# Remove the star_id, gelman_threshold and the P_g and Q values from the results
	res = data[24:].astype(float)
	# Remove ony the star_id. Truth from simulations has no P_g or Q in the data
	t = truth[7:]
	# Go through the map, mode and median plots
	for a, axes in enumerate(axes_list):
		# Go through each of the parameters in the results
		for p, i in enumerate(plot_order):
			ax = axes[p]
			res_param = res[(i-2)*10:((i-1)*10)].copy()


			# Diamonds jitter fix
			if i == pnames.size-1:
				nyquist = 283.25
				constant = nyquist
				mapv = (res_param[0]**2) / constant
				
				mode = (res_param[1]**2) / constant
				# Join with mode to get value of upper and lower bound. Then scale the bounds. Subtract to scaled mode
				mode_down = mode - (((res_param[1] - res_param[2])**2) / constant)
				mode_up = (((res_param[2] + res_param[3])**2) / constant) - mode
				
				med = (res_param[4]**2) / constant
				med_down = med - (((res_param[4] - res_param[5])**2) / constant)
				med_up = (((res_param[4] + res_param[6])**2) / constant) - med

				res_param[0] = mapv
				res_param[1] = mode
				res_param[2] = mode_down
				res_param[3] = mode_up
				res_param[4] = med
				res_param[5] = med_down
				res_param[6] = med_up

			idx = (i-2)*3
			x = t[idx]
			if a == 0:
				y = (res_param[0] - x) / x * 100
				yerr = None
				self_mask = np.full(x.size, False)
			elif a == 1:
				y = (res_param[1] - x) / x * 100
				yerr_m = np.sqrt( (res_param[2] / x)**2 + (t[idx + 1] / x)**2 ) * 100
				yerr_p = np.sqrt( (res_param[3] / x)**2 + (t[idx + 2] / x)**2 ) * 100
				yerr = (yerr_p, yerr_m)
				self_mask = res_param[7].astype(bool)
			elif a == 2:
				y = (res_param[4] - x) / x * 100
				yerr_m = np.sqrt( (res_param[5] / x)**2 + (t[idx + 1] / x)**2 ) * 100
				yerr_p = np.sqrt( (res_param[6] / x)**2 + (t[idx + 2] / x)**2 ) * 100
				yerr = (yerr_p, yerr_m)
				self_mask = res_param[8].astype(bool)
			# any_mask = np.bitwise_xor(res_param[9].astype(bool), self_mask)
			# none_mask = ~np.logical_or(self_mask, any_mask)
			self_mask = res_param[9].astype(bool)
			none_mask = ~self_mask


			plot = ax.scatter(x[self_mask], y[self_mask], zorder=5, c=star_data["Logg"][self_mask], cmap="autumn", edgecolor='black', linestyle=':')
			# plot = ax.scatter(x[any_mask], y[any_mask], zorder=5, c=star_data["Logg"][any_mask], cmap="autumn", edgecolor='black', linestyle='--')
			plot = ax.scatter(x[none_mask], y[none_mask], zorder=5, c=star_data["Logg"][none_mask], cmap="autumn", edgecolor='black', linestyle='-')
			ax.errorbar(x, y, yerr=yerr, fmt="none", marker=None, elinewidth=0.5, color=(0,0,0,1.0), capsize=2)

			bias = np.median(y)
			sigma = np.std(y)
			bias_label = f"Bias = {bias:.2f} %"
			sigma_label = r'1$\sigma$ = {:.2f} %'.format(sigma)
			
			ax.axhline(bias, color='black', label=bias_label, lw=1)
			# ax.fill_between(x, plot_y.mean()+plot_y.std(), plot_y.mean()-plot_y.std(), color="blue", alpha=0.2, label= r"1$\sigma$", facecolor=None)
			ax.axhline(bias + sigma, color='black', lw=1, ls='--', label=sigma_label)
			ax.axhline(bias - sigma, color='black', lw=1, ls='--')

			# Zero line
			ax.axhline(0, color='red', lw=1, ls='--')

			if pnames is None:
				ax.set_title(f"Parameter - {i-1}", fontsize=fsize)
			else:
				ax.set_title(f"{pnames[i]}", fontsize=fsize)
			ax.set_xlabel(r'Diamonds [{:}]'.format(punits[i]), fontsize=fsize-2)
			ax.set_ylabel(r'$\Delta${:}/{:} [%]'.format(pnames[i], pnames[i]), fontsize=fsize-2)
			ax.tick_params(axis='both', which='major', direction='in', labelsize=fsize-2)
			ax.legend(fontsize=fsize-3)
		
		cbar_ax = fig_list[a].add_axes([0.25, 0.10 - np.ceil((pnames.size-2)/2)*0.01, 0.50, 0.01])
		cbar_ax.tick_params(direction='in', labelsize=fsize-2)
		cbar = fig_list[a].colorbar(plot, cax=cbar_ax, orientation='horizontal')
		cbar.set_label(r'log($g$)', fontsize=fsize)

	# plt.show()
	# sys.exit()

	return fig_list

"""
--------------------------------------------------------------------
"""


# --------------------- RUN TESS -----------------------
def run_tess_sim():
	for star in os.listdir(f"{work_path}/tess-sim"):
		for run in os.listdir(f"{work_path}/tess-sim/{star}"):
			for model in [1,2]:	
				lc_file = f"{work_path}/tess-sim/{star}/{run}/model{model}/{star}_{run}.lc"
				sp.run(["python", f"{code_path}/run.py", f"{work_path}/config_model{model}.py", lc_file])

# --------------------- RUN KEPLER ---------------------
def run_kepler():
	for star in os.listdir(f"{work_path}/kepler"):
		for run in os.listdir(f"{work_path}/kepler/{star}"):
			for model in [1,2]:	
				lc_file = f"{work_path}/kepler/{star}/{run}/model{model}/{star}_{run}.lc"
				sp.run(["python", f"{code_path}/run.py", f"{work_path}/config_model{model}.py", lc_file])



# Create kepler folders
def create_kepler_folders():
	old_path = "/mnt/c/work/_old/phd/red_giants/data_diamonds/ready"
	new_path = "/mnt/c/work/astro/gp-paper/kepler"

	quarters = ["Q12_1", "Q12_2", "Q13_1", "Q13_2", "Q14_1", "Q14_2", "Q15_1", "Q15_2", "Q16_1", "Q16_2"]
	for star in os.listdir(old_path):
		os.mkdir(f"{new_path}/kic-{star}")
		for i in range(len(quarters)):
			quarter = quarters[i]
			os.mkdir(f"{new_path}/kic-{star}/{i}")
			for m in [1,2]:
				os.mkdir(f"{new_path}/kic-{star}/{i}/model{m}")
				src = f"{old_path}/{star}/{quarter}.lc"
				dst = f"{new_path}/kic-{star}/{i}/model{m}/kic-{star}_{i}.lc"
				copyfile(src, dst)

# Create tess simulated data folders
def create_tess_sim_folders():
	old_path = "/mnt/c/work/_old/phd/red_giants/data_tess_artificial/ready"
	new_path = "/mnt/c/work/astro/gp-paper/tess-sim"

	for folder in os.listdir(old_path):
		os.mkdir(f"{new_path}/sim-{folder:s}")
		for i in range(10):
			os.mkdir(f"{new_path}/sim-{folder:s}/{i}")
			for m in [1,2]:
				os.mkdir(f"{new_path}/sim-{folder:s}/{i}/model{m}")

				src = f"{old_path}/{folder}/model2/run{i}.lc"
				dst = f"{new_path}/sim-{folder:s}/{i}/model{m}/sim-{folder}_{i}.lc"
				copyfile(src, dst)

if __name__ == "__main__":
	# run_tess_sim()
	# run_kepler()
	# analise_all()
	# join_results()
	# plot_comparison()
	# analyse("tess-sim", "sim-1573736", "1", 2)

	analyse("tess-sim", "sim-1573736", "0", 1)
	# analyse("kepler", "kic-009882316", "4", 2)
	