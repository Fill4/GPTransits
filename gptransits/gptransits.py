''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys, os
import argparse
import logging

__all__ = ['main']

#Internal imports
from .model import Model
from . import mcmc
from . import plot


def main(mean_model, gp_model, settings):

	# Log start time
	init_time = timeit.default_timer()

	#------------------------------------------------------------------
	#	INPUT
	#------------------------------------------------------------------

	# Handle the input flags of the script
	parser = argparse.ArgumentParser(description='Parse arguments')
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('--file', '-f', type=str, dest='input_file', help='Filename with lightcurve data', required=False)#, default=None)
	group.add_argument('--list', '-l', type=str, dest='input_list', help='Filename with list of lightcurve files', required=False)#, default=None)
	parser.add_argument('--output', '-o', dest='output', nargs='?', default=False, help='Whether to write results to a file or not')
	parser.add_argument('--verbose', '-v', dest='verbose', action='store_false', help='Enable verbose mode')
	args = parser.parse_args()

	if args.verbose or settings.verbose:
		logging.basicConfig(format='%(message)s', level=logging.INFO)

	# Code uses first the arguments from argparse and then the values from the settings
	if args.input_file:
		file_list = [args.input_file]
	elif settings.input_file:
		file_list = [settings.input_file]
	elif args.input_list:
		file_list = [line.rstrip() for line in open(args.input_list, 'r')]
	elif settings.input_list:
		file_list = [line.rstrip() for line in open(settings.input_list, 'r')]
	else:
		raise ValueError("Please choose a file or a list")

	# Run main method for each of the stars
	for file in file_list:
		run(file, mean_model, gp_model, args.output, settings)

	# Print execution time
	execution_time = timeit.default_timer() - init_time
	logging.info("Complete execution time: {:.4f} usec".format(execution_time))
	logging.info('End\n')



# Execution for each of the stars. Runs the mcmc and handles the results
def run(file, mean_model, gp_model, output, settings):
	
	# Log start time
	init_star_time = timeit.default_timer()

	filename = os.path.splitext(os.path.basename(file))[0]
	logging.info('Starting {:} ...'.format(filename))

	# Read data from input file and instanciate the GP using the time array and model
	try:
		data = np.loadtxt(file, unpack=True)
	except FileNotFoundError:
		file = "{}/{}".format(os.getcwd(), file)
		try:
			data = np.loadtxt(file, unpack=True)
		except FileNotFoundError:
			logging.error("Couldn't open file: " + file)
			sys.exit(1)
	model = Model(mean_model, gp_model, data, include_errors=True)

	# Run Minimizationn
	#backend.run_minimization(data, priors, plot=plot)

	# Run MCMC
	samples, results = mcmc.run_mcmc(model, settings)

	# Replace model and gp parameters with median from results
	model.gp.gp_model.set_parameters(results[1])
	model.gp.set_parameters()

	# If verbose mode, display the values obtained for each parameter
	names = model.gp.gp_model.get_parameters_names()
	sigma_minus, median, sigma_plus = results
	logging.info(''.join(["{:10}{:3}{:10.4f}\n".format(names[i], "-", median[i]) for i in range(median.size)]))

	#------------------------------------------------------------------
	#	OUTPUT
	#------------------------------------------------------------------

	if output != False:
		if a == None:
			z = open('{}/{}.out'.format(os.path.dirname(file), filename), 'a+')
		else:
			z = open('{}/{}.out'.format(os.path.dirname(file), output), 'a+')
		
		# Write final parameters and uncertainties to output buffer
		output_buffer = "# {:}: {:}\n".format("Filename", filename)
		output_buffer += "# {:>10}{:>12}{:>10}{:>10}\n".format("Parameter", "Median", "-Std", "+Std")

		output_buffer += ''.join(["  {:>10}{:>12.4f}{:>10.4f}{:>10.4f}\n".format(names[i], median[i], median[i]-sigma_minus[i], sigma_plus[i]-median[i]) for i in range(median.size)])

		# output_buffer += "  {:>10}{:>12.4f}\n".format("Std LC", np.std(model.flux))
		output_buffer += '# --------------------------------------------------------------------------\n'

		# Write buffer to results file
		z.write(output_buffer)
		z.close()

	if settings.tess_settings:
		# Extra lines for simulated data
		if os.path.exists('{}/results_model1.dat'.format(os.path.dirname(file))):
			header = False
		else: 
			header = True
		f = open('{}/results_model1.dat'.format(os.path.dirname(file)), 'a+')
		if header:
			f.write("# {:>5}".format("Run") + "".join(["{:>9}{:>8}{:>8}".format(names[i], "-Std", "+Std") for i in range(median.size)]) + "\n")
		f.write("{:>7}".format(filename) + "".join(["{:>9.3f}{:>8.3f}{:>8.3f}".format(median[i], median[i]-sigma_minus[i], sigma_plus[i]-median[i]) for i in range(median.size)]) + "\n") 
		f.close()

	if settings.raw_data_settings:
		# Extra lines for complete simulated lc
		output_path = '{}/tess_artificial_data/full_lc_model1.out'.format(os.getcwd())
		if os.path.exists(output_path):
			header = False
		else: 
			header = True
		f = open(output_path, 'a+')
		if header:
			f.write("# {:>5}".format("Run") + "".join(["{:>9}{:>8}{:>8}".format(names[i], "-Std", "+Std") for i in range(median.size)]) + "\n")
		f.write("{:>7}".format(filename) + "".join(["{:>9.3f}{:>8.3f}{:>8.3f}".format(median[i], median[i]-sigma_minus[i], sigma_plus[i]-median[i]) for i in range(median.size)]) + "\n") 
		f.close()

	if settings.diamonds_settings:
		output_path = '{}/model2_err.out'.format(os.path.dirname(file))
		if os.path.exists(output_path):
			header = False
		else: 
			header = True
		f = open(output_path, 'a+')
		if header:
			f.write("# {:>5}".format("Run") + "".join(["{:>9}{:>8}".format(names[i], "Std") for i in range(median.size)]) + "\n")
		f.write("{:>7}".format(filename) + "".join(["{:>9.3f}{:>8.3f}".format(median[i], (abs(median[i]-sigma_minus[i]) + abs(sigma_plus[i]-median[i])) / 2) for i in range(median.size)]) + "\n") 
		f.close()

	#------------------------------------------------------------------
	#	PLOTS  -  FIX ABSOLUTE PATHS
	#------------------------------------------------------------------
	
	plt.close('all')
	if settings.plot_gp:
		gp_plot = plot.plot_gp(model, data, settings)
	if settings.plot_corner:
		corner_plot = plot.plot_corner(model, samples, settings)
		corner_plot.savefig('{}/{}_m2_err_corner.pdf'.format(os.path.dirname(file), filename))
	if settings.plot_psd:
		psd_plot = plot.plot_psd(model, data, settings, parseval_norm=True)
		psd_plot.savefig('{}/{}_m2_err_psd.pdf'.format(os.path.dirname(file), filename))
	if settings.plots:
		if settings.show_plots:
			plt.show()
	plt.close('all')

	# Print execution time
	execution_time_star = timeit.default_timer() - init_star_time
	logging.info("{:} elapsed time: {:.4f} usec".format(filename, execution_time_star))