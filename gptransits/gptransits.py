''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import os
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
			raise FileNotFoundError("Could not open file: " + file)

	model = Model(mean_model, gp_model, data, include_errors=settings.include_errors)

	# Run MCMC
	samples, results = mcmc.run_mcmc(model, settings)
	
	# Get parameter names and results
	names = model.get_parameters_names()
	p_1, p_16, p_50, p_84, p_99 = results
	# p_50[-1] = p_99[-1]

	# Replace model and gp parameters with median from results
	model.set_parameters(p_50)
	model.gp.update_parameters()

	# If verbose mode, display the values obtained for each parameter
	logging.info(''.join(["{:10}{:3}{:10.4f}\n".format(names[i], "-", p_50[i]) for i in range(p_50.size)]))

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
		output_buffer += "# {:>10}{:>12}{:>10}{:>10}\n".format("Parameter", "p_50", "-Std", "+Std")

		output_buffer += ''.join(["  {:>10}{:>12.4f}{:>10.4f}{:>10.4f}\n".format(names[i][:9], p_50[i], p_50[i]-p_16[i], p_84[i]-p_50[i]) for i in range(p_50.size)])

		# output_buffer += "  {:>10}{:>12.4f}\n".format("Std LC", np.std(model.flux))
		output_buffer += '# --------------------------------------------------------------------------\n'

		# Write buffer to results file
		z.write(output_buffer)
		z.close()

	if any([settings.tess_settings, settings.raw_data_settings, settings.diamonds_settings]):
		# Extra lines for simulated data
		if settings.tess_settings:
			output_path = '{}/model{}.out'.format(os.path.dirname(file), settings.model)
		elif settings.raw_data_settings:
			output_path = '{}/tess_artificial_data/full_lc_model{}.out'.format(os.getcwd(), settings.model)
		elif settings.diamonds_settings:
			output_path = '{}/model{}_new.out'.format(os.path.dirname(file), settings.model)
			
		if os.path.exists(output_path):
			header = False
		else: 
			header = True
		f = open(output_path, 'a+')

		if header:
			# f.write("# {:>6}".format("Run") + "".join(["{:>9}{:>8}{:>8}".format(names[i], "-Std", "+Std") for i in range(p_50.size)]) + "    99th\n")
			f.write("# {:>6}".format("Run") + "".join(["{:>9}{:>8}{:>8}".format(names[i], "-Std", "+Std") for i in range(p_50.size)]) + "\n")

		# f.write("{:>8}".format(filename[:8]) + "".join(["{:>9.3f}{:>8.3f}{:>8.3f}".format(p_50[i], p_50[i]-p_16[i], p_84[i]-p_50[i]) for i in range(p_50.size)]) + "\n") 

		f.write("{:>8}".format(filename[:8]))
		for i in range(p_50.size):
			if names[i] == 'Jitter':
				# f.write("{:>9.3f}{:>8.3f}{:>8.3f}{:>8.3f}".format(p_50[i], p_50[i]-p_16[i], p_84[i]-p_50[i], p_99[i]))
				f.write("{:>9.3f}{:>8.3f}{:>8.3f}".format(p_50[i], p_50[i]-p_16[i], p_84[i]-p_50[i]))
			else:
				f.write("{:>9.3f}{:>8.3f}{:>8.3f}".format(p_50[i], p_50[i]-p_16[i], p_84[i]-p_50[i]))
		f.write('\n')
		f.close()

	#------------------------------------------------------------------
	#	PLOTS  -  FIX ABSOLUTE PATHS
	#------------------------------------------------------------------
	
	plt.close('all')
	if settings.plot_gp:
		gp_plot, zoom_plot = plot.plot_gp(model, settings)
		if settings.save_plots:
			gp_plot.savefig('{}/{}_model{}_gp.pdf'.format(os.path.dirname(file), filename, settings.model))
			zoom_plot.savefig('{}/{}_model{}_gp_zoom.pdf'.format(os.path.dirname(file), filename, settings.model))
	if settings.plot_corner:
		corner_plot = plot.plot_corner(model, samples, settings)
		if settings.save_plots:
			corner_plot.savefig('{}/{}_model{}_corner.pdf'.format(os.path.dirname(file), filename, settings.model))
	if settings.plot_psd:
		psd_plot = plot.plot_psd(model, settings, parseval_norm=True)
		if settings.save_plots:
			psd_plot.savefig('{}/{}_model{}_psd.pdf'.format(os.path.dirname(file), filename, settings.model))
	if settings.plots:
		if settings.show_plots:
			plt.show()
	plt.close('all')

	# Print execution time
	execution_time_star = timeit.default_timer() - init_star_time
	logging.info("{:} elapsed time: {:.4f} usec\n".format(filename, execution_time_star))