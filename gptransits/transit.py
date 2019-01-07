import numpy as np
import batman
from scipy.stats import uniform
from .model import MeanModel


class Transit(MeanModel):
	
	def __init__(self):
		self.npars = 7
		self.batparams = batman.TransitParams()
		self.batparams.u = [0.1, 0.3]
		self.batparams.limb_dark = 'quadratic'
		self.setup_prior()

	def initialize_model(self, time):
		self.set_parameters(self.sample_prior().T)
		self.batmodel = batman.TransitModel(self.batparams, time/(24*3600))

	def set_parameters(self, params):
		self.parameter_array = params
		self.update_batparams()

	def update_batparams(self):
		self.batparams.t0 = self.parameter_array[0]
		self.batparams.per = self.parameter_array[1]
		self.batparams.rp = self.parameter_array[2]
		self.batparams.a = self.parameter_array[3]
		self.batparams.inc = self.parameter_array[4]
		self.batparams.ecc = self.parameter_array[5]
		self.batparams.w = self.parameter_array[6]

	def get_parameters(self):
		return self.parameter_array

	def get_parameters_names(self):
		return np.array(['t0', 'P', 'Rratio', 'a', 'i', 'ecc', 'w'])

	def get_parameters_latex(self):
		return np.array(['t0', 'P', 'Rratio', 'a', 'i', 'ecc', 'w'])		

	def setup_prior(self):
		dist_values = np.vstack([(1, 1), 	# T0
								(2, 1), 	# Period
								(0, .1), 		# Planet Radius
								(1, 4), 		# Semi-major axis
								(70, 20), 	# Orbital inclination
								(0, .4), 	# eccentricity
								(0, 90)]) 	# Longitude Periastron 

		self.priors = uniform(dist_values[:,0], dist_values[:,1])

	def sample_prior(self, num=1):
		return self.priors.rvs([num, self.npars])

	def evaluate_prior(self):
		return np.sum(self.priors.logpdf(self.parameter_array))

	def eval(self):
		flux = self.batmodel.light_curve(self.batparams)
		flux = (flux-1)*1e6
		return flux
