import sys

import numpy as np
from scipy.stats import uniform, norm, reciprocal
import celerite

# Base class for all kernel components for the GPModel
class Component(object):
	def __init__(self, name, params_dict):
		# Get component name
		self.name = name
		# Check that config dictionary as correct number of entries
		if self.npars != len(params_dict["values"].keys()):
			print(f"Component needs to have {self.npars} number of parameters.")
			sys.exit(1)

		if "latex_names" in params_dict:
			if (self.npars != len(params_dict["latex_names"])):
				print(f"Component latex names must have size {self.npars}")
				sys.exit(1)
			self.parameter_latex_names = params_dict["latex_names"]
		if "units" in params_dict:
			if (self.npars != len(params_dict["units"])):
				print(f"Component units must have size {self.npars}")
				sys.exit(1)
			self.parameter_units = params_dict["units"]

		# Setup the priors and init parameters
		self.prior = []
		params_values = params_dict["values"]
		for pname in self.parameter_names:
			if params_values[pname][0]:
				print("Fixing parameter values is not yet supported")
				sys.exit(1)
			if params_values[pname][2] == "uniform":
				self.prior.append(uniform(params_values[pname][3], params_values[pname][4] - params_values[pname][3]))
			elif params_values[pname][2] == "normal":
				self.prior.append(norm(params_values[pname][3], params_values[pname][4]))
			elif params_values[pname][2] == "reciprocal":
				if params_values[pname][3] != 0:
					print("Reciprocal prior cannot have a starting interval value of 0.0")
					sys.exit(1)
				self.prior.append(reciprocal(params_values[pname][3], params_values[pname][4]))
			else:
				print(f"Parameter prior distribution chosen not recognized: {params_values[pname][2]}")
				sys.exit(1)
		self.prior = np.asarray(self.prior)

	def sample_prior(self, num=1):
		return np.hstack([p.rvs([num, 1]) for p in self.prior])

	def lnprior(self, params): # Add non log evaluation of prior
		return np.sum([self.prior[i].logpdf(params[i]) for i in range(self.prior.size)])

	def get_parameters_celerite(self, params):
		raise NotImplementedError

	def get_kernel(self, params):
		raise NotImplementedError
	
	def get_psd(self, params, freq, N, nyquist):
		kernel = self.get_kernel(params)
		# The multiplication fixes the psd calculation from celerite with the constant that correctly normalizaes the value
		psd = kernel.get_psd(2*np.pi*freq) * 2 * np.sqrt(2*np.pi)
		return psd
		

class Granulation(Component):
	def __init__(self, name, config):
		self.npars = 2
		self.parameter_names = ['a_gran', 'b_gran']
		self.parameter_latex_names = [r'$a_{gran}$', r'$b_{gran}$']
		self.parameter_units = ['ppm', r'$\mu$Hz']
		# default_prior = np.array([[20, 400], [10, 200]])

		# Parent contructor to init the priors
		super().__init__(name, config)

	def get_parameters_celerite(self, params):
		a, b = params

		w = b * (2*np.pi)

		# S0 from the PSD equation in Foreman-Mackey 2017
		# S = (2/np.sqrt(np.pi)) * (a**2 / b)

		# S0 from the kernel equation for the granualtion
		# S = a**2 / (2 * np.pi * b * np.cos(-np.pi/4))

		# S0 from updated PSD in celerite using Parseval's Theorem
		S = (np.sqrt(2) / (2 * np.pi)) * (a**2 / b)

		# Test
		# S = np.sqrt(2) * a**2 / w

		# Returning log to comply with celerite
		return np.log([S, w])

	def get_kernel(self, params):
		S, w = self.get_parameters_celerite(params)

		Q = 1.0 / np.sqrt(2.0)
		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=np.log(Q), log_omega0=w)
		kernel.freeze_parameter("log_Q")
		return kernel

class OscillationBump(Component):
	def __init__(self, name, config):
		self.npars = 3
		self.parameter_names = ['P_g', 'Q', 'nu_max']
		self.parameter_latex_names = [r'$P_{g}$', r'$Q_{bump}$', r'$\nu_{max}$']
		self.parameter_units = ['ppm', '', r'$\mu$Hz']
		# default_prior = np.array([[10, 1500], [1.2, 15], [100, 220]])
		
		# Parent contructor to init the priors
		super().__init__(name, config)

	def get_parameters_celerite(self, params):
		P_g, Q, numax = params

		# S0 from the PSD equation in Foreman-Mackey 2017
		# S = P_g / ((Q**2) * np.sqrt(2/np.pi))

		# S0 from updated PSD in celerite using Parseval's Theorem
		S = P_g / (4 * Q**2)

		w = numax * (2*np.pi)

		# return np.array([S, Q, w])
		return np.log([S, Q, w])

	def get_kernel(self, params):
		S, Q, w = self.get_parameters_celerite(params)

		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=Q, log_omega0=w)
		# kernel = celerite.terms.SHOTerm(log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w))

		return kernel

class WhiteNoise(Component):
	
	def __init__(self, name, config):
		self.npars = 1
		self.parameter_names = ['jitter']
		self.parameter_latex_names = [r'White Noise']
		self.parameter_units = ['ppm']
		# default_prior = np.array([[1e-3, 300]])

		# Parent contructor to init the priors
		super().__init__(name, config)

	def get_parameters_celerite(self, params):
		jitter, = params

		# return np.array([jitter])
		return np.log([jitter])

	def get_kernel(self, params):
		jitter, = self.get_parameters_celerite(params)

		kernel = celerite.terms.JitterTerm(log_sigma=jitter)
		# kernel = celerite.terms.JitterTerm(log_sigma=np.log(jitter))
		return kernel

	def get_psd(self, params, freq, N, nyquist):
		jitter, = params

		# PSD = sigma^2 / N points
		# return (self.name, np.full(freq.size, jitter**2 / N))
		
		# PSD = sigma^2 / N points x Parseval normalization constant (2 x sqrt(2 x pi))
		# return (self.name, np.full(freq.size, jitter**2 * (2*np.sqrt(2*np.pi)) / N))
		
		# PSD = sigma
		# return (self.name, np.full(freq.size, jitter))

		# PSD = sigma^2 x 2 x cadence / 10^6        # [ 1 / nyquist = 10^6 / (2 x cadence) ]
		return np.full(freq.size, jitter**2 / nyquist)
