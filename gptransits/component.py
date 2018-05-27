import numpy as np
from scipy.stats import uniform
import celerite

# Base class for all kernel components for the GPModel
class Component(object):
	name = ''
	npars = 0
	parameter_names = []
	default_prior = []

	def __init__(self, *args, **kwargs):
		if 'prior' in kwargs:
			self.prior_settings = np.array(kwargs.get('prior'))
		else:
			self.prior_settings = self.default_prior
		self.setup_prior()

		if len(args):
			if len(args) != self.npars:
				raise ValueError("Expected {0} arguments but got {1}".format(npars, len(args)))
			self.parameter_array = np.array(args)
		else:
			self.parameter_array = self.sample_prior()[0]

	def setup_prior(self):
		means = [prior[0] for prior in self.prior_settings]
		stds = [prior[1]-prior[0] for prior in self.prior_settings]
		self.priors = uniform(means, stds)

	def sample_prior(self, num=1):
		return self.priors.rvs([num, self.npars])

	def eval_prior(self, log=True): # Add non log evaluation of prior
		return sum(self.priors.logpdf(self.parameter_array))

	def get_parameters_celerite(self):
		raise NotImplementedError

	def get_kernel(self):
		raise NotImplementedError
	
	def get_psd(self, freq, npoints):
		kernel = self.get_kernel()
		power = kernel.get_psd(2*np.pi*freq) * 2 * np.sqrt(2*np.pi)
		return (self.name, power)
		

class Granulation(Component):
	name = 'Granulation'
	npars = 2
	parameter_names = ['a_gran', 'b_gran']
	parameter_latex_names = [r'$a_{gran}$', r'$b_{gran}$']
	parameter_units = ['ppm', r'$\mu$Hz']
	default_prior = np.array([[20, 300], [10, 200]])

	def __repr__(self):
		return '{0} ({names[0]}:{values[0]:.3f} {units[0]}, {names[1]}:{values[1]:.3f} {units[1]})'.format(self.name, 
				names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		a, b = self.parameter_array

		# S0 from the PSD equation in Foreman-Mackey 2017
		# S = (2/np.sqrt(np.pi)) * (a**2 / b)

		# S0 from the kernel equation for the granualtion
		# S = a**2 / (2 * np.pi * b * np.cos(-np.pi/4))

		# S0 from updated PSD in celerite using Parseval's Theorem
		S = (np.sqrt(2) / (2 * np.pi)) * (a**2 / b)

		w = b * (2*np.pi)

		# return np.array([S, w])
		return np.log([S, w])

	def get_kernel(self):
		Q = 1.0 / np.sqrt(2.0)
		S, w = self.get_parameters_celerite()

		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=np.log(Q), log_omega0=w)
		# kernel = celerite.terms.SHOTerm(log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w))
		kernel.freeze_parameter("log_Q")
		return kernel

class OscillationBump(Component):
	name = 'OscillationBump'
	npars = 3
	parameter_names = ['P_g', 'Q', 'nu_max']
	parameter_latex_names = [r'$P_{g}$', r'$Q_{bump}$', r'$\nu_{max}$']
	parameter_units = ['ppm', '', r'$\mu$Hz']
	default_prior = np.array([[10, 600], [1.2, 10], [100, 250]])

	def __repr__(self):
		return '{0}({names[0]}:{values[0]:.3f} {units[0]}, {names[1]}:{values[1]:.3f} {units[1]}, {names[2]}:{values[2]:.3f} {units[2]})'.format(self.name, 
				names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		P_g, Q, numax = self.parameter_array

		# S0 from the PSD equation in Foreman-Mackey 2017
		# S = P_g / ((Q**2) * np.sqrt(2/np.pi))

		# S0 from updated PSD in celerite using Parseval's Theorem
		S = P_g / (4 * Q**2)

		w = numax * (2*np.pi)

		# return np.array([S, Q, w])
		return np.log([S, Q, w])

	def get_kernel(self):
		S, Q, w = self.get_parameters_celerite()

		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=Q, log_omega0=w)
		# kernel = celerite.terms.SHOTerm(log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w))

		return kernel

class WhiteNoise(Component):
	name = 'WhiteNoise'
	npars = 1
	parameter_names = ['Jitter']
	parameter_latex_names = ['Jitter']
	parameter_units = ['ppm']
	default_prior = np.array([[1, 200]])

	def __repr__(self):
		return '{0}({names[0]}:{values[0]:.3f} {units[0]})'.format(self.name, names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		jitter, = self.parameter_array

		# return np.array([jitter])
		return np.log([jitter])

	def get_kernel(self):
		jitter, = self.get_parameters_celerite()
		kernel = celerite.terms.JitterTerm(log_sigma=jitter)
		# kernel = celerite.terms.JitterTerm(log_sigma=np.log(jitter))
		return kernel

	def get_psd(self, freq, npoints):
		jitter, = self.parameter_array
		# Return the jitter determined using the sigma^2 value divided by N points
		# return (self.name, np.full(freq.size, jitter**2 / npoints))
		# Return the jitter using sigma^2 / N points multiplied by the parseval constant
		return (self.name, np.full(freq.size, jitter**2 * (2*np.sqrt(2*np.pi)) / npoints))
		# Return the jitter parameter (sigma)
		# return (self.name, np.full(freq.size, jitter))