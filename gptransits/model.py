import numpy as np
import celerite
from .component import Component

# TODO
# Class to implement the logic to integrate both MeanModel and GPModel in the same container
# Purpose is to abstract the logic of the likelihood calculation when dealing with parameters of both models together
class Model(object):
	
	def __init__(self, mean_model, gp_model, data, include_errors=False):
		# If include errors try to get all data from data array
		try: 
			self.time, self.flux, self.error = data
		except ValueError:
			if include_errors:
				raise ValueError("Data needs to have errors to include them")
			else:
				try:
					self.time, self.flux = data
				except ValueError:
					raise ValueError("Data file needs to have at least time and flux columns")

		if isinstance(mean_model, MeanModel):
			self.mean_model = mean_model
		else:
			raise ValueError("First argument must be of type MeanModel")

		self.time -= self.time[0]

		if isinstance(gp_model, GPModel):
			self.gp_model = gp_model
			self.gp_model.time = self.time

			if include_errors:
				self.gp = GP(self.gp_model, self.time, self.error)
			else:
				self.gp = GP(self.gp_model, self.time)
		else:
			raise ValueError("Second argument must be of type GPModel")

	def log_likelihood(self, params):
		self.gp.gp_model.set_parameters(params)
		lnprior = self.gp.gp_model.prior_evaluate()
		if not np.isfinite(lnprior):
			return -np.inf
		self.gp.set_parameters()

		lnlikelihood = self.gp.log_likelihood(self.flux - self.mean_model.eval(params, self.time, self.flux))
		
		return lnprior + lnlikelihood


# TODO
# Parametric model of the data that has a defined functional form
class MeanModel(object):
	
	def eval(self, params, time, flux):
		return 0

# Model that contains the components of the GP. Might be joined with GP in the future.
class GPModel(object):
	def __repr__(self):
		string = 'Model with {0} components:\n'.format(len(self.component_array))
		for component in self.component_array:
			string += repr(component) + '\n'
		return string

	def	__init__(self, *args):
		if len(args):
			self.component_array = []
			for arg in args:
				if isinstance(arg, Component):
					self.component_array.append(arg)
				else:
					raise ValueError("Args must be of type Component")
		else:
			raise ValueError("Model must have at least one component")

	def add(self, *args):
		if len(args):
			for arg in args:
				if isinstance(arg, Component):
					self.component_array.append(arg)
				else:
					raise ValueError("Args must be of type Component")
		else:
			raise ValueError("Must add at least one component to model")

	def set_parameters(self, params):
		i = 0
		for component in self.component_array:
			component.parameter_array = params[i:i+component.npars]
			i += component.npars

	def get_parameters(self):
		return np.hstack([component.parameter_array for component in self.component_array])

	def get_parameters_celerite(self):
		return np.hstack([component.get_parameters_celerite() for component in self.component_array])
		
	def get_parameters_names(self):
		return np.hstack([component.parameter_names for component in self.component_array])

	def get_parameters_latex(self):
		return np.hstack([component.parameter_latex_names for component in self.component_array])

	def get_parameters_units(self):
		return np.hstack([component.parameter_names for component in self.component_array])


	def prior_evaluate(self):
		prior = sum([component.eval_prior() for component in self.component_array])
		if not np.isfinite(prior):
			return -np.inf
		return prior

	def prior_sample(self, num=1):
		return np.hstack([component.sample_prior(num) for component in self.component_array])

	def get_kernel(self):
		kernel = celerite.terms.TermSum()
		for component in self.component_array:
			kernel += component.get_kernel()
		return kernel

	def get_psd(self, time = None):
		if time is None:
			try:
				time = self.time
			except AttributeError:
				raise AttributeError("No time variable present in object")

		cadence = time[1] - time[0]
		nyquist = 1e6 / (2 * cadence)

		time_span = time[-1] - time[0]
		f_sampling = 1 / (time_span / 1e6)

		freq = np.linspace(0.0, nyquist, (nyquist/f_sampling)+1)

		psd_dict = [component.get_psd(freq, time.size, nyquist) for component in self.component_array]
		return [freq, psd_dict]

	def get_kernel_list(self):
		pass

# Class that implements the GP methods of the GPModel and interfaces with celerite methods
class GP(object):
	def __init__(self, gp_model, time, error=1.123e-12): # yerr same as celerite
		if isinstance(gp_model, GPModel):
			self.gp_model = gp_model
		else:
			raise ValueError("model arg must be of type GPModel")
		self.gp = celerite.GP(self.gp_model.get_kernel())
		self.gp.compute(time/1e6, yerr=error)

	def set_parameters(self):
		celerite_params = self.gp_model.get_parameters_celerite()
		self.gp.set_parameter_vector(celerite_params)

	def log_likelihood(self, residuals):
		return self.gp.log_likelihood(residuals)

	def predict(self, y, t=None, return_cov=True, return_var=False):
		return self.gp.predict(y, t, return_cov, return_var)