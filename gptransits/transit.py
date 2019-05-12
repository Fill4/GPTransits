import sys
import logging

import numpy as np
from scipy.stats import uniform, norm, reciprocal
import batman

# Define the transit models
class BatmanModel(object):
	def __init__(self, name, params_dict):
		# self.npars = 9
		# self.parameter_names = ['P', 't0', 'Rrat', 'aR', 'cosi', 'e', 'w', "u1", "u2"]
		# self.parameter_latex_names = ['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'cos($i$)', 'e', 'w', r"$u_1$", r"$u_2$"]
		# self.parameter_units = ["days", "days", "", "", "", "", "", "", ""]

		self.npars = 8
		self.parameter_names = ['P', 't0', 'Rrat', 'aR', 'cosi', 'e', 'w', "u1"]
		self.parameter_latex_names = ['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'cos($i$)', 'e', 'w', r"$u_1$"]
		self.parameter_units = ["days", "days", "", "", "", "", "", ""]

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
				if not params_values[pname][3] != 0:
					print("Reciprocal prior cannot have a starting interval value of 0.0")
					sys.exit(1)
				self.prior.append(reciprocal(params_values[pname][3], params_values[pname][4]))
			else:
				print(f"Parameter prior distribution chosen not recognized: {params_values[pname][2]}")
				sys.exit(1)
		self.prior = np.asarray(self.prior)

		# Init batman stuff
		self.batparams = batman.TransitParams()
		self.batparams.limb_dark = 'linear'

	def get_parameters_names(self):
		return self.parameter_names

	def get_parameters_latex(self):
		return self.parameter_latex_names

	def get_parameters_units(self):
		return self.parameter_units

	def init_model(self, time, cadence, supersample_factor):
		self.update_params(self.sample_prior().T)
		exp_time = cadence - (cadence/supersample_factor)
		self.batmodel = batman.TransitModel(self.batparams, time, supersample_factor=supersample_factor, exp_time=exp_time)

	def sample_prior(self, num=1):
		# return np.array([self.priors[key].rvs(num) for key in self.prior]).T
		return np.hstack([p.rvs([num, 1]) for p in self.prior])

	def update_params(self, params):
		self.batparams.per = params[0]
		self.batparams.t0 = params[1]
		self.batparams.rp = params[2]
		self.batparams.a = params[3]
		self.batparams.inc = np.rad2deg(np.arccos(params[4]))
		self.batparams.ecc = params[5]
		self.batparams.w = params[6]
		self.batparams.u = [params[7]]

	def lnprior(self, params):
		# return sum([self.priors[key].logpdf(params[i]) for i, key in enumerate(self.priors)])
		return np.sum([self.prior[i].logpdf(params[i]) for i in range(self.prior.size)])

	def get_value(self, params):
		self.update_params(params)
		logging.debug("Before batman flux")
		logging.debug(params)
		flux = self.batmodel.light_curve(self.batparams)
		logging.debug("After batman flux")
		return (flux-1)*1e6
	
	# def oversample(self, params, supersample_factor=10):
	# 	self.update_params(params)
	# 	exp_time = self.time[1]-self.time[0]
	# 	m = batman.TransitModel(self.batparams, self.time, supersample_factor=supersample_factor, exp_time=exp_time)
	# 	return m.light_curve(self.batparams)



# class StarryModel(Model):
# 	parameter_names = ("alpha", "ell", "log_sigma2")

# 	def get_value(self, t):
# 		pass
