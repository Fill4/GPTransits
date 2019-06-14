import sys
import logging

import numpy as np
from scipy.stats import uniform, norm, reciprocal

import batman
import pysyzygy as ps

# Define the transit models
class BatmanModel(object):
	def __init__(self, name, params_dict):
		self.npars = 9
		self.parameter_names = np.array(['P', 't0', 'Rrat', 'aR', 'cosib', 'e', 'w', "u1", "u2"])
		self.parameter_latex_names = np.array(['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'cos($i$)', 'e', 'w', r"$u_1$", r"$u_2$"])
		self.parameter_units = np.array(["days", "days", "", "", "", "", "", "", ""])
		self.mask = np.full(self.npars, True)

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
			self.parameter_latex_names = np.array(params_dict["latex_names"])
		if "units" in params_dict:
			if (self.npars != len(params_dict["units"])):
				print(f"Component units must have size {self.npars}")
				sys.exit(1)
			self.parameter_units = np.array(params_dict["units"])
		
		# Setup the priors and init parameters
		self.prior = []
		params_values = params_dict["values"]
		self.pars = np.zeros(self.npars)
		for i, pname in enumerate(self.parameter_names):
			if params_values[pname][0]:
				self.mask[i] = False
				self.pars[i] = params_values[pname][1]
			elif params_values[pname][2] == "uniform":
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
		self.batpars = batman.TransitParams()
		if self.mask[-1] == False and self.pars[-1] == 0.0:
			self.batpars.limb_dark = 'linear'
		else:
			self.batpars.limb_dark = "quadratic"
		logging.debug(f"Transit Model using {self.batpars.limb_dark} limb darkening")

	def get_parameters_names(self):
		return self.parameter_names[self.mask]

	def get_parameters_latex(self):
		return self.parameter_latex_names[self.mask]

	def get_parameters_units(self):
		return self.parameter_units[self.mask]

	def init_model(self, time, cadence, supersample_factor):
		self.update_params(self.sample_prior()[0].T)
		exp_time = cadence - (cadence/supersample_factor)
		self.model = batman.TransitModel(self.batpars, time, supersample_factor=supersample_factor, exp_time=exp_time)

	def sample_prior(self, num=1):
		# return np.array([self.priors[key].rvs(num) for key in self.prior]).T
		return np.hstack([p.rvs([num, 1]) for p in self.prior])

	def update_params(self, params):
		self.pars[self.mask] = params
		self.batpars.per = self.pars[0]
		self.batpars.t0 = self.pars[1]
		self.batpars.rp = self.pars[2]
		self.batpars.a = self.pars[3]
		self.batpars.inc = np.rad2deg(np.arccos(self.pars[4]/self.pars[3]))
		self.batpars.ecc = self.pars[5]
		self.batpars.w = self.pars[6]
		self.batpars.u = [self.pars[7], self.pars[8]]

	def lnprior(self, params):
		# return sum([self.priors[key].logpdf(params[i]) for i, key in enumerate(self.priors)])
		return np.sum([self.prior[i].logpdf(params[i]) for i in range(self.prior.size)])

	def get_value(self, params, time):
		self.update_params(params)

		logging.debug("Before batman flux")
		logging.debug(f"Period: {self.batpars.per}")
		logging.debug(f"Epoch: {self.batpars.t0}")
		logging.debug(f"Radius ratio: {self.batpars.rp}")
		logging.debug(f"Semi-major axis: {self.batpars.a}")
		logging.debug(f"Inclination: {self.batpars.inc}")
		logging.debug(f"Eccentricity: {self.batpars.ecc}")
		logging.debug(f"Arg periastron: {self.batpars.w}")
		logging.debug(f"Limb darkening: {self.batpars.u}")
		
		flux = self.model.light_curve(self.batpars)
		
		logging.debug("After batman flux")
		return (flux-1)*1e6
	
	# def oversample(self, params, supersample_factor=10):
	# 	self.update_params(params)
	# 	exp_time = self.time[1]-self.time[0]
	# 	m = batman.TransitModel(self.pars, self.time, supersample_factor=supersample_factor, exp_time=exp_time)
	# 	return m.light_curve(self.pars)


class PysyzygyModel(object):
	def __init__(self, name, params_dict):
		self.npars = 9
		self.parameter_names = np.array(['P', 't0', 'Rrat', 'aR', 'cosib', 'e', 'w', "u1", "u2"])
		self.parameter_latex_names = np.array(['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'b', 'e', 'w', r"$u_1$", r"$u_2$"])
		self.parameter_units = np.array(["days", "days", "", "", "", "", "", "", ""])
		self.mask = np.full(self.npars, True)

		# self.npars = 8
		# self.parameter_names = ['P', 't0', 'Rrat', 'aR', 'cosi', 'e', 'w', "u1"]
		# self.parameter_latex_names = ['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'cos($i$)', 'e', 'w', r"$u_1$"]
		# self.parameter_units = ["days", "days", "", "", "", "", "", ""]
		# self.mask = np.full(self.npars, True)

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
			self.parameter_latex_names = np.array(params_dict["latex_names"])
		if "units" in params_dict:
			if (self.npars != len(params_dict["units"])):
				print(f"Component units must have size {self.npars}")
				sys.exit(1)
			self.parameter_units = np.array(params_dict["units"])
		
		# Setup the priors and init parameters
		self.prior = []
		params_values = params_dict["values"]
		for i, pname in enumerate(self.parameter_names):
			if params_values[pname][0]:
				self.mask[i] = False
				if i == 5:
					self.ecc = params_values[pname][1]
				elif i == 7:
					self.u1 = params_values[pname][1]
				elif i == 8:
					self.u2 = params_values[pname][1]
				else:
					# logging.error("Fixing parameter values is not yet supported")
					logging.error("Can only fix limb darkening values")
					sys.exit(1)
			elif params_values[pname][2] == "uniform":
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
		self.pars = dict(per = 10, t0 = 0, RpRs = 0.1, aRs = 2, b = 0, ecc = self.ecc, w = 0, u1 = self.u1, u2 = self.u2)
		self.ldmodel = ps.QUADRATIC

	def get_parameters_names(self):
		return self.parameter_names[mask]

	def get_parameters_latex(self):
		return self.parameter_latex_names[mask]

	def get_parameters_units(self):
		return self.parameter_units[mask]

	def init_model(self, time, cadence, supersample_factor):
		self.cadence = cadence
		self.supersamp = supersample_factor

	def sample_prior(self, num=1):
		# return np.array([self.priors[key].rvs(num) for key in self.prior]).T
		return np.hstack([p.rvs([num, 1]) for p in self.prior])

	def update_params(self, params):

		self.pars["per"] = params[0]
		self.pars["t0"] = params[1]
		self.pars["RpRs"] = params[2]
		self.pars["aRs"]= params[3]
		self.pars["b"] = params[4]
		# self.pars["ecc"] = params[5]
		self.pars["w"]= np.deg2rad(params[5])
		# self.pars["u1"]= params[7]
		# self.pars["u2"]= params[8]
		logging.debug(self.pars)
		self.model = ps.Transit(**self.pars, ldmodel=self.ldmodel, exppts=self.supersamp)

	def lnprior(self, params):
		# return sum([self.priors[key].logpdf(params[i]) for i, key in enumerate(self.priors)])
		return np.sum([self.prior[i].logpdf(params[i]) for i in range(self.prior.size)])

	def get_value(self, params, time):
		self.update_params(params)

		logging.debug("Before pysygyzy flux")
		logging.debug(f"Period: {self.batpars.per}")
		logging.debug(f"Epoch: {self.batpars.t0}")
		logging.debug(f"Radius ratio: {self.batpars.rp}")
		logging.debug(f"Semi-major axis: {self.batpars.a}")
		logging.debug(f"Inclination: {self.batpars.inc}")
		logging.debug(f"Eccentricity: {self.batpars.ecc}")
		logging.debug(f"Arg periastron: {self.batpars.w}")
		logging.debug(f"Limb darkening: {self.batpars.u}")

		flux = self.model(time, param="unbinned")

		logging.debug("After pysygyzy flux")
		return (flux-1)*1e6
	
	# def oversample(self, params, supersample_factor=10):
	# 	self.update_params(params)
	# 	exp_time = self.time[1]-self.time[0]
	# 	m = batman.TransitModel(self.pars, self.time, supersample_factor=supersample_factor, exp_time=exp_time)
	# 	return m.light_curve(self.pars)
