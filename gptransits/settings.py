class Settings(object):
	def __init__(self):
		self.plot_gp = False
		self.plot_corner = False
		self.plot_psd = False
		self.plots = any([self.plot_gp, self.plot_corner, self.plot_psd])

		self.burnin = 500
		self.iterations = 2000
		self.nwalkers = 20