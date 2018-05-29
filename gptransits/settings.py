class Settings(object):
	def __init__(self):
		self.input_file = None
		self.input_list = None

		self.output = False
		self.output_file = None

		self.verbose = False
		self.include_errors = False

		self.burnin = 500
		self.iterations = 2000
		self.nwalkers = 20
		
		self.plot_gp = False
		self.plot_corner = False
		self.plot_psd = False
		
		self.tess_settings = False
		self.raw_data_settings = False
		self.diamonds_settings = False

	@property
	def plots(self):
		return any([self.plot_gp, self.plot_corner, self.plot_psd])

