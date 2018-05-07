from setuptools import setup, find_packages
setup(
    name='gptransits',
    version='0.2',
    author='Filipe Pereira',
    author_email='filipe.pereira@astro.up.pt',
    license='MIT',

    packages=['gptransits'],
    install_requires=[	'matplotlib', 'numpy', 'emcee >= 3.0.0.dev0', 'celerite >= 0.3.0', 'scipy >= 1.0.0', 'astropy >= 3.0.1', 
    					'h5py >= 2.8.0rc1', 'tqdm', 'corner'],
    description='Fit time series to light curves with correlated noise modelled by gaussian processes',
    
)