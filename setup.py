
from setuptools import setup
from os import path

# Load the __version__ variable without importing the package already
exec(open('gptransits/version.py').read())

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="gptransits",
    version=__version__,
    author="Filipe Pereira",
    author_email="filipe.pereira@astro.up.pt",
    license="MIT",
    packages=["gptransits"],
    description="Fit planetary transits and stellar signals at the same time with the help of gaussian processes",
    long_description=long_description,
    long_description_content_type='text/markdown',

    python_requires=">=3.6",
    setup_requires=[
        "numpy",
        "pybind11",
    ],
    install_requires=[ 
        "numpy",
        "batman-package",
        "pybind11",
        "scipy",
        "matplotlib",
        "tqdm",
        "corner",
        "astropy>=3.0.0",
        'emcee @ git+https://github.com/dfm/emcee',
        "celerite==0.3.0",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
)