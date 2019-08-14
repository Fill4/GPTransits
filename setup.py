
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
    url="https://github.com/Fill4/gptransits",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

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
        'emcee==3.0rc2',
        "celerite==0.3.0",
    ],
    include_package_data=True,
    zip_safe=True,
)