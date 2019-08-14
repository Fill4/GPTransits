
from setuptools import setup

# Load the __version__ variable without importing the package already
exec(open('gptransits/version.py').read())

setup(
    name="gptransits",
    version=__version__,
    author="Filipe Pereira",
    author_email="filipe.pereira@astro.up.pt",
    license="MIT",
    packages=["gptransits"],
    description="Fit planetary transits and stellar signals at the same time with the help of gaussian processes",

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