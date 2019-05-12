
from setuptools import setup
import gptransits

setup(
    name="gptransits",
    version=gptransits.__version__,
    author="Filipe Pereira",
    author_email="filipe.pereira@astro.up.pt",
    license="MIT",
    packages=["gptransits"],
    description="Fit planetary transits and stellar signals at the same time with the help of gaussian processes",
    long_description=open("README.md").read(),

    python_requires=">=3.6",
    install_requires=["matplotlib", "numpy", "scipy", "celerite", "batman-package", "astropy", "tqdm", "corner", 
                        "emcee @ git+https://github.com/dfm/emcee"],
    
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
)