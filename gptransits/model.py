#!/usr/bin/env python3
"""
Fits a transit model together with a Gaussian Process model created by celerite to lightcurve observations
"""

import sys
import copy
import pickle
from pathlib import Path
import logging
import argparse
import importlib

import numpy as np
import matplotlib.pyplot as plt
import emcee
import celerite
from multiprocessing.pool import Pool

# local imports
from .transit import BatmanModel
from .gp import GPModel
from .settings import Settings

__all__ = ["Model"]

class Model():
    lc_file = None
    config_file = None
    config = None
    settings = None

    # Add these variables as class variables to make them global for multithreading.Pool
    time = None
    flux = None
    flux_err = None

    mean_model = None
    mean_model_ndim = 0
    has_mean_model = False

    gp_model = None
    gp_model_ndim = 0
    gp = None

    @classmethod
    def __init__(cls, lc_file, config_file):
        cls.lc_file = Path(lc_file)
        cls.config_file = Path(config_file)

        # Get config from file
        try:
            spec = importlib.util.spec_from_file_location("config", cls.config_file)
            cls.config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.config)
        except Exception:
            logging.exception(f"ERROR: Couldn't import {cls.config_file.stem}")
            sys.exit(1)
        sys.modules["config"] = cls.config

        if hasattr(cls.config, "settings"):
            cls.settings = cls.config.settings
        else:
            cls.settings = Settings()
        
        # Setup logging according to args
        if cls.settings.log_to_file:
            logging.basicConfig(format='%(levelname)s: %(message)s', filename="log.txt", filemode="w", level=cls.settings.log_level)
        else:
            logging.basicConfig(format='%(levelname)s: %(message)s', level=cls.settings.log_level)

        # Set random state early to control the initial random sample and also emcee iterations
        np.random.seed(cls.settings.seed)

    # Defines the likelihood function for emcee. Classmethod (global) to allow for multithreading
    @classmethod
    def log_likelihood(cls, params):
        gp_lnprior = cls.gp_model.lnprior(params[:cls.gp_model_ndim])
        if cls.has_mean_model:
            mean_lnprior = cls.mean_model.lnprior(params[cls.gp_model_ndim:])
        else:
            mean_lnprior = 0
        if not (np.isfinite(gp_lnprior) and np.isfinite(mean_lnprior)):
            return -np.inf
        
        # Convert params to celerite model and update gp object
        gp_params = cls.gp_model.get_parameters_celerite(params[:cls.gp_model_ndim])
        cls.gp.set_parameter_vector(gp_params)

        if cls.has_mean_model:
            lnlikelihood = cls.gp.log_likelihood(cls.flux - cls.mean_model.get_value(params[cls.gp_model_ndim:]))
        else:
            lnlikelihood = cls.gp.log_likelihood(cls.flux)
            
        # return mean_lnprior + gp_lnprior + lnlikelihood
        return gp_lnprior + lnlikelihood
    
    @classmethod
    def run(cls):
        # Read data from file
        logging.info(f'Working in directory: {cls.lc_file.parent.resolve()}')
        logging.info(f'Reading data from: {cls.lc_file.name}')
        try:
            time, flux, flux_err = np.loadtxt(cls.lc_file, unpack=True)
        except ValueError:
            try:
                time, flux = np.loadtxt(cls.lc_file, unpack=True)
            except ValueError:
                logging.error(f"Error: Couldn't load data from the lc_file: {cls.lc_file}")
                sys.exit(1)
        if cls.settings.include_errors:
            if flux_err is None:
                logging.error(f"Error: Need a flux_err column when include_errors is True")
                sys.exit(1)

        # Convert time to days and flux and err to ppm (if needed). Make copy to separate memory of arrays for C
        cls.time = time.copy()                    # keep days
        # cls.time = time.copy() / (24.*3600.)        # seconds to days
        # cls.flux = flux.copy()                      # keep ppm
        cls.flux = (flux.copy() - 1.0) * 1e6      # frac to ppm
        if cls.settings.include_errors:
            # cls.flux_err = flux_err.copy()          # keep ppm
            cls.flux_err = flux_err.copy() * 1e6  # frac to ppm

        # Setup a transit model if there is configs in file ,otherwise setup empty params
        if hasattr(cls.config, "transit"):
            cls.mean_model = BatmanModel(cls.config.transit["name"], cls.config.transit["params"])
            cls.mean_model.init_model(cls.time, cls.time[1]-cls.time[0], 2)
            cls.mean_model_ndim = cls.mean_model.npars
            cls.has_mean_model = True

        # Setup gp model. Create a setup_gp_model() function that reads in an external config file
        cls.gp_model = GPModel(cls.config.gp)
        kernel = cls.gp_model.get_kernel(cls.gp_model.sample_prior()[0])
        cls.gp_model_ndim = kernel.get_parameter_vector().size

        # Setup celerite gp with kernel. Time goes in microseconds for the muHz parameters in the GP
        cls.gp = celerite.GP(kernel)
        days_to_microsec = (24*3600) / 1e6
        if cls.settings.include_errors:
            cls.gp.compute(cls.time*days_to_microsec, yerr=cls.flux_err)
        else:
            cls.gp.compute(cls.time*days_to_microsec)


        # -------------------- MCMC ----------------------------
        # Setup mcmc
        ndim = cls.gp_model_ndim + cls.mean_model_ndim
        nwalkers =  ndim * 4
        nsteps = cls.settings.num_steps
        pool = Pool(cls.settings.num_threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, cls.log_likelihood, pool=pool)

        # Sample priors to get initial values for all walkers
        init_gp_params = cls.gp_model.sample_prior(num=nwalkers)
        if cls.mean_model is not None:
            init_mean_params = cls.mean_model.sample_prior(num=nwalkers)
            init_params = np.hstack([init_gp_params, init_mean_params])
        else:
            init_params = init_gp_params

        # Run mcmc
        logging.info(f"Runnning MCMC on {cls.settings.num_threads} processes ...")
        sampler.run_mcmc(init_params, nsteps, progress=True)

        # Save data
        if cls.settings.save:
            # If there is no output folder, create it
            if not (lc_file.parent / "output").is_dir():
                Path.mkdir(lc_file.parent / "output")

            logging.info(f"Saving chain")
            with open(lc_file.parent / "output" / "chain.pk", "wb") as f:
                pickle.dump(sampler.chain, f, protocol=-1)

            logging.info(f"Saving lnprobability")
            with open(lc_file.parent / "output" / "lnprobability.pk", "wb") as f:
                pickle.dump(sampler.lnprobability, f, protocol=-1)


if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("lc_file")
    parser.add_argument("config_file")
    args = parser.parse_args()
    
    model = Model(args.lc_file, args.config_file)
    model.run()