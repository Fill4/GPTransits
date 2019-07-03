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


# Model imports
from .transit import BatmanModel, PysyzygyModel
from .gp import GPModel
from .settings import Settings

# Analysis imports
from .convergence import geweke, gelman_rubin, gelman_brooks
from .stats import mapv, mode, hpd
from .plot import *

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

    lnlike_func = None

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

        cls.load_data()
        cls.setup_models()

    @classmethod
    def load_data(cls):
        # Read data from file
        logging.info(f'Reading data from: {cls.lc_file.name}')
        logging.info(f'Working in directory: {cls.lc_file.parent.resolve()}')
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

    @classmethod
    def setup_models(cls):
        # Setup a transit model if there is configs in file ,otherwise setup empty params
        if hasattr(cls.config, "transit"):
            cls.mean_model = BatmanModel(cls.config.transit["name"], cls.config.transit["params"])
            # cls.mean_model = PysyzygyModel(cls.config.transit["name"], cls.config.transit["params"])
            cls.mean_model.init_model(cls.time, cls.time[1]-cls.time[0], 6)
            cls.mean_model_ndim = cls.mean_model.mask.sum()
            cls.has_mean_model = True

        if hasattr(cls.config, "gp"):
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


        if cls.mean_model is not None and cls.gp_model is not None:
            cls.lnlike_func = cls.full_lnlike
            logging.info("Model with GP and transit")
        elif cls.gp_model is not None:
            cls.lnlike_func = cls.gp_lnlike
            logging.info("Model with GP")
        elif cls.mean_model is not None:
            cls.lnlike_func = cls.transit_lnlike
            logging.info("Model with transit")
        else:
            logging.error("Need to define at least a gp or transit model")
            sys.exit()
    
    @classmethod
    def run(cls):
        # -------------------- MCMC ----------------------------
        # MCMC settings
        ndim = cls.gp_model_ndim + cls.mean_model_ndim
        nwalkers =  ndim * 4
        nsteps = cls.settings.num_steps

        # Ability to restart progress in the middle of run_mcmc
        # If there is a backend and we use it, set init_params to None so that run_mcmc uses the ones from the backend
        if cls.settings.save:
            if not (cls.lc_file.parent / "output").is_dir():
                Path.mkdir(cls.lc_file.parent / "output")
            filename = cls.lc_file.parent / "output" / "chain.hdf5"
            backend = emcee.backends.HDFBackend(str(filename))

            if filename.is_file():
                if backend.iteration >= nsteps:
                    logging.info("Number of iterations is equal or lower than the one found in the backend")
                    return
                    # sys.exit(4)
                nsteps = nsteps - backend.iteration
                init_params = None
                set_params = False
            else:
                set_params = True

            # TODO: Add flag/setting to reset backend. Will need to init priors in that case
            # backend.reset(nwalkers, ndim)
        # Otherwise set backend to None and just init the params from the prior
        else:
            backend=None
            set_params = True

        # If the initial params are not taken from the backend init them from the prior
        if set_params:
            # Sample priors to get initial values for all walkers
            if cls.gp_model is not None and cls.mean_model is not None:
                init_gp_params = cls.gp_model.sample_prior(num=nwalkers)
                init_mean_params = cls.mean_model.sample_prior(num=nwalkers)
                init_params = np.hstack([init_gp_params, init_mean_params])
            elif cls.mean_model is not None:
                init_params = cls.mean_model.sample_prior(num=nwalkers)
            elif cls.gp_model is not None:
                init_params = cls.gp_model.sample_prior(num=nwalkers)

        # Multiprocessing settings and sampler initialization
        # cls.settings.num_threads = 6 # Just a small hack before things are changed. TODO: Need global override option
        if cls.settings.num_threads != 1:
            # with Pool(cls.settings.num_threads) as pool:
            pool = Pool(cls.settings.num_threads)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, cls.lnlike_func, pool=pool, backend=backend)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, cls.lnlike_func, backend=backend)

        # Run mcmc
        logging.info(f"Runnning MCMC on {cls.settings.num_threads} processes for {nsteps} iterations...")
        try:
            sampler.run_mcmc(init_params, nsteps, progress=True)
        except KeyboardInterrupt:
            logging.exception("User exited on Ctrl-C from run_mcmc")
            sys.exit(3)

        # Save data
        if cls.settings.save:
            # If there is no output folder, create it
            if not (cls.lc_file.parent / "output").is_dir():
                Path.mkdir(cls.lc_file.parent / "output")

            logging.info(f"Saving chain and lnprobability")
            with open(cls.lc_file.parent / "output" / "chain.pk", "wb") as f:
                pickle.dump(sampler.chain, f, protocol=-1)
            with open(cls.lc_file.parent / "output" / "lnprobability.pk", "wb") as f:
                pickle.dump(sampler.lnprobability, f, protocol=-1)


    # -------------------------------------------------------------------------------------------
    # LIKELIHOOD FUNCTIONS
    # -------------------------------------------------------------------------------------------
    # Defines the likelihood function for emcee. Classmethod (global) to allow for multithreading
    @classmethod
    def full_lnlike(cls, params):
        gp_lnprior = cls.gp_model.lnprior(params[:cls.gp_model_ndim])
        mean_lnprior = cls.mean_model.lnprior(params[cls.gp_model_ndim:])
        if not (np.isfinite(gp_lnprior) and np.isfinite(mean_lnprior)):
            return -np.inf
        
        # Convert params to celerite model and update gp object
        gp_params = cls.gp_model.get_parameters_celerite(params[:cls.gp_model_ndim])
        cls.gp.set_parameter_vector(gp_params)

        mean = cls.mean_model.get_value(params[cls.gp_model_ndim:], cls.time)
        lnlikelihood = cls.gp.log_likelihood(cls.flux - mean)
            
        return mean_lnprior + gp_lnprior + lnlikelihood

    @classmethod
    def gp_lnlike(cls, params):
        gp_lnprior = cls.gp_model.lnprior(params)
        if not np.isfinite(gp_lnprior):
            return -np.inf
        
        # Convert params to celerite model and update gp object
        cls.gp.set_parameter_vector(cls.gp_model.get_parameters_celerite(params))

        lnlikelihood = cls.gp.log_likelihood(cls.flux)
            
        return gp_lnprior + lnlikelihood

    @classmethod
    def transit_lnlike(cls, params):
        mean_lnprior = -cls.mean_model.lnprior(params[cls.gp_model_ndim:])
        if not np.isfinite(mean_lnprior):
            return -np.inf
    
        residuals = cls.flux - cls.mean_model.get_value(params, cls.time)
        # TODO: Needs white noise component added here maybe
        if cls.flux_err is not None:
            lnlikelihood = np.sum(-0.5 * (residuals**2 / cls.flux_err**2))
        else:
            lnlikelihood = np.sum(-0.5 * residuals**2)
        
        # if lnlikelihood > -18.5:
        # logging.info(f"lnlikelihood: {lnlikelihood}")

        return mean_lnprior + lnlikelihood


    # -------------------------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------------------------
    @classmethod
    def analysis(cls, plot=True, fout=None):
        # Setup folder names and load chain and posterior from the pickle files
        logging.info(f"Running analysis on: {cls.lc_file.stem} ...")
        
        output_folder = cls.lc_file.parent / "output"
        logging.info(f"Fetching data from: {output_folder}")
        if not output_folder.is_dir():
            logging.error(f"No directory with target data to analyse: {cls.lc_file.parent / 'output'}")
            sys.exit(5)

        figure_folder = cls.lc_file.parent / "figures"
        if not figure_folder.is_dir():
            figure_folder.mkdir()

        # Read in all data
        # try:
        logging.info(f"{output_folder}/chain.pk")
        with open(f"{output_folder}/chain.pk", "rb") as f:
            chain = pickle.load(f)
        with open(f"{output_folder}/lnprobability.pk", "rb") as p:
            posterior = pickle.load(p)
        # except Exception:
        #     logging.error("Can't open chain or lnprobability file")
        #     sys.exit(7)

        if cls.gp_model is not None and cls.mean_model is not None:
            gp_names = cls.gp_model.get_parameters_latex()
            transit_names = cls.mean_model.get_parameters_latex()
            names = np.hstack([gp_names, transit_names])
        elif cls.mean_model is not None:
            names = cls.mean_model.get_parameters_latex()
        elif cls.gp_model is not None:
            names = cls.gp_model.get_parameters_latex()

        output = f"{cls.lc_file.stem:>13s}"

        # Run burn-in diagnostics (Geweke)
        logging.info(f"Calculating Geweke diagnostic ...")
        starts, zscores = geweke(chain)
        chains_over = np.max(zscores, axis=2) > 2
        num_chains_over = np.sum(chains_over, axis=1)
        start_index = np.where(num_chains_over == np.min(num_chains_over))[0][0]
        if start_index == 0:
            start_index = 1
        
        walkers_mask = ~chains_over[start_index]
        geweke_flag = False
        if np.sum(chains_over[start_index])/chain.shape[0] > 0.4:
            walkers_mask[:] = True
            geweke_flag = True
        num_walkers = np.sum(walkers_mask)

        reduced_chain = chain[walkers_mask,starts[start_index]:,:]
        reduced_posterior = posterior[starts[start_index]:,walkers_mask]
        
        output = f"{output} {str(geweke_flag):>7s} {num_walkers:>4d} {starts[start_index]/chain.shape[1]:>5.2f}"

        # Cut chains with lower posterior
        logging.info(f"Removing lower posterior chains ...")
        posterior_mask = (np.median(reduced_posterior, axis=0) - np.median(reduced_posterior)) > -np.std(reduced_posterior)

        reduced_chain = reduced_chain[posterior_mask,:,:]
        reduced_posterior = reduced_posterior[:,posterior_mask]

        # Check consistency between chains
        logging.info(f"Calculating Gelman-Rubin diagnostic ...")
        r_hat, _, _ = gelman_rubin(reduced_chain)
        r_mvar = gelman_brooks(reduced_chain)

        logging.info(f"Calculating Gelman-Brooks diagnostic ...")
        r_hat_str = "".join([" {:>9.5f}".format(ri) for ri in r_hat])
        output = f"{output} {r_mvar.max():>9.5f}{r_hat_str}"


        logging.info(f"Calculating parameter statistics ...")
        params = {}
        samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
        params["median"] = np.median(samples, axis=0)
        params["hpd_down"], params["hpd_up"] = hpd(reduced_chain, level=0.683)
        hpd_99_down, hpd_99_up = hpd(reduced_chain, level=0.99)
        params["hpd_99_interval"] = (hpd_99_up - hpd_99_down)
        params["mapv"] = mapv(reduced_chain, reduced_posterior)
        params["modes"] = mode(chain)

        results_str = "".join([f"{params['mapv'][i]:>15.10f} {params['modes'][i]:>15.10f} {params['median'][i]:>15.10f} {params['hpd_down'][i]:>15.10f} {params['hpd_up'][i]:>15.10f} {params['hpd_99_interval'][i]:>15.10f}" for i in range(params['median'].size)])
        output = f"{output}{results_str}\n"

        """
        # Get the median and 68% intervals for each of the parameters
        # logging.info(f"Calculating medians and stds ...")
        # samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
        # percentiles = np.percentile(samples.T, [50,16,84], axis=1)
        # median = percentiles[0]
        # lower = median - percentiles[1]
        # upper = percentiles[2] - median

        # results_str = "".join([f" {median[i]:>15.10f} {lower[i]:>15.10f} {upper[i]:>15.10f}" for i in range(median.size)])
        # output = f"{output}{results_str}\n"
        """

        if plot:
            logging.info(f"Plotting Gelman-Rubin analysis ...")
            gelman_fig = gelman_rubin_plot(reduced_chain, pnames=names)
            gelman_fig.savefig(f"{figure_folder}/gelman_plot.pdf")

            logging.info(f"Plotting parameter histograms ...")
            parameter_fig = parameter_hist(chain, params, pnames=names)
            parameter_fig.savefig(f"{figure_folder}/parameter_hist.pdf")

            logging.info(f"Plotting corner ...")
            corner_fig = corner_plot(reduced_chain, pnames=names, downsample=5)
            corner_fig.savefig(f"{figure_folder}/corner_plot.pdf")

            logging.info(f"Plotting posterior histogram ...")
            posterior_fig = posterior_hist(reduced_posterior)
            posterior_fig.savefig(f"{figure_folder}/posterior_hist.pdf")

            logging.info(f"Plotting traces ...")
            trace_fig = trace_plot(chain, posterior, pnames=names, downsample=10)
            trace_fig.savefig(f"{figure_folder}/trace_plot.pdf")

            # Plot the GP dist, and PSD of the distributions
            logging.info(f"Plotting GP ...")
            gp_fig, gp_zoom_fig = gp_plot(cls.gp_model, cls.mean_model, params, cls.time, cls.flux, cls.flux_err, offset=0.05, oversample=10)
            gp_fig.savefig(f"{figure_folder}/gp_plot.pdf")
            gp_zoom_fig.savefig(f"{figure_folder}/gp_zoom_plot.pdf")

            if cls.gp_model is not None:
                logging.info(f"Plotting PSD ...")
                psd_fig = psd_plot(cls.gp_model, params, cls.time, cls.flux, include_data=True, parseval_norm=True)
                psd_fig.savefig(f"{figure_folder}/psd_plot.pdf")

        # Output to file
        if fout is not None:
            logging.info(f"Writing output to file {fout}...")
            with open(fout, "a+") as o:
                o.write(output)
        
        logging.info(f"{'-'*40}")
        plt.close("all")

        return output


if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("lc_file")
    parser.add_argument("config_file")
    args = parser.parse_args()
    
    model = Model(args.lc_file, args.config_file)
    model.run()