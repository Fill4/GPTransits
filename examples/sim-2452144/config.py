import numpy as np
import os
from gptransits import Settings

settings = Settings()
settings.seed = 42                      # RNG number seed
settings.include_errors = False         # Include lightcurve errors in calculation. Set to false as this is simulated data
settings.save = False                    # Save the mcmc iterations in a file
settings.num_threads = os.cpu_count()   # Use all cores available
settings.num_steps = 20000              # Number of iterations for the mcmc

# Define GP error model
# Allowed types: [oscillation_bump, granulation, white_noise]
# Param format: [fix_value, value, distribution, dist_arg_1, dist_arg_2]
gp = [ {
        "type": "oscillation_bump",
        "name": "Oscillation Envelope",
        "params": {
            "values": {
                "P_g": [False, None, "uniform", 40, 600],
                "Q": [False, None, "uniform", 1.0, 8],
                "nu_max": [False, None, "uniform", 100, 200]
            }
        }
    }, {
        "type": "granulation",
        "name": "Mesogranulation",
        "params": {
            "latex_names": [r'$a_{gran}$', r'$b_{gran}$'],
            "values": {
                "a_gran": [False, None, "uniform", 50, 500],
                "b_gran": [False, None, "uniform", 10, 80]
            }
        }
    }, {
        "type": "white_noise",
        "name": "Shot Noise",
        "params": {
            "values": {
                "jitter": [False, None, "uniform", 30, 350]
            }
        }
    }
]

# Define parametric model
# Allowed types: [batman_model]
# Only works with batman transit right now
transit = {
    "type": "batman_model",
    "name": "Planet",
    "params": {
        "values": {
            "P": [False, None, "uniform", 5.7, 6.2],
            "t0": [False, None, "uniform", 2.8, 3.3],
            "Rrat": [False, None, "reciprocal", 0.001, 0.1],
            "aR": [False, None, "reciprocal", 1.0, 15.0],
            "cosib": [False, None, "uniform", 0.0, 1.0],
            "e": [True, 0.0, "uniform", 0.0, 1.0],
            "w": [True, 90, "uniform", 0.0, 90.0],
            "u1": [True, 0.4996, "uniform", 0.1, 0.9],
            "u2": [True, 0.0847, "reciprocal", 0.001, 0.5],
        }
    }
}
