import numpy as np
import os
from gptransits import Settings

settings = Settings()
settings.seed = 42                      # RNG number seed
settings.include_errors = True          # Include lightcurve errors in calculation
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
                "P_g": [False, None, "uniform", 10, 2000],
                "Q": [False, None, "uniform", 1.0, 20],
                "nu_max": [False, None, "uniform", 80, 220]
            }
        }
    }, {
        "type": "granulation",
        "name": "Mesogranulation",
        "params": {
            "latex_names": [r'$a_{gran}$', r'$b_{gran}$'],
            "values": {
                "a_gran": [False, None, "uniform", 10, 400],
                "b_gran": [False, None, "uniform", 10, 200]
            }
        }
    }, {
        "type": "white_noise",
        "name": "Shot Noise",
        "params": {
            "values": {
                "jitter": [False, None, "uniform", 0.1, 400]
            }
        }
    }
]
