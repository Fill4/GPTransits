import numpy as np
import os
from gptransits import Settings

settings = Settings()
settings.seed = 42
settings.include_errors = False
settings.save = True
settings.num_threads = os.cpu_count()
settings.num_steps = 10000

# Allowed types: [oscillation_bump, granulation, white_noise]
# Param format: [fix_value, value, distribution, dist_arg_1, dist_arg_2]
gp = [ {
        "type": "oscillation_bump",
        "name": "Oscillation Envelope",
        "params": {
            "values": {
                "P_g": [False, None, "uniform", 40, 600],
                "Q": [False, None, "uniform", 1.0, 8],
                "nu_max": [False, None, "uniform", 120, 180]
            }
        }
    }, {
        "type": "granulation",
        "name": "Mesogranulation",
        "params": {
            "latex_names": [r'$a_{gran}$', r'$b_{gran}$'],
            "values": {
                "a_gran": [False, None, "uniform", 50, 300],
                "b_gran": [False, None, "uniform", 30, 80]
            }
        }
    }, {
        "type": "white_noise",
        "name": "Shot Noise",
        "params": {
            "values": {
                "jitter": [False, None, "uniform", 50, 300]
            }
        }
    }
]

# Only works with 1 transit type right now
transit = {
    "type": "batman_model",
    "name": "Planet",
    "params": {
        "values": {
            "P": [False, None, "uniform", 8.3, 9.3],
            "t0": [False, None, "uniform", 4.7, 5.7],
            "Rrat": [False, None, "reciprocal", 0.001, 0.1],
            "aR": [False, None, "reciprocal", 1.0, 15.0],
            "cosi": [False, None, "uniform", 0.0, 1.0],
            "e": [False, None, "uniform", 0.0, 1.0],
            "w": [False, None, "uniform", 0.0, 90.0],
            "u1": [False, None, "uniform", 0.1, 0.9]
            # "u2": [False, None, "reciprocal", 0.001, 0.5]
        }
    }
}
