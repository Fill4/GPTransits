gptransits
==========

### Fit planetary transits and stellar signals at the same time with the help of gaussian processes

### Code usage
~~~~
import gptransits

model = gptransits.Model("lightcurve_file.txt", "config_file.py")
model.run()
model.analysis(plot=True, fout="results_file.txt")
~~~~

### TODO
- [x] Add transit fitting to code
- [x] Refactor code to use multithreading.Pool efficiently
- [x] Redo helper.py to work correctly with new Model and analyze the saved chains
- [x] Compare transit results with and without GP
- [ ] Remove unnecessary code from helper and other convergence places
- [ ] Evaluate keeping all parameters even without convergence. Test if converged are equal to usual
- [ ] Add limb darkening parametrization from Kipping 2013
- [ ] Look at exoplanet for more parametrizations and possible improvements
- [ ] Compare GP results with and without transit
- [ ] Add example with simulated transit and real/simulated stellar
- [ ] Add proper documentation