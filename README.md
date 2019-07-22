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
