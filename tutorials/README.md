Tutorials
----------

Here is some simple script for PWA.
The model is described in [config.yml](config.yml), a simple version of the process:
```math
A \rightarrow B C D,
```
including such three decay chain
```math
A \rightarrow R_{BC} D, R_{BC} \rightarrow B C;
```
```math
A \rightarrow R_{BD} C, R_{BD} \rightarrow B D;
```
```math
A \rightarrow R_{CD} B, R_{CD} \rightarrow C D.
```

1. [gen_toy.py](gen_toy.py)

    Using `python gen_toy.py`, we can get data sample and phase space (PHSP) sample of model in [config.yml](config.yml) and parameters in [gen_params.json](gen_params.json).

2. [fit.py](fit.py)

    Using `python fit.py`, we can do a simple fit for [config.yml](config.yml). The fit parameters are stored in `final_params.json`. This script also include the function of following scripts (plot_vars.py, cal_errors.py, cal_fitfracs.py). 

3. [plot_vars.py](plot_vars.py)

    Using `python plot_vars.py`, we can get partial wave distribution plots of the fit parameters.

4. [cal_errors.py](cal_errors.py)
    
    Using `python cal_errors.py`, we can get the fit errors about `final_params.json`.

5. [cal_fitfracs.py](cal_fitfracs.py)

    Using `python cal_fitfracs.py`, we can get the fit fractions of the fit.

