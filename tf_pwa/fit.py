import tensorflow as tf
import time
import numpy as np
import json


from iminuit import Minuit
def fit_minuit(fcn,bounds_dict={},hesse=True,minos=False):
    var_args = {}
    var_names = fcn.model.Amp.vm.trainable_vars
    for i in var_names:
        var_args[i] = fcn.model.Amp.vm.get(i).numpy()
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        var_args["error_" + i] = 0.1
    m = Minuit(fcn, forced_parameters=var_names, errordef=0.5, grad=fcn.grad, print_level=2, use_array_call=True,
               **var_args)
    print("########## begin MIGRAD")
    now = time.time()
    m.migrad()  # (ncall=10000))#,precision=5e-7))
    print("MIGRAD Time", time.time() - now)
    if hesse:
        print("########## begin HESSE")
        now = time.time()
        m.hesse()
        print("HESSE Time", time.time() - now)
    if minos:
        print("########## begin MINOS")
        now = time.time()
        m.minos()  # (var="")
        print("MINOS Time", time.time() - now)
    return m


from scipy.optimize import minimize,BFGS,basinhopping
def fit_scipy(fcn, method="BFGS",bounds_dict={}, **kwargs):
    points = []
    nlls = []
    maxiter = 2000
    if method in ["BFGS","CG","Nelder-Mead"]:
        def callback(x):
            points.append([float(i) for i in fcn.model.Amp.get_all_val()])
            nlls.append(float(fcn.cached_nll))
            if len(nlls) > maxiter:
                return False, {"nlls": nlls, "points": points}
                raise Exception("Reached the largest iterations: {}".format(maxiter))
            print(fcn.cached_nll)
        fcn.model.Amp.set_bound(bounds_dict)
        f_g = fcn.model.Amp.trans_fcn_grad(fcn.nll_grad)
        fitres = minimize(f_g, np.array(fcn.model.Amp.get_all_val(True)), method=method, jac=True, callback=callback,
                     options={"disp": 1})
    elif method in ["L-BFGS-B"]:
        def callback(x):
            points.append([float(i) for i in x])
            nlls.append(float(fcn.cached_nll))
        bnds = []
        for name in fcn.model.Amp.trainable_vars:
            if name in bounds_dict:
                bnds.append(bounds_dict[name])
            else:
                bnds.append((None, None))
        fitres = minimize(fcn.nll_grad, fcn.model.Amp.get_all_val(), method=method, jac=True, bounds=bnds, callback=callback,
                     options={"disp": 1, "maxcor": 10000, "ftol": 1e-15, "maxiter": maxiter})
    elif method in ["basinhopping"]:
        def callback(x):
            points.append([float(i) for i in fcn.model.Amp.get_all_val()])
            nlls.append(float(fcn.cached_nll))
            print(fcn.cached_nll)
        fcn.model.Amp.set_bound(bounds_dict)
        f_g = fcn.model.Amp.trans_fcn_grad(fcn.nll_grad)
        if "niter" in kwargs:
            niter = kwargs["niter"]
        else:
            niter = 1
        fitres = basinhopping(f_g,np.array(fcn.model.Amp.get_all_val(True)),niter=niter,stepsize=3.0,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True},"callback":callback})
    else:
        raise Exception("unknown method")

    return fitres, nlls, points


#import pymultinest
def fit_multinest(model):
    pass