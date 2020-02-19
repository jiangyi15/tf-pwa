from .model import Cache_Model,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from .angle import cal_ang_file,cal_ang_file4
from .utils import load_config_file,flatten_np_data,pprint,error_print


from iminuit import Minuit
def fit_minuit(model,bounds_dict={},hesse=True,minos=False):
    fcn = FCN(model)
    var_args = {}
    var_names = model.Amp.trainable_vars
    for i in var_names:
        var_args[i] = model.Amp.get(i).numpy()
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
def fit_scipy(model, method="BFGS", **kwargs):
    fcn = FCN(model)
    points = []
    nlls = []
    maxiter = 2000
    if method in ["BFGS","CG","Nelder-Mead"]:
        def callback(x):
            points.append([float(i) for i in model.Amp.get_all_val()])
            nlls.append(float(fcn.cached_nll))
            if len(nlls) > maxiter:
                return False, {"nlls": nlls, "points": points}
                raise Exception("Reached the largest iterations: {}".format(maxiter))
            print(fcn.cached_nll)
        f_g = model.Amp.trans_fcn_grad(fcn.nll_grad)
        fitres = minimize(f_g, np.array(model.Amp.get_all_val(True)), method=method, jac=True, callback=callback,
                     options={"disp": 1})
        model.Amp.set_all(fitres.x,bound_trans=True)
    elif method in ["L-BFGS-B"]:
        def callback(x):
            points.append([float(i) for i in x])
            nlls.append(float(fcn.cached_nll))
        bnds = []
        for name in model.Amp.trainable_vars:
            if name in model.Amp.bounds_dict:
                bnds.append(model.Amp.bounds_dict[name])
            else:
                bnds.append((None, None))
        fitres = minimize(fcn.nll_grad, model.Amp.get_all_val(), method=method, jac=True, bounds=bnds, callback=callback,
                     options={"disp": 1, "maxcor": 10000, "ftol": 1e-15, "maxiter": maxiter})
    elif method in ["basinhopping"]:
        def callback(x):
            points.append([float(i) for i in model.Amp.get_all_val()])
            nlls.append(float(fcn.cached_nll))
            print(fcn.cached_nll)
        f_g = model.Amp.trans_fcn_grad(fcn.nll_grad)
        if "niter" in kwargs:
            niter = kwargs["niter"]
        else:
            niter = 1
        fitres = basinhopping(f_g,np.array(model.Amp.get_all_val(True)),niter=niter,stepsize=3.0,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True},"callback":callback})
        model.Amp.set_all(fitres.x,bound_trans=True)
    else:
        raise Exception("unknown method")

    return fitres, nlls, points


#import pymultinest
def fit_multinest(model):
    pass
