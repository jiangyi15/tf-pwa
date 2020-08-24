import tensorflow as tf
import time
import numpy as np
import json
from .fit_improve import minimize as my_minimize, Cached_FG


def fit_minuit(fcn,bounds_dict={},hesse=True,minos=False, **kwargs):
    try:
        from iminuit import Minuit
    except ImportError:
        raise RuntimeError("You haven't installed iminuit so you can't use Minuit to fit.")
    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    var_args = {}
    var_names = fcn.vm.trainable_vars
    for i in var_names:
        var_args[i] = fcn.vm.get(i).numpy()
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        var_args["error_" + i] = 0.1

    f_g = Cached_FG(fcn.nll_grad)
    m = Minuit(f_g.fun, forced_parameters=var_names, errordef=0.5, grad=f_g.grad, print_level=2, use_array_call=True,
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
    ndf = len(m.list_of_vary_param())
    ret = FitResult(dict(m.values), fcn, m.fval, ndf = ndf, success = m.migrad_ok())
    ret.set_error(dict(m.errors))
    return ret


from scipy.optimize import minimize,BFGS,basinhopping
def fit_scipy(fcn, method="BFGS",bounds_dict={}, check_grad=False, improve=False, maxiter=None):
    """

    :param fcn:
    :param method:
    :param bounds_dict:
    :param kwargs:
    :return:
    """
    args_name = fcn.vm.trainable_vars
    x0 = []
    bnds = []
    for name, i in zip(args_name, fcn.vm.trainable_variables):
        x0.append(i.numpy())
        if name in bounds_dict:
            bnds.append(bounds_dict[name])
        else:
            bnds.append((None, None))

    points = []
    nlls = []
    hess_inv = None
    now = time.time()
    if maxiter is None:
        maxiter = max(100 * len(x0), 2000)
    min_nll = 0.0
    ndf = 0
    #maxiter = 0
    def v_g2(x0):
        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        nll, gs0 = f_g(x0)
        gs = []
        for i, name in enumerate(args_name):
            x0[i] += 1e-5
            nll0, _ = f_g(x0)
            x0[i] -= 2e-5
            nll1, _ = f_g(x0)
            x0[i] += 1e-5
            gs.append((nll0-nll1)/2e-5)
        return nll, np.array(gs)
    
    if check_grad:
        print("checking gradients ...")
        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        nll, gs0 = f_g(x0)
        _, gs = v_g2(x0)
        for i, name in enumerate(args_name):
            print(args_name[i], gs[i], gs0[i])

    if method in ["BFGS", "CG", "Nelder-Mead", "test"]:
        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise Exception("x too large: {}".format(x_p))
            points.append(fcn.vm.get_all_val())
            nlls.append(float(fcn.cached_nll))
            # if len(nlls) > maxiter:
            #    with open("fit_curve.json", "w") as f:
            #        json.dump({"points": points, "nlls": nlls}, f, indent=2)
            #    pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
            print(fcn.cached_nll)

        #bd = Bounds(bnds)
        fcn.vm.set_bound(bounds_dict)

        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        x0 = np.array(fcn.vm.get_all_val(True))
        # s = minimize(f_g, x0, method='trust-constr', jac=True, hess=BFGS(), options={'gtol': 1e-4, 'disp': True})
        if method == "test":
            s = my_minimize(f_g, x0, method=method,
                        jac=True, callback=callback, options={"disp": 1, "gtol": 1e-3, "maxiter": maxiter})
        else:
            s = minimize(f_g, x0, method=method,
                        jac=True, callback=callback, options={"disp": 1, "gtol": 1e-3, "maxiter": maxiter})
        while improve and not s.success:
            min_nll = s.fun
            maxiter -= s.nit
            s = minimize(f_g, s.x, method=method,
                     jac=True, callback=callback, options={"disp": 1, "gtol": 1e-2, "maxiter": maxiter})
            if hasattr(s, "hess_inv"):
                edm = np.dot(np.dot(s.hess_inv, s.jac), s.jac)
            else:
                break
            if edm < 1e-5:
                print("edm: ", edm)
                s.message = "Edm allowed"
                break
            if abs(s.fun - min_nll) < 1e-3:
                break
        print(s)
        
        #xn = s.x  # fcn.vm.get_all_val()  # bd.get_y(s.x)
        fcn.vm.set_all(s.x)
        ndf = s.x.shape[0]
        min_nll = s.fun
        success = s.success
        hess_inv = s.hess_inv
        fcn.vm.remove_bound()
        xn = fcn.vm.get_all_val()
    elif method in ["L-BFGS-B"]:
        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise Exception("x too large: {}".format(x_p))
            points.append([float(i) for i in x])
            nlls.append(float(fcn.cached_nll))

        s = minimize(fcn.nll_grad, np.array(x0), method=method, jac=True, bounds=bnds, callback=callback,
                     options={"disp": 1, "maxcor": 50, "ftol": 1e-15, "maxiter": maxiter})
        xn = s.x
        ndf = s.x.shape[0]
        min_nll = s.fun
        success = s.success
    elif method in ["iminuit"]:
        m = fit_minuit(fcn)
        return m
    else:
        raise Exception("unknown method")
    if check_grad:
        print("checking gradients ...")
        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        _, gs0 = f_g(xn)
        gs = []
        for i, name in enumerate(args_name):
            xn[i] += 1e-5
            nll0, _ = f_g(xn)
            xn[i] -= 2e-5
            nll1, _ = f_g(xn)
            xn[i] += 1e-5
            gs.append((nll0-nll1)/2e-5)
            print(args_name[i], gs[i], gs0[i])
    fcn.vm.set_all(xn)
    params = fcn.vm.get_all_dic()
    return FitResult(params, fcn, min_nll, ndf = ndf, success=success, hess_inv=hess_inv)


#import pymultinest
def fit_multinest(fcn):
    pass



class FitResult(object):
    def __init__(self, params, fcn, min_nll, ndf=0, success=True, hess_inv=None):
        self.params = params
        self.error = {}
        self.fcn = fcn
        self.min_nll = float(min_nll)
        self.ndf = int(ndf)
        self.success = success
        self.hess_inv = hess_inv

    def save_as(self, file_name):
        s = {"value": self.params, "error": self.error, "status": {"success":self.success,"NLL":self.min_nll,"Ndf":self.ndf}}
        with open(file_name, "w") as f:
            json.dump(s, f, indent=2)

    def set_error(self, error):
        self.error = error.copy()
