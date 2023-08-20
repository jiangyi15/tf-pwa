import json
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import BFGS, basinhopping, minimize

from .fit_improve import Cached_FG
from .fit_improve import minimize as my_minimize
from .utils import time_print


class LargeNumberError(ValueError):
    pass


def fit_minuit(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):
    try:
        import iminuit
    except ImportError:
        raise RuntimeError(
            "You haven't installed iminuit so you can't use Minuit to fit."
        )
    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    if int(iminuit.__version__[0]) < 2:
        return fit_minuit_v1(
            fcn, bounds_dict=bounds_dict, hesse=hesse, minos=minos, **kwargs
        )
    return fit_minuit_v2(
        fcn, bounds_dict=bounds_dict, hesse=hesse, minos=minos, **kwargs
    )


def fit_minuit_v1(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):

    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    from iminuit import Minuit

    var_args = {}
    var_names = fcn.vm.trainable_vars
    for i in var_names:
        var_args[i] = fcn.vm.get(i)
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        var_args["error_" + i] = 0.1

    f_g = Cached_FG(fcn.nll_grad)
    m = Minuit(
        f_g.fun,
        name=var_names,
        errordef=0.5,
        grad=f_g.grad,
        print_level=2,
        use_array_call=True,
        **var_args,
    )
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
    ret = FitResult(
        dict(m.values), fcn, m.fval, ndf=ndf, success=m.migrad_ok()
    )
    ret.set_error(dict(m.errors))
    return ret


def fit_minuit_v2(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):

    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    from iminuit import Minuit

    var_args = {}
    var_names = fcn.vm.trainable_vars
    x0 = []
    for i in var_names:
        x0.append(fcn.vm.get(i))
        var_args[i] = fcn.vm.get(i)
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        # var_args["error_" + i] = 0.1

    f_g = Cached_FG(fcn.nll_grad)
    m = Minuit(
        f_g.fun,
        np.array(x0),
        name=var_names,
        grad=f_g.grad,
    )
    m.strategy = 0
    for i in var_names:
        if i in bounds_dict:
            m.limits[i] = bounds_dict[i]
    m.errordef = 0.5
    m.print_level = 2
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
    ndf = len(var_names)
    ret = FitResult(
        dict(zip(var_names, m.values)), fcn, m.fval, ndf=ndf, success=m.valid
    )
    # print(m.errors)
    ret.set_error(dict(zip(var_names, m.errors)))
    return ret


def fit_root_fitter(fcn):
    from array import array

    import ROOT

    var_names = fcn.vm.trainable_vars
    x0 = []
    for i in var_names:
        x0.append(fcn.vm.get(i))
    f_g = Cached_FG(fcn.nll_grad)

    class MyMultiGenFCN(ROOT.Math.IMultiGenFunction):
        def NDim(self):
            return len(x0)

        def DoEval(self, x):
            return f_g.fun(x)

        def Clone(self):
            x = MyMultiGenFCN()
            ROOT.SetOwnership(x, False)
            return x

    fitter = ROOT.Fit.Fitter()
    myMultiGenFCN = MyMultiGenFCN()
    params = array("d", x0)
    fitter.FitFCN(myMultiGenFCN, params)
    fit_result = fitter.Result()
    x = dict(zip(var_names, [fit_result.Parameter(i) for i in range(len(x0))]))
    xerr = dict(zip(var_names, [fit_result.Error(i) for i in range(len(x0))]))
    ndf = len(var_names)
    ret = FitResult(
        x, fcn, fit_result.MinFcnValue(), ndf=ndf, success=fit_result.IsValid()
    )
    ret.set_error(xerr)
    return ret


def fit_scipy(
    fcn,
    method="BFGS",
    bounds_dict={},
    check_grad=False,
    improve=False,
    maxiter=None,
    jac=True,
    callback=None,
    standard_complex=True,
    grad_scale=1.0,
    gtol=1e-3,
):
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
    ndf = fcn.vm.get_all_val(True)
    # maxiter = 0
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
            gs.append((nll0 - nll1) / 2e-5)
        return nll, np.array(gs)

    if check_grad:
        print("checking gradients ...")
        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        nll, gs0 = f_g(x0)
        _, gs = v_g2(x0)
        for i, name in enumerate(args_name):
            print(args_name[i], gs[i], gs0[i])

    callback_inner = lambda x, y: None
    if callback is not None:
        callback_inner = callback

    if method in ["BFGS", "CG", "Nelder-Mead", "test"]:

        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise LargeNumberError("x too large: {}".format(x_p))
            points.append(fcn.vm.get_all_val())
            nlls.append(float(fcn.cached_nll))
            # if len(nlls) > maxiter:
            #    with open("fit_curve.json", "w") as f:
            #        json.dump({"points": points, "nlls": nlls}, f, indent=2)
            #    pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
            callback_inner(x, fcn)
            print(fcn.cached_nll)

        # bd = Bounds(bnds)
        fcn.vm.set_bound(bounds_dict)

        f_g = fcn.vm.trans_fcn_grad(fcn.nll_grad)
        f_g = Cached_FG(f_g, grad_scale=grad_scale)
        # print(f_g)
        x0 = np.array(fcn.vm.get_all_val(True))
        # print(x0, fcn.vm.get_all_dic())
        # s = minimize(f_g, x0, method='trust-constr', jac=True, hess=BFGS(), options={'gtol': 1e-4, 'disp': True})
        if method == "test":
            try:
                s = my_minimize(
                    f_g,
                    x0,
                    method=method,
                    jac=True,
                    callback=callback,
                    options={"disp": 1, "gtol": gtol, "maxiter": maxiter},
                )
            except LargeNumberError:
                return except_result(fcn, x0.shape[0])
        elif jac is not True:
            try:
                s = minimize(
                    lambda x: float(fcn(x)),
                    x0,
                    method=method,
                    jac=jac,
                    callback=callback,
                    options={"disp": 1, "gtol": gtol, "maxiter": maxiter},
                )
            except LargeNumberError:
                return except_result(fcn, x0.shape[0])
        else:
            try:
                s = minimize(
                    f_g,
                    x0,
                    method=method,
                    jac=True,
                    callback=callback,
                    options={"disp": 1, "gtol": gtol, "maxiter": maxiter},
                )
            except LargeNumberError:
                return except_result(fcn, x0.shape[0])

        while improve and not s.success:
            min_nll = s.fun
            maxiter -= s.nit
            s = minimize(
                f_g,
                s.x,
                method=method,
                jac=True,
                callback=callback,
                options={"disp": 1, "gtol": gtol * 10, "maxiter": maxiter},
            )
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
        fcn.vm.set_trans_var(s.x)  # make sure fit results same as variable
        print(s)
        # xn = s.x  # fcn.vm.get_all_val()  # bd.get_y(s.x)

        # fcn.vm.set_all(s.x, True)
        ndf = s.x.shape[0]
        min_nll = s.fun
        success = s.success
        hess_inv = fcn.vm.trans_error_matrix(s.hess_inv / grad_scale, s.x)
        fcn.vm.remove_bound()

        xn = fcn.vm.get_all_val()
    elif method in ["L-BFGS-B"]:

        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise LargeNumberError("x too large: {}".format(x_p))
            points.append([float(i) for i in x])
            nlls.append(float(fcn.cached_nll))

        try:

            s = minimize(
                fcn.nll_grad,
                np.array(x0),
                method=method,
                jac=True,
                bounds=bnds,
                callback=callback,
                options={
                    "disp": 1,
                    "maxcor": 50,
                    "ftol": 1e-15,
                    "maxiter": maxiter,
                },
            )
        except LargeNumberError:
            return except_result(fcn, len(x0))
        xn = s.x
        fcn.vm.set_var(xn)
        print(s)
        ndf = s.x.shape[0]
        min_nll = s.fun
        success = s.success
    elif method in ["Newton-CG", "trust-krylov", "trust-ncg", "trust-exact"]:
        fcn.vm.set_bound(bounds_dict)
        return fit_newton_cg(fcn, method, False)
    elif method in ["Newton-CG-p", "trust-krylov-p", "trust-ncg-p"]:
        fcn.vm.set_bound(bounds_dict)
        return fit_newton_cg(fcn, method[:-2], True)
    elif method in ["iminuit"]:
        m = fit_minuit(fcn)
        return m
    elif method in ["root"]:
        m = fit_root_fitter(fcn)
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
            gs.append((nll0 - nll1) / 2e-5)
            print(args_name[i], gs[i], gs0[i])
    if standard_complex:
        fcn.vm.standard_complex()
    params = fcn.get_params()  # vm.get_all_dic()
    return FitResult(
        params, fcn, min_nll, ndf=ndf, success=success, hess_inv=hess_inv
    )


def fit_newton_cg(
    fcn, method="Newton-CG", use_hessp=False, check_hess=False, gtol=1e-4
):
    vm = fcn.vm

    points = []

    def callback(x):
        points.append(fcn.vm.get_all_val())
        # if len(nlls) > maxiter:
        #    with open("fit_curve.json", "w") as f:
        #        json.dump({"points": points, "nlls": nlls}, f, indent=2)
        #    pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
        print(fcn.cached_nll)

    # bd = Bounds(bnds)

    f_g = vm.trans_fcn_grad(fcn.nll_grad)
    if use_hessp:
        hessp = vm.trans_grad_hessp(fcn.grad_hessp)
    else:
        hess = vm.trans_f_grad_hess(fcn.nll_grad_hessian)
        hess = time_print(hess)
    f_g = Cached_FG(f_g)

    x0 = np.array(vm.get_all_val(True))

    if check_hess:
        # check if hessp works well
        # hess(x0, x0)
        gs = []
        for i, _ in enumerate(x0):
            x0[i] += 1e-3
            _, g1 = f_g(x0)
            x0[i] -= 2e-3
            _, g2 = f_g(x0)
            x0[i] += 1e-3
            gs.append((g1 - g2) / 2e-3)
        gs = np.array(gs)

        # print(gs)
        if use_hessp:
            p = np.random.random(x0.shape)
            print("check", hessp(x0, p)[1], "==", np.dot(gs, p))
            p = np.random.random(x0.shape)
            print("check", hessp(x0, p)[1], "==", np.dot(gs, p))
        else:
            print(hess(x0)[2] - gs)

    if use_hessp:
        s = minimize(
            f_g, x0, jac=True, hessp=lambda x, p: hessp(x, p)[1], method=method
        )
    else:
        s = minimize(
            f_g, x0, jac=True, hess=lambda x: hess(x)[2], method=method
        )
    fcn.vm.set_trans_var(s.x)
    xn = s.x
    ndf = s.x.shape[0]
    min_nll = s.fun
    if not s.success:
        if np.min(np.abs(s.jac)) < gtol:
            s.success = True
            s.message = s.message + "\n But gradients allow"
    success = s.success
    print(s)
    # fcn.vm.set_all(xn)
    params = fcn.get_params()
    return FitResult(
        params, fcn, min_nll, ndf=ndf, success=success, hess_inv=None
    )


def except_result(fcn, ndf):
    params = fcn.vm.get_all_dic()
    return FitResult(
        params, fcn, float(fcn.cached_nll), ndf=ndf, success=False
    )


# import pymultinest
def fit_multinest(fcn):
    pass


class FitResult(object):
    def __init__(
        self, params, fcn, min_nll, ndf=0, success=True, hess_inv=None
    ):
        self.params = params
        self.error = {}
        self.fcn = fcn
        self.min_nll = float(min_nll)
        self.ndf = int(ndf)
        self.success = success
        self.hess_inv = hess_inv

    def save_as(self, file_name, save_hess=False):
        s = {
            "value": self.params,
            "error": self.error,
            "status": {
                "success": self.success,
                "NLL": self.min_nll,
                "Ndf": self.ndf,
            },
        }
        if save_hess and self.hess_inv is not None:
            s["free_params"] = [str(i) for i in self.error]
            s["hess_inv"] = [[float(j) for j in i] for i in self.hess_inv]
        with open(file_name, "w") as f:
            json.dump(s, f, indent=2)

    def set_error(self, error):
        self.error = error.copy()
