import contextlib
import copy
import functools
import itertools
import json
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import BFGS, basinhopping, minimize

from tf_pwa.amp import (
    AmplitudeModel,
    DecayChain,
    DecayGroup,
    get_decay,
    get_particle,
)
from tf_pwa.applications import (
    cal_hesse_error,
    corr_coef_matrix,
    fit,
    fit_fractions,
)
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.data import (
    data_index,
    data_shape,
    data_split,
    load_data,
    save_data,
)
from tf_pwa.fit import FitResult
from tf_pwa.fit_improve import minimize as my_minimize
from tf_pwa.model import FCN, CombineFCN, Model, Model_new
from tf_pwa.model.cfit import Model_cfit
from tf_pwa.particle import split_particle_type
from tf_pwa.root_io import has_uproot, save_dict_to_root
from tf_pwa.utils import time_print
from tf_pwa.variable import Variable, VarsManager

from .config_loader import ConfigLoader
from .decay_config import DecayConfig


class MultiConfig(object):
    def __init__(self, file_names, vm=None, total_same=False, share_dict={}):
        if vm is None:
            self.vm = VarsManager()
            # print(self.vm)
        else:
            self.vm = vm
        self.total_same = total_same
        self.configs = [
            ConfigLoader(i, vm=self.vm, share_dict=share_dict)
            for i in file_names
        ]
        self.bound_dic = {}
        self.gauss_constr_dic = {}
        self._neglect_when_set_params = []
        self.cached_fcn = {}
        self.inv_he = None

    def get_all_data(self):
        return [i.get_all_data() for i in self.configs]

    def get_amplitudes(self, vm=None):
        if not self.total_same:
            amps = [
                j.get_amplitude(name="s" + str(i), vm=vm)
                for i, j in enumerate(self.configs)
            ]
        else:
            amps = [j.get_amplitude(vm=vm) for j in self.configs]
        for i in self.configs:
            self.bound_dic.update(i.bound_dic)
            self.gauss_constr_dic.update(i.gauss_constr_dic)
            for j in i._neglect_when_set_params:
                if j not in self._neglect_when_set_params:
                    self._neglect_when_set_params.append(j)
        return amps

    """
    def _get_models(self, vm=None): # get_model is useless to users given get_fcn and get_amplitude
        if not self.total_same:
            models = [j._get_model(name="s"+str(i), vm=vm)
                      for i, j in enumerate(self.configs)]
        else:
            models = [j._get_model(vm=vm) for j in self.configs]
        return models
    """

    def get_fcns(self, datas=None, vm=None, batch=65000):
        if datas is not None:
            if not self.total_same:
                fcns = [
                    i[1].get_fcn(
                        name="s" + str(i[0]),
                        all_data=j,
                        vm=vm,
                        batch=batch,
                    )
                    for i, j in zip(enumerate(self.configs), datas)
                ]
            else:
                fcns = [
                    j.get_fcn(
                        all_data=data,
                        vm=vm,
                        batch=batch,
                    )
                    for data, j in zip(datas, self.configs)
                ]
        else:
            if not self.total_same:
                fcns = [
                    j.get_fcn(
                        name="s" + str(i),
                        vm=vm,
                        batch=batch,
                    )
                    for i, j in enumerate(self.configs)
                ]
            else:
                fcns = [j.get_fcn(vm=vm, batch=batch) for j in self.configs]
        return fcns

    def get_fcn(self, datas=None, vm=None, batch=65000):
        if datas is None:
            if vm in self.cached_fcn:
                return self.cached_fcn[vm]
        fcns = self.get_fcns(datas=datas, vm=vm, batch=batch)
        fcn = CombineFCN(fcns=fcns, gauss_constr=self.gauss_constr_dic)
        if datas is None:
            self.cached_fcn[vm] = fcn
        return fcn

    def get_args_value(self, bounds_dict):
        args = {}
        args_name = self.vm.trainable_vars
        x0 = []
        bnds = []

        for i in self.vm.trainable_variables:
            args[i.name] = i.numpy()
            x0.append(i.numpy())
            if i.name in bounds_dict:
                bnds.append(bounds_dict[i.name])
            else:
                bnds.append((None, None))
            args["error_" + i.name] = 0.1

        return args_name, x0, args, bnds

    @time_print
    def fit(
        self,
        datas=None,
        batch=65000,
        method="BFGS",
        maxiter=None,
        print_init_nll=False,
        callback=None,
    ):
        fcn = self.get_fcn(datas=datas)
        # fcn.gauss_constr.update({"Zc_Xm_width": (0.177, 0.03180001857)})
        print("\n########### initial parameters")
        print(json.dumps(fcn.get_params(), indent=2), flush=True)
        if print_init_nll:
            print("initial NLL: ", fcn({}))
        self.fit_params = fit(
            fcn=fcn,
            method=method,
            bounds_dict=self.bound_dic,
            maxiter=maxiter,
            callback=callback,
        )
        if self.fit_params.hess_inv is not None:
            self.inv_he = self.fit_params.hess_inv
        return self.fit_params

    def reinit_params(self):
        self.get_fcn().vm.refresh_vars(self.bound_dic)

    def get_params_error(
        self, params=None, datas=None, batch=10000, using_cached=False
    ):
        if params is None:
            params = {}
        if hasattr(params, "params"):
            params = getattr(params, "params")

        if using_cached and self.inv_he is not None:
            hesse_error = np.sqrt(np.fabs(self.inv_he.diagonal())).tolist()
        else:
            fcn = self.get_fcn(datas, batch=batch)
            hesse_error, self.inv_he = cal_hesse_error(
                fcn, params, check_posi_def=True, save_npy=True
            )
        # hesse_error, self.inv_he = cal_hesse_error(
        # fcn, params, check_posi_def=True, save_npy=True
        # )
        print("hesse_error:", hesse_error)
        err = dict(zip(self.vm.trainable_vars, hesse_error))
        if hasattr(self, "fit_params"):
            self.fit_params.set_error(err)
        return err

    def get_params(self, trainable_only=False):
        # _amps = self.get_fcn()
        return self.vm.get_all_dic(trainable_only)

    def set_params(self, params, neglect_params=None):
        _amps = self.get_amplitudes()
        if isinstance(params, str):
            if params == "":
                return False
            try:
                with open(params) as f:
                    params = yaml.safe_load(f)
            except Exception as e:
                print(e)
                return False
        if hasattr(params, "params"):
            params = params.params
        if isinstance(params, dict):
            if "value" in params:
                params = params["value"]
        ret = params.copy()
        if neglect_params is None:
            neglect_params = self._neglect_when_set_params
        if len(neglect_params) != 0:
            warnings.warn(
                "Neglect {} when setting params.".format(neglect_params)
            )
            for v in params:
                if v in self._neglect_when_set_params:
                    del ret[v]
        self.vm.set_all(ret)
        return True

    @contextlib.contextmanager
    def params_trans(self):
        with self.vm.error_trans(self.inv_he) as f:
            yield f

    def save_params(self, file_name):
        params = self.get_params()
        val = {k: float(v) for k, v in params.items()}
        with open(file_name, "w") as f:
            json.dump(val, f, indent=2)
