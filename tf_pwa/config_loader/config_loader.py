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
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import BFGS, basinhopping, minimize

from tf_pwa.adaptive_bins import AdaptiveBound, cal_chi2
from tf_pwa.amp import (
    AmplitudeModel,
    DecayChain,
    DecayGroup,
    HelicityDecay,
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
    data_merge,
    data_shape,
    data_split,
    data_to_numpy,
    load_data,
    save_data,
)
from tf_pwa.fit import FitResult
from tf_pwa.fit_improve import minimize as my_minimize
from tf_pwa.model import FCN, CombineFCN, MixLogLikehoodFCN, Model, Model_new
from tf_pwa.model.cfit import Model_cfit, Model_cfit_cached
from tf_pwa.model.opt_int import ModelCachedAmp, ModelCachedInt
from tf_pwa.particle import split_particle_type
from tf_pwa.root_io import has_uproot, save_dict_to_root
from tf_pwa.utils import time_print
from tf_pwa.variable import Variable, VarsManager

from .base_config import BaseConfig
from .data import load_data_mode
from .decay_config import DecayConfig


class ConfigLoader(BaseConfig):
    """class for loading config.yml"""

    def __init__(self, file_name, vm=None, share_dict=None):
        if share_dict is None:
            share_dict = {}
        super().__init__(file_name, share_dict)
        self.config["data"] = self.config.get("data", {})
        self.share_dict = share_dict
        self.decay_config = DecayConfig(self.config, share_dict)
        self.dec = self.decay_config.dec
        self.particle_map, self.particle_property = (
            self.decay_config.particle_map,
            self.decay_config.particle_property,
        )
        self.top, self.finals = self.decay_config.top, self.decay_config.finals
        self.full_decay = self.decay_config.full_decay
        self.decay_struct = self.decay_config.decay_struct
        if vm is None:
            vm = VarsManager()
        self.vm = vm
        self.amps = {}
        self.cached_data = None
        self.bound_dic = {}
        self.gauss_constr_dic = {}
        self.init_value = {}
        self.plot_params = PlotParams(
            self.config.get("plot", {}), self.decay_struct
        )
        self._neglect_when_set_params = []
        self.data = load_data_mode(self["data"], self.decay_struct)
        self.inv_he = None
        self._Ngroup = 1
        self.cached_fcn = {}
        self.extra_constrains = {}
        self.resolution_size = self.config.get("data", {}).get(
            "resolution_size", 1
        )

    @staticmethod
    def load_config(file_name, share_dict={}):
        if isinstance(file_name, dict):
            return copy.deepcopy(file_name)
        if isinstance(file_name, str):
            if file_name in share_dict:
                return ConfigLoader.load_config(share_dict[file_name])
            with open(file_name) as f:
                ret = yaml.load(f, yaml.FullLoader)
            return ret
        raise TypeError("not support config {}".format(type(file_name)))

    def get_data_file(self, idx):
        if idx in self.config["data"]:
            ret = self.config["data"][idx]
        else:
            ret = None
        return ret

    def get_dat_order(self, standard=False):
        order = self.config["data"].get("dat_order", None)
        if order is None:
            order = list(self.decay_struct.outs)
        else:
            order = [get_particle(str(i)) for i in order]
        if not standard:
            return order

        re_map = self.decay_struct.get_chains_map()

        def particle_item():
            for j in re_map:
                for k, v in j.items():
                    for s, l in v.items():
                        yield s, l

        new_order = []
        for i in order:
            for s, l in particle_item():
                if str(l) == str(i):
                    new_order.append(s)
                    break
            else:
                new_order.append(i)
        return new_order

    def get_data_mode(self):
        data_config = self.config.get("data", {})
        data = data_config.get("data", "")
        if isinstance(data, str):
            mode = "single"
        elif isinstance(data, list):
            data_i = data[0]
            if isinstance(data_i, str):
                return "single"
            elif isinstance(data_i, list):
                return "multi"

    @functools.lru_cache()
    def get_data(self, idx):
        return self.data.get_data(idx)

    def load_cached_data(self, file_name=None):
        return self.data.load_cached_data(file_name)

    def save_cached_data(self, data, file_name=None):
        self.data.save_cached_data(data, file_name=file_name)

    def get_all_data(self):
        datafile = ["data", "phsp", "bg", "inmc"]
        self.load_cached_data()
        data, phsp, bg, inmc = [self.get_data(i) for i in datafile]
        self._Ngroup = len(data)
        assert len(phsp) == self._Ngroup
        if bg is None:
            bg = [None] * self._Ngroup
        if inmc is None:
            inmc = [None] * self._Ngroup
        assert len(bg) == self._Ngroup
        assert len(inmc) == self._Ngroup
        self.save_cached_data(dict(zip(datafile, [data, phsp, bg, inmc])))
        return data, phsp, bg, inmc

    def get_data_index(self, sub, name):
        return self.plot_params.get_data_index(sub, name)

    def get_phsp_noeff(self):
        if "phsp_noeff" in self.config["data"]:
            phsp_noeff = self.get_data("phsp_noeff")
            assert len(phsp_noeff) == 1
            return phsp_noeff[0]
        warnings.warn(
            "No data file as 'phsp_noeff', using the first 'phsp' file instead."
        )
        return self.get_data("phsp")[0]

    def get_phsp_plot(self):
        if "phsp_plot" in self.config["data"]:
            assert len(self.config["data"]["phsp_plot"]) == len(
                self.config["data"]["phsp"]
            )
            return self.get_data("phsp_plot")
        return self.get_data("phsp")

    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @functools.lru_cache()
    def get_amplitude(self, vm=None, name=""):
        use_tf_function = self.config.get("data", {}).get(
            "use_tf_function", False
        )
        decay_group = self.full_decay
        self.check_valid_jp(decay_group)
        if vm is None:
            vm = self.vm
        if vm in self.amps:
            return self.amps[vm]
        amp = AmplitudeModel(
            decay_group, vm=vm, name=name, use_tf_function=use_tf_function
        )
        self.add_constraints(amp)
        self.amps[vm] = amp
        return amp

    def check_valid_jp(self, decay_group):
        for decay_chain in decay_group:
            for dec in decay_chain:
                if isinstance(dec, HelicityDecay):
                    dec.check_valid_jp()

    def add_constraints(self, amp):
        constrains = self.config.get("constrains", {})
        if constrains is None:
            constrains = {}
        self.add_decay_constraints(amp, constrains.get("decay", {}))
        self.add_particle_constraints(amp, constrains.get("particle", {}))
        self.add_fix_var_constraints(amp, constrains.get("fix_var", {}))
        self.add_free_var_constraints(amp, constrains.get("free_var", []))
        self.add_var_range_constraints(amp, constrains.get("var_range", {}))
        self.add_var_equal_constraints(amp, constrains.get("var_equal", []))
        for k, v in self.extra_constrains.items():
            v(amp, constrains.get(k, {}))

    def register_extra_constrains(self, name, f=None):
        """
        add extra_constrains
        """

        def _reg(g):
            self.extra_constrains[name] = g
            return g

        if f is None:
            return _reg
        else:
            return _reg(f)

    def add_fix_var_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        for k, v in dic.items():
            print("fix var: ", k, "=", v)
            amp.vm.set_fix(k, v)

    def add_free_var_constraints(self, amp, dic=None):
        if dic is None:
            dic = []
        for k in dic:
            print("free var: ", k)
            amp.vm.set_fix(k, unfix=True)

    def add_var_range_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        for k, v in dic.items():
            print("variable range: ", k, " in ", v)
            self.bound_dic[k] = v

    def add_var_equal_constraints(self, amp, dic=None):
        if dic is None:
            dic = []
        for k in dic:
            print("same value:", k)
            amp.vm.set_same(k)

    def add_decay_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        fix_total_idx = dic.get("fix_chain_idx", 0)
        fix_total_val = dic.get("fix_chain_val", np.random.uniform(0, 2))

        fix_decay = amp.decay_group.get_decay_chain(fix_total_idx)
        # fix which total factor
        fix_decay.total.set_fix_idx(fix_idx=0, fix_vals=(fix_total_val, 0.0))

    def add_particle_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}

        res_dec = {}
        for d in amp.decay_group:
            for p_i in d.inner:
                i = str(p_i)
                res_dec[i] = d
                prefix_map = {
                    "m0": "mass",
                    "g0": "width",
                    "m_": "mass_",
                    "g_": "width_",
                }
                particle_config = self.config["particle"][i].copy()

                params_dic = self.config["particle"][i].get("params", None)
                if params_dic is None:
                    params_dic = {}
                for name in list(particle_config):
                    for prefix_i in prefix_map.keys():
                        if name.startswith(prefix_i):
                            name2 = (
                                prefix_map[prefix_i] + name[len(prefix_i) :]
                            )
                            params_dic[name2] = particle_config[name]
                    for prefix_i in prefix_map.values():
                        if name.startswith(prefix_i):
                            params_dic[name] = particle_config[name]

                variable_prefix = p_i.get_variable_name()

                set_prefix_constrains(self.vm, p_i, params_dic, self)

                simple_map = {"m": "mass", "g": "width"}

                gauss_constr = particle_config.get("gauss_constr", None)
                if gauss_constr is not None:
                    assert isinstance(gauss_constr, dict)
                    for k, v in gauss_constr.items():
                        if v:
                            name = simple_map.get(k, k)
                            full_name = variable_prefix + name
                            var0 = self.vm.get(full_name)
                            self.gauss_constr_dic[full_name] = (
                                var0.value,
                                v,
                            )
                        else:
                            raise Exception(
                                f"Need sigma of {k} of {p_i} when adding gaussian constraint"
                            )

                if isinstance(p_i.mass, Variable) or isinstance(
                    p_i.width, Variable
                ):
                    if (
                        "float" in self.config["particle"][i]
                        and self.config["particle"][i]["float"]
                    ):
                        if "m" in self.config["particle"][i]["float"]:
                            p_i.mass.freed()  # set_fix(i+'_mass',unfix=True)
                            if "m_max" in self.config["particle"][i]:
                                upper = self.config["particle"][i]["m_max"]
                            # elif m_sigma is not None:
                            #    upper = self.config["particle"][i]["m0"] + 10 * m_sigma
                            else:
                                upper = None
                            if "m_min" in self.config["particle"][i]:
                                lower = self.config["particle"][i]["m_min"]
                            # elif m_sigma is not None:
                            #    lower = self.config["particle"][i]["m0"] - 10 * m_sigma
                            else:
                                lower = None
                            self.bound_dic[str(p_i.mass)] = (lower, upper)
                        else:
                            self._neglect_when_set_params.append(str(p_i.mass))
                        if "g" in self.config["particle"][i]["float"]:
                            p_i.width.freed()  # amp.vm.set_fix(i+'_width',unfix=True)
                            if "g_max" in self.config["particle"][i]:
                                upper = self.config["particle"][i]["g_max"]
                            # elif g_sigma is not None:
                            #    upper = self.config["particle"][i]["g0"] + 10 * g_sigma
                            else:
                                upper = None
                            if "g_min" in self.config["particle"][i]:
                                lower = self.config["particle"][i]["g_min"]
                            # elif g_sigma is not None:
                            #    lower = self.config["particle"][i]["g0"] - 10 * g_sigma
                            else:
                                lower = None
                            self.bound_dic[str(p_i.width)] = (lower, upper)
                        else:
                            self._neglect_when_set_params.append(
                                str(p_i.width)
                            )
                    else:
                        self._neglect_when_set_params.append(
                            i + "_mass"
                        )  # p_i.mass.name
                        self._neglect_when_set_params.append(
                            i + "_width"
                        )  # p_i.width.name

                # share helicity variables
                if "coef_head" in self.config["particle"][i]:
                    coef_head = self.config["particle"][i]["coef_head"]
                    if coef_head in res_dec:
                        d_coef_head = res_dec[coef_head]
                        for j, h in zip(d, d_coef_head):
                            if i in [str(jj) for jj in j.outs] or i is str(
                                j.core
                            ):
                                h.g_ls.sameas(j.g_ls)
                        # share total radium
                        d_coef_head.total.r_shareto(d.total)
                    else:
                        self.config["particle"][coef_head]["coef_head"] = i

        equal_params = dic.get("equal", {})
        for k, v in equal_params.items():
            for vi in v:
                a = []
                for i in amp.decay_group.resonances:
                    if str(i) in vi:
                        a.append(i)
                a0 = a.pop(0)
                arg = getattr(a0, k)
                for i in a:
                    arg_i = getattr(i, k)
                    if isinstance(arg_i, Variable):
                        arg_i.sameas(arg)

    @functools.lru_cache()
    def _get_model(self, vm=None, name=""):
        amp = self.get_amplitude(vm=vm, name=name)
        model_name = self.config["data"].get("model", "auto")
        w_bkg, w_inmc = self._get_bg_weight()
        model = []
        if model_name == "cfit":
            bg_function = self.config["data"].get("bg_function", None)
            eff_function = self.config["data"].get("eff_function", None)
            w_bkg = self.config["data"]["bg_frac"]
            if not isinstance(w_bkg, list):
                w_bkg = [w_bkg]
            for wb in w_bkg:
                if self.config["data"].get("cached_amp", False):
                    model.append(
                        Model_cfit_cached(amp, wb, bg_function, eff_function)
                    )
                else:
                    model.append(
                        Model_cfit(amp, wb, bg_function, eff_function)
                    )
        elif "inmc" in self.config["data"]:
            float_wmc = self.config["data"].get(
                "float_inmc_ratio_in_pdf", False
            )
            if not isinstance(float_wmc, list):
                float_wmc = [float_wmc] * self._Ngroup
            assert len(float_wmc) == self._Ngroup
            for wb, wi, fw in zip(w_bkg, w_inmc, float_wmc):
                model.append(Model_new(amp, wb, wi, fw))
        elif self.config["data"].get("cached_int", False):
            for wb in w_bkg:
                model.append(ModelCachedInt(amp, wb))
        elif self.config["data"].get("cached_amp", False):
            for wb in w_bkg:
                model.append(ModelCachedAmp(amp, wb))
        else:
            for wb in w_bkg:
                model.append(
                    Model(amp, wb, resolution_size=self.resolution_size)
                )
        return model

    def _get_bg_weight(self, data=None, bg=None, display=True):
        w_bkg = self.config["data"].get("bg_weight", 0.0)
        if not isinstance(w_bkg, list):
            w_bkg = [w_bkg] * self._Ngroup
        assert len(w_bkg) == self._Ngroup
        w_inmc = self.config["data"].get("inject_ratio", 0.0)
        if not isinstance(w_inmc, list):
            w_inmc = [w_inmc] * self._Ngroup
        assert len(w_inmc) == self._Ngroup
        weight_scale = self.config["data"].get("weight_scale", False)  # ???
        if weight_scale:
            data = data if data is not None else self.get_data("data")
            bg = bg if bg is not None else self.get_data("bg")
            tmp = []
            for wb, dt, sb in zip(w_bkg, data, bg):
                if isinstance(wb, str):
                    wb = self.data.load_weight_file(wb)
                tmp.append(wb * data_shape(dt) / data_shape(sb))
            w_bkg = tmp
            if display:
                print("background weight:", w_bkg)
        else:
            tmp = []
            for wb in w_bkg:
                if isinstance(wb, str):
                    wb = self.data.load_weight_file(wb)
                tmp.append(wb)
            w_bkg = tmp
        return w_bkg, w_inmc

    def get_fcn(self, all_data=None, batch=65000, vm=None, name=""):
        if all_data is None:
            if vm in self.cached_fcn:
                return self.cached_fcn[vm]
            data, phsp, bg, inmc = self.get_all_data()
        else:
            data, phsp, bg, inmc = all_data
        self._Ngroup = len(data)
        if inmc is None:
            inmc = [None] * self._Ngroup
        if bg is None:
            bg = [None] * self._Ngroup
        model = self._get_model(vm=vm, name=name)
        fcns = []
        # print(self.config["data"].get("using_mix_likelihood", False))
        if self.config["data"].get("using_mix_likelihood", False):
            print("  Using Mix Likelihood")
            fcn = MixLogLikehoodFCN(
                model,
                data,
                phsp,
                bg=bg,
                batch=batch,
                gauss_constr=self.gauss_constr_dic,
            )
            if all_data is None:
                self.cached_fcn[vm] = fcn
            return fcn
        for md, dt, mc, sb, ij in zip(model, data, phsp, bg, inmc):
            if self.config["data"].get("model", "auto") == "cfit":
                fcns.append(
                    FCN(
                        md,
                        dt,
                        mc,
                        batch=batch,
                        inmc=ij,
                        gauss_constr=self.gauss_constr_dic,
                    )
                )
            else:
                fcns.append(
                    FCN(
                        md,
                        dt,
                        mc,
                        bg=sb,
                        batch=batch,
                        inmc=ij,
                        gauss_constr=self.gauss_constr_dic,
                    )
                )
        if len(fcns) == 1:
            fcn = fcns[0]
        else:
            fcn = CombineFCN(fcns=fcns, gauss_constr=self.gauss_constr_dic)
        if all_data is None:
            self.cached_fcn[vm] = fcn
        return fcn

    def get_ndf(self):
        amp = self.get_amplitude()
        args_name = amp.vm.trainable_vars
        return len(args_name)

    @staticmethod
    def reweight_init_value(amp, phsp, ns=None):
        """reset decay chain total and make the integration to be ns"""
        total = [i.total for i in amp.decay_group]
        n_phsp = data_shape(phsp)
        weight = np.array(phsp.get("weight", [1] * n_phsp))
        sw = np.sum(weight)
        if ns is None:
            ns = [1] * len(total)
        elif isinstance(ns, (int, float)):
            ns = [ns / len(total)] * len(total)
        for i in total:
            i.set_rho(1.0)
        pw = amp.partial_weight(phsp)
        for i, w, ni in zip(total, pw, ns):
            i.set_rho(np.sqrt(ni / np.sum(weight * w) * sw))

    @time_print
    def fit(
        self,
        data=None,
        phsp=None,
        bg=None,
        inmc=None,
        batch=65000,
        method="BFGS",
        check_grad=False,
        improve=False,
        reweight=False,
        maxiter=None,
    ):
        if data is None and phsp is None:
            data, phsp, bg, inmc = self.get_all_data()
            fcn = self.get_fcn(batch=batch)
        else:
            fcn = self.get_fcn([data, phsp, bg, inmc], batch=batch)
        # print("sss")
        amp = self.get_amplitude()
        print("decay chains included: ")
        for i in self.full_decay:
            ls_list = [getattr(j, "get_ls_list", lambda x: None)() for j in i]
            print("  ", i, " ls: ", *ls_list)
        if reweight:
            ConfigLoader.reweight_init_value(
                amp, phsp[0], ns=data_shape(data[0])
            )

        print("\n########### initial parameters")
        print(json.dumps(amp.get_params(), indent=2), flush=True)
        print("initial NLL: ", fcn({}))  # amp.get_params()))
        # fit configure
        # self.bound_dic[""] = (,)
        self.fit_params = fit(
            fcn=fcn,
            method=method,
            bounds_dict=self.bound_dic,
            check_grad=check_grad,
            improve=False,
            maxiter=maxiter,
        )
        if self.fit_params.hess_inv is not None:
            self.inv_he = self.fit_params.hess_inv
        return self.fit_params

    def reinit_params(self):
        vm = self.get_amplitude().vm
        vm.refresh_vars(init_val=self.init_value, bound_dic=self.bound_dic)

    def fitNtimes(self, N, *args, **kwargs):
        for i in range(N):
            self.reinit_params()
            fit_result = self.fit(*args, **kwargs)
            fit_pars = json.dumps(fit_result.params, indent=2)
            print(fit_pars, flush=True)

    def get_params_error(
        self,
        params=None,
        data=None,
        phsp=None,
        bg=None,
        inmc=None,
        batch=10000,
        using_cached=False,
    ):
        if params is None:
            params = {}
        if data is None:
            data, phsp, bg, inmc = self.get_all_data()
        if hasattr(params, "params"):
            params = getattr(params, "params")
        fcn = self.get_fcn([data, phsp, bg, inmc], batch=batch)
        if using_cached and self.inv_he is not None:
            hesse_error = np.sqrt(np.fabs(self.inv_he.diagonal())).tolist()
        else:
            hesse_error, self.inv_he = cal_hesse_error(
                fcn, params, check_posi_def=True, save_npy=True
            )
        # print("parameters order")
        # print(fcn.model.Amp.vm.trainable_vars)
        # print("error matrix:")
        # print(self.inv_he)
        # print("correlation matrix:")
        # print(corr_coef_matrix(self.inv_he))
        print("hesse_error:", hesse_error)
        err = dict(zip(fcn.vm.trainable_vars, hesse_error))
        if hasattr(self, "fit_params"):
            self.fit_params.set_error(err)
        return err

    @classmethod
    def register_function(cls, name=None):
        def _f(f):
            my_name = name
            if my_name is None:
                my_name = f.__name__
            if hasattr(cls, my_name):
                warnings.warn("override function {}".format(name))
            setattr(cls, my_name, f)
            return f

        return _f

    def get_chain(self, idx):
        decay_group = self.full_decay
        return decay_group.get_decay_chain(idx)

    def get_chain_property(self, idx, display=True):
        """Get chain name and curve style in plot"""
        chain = self.get_chain(idx)
        for i in chain:
            curve_style = i.curve_style
            break
        combine = []
        for i in chain:
            if i.core == chain.top:
                combine = list(i.outs)
        names = []
        displays = []
        for i in combine:
            pro = self.particle_property[str(i)]
            names.append(str(i))
            displays.append(pro.get("display", str(i)))
        if display:
            return " ".join(displays), curve_style
        return "_".join(names), curve_style

    def cal_fitfractions(
        self, params={}, mcdata=None, res=None, exclude_res=[], batch=25000
    ):
        if hasattr(params, "params"):
            params = getattr(params, "params")
        if mcdata is None:
            mcdata = self.get_phsp_noeff()
        amp = self.get_amplitude()
        if res is None:
            res = sorted(
                list(set([str(i) for i in amp.res]) - set(exclude_res))
            )
        frac, err_frac = fit_fractions(
            amp, mcdata, self.inv_he, params, batch, res
        )
        return frac, err_frac

    def cal_signal_yields(self, params={}, mcdata=None, batch=25000):
        if hasattr(params, "params"):
            params = getattr(params, "params")
        if mcdata is None:
            mcdata = self.get_data("phsp")
        amp = self.get_amplitude()
        fracs = [
            fit_fractions(amp, i, self.inv_he, params, batch) for i in mcdata
        ]
        data = self.get_data("data")
        bg = self.get_data("bg")
        if bg is None:
            N_total = [data_shape(i) for i in data]
            for i in data:
                N_data = data_shape(i)
                N_total.append((N_data, np.sqrt(N_data)))
        else:
            bg_weight, _ = self._get_bg_weight(data, bg)
            N_total = []
            for i, j, w in zip(data, bg, bg_weight):
                N_data = data_shape(i)
                N_bg = data_shape(j)
                N_total.append(
                    (N_data - w * N_bg, np.sqrt(N_data + w * w * N_bg))
                )

        N_sig_s = []
        for frac_e, N_e in zip(fracs, N_total):
            frac, frac_err = frac_e
            N, N_err = N_e
            N_sig = {}
            for i in frac:
                N_sig[i] = (
                    frac[i] * N,
                    np.sqrt(
                        (N * frac_err.get(i, 0.0)) ** 2
                        + (N_err * frac[i]) ** 2
                    ),
                )
            N_sig_s.append(N_sig)
        return N_sig_s

    def likelihood_profile(self, var, var_min, var_max, N=100):
        params = self.get_params()
        var0 = params[var]
        delta_var = (var_max - var_min) / N
        vm = self.get_amplitude().vm
        unfix = var in vm.get_all_dic(True)
        nlls_up = []
        vars_up = []
        while var0 <= var_max:
            vm.set_fix(var, var0)
            fit_result = self.fit()
            vars_up.append(var0)
            nlls_up.append(fit_result.min_nll)
            var0 += delta_var
        self.set_params(params)
        var0 = params[var] - delta_var
        vars_down = []
        nlls_down = []
        while var0 >= var_min:
            vm.set_fix(var, var0)
            fit_result = self.fit()
            vars_down.append(var0)
            nlls_down.append(fit_result.min_nll)
            var0 -= delta_var
        self.set_params(params)
        vm.set_fix(var, params[var], unfix=unfix)
        return vars_down[::-1] + vars_up, nlls_down[::-1] + nlls_up

    def get_params(self, trainable_only=False):
        return self.get_amplitude().get_params(trainable_only)

    def set_params(self, params, neglect_params=None):
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
        amplitude = self.get_amplitude()
        ret = params.copy()
        if neglect_params is None:
            neglect_params = self._neglect_when_set_params
        if len(neglect_params) != 0:
            # warnings.warn("Neglect {} when setting params.".format(neglect_params))
            for v in params:
                if v in self._neglect_when_set_params:
                    del ret[v]
        amplitude.set_params(ret)
        return True

    def save_params(self, file_name):
        params = self.get_params()
        val = {k: float(v) for k, v in params.items()}
        with open(file_name, "w") as f:
            json.dump(val, f, indent=2)


def set_prefix_constrains(vm, base, params_dic, self):
    prefix = base.get_variable_name()
    p_list = []
    for v in params_dic:
        vname = v
        for tail in ["_range", "_sigma", "_free", "_constr", "_min", "_max"]:
            if v.endswith(tail):
                vname = v[: -len(tail)]
                break

        if vname not in p_list:
            # print(vname, v)
            p_list.append(vname)
            vv = base.get_var(vname)
            # print(vv, prefix + vname)
            # if isinstance(vv, Variable):# getattr(p_i, vname)
            if vv is None:
                continue
            p_sigma = params_dic.get(vname + "_sigma", None)
            if vname in params_dic and params_dic[vname] is not None:
                p_value = params_dic[vname]
                vv.set_value(p_value)
                if p_sigma is None:
                    self.init_value[vname] = p_value
                else:
                    self.init_value[vname] = [p_value, p_sigma]
            else:
                p_value = None
            p_free = params_dic.get(vname + "_free", None)
            if p_free:
                vv.freed()
            elif p_free is False:
                vv.fixed()
            p_range = vname + "_range"
            if p_range in params_dic and params_dic[p_range] is not None:
                lower, upper = params_dic[p_range]
                self.bound_dic[vv.name] = (lower, upper)
                # vm.set_bound({vv.name: (lower, upper)})
            else:
                lower = params_dic.get(vname + "_min")
                upper = params_dic.get(vname + "_max")
                # print(lower, upper)
                if lower is not None or upper is not None:
                    self.bound_dic[vv.name] = (lower, upper)
                    # vm.set_bound({vv.name: (lower, upper)})

                # self.bound_dic[vv.name] = (lower, upper)
            # elif p_sigma is not None and p_value is not None:
            #    p_10sigma = 10 * p_sigma
            #    self.bound_dic[vv.name] = (
            #        p_value - p_10sigma,
            #        p_value + p_10sigma,
            #    )
            p_constr = vname + "_constr"
            if p_constr in params_dic and params_dic[p_constr] is not None:
                if params_dic[p_constr]:
                    if p_value is None:
                        raise Exception(
                            "Need central value of {0} of {1} when adding gaussian constraint".format(
                                vname, prefix
                            )
                        )
                    if p_sigma is None:
                        raise Exception(
                            "Need sigma of {0} of {1} when adding gaussian constraint".format(
                                vname, prefix
                            )
                        )
                    self.gauss_constr_dic[vv.name] = (
                        params_dic[vname],
                        p_sigma,
                    )


def validate_file_name(s):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    name = re.sub(rstr, "_", s)
    return name


class PlotParams(dict):
    def __init__(self, plot_config, decay_struct):
        self.config = plot_config
        self.defaults_config = {}
        self.defaults_config.update(self.config.get("config", {}))
        self.decay_struct = decay_struct
        chain_map = self.decay_struct.get_chains_map()
        self.re_map = {}
        for i in chain_map:
            for _, j in i.items():
                for k, v in j.items():
                    self.re_map[v] = k
        self.params = []
        for i in self.get_mass_vars():
            self.params.append(i)
        for i in self.get_angle_vars():
            self.params.append(i)

    def get_data_index(self, sub, name):
        dec = self.decay_struct.topology_structure()
        if sub == "mass":
            p = get_particle(name)
            return "particle", self.re_map.get(p, p), "m"
        if sub == "p":
            p = get_particle(name)
            return "particle", self.re_map.get(p, p), "p"
        if sub == "angle":
            name_i = name.split("/")
            de_i = self.decay_struct.get_decay_chain(name_i)
            p = get_particle(name_i[-1])
            for i in de_i:
                if p in i.outs:
                    de = i
                    break
            else:
                raise IndexError("not found such decay {}".format(name))
            return (
                "decay",
                de_i.standard_topology(),
                self.re_map.get(de, de),
                self.re_map.get(p, p),
                "ang",
            )
        if sub == "aligned_angle":
            name_i = name.split("/")
            de_i = self.decay_struct.get_decay_chain(name_i)
            p = get_particle(name_i[-1])
            for i in de_i:
                if p in i.outs:
                    de = i
                    break
            else:
                raise IndexError("not found such decay {}".format(name))
            return (
                "decay",
                de_i.standard_topology(),
                self.re_map.get(de, de),
                self.re_map.get(p, p),
                "aligned_angle",
            )
        raise ValueError("unknown sub {}".format(sub))

    def get_mass_vars(self):
        mass = self.config.get("mass", {})
        x = sy.symbols("x")
        for k, v in mass.items():
            display = v.get("display", "M({})".format(k))
            upper_ylim = v.get("upper_ylim", None)
            xrange = v.get("range", None)
            trans = v.get("trans", None)
            if trans is None:
                trans = lambda x: x
            else:
                trans = sy.sympify(trans)
                trans = sy.lambdify(x, trans, modules="numpy")
            units = v.get("units", "GeV")
            bins = v.get("bins", self.defaults_config.get("bins", 50))
            legend = v.get("legend", self.defaults_config.get("legend", True))
            yscale = v.get(
                "yscale", self.defaults_config.get("yscale", "linear")
            )
            yield {
                "name": "m_" + k,
                "display": display,
                "upper_ylim": upper_ylim,
                "idx": (
                    "particle",
                    self.re_map.get(get_particle(k), get_particle(k)),
                    "m",
                ),
                "legend": legend,
                "range": xrange,
                "bins": bins,
                "trans": trans,
                "units": units,
                "yscale": yscale,
            }

    def get_angle_vars(self):
        ang = self.config.get("angle", {})
        for k, i in ang.items():
            names = k.split("/")
            name = names[0]
            number_decay = True
            if len(names) > 1:
                try:
                    count = int(names[-1])
                except ValueError:
                    number_decay = False
            else:
                count = 0
            if number_decay:
                decay_chain, decay = None, None
                part = self.re_map.get(get_particle(name), get_particle(name))
                for decs in self.decay_struct:
                    for dec in decs:
                        if dec.core == get_particle(name):
                            decay = dec.core.decay[count]
                            for j in self.decay_struct:
                                if decay in j:
                                    decay_chain = j.standard_topology()
                            decay = self.re_map.get(decay, decay)
                part = decay.outs[0]
            else:
                _, decay_chain, decay, part, _ = self.get_data_index(
                    "angle", k
                )
            for j, v in i.items():
                display = v.get("display", j)
                upper_ylim = v.get("upper_ylim", None)
                theta = j
                trans = lambda x: x
                if "cos" in j:
                    theta = j[4:-1]
                    trans = np.cos
                bins = v.get("bins", self.defaults_config.get("bins", 50))
                xrange = v.get("range", None)
                legend = v.get(
                    "legend", self.defaults_config.get("legend", False)
                )
                yscale = v.get(
                    "yscale", self.defaults_config.get("yscale", "linear")
                )
                yield {
                    "name": validate_file_name(k + "_" + j),
                    "display": display,
                    "upper_ylim": upper_ylim,
                    "idx": ("decay", decay_chain, decay, part, "ang", theta),
                    "trans": trans,
                    "bins": bins,
                    "range": xrange,
                    "legend": legend,
                    "yscale": yscale,
                }

    def get_params(self, params=None):
        if params is None:
            return self.params
        if isinstance(params, str):
            params = [params]
        params_list = []
        for i in self.params:
            if i["display"] in params:
                params_list.append(i)
        return params_list
