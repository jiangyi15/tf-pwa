import yaml
import json
from tf_pwa.amp import get_particle, get_decay, DecayChain, DecayGroup, AmplitudeModel
from tf_pwa.particle import split_particle_type
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.model import Model, Model_new, FCN, CombineFCN
from tf_pwa.model.cfit import Model_cfit
import re
import functools
import time
from scipy.interpolate import interp1d
from scipy.optimize import minimize, BFGS, basinhopping
import numpy as np
import matplotlib.pyplot as plt
from tf_pwa.data import data_index, data_shape, data_split, load_data, save_data
from tf_pwa.variable import VarsManager
from tf_pwa.utils import time_print
import itertools
import os
import sympy as sy
from tf_pwa.root_io import save_dict_to_root, has_uproot
import warnings
from scipy.optimize import BFGS
from tf_pwa.fit_improve import minimize as my_minimize
from tf_pwa.applications import fit, cal_hesse_error, corr_coef_matrix, fit_fractions
from tf_pwa.fit import FitResult
from tf_pwa.variable import Variable
import copy

from .decay_config import DecayConfig
from .data import load_data_mode

class ConfigLoader(object):
    """class for loading config.yml"""

    def __init__(self, file_name, vm=None, share_dict={}):
        self.share_dict = share_dict
        self.config = self.load_config(file_name)
        self.decay_config = DecayConfig(self.config, share_dict)
        self.dec = self.decay_config.dec
        self.particle_map, self.particle_property = self.decay_config.particle_map, self.decay_config.particle_property
        self.top, self.finals = self.decay_config.top, self.decay_config.finals
        self.full_decay = self.decay_config.full_decay
        self.decay_struct = self.decay_config.decay_struct
        self.vm = vm
        self.amps = {}
        self.cached_data = None
        self.bound_dic = {}
        self.gauss_constr_dic = {}
        self.plot_params = PlotParams(self.config.get("plot", {}), self.decay_struct)
        self._neglect_when_set_params = []
        self.data = load_data_mode(self.config.get("data", None), self.decay_struct)

    @staticmethod
    def load_config(file_name, share_dict={}):
        if isinstance(file_name, dict):
            return copy.deepcopy(file_name)
        if isinstance(file_name, str):
            if file_name in share_dict:
                return ConfigLoader.load_config(share_dict[file_name])
            with open(file_name) as f:
                ret = yaml.safe_load(f)
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
            mode =  "single"
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
        warnings.warn("No data file as 'phsp_noeff', using the first 'phsp' file instead.")
        return self.get_data("phsp")[0]

    def get_phsp_plot(self):
        if "phsp_plot" in self.config["data"]:
            assert len(self.config["data"]["phsp_plot"]) == len(self.config["data"]["phsp"])
            return self.get_data("phsp_plot")
        return self.get_data("phsp")

    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @functools.lru_cache()
    def get_amplitude(self, vm=None, name=""):
        use_tf_function = self.config.get("data",{}).get("use_tf_function", False)
        decay_group = self.full_decay
        if vm is None:
            vm = self.vm
        if vm in self.amps:
            return self.amps[vm]
        amp = AmplitudeModel(decay_group, vm=vm, name=name, use_tf_function=use_tf_function)
        self.add_constraints(amp)
        self.amps[vm] = amp
        return amp

    def add_constraints(self, amp):
        constrains = self.config.get("constrains", {})
        if constrains is None:
            constrains = {}
        self.add_decay_constraints(amp, constrains.get("decay", {}))
        self.add_particle_constraints(amp, constrains.get("particle", {}))
        self.add_fix_var_constraints(amp, constrains.get("fix_var", {}))
        self.add_var_range_constraints(amp, constrains.get("var_range", {}))

    def add_fix_var_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        for k, v in dic.items():
            print("fix var: ", k, "=", v)
            amp.vm.set_fix(k, v)

    def add_var_range_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        for k, v in dic.items():
            print("variable range: ", k, " in ", v)
            self.bound_dic[k] = v

    def add_decay_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}
        fix_total_idx = dic.get("fix_chain_idx", 0)
        fix_total_val = dic.get("fix_chain_val", np.random.uniform(0,2))

        fix_decay = amp.decay_group.get_decay_chain(fix_total_idx)
        # fix which total factor
        fix_decay.total.set_fix_idx(fix_idx=0, fix_vals=(fix_total_val,0.0))

    def add_particle_constraints(self, amp, dic=None):
        if dic is None:
            dic = {}

        res_dec = {}
        for d in amp.decay_group:
            for p_i in d.inner:
                i = str(p_i)
                res_dec[i] = d
                # free mass and width and set bounds
                m_sigma = self.config['particle'][i].get("m_sigma", None)
                g_sigma = self.config['particle'][i].get("g_sigma", None)
                if "gauss_constr" in self.config['particle'][i] and self.config['particle'][i]["gauss_constr"]:
                    if 'm' in self.config['particle'][i]["gauss_constr"]:
                        if m_sigma is None:
                            raise Exception("Need sigma of mass of {} when adding gaussian constraint".format(i))
                        self.gauss_constr_dic[p_i.mass.name] = (self.config['particle'][i]["m0"], m_sigma)
                    if 'g' in self.config['particle'][i]["gauss_constr"]:
                        if g_sigma is None:
                            raise Exception("Need sigma of width of {} when adding gaussian constraint".format(i))
                        self.gauss_constr_dic[p_i.width.name] = (self.config['particle'][i]["g0"], g_sigma)
                if "float" in self.config['particle'][i] and self.config['particle'][i]["float"]:
                    if 'm' in self.config['particle'][i]["float"]:
                        p_i.mass.freed() # set_fix(i+'_mass',unfix=True)
                        if "m_max" in self.config['particle'][i]:
                            upper = self.config['particle'][i]["m_max"]
                        elif m_sigma is not None:
                            upper = self.config['particle'][i]["m0"] + 10 * m_sigma
                        else:
                            upper = None
                        if "m_min" in self.config['particle'][i]:
                            lower = self.config['particle'][i]["m_min"]
                        elif m_sigma is not None:
                            lower = self.config['particle'][i]["m0"] - 10 * m_sigma
                        else:
                            lower = None
                        self.bound_dic[p_i.mass.name] = (lower,upper)
                    else:
                        self._neglect_when_set_params.append(p_i.mass.name)
                    if 'g' in self.config['particle'][i]["float"]:
                        p_i.width.freed() # amp.vm.set_fix(i+'_width',unfix=True)
                        if "g_max" in self.config['particle'][i]:
                            upper = self.config['particle'][i]["g_max"]
                        elif g_sigma is not None:
                            upper = self.config['particle'][i]["g0"] + 10 * g_sigma
                        else:
                            upper = None
                        if "g_min" in self.config['particle'][i]:
                            lower = self.config['particle'][i]["g_min"]
                        elif g_sigma is not None:
                            lower = self.config['particle'][i]["g0"] - 10 * g_sigma
                        else:
                            lower = None
                        self.bound_dic[p_i.width.name] = (lower,upper)
                    else:
                        self._neglect_when_set_params.append(p_i.width.name)
                #else:
                    #self._neglect_when_set_params.append(p_i.mass.name) #p_i.mass.name
                    #self._neglect_when_set_params.append(p_i.mass.name) #p_i.width.name
                else:
                    self._neglect_when_set_params.append(i+'_mass') #p_i.mass.name
                    self._neglect_when_set_params.append(i+'_width') #p_i.width.name
                # share helicity variables 
                if "coef_head" in self.config['particle'][i]:
                    coef_head = self.config['particle'][i]["coef_head"]
                    if coef_head in res_dec:
                        d_coef_head = res_dec[coef_head]
                        for j,h in zip(d,d_coef_head):
                            if i in [str(jj) for jj in j.outs] or i is str(j.core):
                                h.g_ls.sameas(j.g_ls)
                        # share total radium
                        d_coef_head.total.r_shareto(d.total)
                    else:
                        self.config['particle'][coef_head]["coef_head"] = i

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
        w_bkg, w_inmc = self._get_bg_weight()
        model = []
        if "inmc" in self.config["data"]:
            float_wmc = self.config["data"].get("float_inmc_ratio_in_pdf", False)
            if not isinstance(float_wmc, list):
                float_wmc = [float_wmc] * self._Ngroup
            assert len(float_wmc) == self._Ngroup
            for wb, wi, fw in zip(w_bkg, w_inmc, float_wmc):
                model.append(Model_new(amp, wb, wi, fw))
        else:
            for wb in w_bkg:
                model.append(Model(amp, wb))
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
        weight_scale = self.config["data"].get("weight_scale", False) #???
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
            data, phsp, bg, inmc = self.get_all_data()
        else:
            data, phsp, bg, inmc = all_data
        self._Ngroup = len(data)
        model = self._get_model()
        fcns = []
        for md, dt, mc, sb, ij in zip(model, data, phsp, bg, inmc):
            fcns.append(FCN(md, dt, mc, bg=sb, batch=batch, inmc=ij, gauss_constr=self.gauss_constr_dic))
        if len(fcns) == 1:
            fcn = fcns[0]
        else:
            fcn = CombineFCN(fcns=fcns, gauss_constr=self.gauss_constr_dic)
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
            ns = [ns/len(total)] * len(total)
        for i in total:
            i.set_rho(1.0)
        pw = amp.partial_weight(phsp)
        for i, w, ni in zip(total, pw, ns):
            i.set_rho(np.sqrt(ni / np.sum(weight * w) * sw))

    @time_print
    def fit(self, data=None, phsp=None, bg=None, inmc=None, batch=65000, method="BFGS", check_grad=False, improve=False, reweight=False, maxiter=None):
        if data is None and phsp is None:
            data, phsp, bg, inmc = self.get_all_data()
        amp = self.get_amplitude()
        fcn = self.get_fcn([data, phsp, bg, inmc], batch=batch)
        print("decay chains included: ")
        for i in self.full_decay:
            ls_list = [getattr(j, "get_ls_list", lambda x:None)() for j in i]
            print("  ", i, " ls: ", *ls_list)
        if reweight:
            ConfigLoader.reweight_init_value(amp, phsp, ns=data_shape(data))

        print("\n########### initial parameters")
        print(json.dumps(amp.get_params(), indent=2))
        print("initial NLL: ", fcn({})) # amp.get_params()))
        # fit configure
        # self.bound_dic[""] = (,)
        self.fit_params = fit(fcn=fcn, method=method, bounds_dict=self.bound_dic, check_grad=check_grad, improve=False, maxiter=maxiter)
        return self.fit_params

    def get_params_error(self, params=None, data=None, phsp=None, bg=None, batch=10000):
        if params is None:
            params = {}
        if data is None:
            data, phsp, bg, inmc = self.get_all_data()
        if hasattr(params, "params"):
            params = getattr(params, "params")
        fcn = self.get_fcn([data, phsp, bg, inmc], batch=batch)
        hesse_error, self.inv_he = cal_hesse_error(fcn, params, check_posi_def=True, save_npy=True)
        #print("parameters order")
        #print(fcn.model.Amp.vm.trainable_vars)
        #print("error matrix:")
        #print(self.inv_he)
        #print("correlation matrix:")
        #print(corr_coef_matrix(self.inv_he))
        print("hesse_error:", hesse_error)
        err = dict(zip(fcn.vm.trainable_vars, hesse_error))
        if hasattr(self, "fit_params"):
            self.fit_params.set_error(err)
        return err

    def plot_partial_wave(self, params=None, data=None, phsp=None, bg=None, prefix="figure/", save_root=False, **kwargs):
        if params is None:
            params = {}
        if hasattr(params, "params"):
            params = getattr(params, "params")
        pathes = prefix.rstrip('/').split('/')
        path = ""
        for p in pathes:
            path += p+'/'
            if not os.path.exists(path):
                os.mkdir(path)
        if data is None:
            data = self.get_data("data")
            bg = self.get_data("bg")
            phsp = self.get_phsp_plot()
            if bg is None:
                bg = [bg] * len(data)
        amp = self.get_amplitude()
        self._Ngroup = len(data)
        ws_bkg, ws_inmc = self._get_bg_weight(data, bg)
        chain_property = []
        for i in range(len(self.full_decay.chains)):
            label, curve_style = self.get_chain_property(i)
            chain_property.append([i, label, curve_style])
        plot_var_dic = {}
        for conf in self.plot_params.get_params():
            name = conf.get("name")
            display = conf.get("display", name)
            upper_ylim = conf.get("upper_ylim", None)
            idx = conf.get("idx")
            trans = conf.get("trans", lambda x: x)
            has_legend = conf.get("legend", False)
            xrange = conf.get("range", None)
            bins = conf.get("bins", None)
            units = conf.get("units", "")
            plot_var_dic[name] = {"display": display, "upper_ylim": upper_ylim, "legend": has_legend,
                "idx": idx, "trans": trans, "range": xrange, "bins": bins, "units": units}
        if self._Ngroup == 1:
            data_dict, phsp_dict, bg_dict = self._cal_partial_wave(amp, params, data[0], phsp[0], bg[0], ws_bkg[0], path, plot_var_dic, chain_property, save_root=save_root, **kwargs)
            self._plot_partial_wave(data_dict, phsp_dict, bg_dict, path, plot_var_dic, chain_property, **kwargs)
        else:
            combine_plot = self.config["plot"].get("combine_plot", True)
            if not combine_plot:
                for dt, mc, sb, w_bkg, i in zip(data, phsp, bg, ws_bkg, range(self._Ngroup)):
                    data_dict, phsp_dict, bg_dict = self._cal_partial_wave(amp, params, dt, mc, sb, w_bkg, path+'d{}_'.format(i), plot_var_dic, chain_property, save_root=save_root, **kwargs)
                    self._plot_partial_wave(data_dict, phsp_dict, bg_dict, path+'d{}_'.format(i), plot_var_dic, chain_property, **kwargs)
            else:

                for dt, mc, sb, w_bkg, i in zip(data, phsp, bg, ws_bkg, range(self._Ngroup)):
                    data_dict, phsp_dict, bg_dict = self._cal_partial_wave(amp, params, dt, mc, sb, w_bkg, path+'d{}_'.format(i), plot_var_dic, chain_property, save_root=save_root, **kwargs)
                    #self._plot_partial_wave(data_dict, phsp_dict, bg_dict, path+'d{}_'.format(i), plot_var_dic, chain_property, **kwargs)
                    if i == 0:
                        datas_dict = {}
                        for ct in data_dict:
                            datas_dict[ct] = [data_dict[ct]]
                        phsps_dict = {}
                        for ct in phsp_dict:
                            phsps_dict[ct] = [phsp_dict[ct]]
                        bgs_dict = {}
                        for ct in bg_dict:
                            bgs_dict[ct] = [bg_dict[ct]]
                    else:
                        for ct in data_dict:
                            datas_dict[ct].append(data_dict[ct])
                        for ct in phsp_dict:
                            phsps_dict[ct].append(phsp_dict[ct])
                        for ct in bg_dict:
                            bgs_dict[ct].append(bg_dict[ct])
                for ct in datas_dict:
                    datas_dict[ct] = np.concatenate(datas_dict[ct])
                for ct in phsps_dict:
                    phsps_dict[ct] = np.concatenate(phsps_dict[ct])
                for ct in bgs_dict:
                    bgs_dict[ct] = np.concatenate(bgs_dict[ct])
                self._plot_partial_wave(datas_dict, phsps_dict, bgs_dict, path+'com_', plot_var_dic, chain_property, **kwargs)
                if has_uproot and save_root:
                    save_dict_to_root([datas_dict, phsps_dict, bgs_dict], file_name=path+"variables_com.root", tree_name=["data", "fitted", "sideband"])
                    print("Save root file "+prefix+"com_variables.root")

    def _cal_partial_wave(self, amp, params, data, phsp, bg, w_bkg, prefix, plot_var_dic, chain_property,
                            save_root=False, bin_scale=3, **kwargs):
        data_dict = {}
        phsp_dict = {}
        bg_dict = {}
        with amp.temp_params(params):
            total_weight = amp(phsp) * phsp.get("weight", 1.0)
            data_weight = data.get("weight", None)
            if data_weight is None:
                n_data = data_shape(data)
            else:
                n_data = np.sum(data_weight)
            if bg is None:
                norm_frac = n_data / np.sum(total_weight)
            else:
                if isinstance(w_bkg, float):
                    norm_frac = (n_data - w_bkg *
                                data_shape(bg)) / np.sum(total_weight)
                else:
                    norm_frac = (n_data + np.sum(w_bkg))/np.sum(total_weight)
            weights = amp.partial_weight(phsp)
            data_weights = data.get("weight", [1.0]*data_shape(data))
            data_dict["data_weights"] = data_weights
            phsp_weights = total_weight*norm_frac
            phsp_dict["MC_total_fit"] = phsp_weights # MC total weight
            if bg is not None:
                if isinstance(w_bkg, float):
                    bg_weight = [w_bkg] * data_shape(bg)
                else:
                    bg_weight = -w_bkg
                bg_dict["sideband_weights"] = bg_weight # sideband weight
            for i, label, _ in chain_property:
                weight_i = weights[i] * norm_frac * bin_scale * phsp.get("weight", 1.0)
                phsp_dict["MC_{0}_{1}_fit".format(i, label)] = weight_i # MC partial weight
            for name in plot_var_dic:
                idx = plot_var_dic[name]["idx"]
                trans = plot_var_dic[name]["trans"]

                data_i = trans(data_index(data, idx))
                data_dict[name] = data_i # data variable

                phsp_i = trans(data_index(phsp, idx))
                phsp_dict[name+"_MC"] = phsp_i # MC
                
                if bg is not None:
                    bg_i = trans(data_index(bg, idx))
                    bg_dict[name+"_sideband"] = bg_i # sideband
                
        if has_uproot and save_root:
            save_dict_to_root([data_dict, phsp_dict, bg_dict], file_name=prefix+"variables.root", tree_name=["data", "fitted", "sideband"])
            print("Save root file "+prefix+"variables.root")
        return data_dict, phsp_dict, bg_dict

    def _plot_partial_wave(self, data_dict, phsp_dict, bg_dict, prefix, plot_var_dic, chain_property,
                        plot_delta=False, plot_pull=False, save_pdf=False, bin_scale=3, single_legend=False, format="png", **kwargs):
        #cmap = plt.get_cmap("jet")
        #N = 10
        #colors = [cmap(float(i) / (N+1)) for i in range(1, N+1)]
        colors = ["red", "orange", "purple", "springgreen", "y", "green", "blue", "c"]
        linestyles = ['-', '--', '-.', ':']

        data_weights = data_dict["data_weights"]
        if bg_dict:
            bg_weight = bg_dict["sideband_weights"]
        phsp_weights = phsp_dict["MC_total_fit"]
        for name in plot_var_dic:
            data_i = data_dict[name]
            phsp_i = phsp_dict[name+"_MC"]
            if bg_dict:
                bg_i = bg_dict[name+"_sideband"]

            display = plot_var_dic[name]["display"]
            upper_ylim = plot_var_dic[name]["upper_ylim"]
            has_legend = plot_var_dic[name]["legend"]
            bins = plot_var_dic[name]["bins"]
            units = plot_var_dic[name]["units"]
            xrange = plot_var_dic[name]["range"]
            if xrange is None:
                xrange = [np.min(data_i) - 0.1, np.max(data_i) + 0.1]
            data_x, data_y, data_err = hist_error(data_i, bins=bins, weights=data_weights,xrange=xrange)
            fig = plt.figure()
            if plot_delta or plot_pull:
                ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            else:
                ax = fig.add_subplot(1, 1, 1)

            ax.errorbar(data_x, data_y, yerr=data_err, fmt=".",
                        zorder=-2, label="data", color="black")  #, capsize=2)

            if bg_dict:
                ax.hist(bg_i, weights=bg_weight,
                        label="back ground", bins=bins, range=xrange, histtype="stepfilled", alpha=0.5, color="grey")
                mc_i = np.concatenate([bg_i, phsp_i])
                mc_weights = np.concatenate([bg_weight, phsp_weights])
                fit_y, fit_x, _ = ax.hist(mc_i, weights=mc_weights, range=xrange,
                                            histtype="step", label="total fit", bins=bins, color="black")
            else:
                mc_i = phsp_i
                fit_y, fit_x, _ = ax.hist(phsp_i, weights=phsp_weights, range=xrange, histtype="step", 
                                            label="total fit", bins=bins, color="black")
            
            # plt.hist(data_i, label="data", bins=50, histtype="step")
            style = itertools.product(colors, linestyles)
            for i, label, curve_style in chain_property:
                weight_i = phsp_dict["MC_{0}_{1}_fit".format(i, label)]
                x, y = hist_line(phsp_i, weights=weight_i, xrange=xrange, bins=bins*bin_scale)
                if curve_style is None:
                    color, ls = next(style)
                    ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1)
                else:
                    ax.plot(x, y, curve_style, label=label, linewidth=1)

            ax.set_ylim((0, upper_ylim))
            ax.set_xlim(xrange)
            if has_legend:
                leg = ax.legend(frameon=False, labelspacing=0.1, borderpad=0.0)
            ax.set_title(display, fontsize='xx-large')
            ax.set_xlabel(display + units)
            ax.set_ylabel("Events/{:.3f}{}".format((max(data_x) - min(data_x))/bins, units))
            if plot_delta or plot_pull:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax2 = plt.subplot2grid((4, 1), (3, 0),  rowspan=1)
                y_err = fit_y - data_y
                if plot_pull:
                    _epsilon = 1e-10
                    with np.errstate(divide='ignore', invalid='ignore'):
                        fit_err = np.sqrt(fit_y)
                        y_err = y_err/fit_err
                    y_err[fit_err<_epsilon] = 0.0
                ax2.plot(data_x, y_err, color="r")
                ax2.plot([data_x[0], data_x[-1]], [0, 0], color="r")
                if plot_pull:
                    ax2.set_ylabel("pull")
                    ax2.set_ylim((-5, 5))
                else:
                    ax2.set_ylabel("$\\Delta$Events")
                    ax2.set_ylim((-max(abs(y_err)),
                                max(abs(y_err))))
                ax.set_xlabel("")
                ax2.set_xlabel(display + units)
                if xrange is not None:
                    ax2.set_xlim(xrange)
            #ax.set_yscale("log")
            #ax.set_ylim([0.1, 1e3])
            fig.savefig(prefix+name+"."+format, dpi=300)
            if single_legend:
                export_legend(ax, prefix + "legend.{}".format(format))
            if save_pdf:
                fig.savefig(prefix+name+".pdf", dpi=300)
                if single_legend:
                    export_legend(ax, prefix + "legend.pdf")
            print("Finish plotting "+prefix+name)
            plt.close(fig)
                

        twodplot = self.config["plot"].get("2Dplot", {})
        for k, i in twodplot.items():
            var1, var2 = k.split('&')
            var1 = var1.rstrip()
            var2 = var2.lstrip()
            k = var1 + "_vs_" + var2
            display = i.get("display", k)
            plot_figs = i["plot_figs"]
            name1, name2 = display.split('vs')
            name1 = name1.rstrip()
            name2 = name2.lstrip()
            range1 = plot_var_dic[var1]["range"]
            data_1 = data_dict[var1]
            phsp_1 = phsp_dict[var1+"_MC"]
            range2 = plot_var_dic[var2]["range"]
            data_2 = data_dict[var2]
            phsp_2 = phsp_dict[var2+"_MC"]

            # data
            if "data" in plot_figs:
                plt.scatter(data_1,data_2,s=1,alpha=0.8,label='data')
                plt.xlabel(name1); plt.ylabel(name2); plt.title(display, fontsize='xx-large'); plt.legend()
                plt.xlim(range1); plt.ylim(range2)
                plt.savefig(prefix+k+'_data')
                plt.clf()
                print("Finish plotting 2D data "+prefix+k)
            # sideband
            if "sideband" in plot_figs:
                if bg_dict:
                    bg_1 = bg_dict[var1+"_sideband"]
                    bg_2 = bg_dict[var2+"_sideband"]
                    plt.scatter(bg_1,bg_2,s=1,c='g',alpha=0.8,label='sideband')
                    plt.xlabel(name1); plt.ylabel(name2); plt.title(display, fontsize='xx-large'); plt.legend()
                    plt.xlim(range1); plt.ylim(range2)
                    plt.savefig(prefix+k+'_bkg')
                    plt.clf()
                    print("Finish plotting 2D sideband "+prefix+k)
                else:
                    print("There's no bkg input")
            # fit pdf
            if "fitted" in plot_figs:
                plt.hist2d(phsp_1,phsp_2,bins=100,weights=phsp_weights)
                plt.xlabel(name1); plt.ylabel(name2); plt.title(display, fontsize='xx-large'); plt.colorbar()
                plt.xlim(range1); plt.ylim(range2)
                plt.savefig(prefix+k+'_fitted')
                plt.clf()
                print("Finish plotting 2D fitted "+prefix+k)

    def get_chain(self, idx):
        decay_group = self.full_decay
        return decay_group.get_decay_chain(idx)

    def get_chain_property(self, idx):
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
        for i in combine:
            pro = self.particle_property[str(i)]
            names.append(pro.get("display", str(i)))
        return " ".join(names), curve_style

    def cal_fitfractions(self, params={}, mcdata=None, batch=25000):
        if hasattr(params, "params"):
            params = getattr(params, "params")
        if mcdata is None:
            mcdata = self.get_phsp_noeff()
        amp = self.get_amplitude()
        frac, err_frac = fit_fractions(amp, mcdata, self.inv_he, params, batch)
        return frac, err_frac

    def cal_signal_yields(self, params={}, mcdata=None, batch=25000):
        if hasattr(params, "params"):
            params = getattr(params, "params")
        if mcdata is None:
            mcdata = self.get_data("phsp")
        amp = self.get_amplitude()
        fracs = [fit_fractions(amp, i, self.inv_he, params, batch) for i in mcdata]
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
                N_total.append((N_data - w * N_bg, np.sqrt(N_data + w*w * N_bg)))

        N_sig_s = []
        for frac_e, N_e in zip(fracs, N_total):
            frac, frac_err = frac_e
            N, N_err = N_e
            N_sig = {}
            for i in frac:
                N_sig[i] = (frac[i] * N, np.sqrt((N*frac_err[i])**2 + (N_err*frac[i])**2))
            N_sig_s.append(N_sig)
        return N_sig_s

    def get_params(self, trainable_only=False):
        return self.get_amplitude().get_params(trainable_only)

    def set_params(self, params, neglect_params=None):
        if isinstance(params, str):
            with open(params) as f:
                params = yaml.safe_load(f)
        if hasattr(params, "params"):
            params = params.params
        if isinstance(params, dict):
            if "value" in params:
                params = params["value"]
        amplitude = self.get_amplitude()
        ret = params.copy()
        if neglect_params is None:
            neglect_params = self._neglect_when_set_params
        if neglect_params.__len__() is not 0:
            warnings.warn("Neglect {} when setting params.".format(neglect_params))
            for v in params:
                if v in self._neglect_when_set_params:
                    del ret[v]
        amplitude.set_params(ret)

    def save_params(self, file_name):
        params = self.get_params()
        val = {k: float(v) for k,v in params.items()}
        with open(file_name, "w") as f:
            json.dump(val, f, indent=2)


def validate_file_name(s):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    name = re.sub(rstr, "_", s)
    return name




def hist_error(data, bins=50, xrange=None, weights=1.0, kind="poisson"):
    if not hasattr(weights, "__len__"):
        weights = [weights] * data.__len__()
    data_hist = np.histogram(data, bins=bins, weights=weights, range=xrange)
    # ax.hist(fd(data[idx].numpy()),range=xrange,bins=bins,histtype="step",label="data",zorder=99,color="black")
    data_y, data_x = data_hist[0:2]
    data_x = (data_x[:-1]+data_x[1:])/2
    if kind == "poisson":
        data_err = np.sqrt(np.abs(data_y))
    elif kind == "binomial":
        n = data.shape[0]
        p = data_y / n
        data_err = np.sqrt(p*(1-p)*n)
    else:
        raise ValueError("unknown error kind {}".format(kind))
    return data_x, data_y, data_err


def hist_line(data, weights, bins, xrange=None, inter=1, kind="quadratic"):
    """interpolate data from hostgram into a line"""
    y, x = np.histogram(data, bins=bins, range=xrange, weights=weights)
    x = (x[:-1] + x[1:])/2
    if xrange is None:
        xrange = (np.min(data), np.max(data))
    func = interp1d(x, y, kind=kind)
    num = data.shape[0] * inter
    x_new = np.linspace(np.min(x), np.max(x), num=num, endpoint=True)
    y_new = func(x_new)
    return x_new, y_new


def export_legend(ax, filename="legend.pdf",ncol=1):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center')
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    plt.close(fig2)
    plt.close(fig)


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
            return "decay", de_i.standard_topology(), self.re_map.get(de, de), self.re_map.get(p, p), "ang"
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
            return "decay", de_i.standard_topology(), self.re_map.get(de, de), self.re_map.get(p, p), "aligned_angle"
        raise ValueError("unknown sub {}".format(sub))

    def get_mass_vars(self):
        mass = self.config.get("mass", {})
        x = sy.symbols('x')
        for k, v in mass.items():
            display = v.get("display", "M({})".format(k))
            upper_ylim = v.get("upper_ylim", None)
            xrange = v.get("range", None)
            trans = v.get("trans", 'x')
            trans = sy.sympify(trans)
            trans = sy.lambdify(x,trans)
            units = v.get("units", "GeV")
            bins = v.get("bins", self.defaults_config.get("bins", 50))
            legend = v.get("legend", self.defaults_config.get("legend", True))
            yield {"name": "m_"+k, "display": display, "upper_ylim": upper_ylim,
                   "idx": ("particle", self.re_map.get(get_particle(k), get_particle(k)), "m"),
                   "legend": legend, "range": xrange, "bins": bins,
                   "trans": trans, "units": units}

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
                _, decay_chain, decay, part, _ = self.get_data_index("angle", k)
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
                legend = v.get("legend", self.defaults_config.get("legend", False))
                yield {"name": validate_file_name(k+"_"+j), "display": display, "upper_ylim": upper_ylim,
                       "idx": ("decay", decay_chain, decay, part, "ang", theta),
                       "trans": trans, "bins": bins, "range": xrange, "legend": legend}

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
