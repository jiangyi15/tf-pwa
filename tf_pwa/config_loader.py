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
from .fit_improve import minimize as my_minimize
from .applications import fit, cal_hesse_error, corr_coef_matrix, fit_fractions
from .fit import FitResult
from .variable import Variable


class ConfigLoader(object):
    """class for loading config.yml"""

    def __init__(self, file_name, vm=None):
        self.config = self.load_config(file_name)
        self.particle_key_map = {
            "Par": "P",
            "m0": "mass",
            "g0": "width",
            "J": "J",
            "P": "P",
            "spins": "spins",
            "bw": "model",
            "model": "model",
            "bw_l": "bw_l",
            "running_width": "running_width"
        }
        self.decay_key_map = {
            "model": "model"
        }
        self.dec = self.decay_item(self.config["decay"])
        self.particle_map, self.particle_property, self.top, self.finals = self.particle_item(
            self.config["particle"])
        self.full_decay = DecayGroup(self.get_decay_struct(
            self.dec, self.particle_map, self.particle_property, self.top, self.finals))
        self.decay_struct = DecayGroup(self.get_decay_struct(self.dec))
        self.vm = vm
        self.amps = {}
        self.cached_data = None
        self.bound_dic = {}
        self.plot_params = PlotParams(self.config["plot"], self.decay_struct)

    @staticmethod
    def load_config(file_name):
        if isinstance(file_name, dict):
            return file_name
        with open(file_name) as f:
            ret = yaml.safe_load(f)
        return ret

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

    @functools.lru_cache()
    def get_data(self, idx):
        if self.cached_data is not None:
            data = self.cached_data.get(idx, None)
            if data is not None:
                # print(data.keys())
                return data
        files = self.get_data_file(idx)
        if files is None:
            return None
        order = self.get_dat_order()
        center_mass = self.config["data"].get("center_mass", True)
        r_boost = self.config["data"].get("r_boost", False)
        data = prepare_data_from_decay(files, self.decay_struct, order, center_mass=center_mass, r_boost=r_boost)
        if idx == "bg":
            return data
        weight = self.config["data"].get(idx+"_weight", None)
        if weight is not None:
            if isinstance(weight, float):
                data["weight"] = np.array([weight] * data_shape(data))
            else:  # weight files
                weight = self.load_weight_file(weight)
                data["weight"] = weight[:data_shape(data)]
        # print(data.keys())
        return data

    def load_weight_file(self, weight_files):
        ret = []
        if isinstance(weight_files, list):
            for i in weight_files:
                data = np.loadtxt(i).reshape((-1,))
                ret.append(data)
        elif isinstance(weight_files, str):
            data = np.loadtxt(i).reshape((-1,))
            ret.append(data)
        else:
            raise TypeError("weight files must be string of list of strings, not {}".format(type(weight_files)))
        if len(ret) == 1:
            return ret[0]
        return np.concatenate(ret)

    def load_cached_data(self, file_name=None):
        if file_name is None:
            file_name = self.config["data"].get("cached_data", None)
        if file_name is not None and os.path.exists(file_name):
            if self.cached_data is None:
                self.cached_data = load_data(file_name)
                print("load cached_data {}".format(file_name))
                # print(self.cached_data["data"]["decay"].keys())
    
    def save_cached_data(self, data, file_name=None):
        if file_name is None:
            file_name = self.config["data"].get("cached_data", None)
        if file_name is not None:
            if not os.path.exists(file_name):
                save_data(file_name, data)
                print("save cached_data {}".format(file_name))

    def get_all_data(self):
        datafile = ["data", "phsp", "bg", "inmc"]
        self.load_cached_data()
        all_data = [self.get_data(i) for i in datafile]
        self.save_cached_data(dict(zip(datafile, all_data)))
        return all_data

    def get_phsp_noeff(self):
        if "phsp_noeff" in self.config["data"]:
            return self.get_data("phsp_noeff")
        return self.get_data("phsp")

    def get_phsp_plot(self):
        if "phsp_plot" in self.config["data"]:
            return self.get_data("phsp_plot")
        return self.get_data("phsp")

    def get_decay(self, full=True):
        if full:
            return self.full_decay
        else:
            return self.decay_struct

    @staticmethod
    def _list2decay(core, outs):
        parts = []
        params = {}
        for j in outs:
            if isinstance(j, dict):
                for k, v in j.items():
                    params[k] = v
            else:
                parts.append(j)
        dec = {"core": core, "outs": parts, "params": params}
        return dec

    @staticmethod
    def decay_item(decay_dict):
        decs = []
        for core, outs in decay_dict.items():
            is_list = [isinstance(i, list) for i in outs]
            if all(is_list):
                for i in outs:
                    dec = ConfigLoader._list2decay(core, i)
                    decs.append(dec)
            else:
                dec = ConfigLoader._list2decay(core, outs)
                decs.append(dec)
        return decs

    @staticmethod
    def _do_include_dict(d, o):
        s = ConfigLoader.load_config(o)
        for i in s:
            if i not in d:
                d[i] = s[i]

    @staticmethod
    def particle_item_list(particle_list):
        particle_map = {}
        particle_property = {}
        for particle, candidate in particle_list.items():
            if isinstance(candidate, list):  # particle map
                if len(candidate) == 0:
                    particle_map[particle] = []
                for i in candidate:
                    if isinstance(i, str):
                        particle_map[particle] = particle_map.get(
                            particle, []) + [i]
                    elif isinstance(i, dict):
                        map_i, pro_i = ConfigLoader.particle_item_list(i)
                        for k, v in map_i.items():
                            particle_map[k] = particle_map.get(k, []) + v
                        particle_property.update(pro_i)
                    else:
                        raise ValueError(
                            "value of particle map {} is {}".format(i, type(i)))
            elif isinstance(candidate, dict):
                particle_property[particle] = candidate
            else:
                raise ValueError("value of particle {} is {}".format(
                    particle, type(candidate)))
        return particle_map, particle_property

    @staticmethod
    def particle_item(particle_list):
        top = particle_list.pop("$top", None)
        finals = particle_list.pop("$finals", None)
        includes = particle_list.pop("$include", None)
        if includes:
            if isinstance(includes, list):
                for i in includes:
                    ConfigLoader._do_include_dict(particle_list, i)
            elif isinstance(includes, str):
                ConfigLoader._do_include_dict(particle_list, includes)
            else:
                raise ValueError("$include must be string or list of string not {}"
                                 .format(type(includes)))
        particle_map, particle_property = ConfigLoader.particle_item_list(
            particle_list)

        if isinstance(top, dict):
            particle_property.update(top)
        if isinstance(finals, dict):
            particle_property.update(finals)
        return particle_map, particle_property, top, finals

    def rename_params(self, params, is_particle=True):
        ret = {}
        if is_particle:
            key_map = self.particle_key_map
        else:
            key_map = self.decay_key_map
        for k, v in params.items():
            ret[key_map.get(k, k)] = v
        return ret

    def get_decay_struct(self, decay, particle_map=None, particle_params=None, top=None, finals=None):
        """  get decay structure for decay dict"""
        particle_map = particle_map if particle_map is not None else {}
        particle_params = particle_params if particle_params is not None else {}

        particle_set = {}

        def add_particle(name):
            if name in particle_set:
                return particle_set[name]
            params = particle_params.get(name, {})
            params = self.rename_params(params)
            part = get_particle(name, **params)
            particle_set[name] = part
            return part

        def wrap_particle(name):
            name_list = particle_map.get(name, [name])
            return [add_particle(i) for i in name_list]

        def all_combine(out):
            if len(out) < 1:
                yield []
            else:
                for i in out[0]:
                    for j in all_combine(out[1:]):
                        yield [i] + j

        decs = []
        for dec in decay:
            core = wrap_particle(dec["core"])
            outs = [wrap_particle(j) for j in dec["outs"]]
            for i in core:
                for j in all_combine(outs):
                    dec_i = get_decay(i, j, **dec["params"])
                    decs.append(dec_i)

        top_tmp, finals_tmp = set(), set()
        if top is None or finals is None:
            top_tmp, res, finals_tmp = split_particle_type(decs)
        if top is None:
            top_tmp = list(top_tmp)
            assert len(top_tmp) == 1, "not only one top particle"
            top = list(top_tmp)[0]
        else:
            if isinstance(top, str):
                top = particle_set[top]
            elif isinstance(top, dict):
                keys = list(top.keys())
                assert len(keys) == 1
                top = particle_set[keys.pop()]
            else:
                return particle_set[str(top)]
        if finals is None:
            finals = list(finals_tmp)
        elif isinstance(finals, (list, dict)):
            finals = [particle_set[i] for i in finals]
        else:
            raise TypeError("{}: {}".format(finals, type(finals)))

        dec_chain = top.chain_decay()
        ret = []
        for i in dec_chain:
            if sorted(DecayChain(i).outs) == sorted(finals):
                ret.append(i)
        return ret

    @functools.lru_cache()
    def get_amplitude(self, vm=None, name=""):
        decay_group = self.full_decay
        if vm is None:
            vm = self.vm
        if vm in self.amps:
            return self.amps[vm]
        amp = AmplitudeModel(decay_group, vm=vm, name=name)
        self.add_constraints(amp)
        self.amps[vm] = amp
        return amp

    def add_constraints(self, amp):
        constrains = self.config.get("constrains", {})
        if constrains is None:
            constrains = {}
        self.add_decay_constraints(amp, constrains.get("decay", {}))
        self.add_particle_constraints(amp, constrains.get("particle", {}))

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
                if "float" in self.config['particle'][i] and self.config['particle'][i]["float"]:
                    if 'm' in self.config['particle'][i]["float"]:
                        p_i.mass.freed() # set_fix(i+'_mass',unfix=True)
                        upper = self.config['particle'][i]["m_max"] if "m_max" in self.config['particle'][i] else None
                        lower = self.config['particle'][i]["m_min"] if "m_min" in self.config['particle'][i] else None
                        self.bound_dic[p_i.mass.name] = (lower,upper)
                    if 'g' in self.config['particle'][i]["float"]:
                        p_i.width.freed() # amp.vm.set_fix(i+'_width',unfix=True)
                        upper = self.config['particle'][i]["g_max"] if "g_max" in self.config['particle'][i] else None
                        lower = self.config['particle'][i]["g_min"] if "g_min" in self.config['particle'][i] else None
                        self.bound_dic[p_i.width.name] = (lower,upper)
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
    def get_model(self, vm=None, name=""):
        amp = self.get_amplitude(vm=vm, name=name)
        w_bkg, w_inmc = self.get_bg_weight()
        if "inmc" in self.config["data"]:
            float_wmc = self.config["data"].get("float_inmc_ratio_in_pdf", False)
            model = Model_new(amp, w_bkg, w_inmc, float_wmc)
        else:
            model = Model(amp, w_bkg)
        return model
    
    def get_bg_weight(self, data=None, bg=None, display=True):
        w_bkg = self.config["data"].get("bg_weight", 0.0)
        w_inmc = self.config["data"].get("inject_ratio", 0.0)
        weight_scale = self.config["data"].get("weight_scale", False)
        if weight_scale:
            data = data if data is not None else self.get_data("data")
            bg = bg if bg is not None else self.get_data("bg")
            w_bkg = w_bkg * data_shape(data) / data_shape(bg)
            if display:
                print("background weight:", w_bkg)
        return w_bkg, w_inmc

    def get_fcn(self, batch=65000, vm=None, name=""):
        model = self.get_model()
        data, phsp, bg, inmc = self.get_all_data()
        fcn = FCN(model, data, phsp, bg=bg, batch=batch, inmc=inmc)
        """model = self.get_model(vm, name="")
        for i in self.full_decay:
            print(i)
        data, phsp, bg, inmc = self.get_all_data()
        fcn = FCN(model, data, phsp, bg=bg, inmc=inmc, batch=batch)"""
        return fcn
    
    def get_ndf(self):
        model = self.get_model()
        args_name = model.Amp.vm.trainable_vars
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
    def fit(self, data=None, phsp=None, bg=None, inmc=None, batch=65000, method="BFGS", check_grad=False, improve=False, reweight=False):
        model = self.get_model()
        if data is None and phsp is None:
            data, phsp, bg, inmc = self.get_all_data()
        print("decay chains included: ")
        for i in self.full_decay:
            ls_list = [getattr(j, "get_ls_list", lambda x:None)() for j in i]
            print("  ", i, " ls: ", *ls_list)
        if reweight:
            ConfigLoader.reweight_init_value(model.Amp, phsp, ns=data_shape(data))
        fcn = FCN(model, data, phsp, bg=bg, batch=batch, inmc=inmc)

        print("\n########### initial parameters")
        print(json.dumps(model.get_params(), indent=2))
        print("initial NLL: ", fcn.nll_grad()[0]) # model.get_params()))
        # fit configure
        # self.bound_dic[""] = (,)
        self.fit_params = fit(fcn=fcn, method=method, bounds_dict=self.bound_dic, check_grad=check_grad, improve=False)
        return self.fit_params

    def get_params_error(self, params=None, data=None, phsp=None, bg=None, batch=10000):
        if params is None:
            params = {}
        if data is None:
            data, phsp, bg, inmc = self.get_all_data()
        if hasattr(params, "params"):
            params = getattr(params, "params")
        fcn = FCN(self.get_model(), data, phsp, bg=bg, batch=batch, inmc=inmc)
        hesse_error, self.inv_he = cal_hesse_error(fcn, params, check_posi_def=True, save_npy=True)
        #print("parameters order")
        #print(fcn.model.Amp.vm.trainable_vars)
        #print("error matrix:")
        #print(self.inv_he)
        #print("correlation matrix:")
        #print(corr_coef_matrix(self.inv_he))
        print("hesse_error:", hesse_error)
        err = dict(zip(fcn.model.Amp.vm.trainable_vars, hesse_error))
        if hasattr(self, "fit_params"):
            self.fit_params.set_error(err)
        return err

    def plot_partial_wave(self, params=None, data=None, phsp=None, bg=None, prefix="figure/", 
                          plot_delta=False, plot_pull=False, save_pdf=False, save_root=False, bin_scale=3):
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        if params is None:
            params = {}
        if data is None:
            data = self.get_data("data")
            bg = self.get_data("bg")
            phsp = self.get_phsp_plot()
        if hasattr(params, "params"):
            params = getattr(params, "params")
        amp = self.get_amplitude()
        w_bkg, w_inmc = self.get_bg_weight(data, bg)
        #cmap = plt.get_cmap("jet")
        #N = 10
        #colors = [cmap(float(i) / (N+1)) for i in range(1, N+1)]
        colors = ["red", "orange", "purple", "springgreen", "y", "green", "blue", "c"]
        linestyles = ['-', '--', '-.', ':']
        root_dict = {}
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
                norm_frac = (n_data - w_bkg *
                             data_shape(bg)) / np.sum(total_weight)
            weights = amp.partial_weight(phsp)
            plot_var_dic = {}
            for conf in self.plot_params.get_params():
                name = conf.get("name")
                display = conf.get("display", name)
                upper_ylim = conf.get("upper_ylim", None)
                idx = conf.get("idx")
                trans = conf.get("trans", lambda x: x)
                has_lengend = conf.get("legend", False)
                xrange = conf.get("range", None)
                bins = conf.get("bins", None)
                units = conf.get("units", "")
                fig = plt.figure()
                if plot_delta or plot_pull:
                    ax = plt.subplot2grid((4, 1), (0, 0),  rowspan=3)
                else:
                    ax = fig.add_subplot(1, 1, 1)
                data_i = trans(data_index(data, idx))
                if xrange is None:
                    xrange = [np.min(data_i) - 0.1, np.max(data_i) + 0.1]
                phsp_i = trans(data_index(phsp, idx))
                data_weights = data.get("weight", [1.0]*data_shape(data))
                data_x, data_y, data_err = hist_error(data_i, bins=bins, weights=data_weights,xrange=xrange)
                ax.errorbar(data_x, data_y, yerr=data_err, fmt=".",
                            zorder=-2, label="data", color="black")  #, capsize=2)
                if bg is not None:
                    bg_i = trans(data_index(bg, idx))
                    bg_weight = np.ones_like(bg_i)*w_bkg
                    ax.hist(bg_i, weights=bg_weight,
                            label="back ground", bins=bins, range=xrange, histtype="stepfilled", alpha=0.5, color="grey")
                    mc_i = np.concatenate([bg_i, phsp_i])
                    mc_weights = np.concatenate([bg_weight, total_weight*norm_frac])
                    fit_y, fit_x, _ = ax.hist(mc_i, weights=mc_weights, range=xrange,
                                              histtype="step", label="total fit", bins=bins, color="black")
                else:
                    mc_i = phsp_i
                    mc_weights = total_weight*norm_frac
                    fit_y, fit_x, _ = ax.hist(phsp_i, weights=mc_weights, range=xrange, histtype="step", 
                                              label="total fit", bins=bins, color="black")
                # plt.hist(data_i, label="data", bins=50, histtype="step")
                style = itertools.product(colors, linestyles)
                for i, j in enumerate(weights):
                    # print(phsp.get("weight", 1.0))
                    x, y = hist_line(phsp_i, weights=j * norm_frac*bin_scale*phsp.get("weight", 1.0), xrange=xrange, bins=bins*bin_scale)
                    label, curve_style = self.get_chain_property(i)
                    if curve_style is None:
                        color, ls = next(style)
                        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1)
                    else:
                        ax.plot(x, y, curve_style, label=label, linewidth=1)

                ax.set_ylim((0, upper_ylim))
                xlimin, xlimax = ax.set_xlim(xrange)
                if has_lengend:
                    ax.legend(frameon=False, labelspacing=0.1, borderpad=0.0)
                ax.set_title(display)
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
                fig.savefig(prefix+name, dpi=300)
                if save_pdf:
                    fig.savefig(prefix+name+".pdf", dpi=300)
                print("Finish plotting "+prefix+name)
                plt.close(fig)
                plot_var_dic[name] = {"idx": idx, "trans": trans, "range": [xlimin, xlimax]}
                root_dict[name] = data_i
                root_dict[name+"_weights"] = data_weights
                #root_dict[name+"_mc"] = mc_i
                #root_dict[name+"_mc_weight"] = mc_weights
                #if bg is not None:
                    #root_dict[name+"_bg"] = bg_i
                    #root_dict[name+"_bg_weights"] = bg_weight

            twodplot = self.config["plot"].get("2Dplot", {})
            for k, i in twodplot.items():
                var1, var2 = k.split('&')
                var1 = var1.rstrip()
                var2 = var2.lstrip()
                k = var1 + "_vs_" + var2
                display = i["display"]
                plot_figs = i["plot_figs"]
                name1, name2 = display.split('vs')
                name1 = name1.rstrip()
                name2 = name2.lstrip()
                idx1 = plot_var_dic[var1]["idx"]
                trans1 = plot_var_dic[var1]["trans"]
                range1 = plot_var_dic[var1]["range"]
                data_1 = trans1(data_index(data, idx1))
                phsp_1 = trans1(data_index(phsp, idx1))
                idx2 = plot_var_dic[var2]["idx"]
                trans2 = plot_var_dic[var2]["trans"]
                range2 = plot_var_dic[var2]["range"]
                data_2 = trans2(data_index(data, idx2))
                phsp_2 = trans2(data_index(phsp, idx2))
                # data
                if "data" in plot_figs:
                    plt.scatter(data_1,data_2,s=1,alpha=0.8,label='data')
                    plt.xlabel(name1); plt.ylabel(name2); plt.title(display); plt.legend()
                    plt.xlim(range1); plt.ylim(range2)
                    plt.savefig(prefix+k+'_data')
                    plt.clf()
                    print("Finish plotting 2D data "+prefix+k)
                # sideband
                if "sideband" in plot_figs:
                    if bg is None:
                        print("There's no bkg input")
                    else:
                        bg_1 = trans1(data_index(bg, idx1))
                        bg_2 = trans2(data_index(bg, idx2))
                        plt.scatter(bg_1,bg_2,s=1,c='g',alpha=0.8,label='sideband')
                        plt.xlabel(name1); plt.ylabel(name2); plt.title(display); plt.legend()
                        plt.xlim(range1); plt.ylim(range2)
                        plt.savefig(prefix+k+'_bkg')
                        plt.clf()
                        print("Finish plotting 2D sideband "+prefix+k)
                # fit pdf
                if "fitted" in plot_figs:
                    plt.hist2d(phsp_1,phsp_2,bins=100,weights=total_weight*norm_frac)
                    plt.xlabel(name1); plt.ylabel(name2); plt.title(display); plt.colorbar()
                    plt.xlim(range1); plt.ylim(range2)
                    plt.savefig(prefix+k+'_fitted')
                    plt.clf()
                    print("Finish plotting 2D fitted "+prefix+k)
        if has_uproot and save_root:
            save_dict_to_root(root_dict, file_name=prefix+"variables.root", tree_name="Tree")
            print("Save root file "+prefix+"variables.root")

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
        model = self.get_model()
        frac, err_frac = fit_fractions(model, mcdata, self.inv_he, params, batch)
        return frac, err_frac

    def get_params(self):
        return self.get_amplitude().get_params()

    def set_params(self, params):
        if isinstance(params, str):
            with open(params) as f:
                params = yaml.safe_load(f)
        if hasattr(params, "params"):
            params = params.params
        if isinstance(params, dict):
            if "value" in params:
                params = params["value"]
        self.get_amplitude().set_params(params)


def validate_file_name(s):
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    name = re.sub(rstr, "_", s)
    return name


class MultiConfig(object):
    def __init__(self, file_names, vm=None, total_same=False):
        if vm is None:
            self.vm = VarsManager()
            print(self.vm)
        else:
            self.vm = vm
        self.total_same = total_same
        self.configs = [ConfigLoader(i, vm=self.vm) for i in file_names]

    def get_amplitudes(self, vm=None):
        if not self.total_same:
            amps = [j.get_amplitude(name="s"+str(i), vm=vm)
                    for i, j in enumerate(self.configs)]
        else:
            amps = [j.get_amplitude(vm=vm) for j in self.configs]
        return amps

    def get_models(self, vm=None):
        if not self.total_same:
            models = [j.get_model(name="s"+str(i), vm=vm)
                      for i, j in enumerate(self.configs)]
        else:
            models = [j.get_model(vm=vm) for j in self.configs]
        return models

    def get_fcns(self, vm=None, batch=65000):
        if not self.total_same:
            fcns = [j.get_fcn(name="s"+str(i), vm=vm, batch=batch)
                    for i, j in enumerate(self.configs)]
        else:
            fcns = [j.get_fcn(vm=vm, batch=batch) for j in self.configs]
        return fcns

    def get_fcn(self, vm=None, batch=65000):
        fcns = self.get_fcns(vm=vm, batch=batch)
        return CombineFCN(fcns=fcns)

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

    def fit(self, batch=65000, method="BFGS"):
        fcn = self.get_fcn()
        print("\n########### initial parameters")
        print(json.dumps(fcn.get_params(), indent=2))
        print("initial NLL: ", fcn({}))
        # fit configure
        bounds_dict = {}
        args_name, x0, args, bnds = self.get_args_value(bounds_dict)

        points = []
        nlls = []
        now = time.time()
        maxiter = 1000
        min_nll = 0.0
        ndf = 0

        if method in ["BFGS", "CG", "Nelder-Mead"]:
            def callback(x):
                if np.fabs(x).sum() > 1e7:
                    x_p = dict(zip(args_name, x))
                    raise Exception("x too large: {}".format(x_p))
                points.append(self.vm.get_all_val())
                nlls.append(float(fcn.cached_nll))
                # if len(nlls) > maxiter:
                #    with open("fit_curve.json", "w") as f:
                #        json.dump({"points": points, "nlls": nlls}, f, indent=2)
                #    pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
                print(fcn.cached_nll)

            self.vm.set_bound(bounds_dict)
            f_g = self.vm.trans_fcn_grad(fcn.nll_grad)
            s = minimize(f_g, np.array(self.vm.get_all_val(True)), method=method,
                         jac=True, callback=callback, options={"disp": 1, "gtol": 1e-4, "maxiter": maxiter})
            xn = s.x  # self.vm.get_all_val()  # bd.get_y(s.x)
            ndf = s.x.shape[0]
            min_nll = s.fun
            if hasattr(s, "hess_inv"):
                self.inv_he = s.hess_inv
            success = s.success
        elif method in ["L-BFGS-B"]:
            def callback(x):
                if np.fabs(x).sum() > 1e7:
                    x_p = dict(zip(args_name, x))
                    raise Exception("x too large: {}".format(x_p))
                points.append([float(i) for i in x])
                nlls.append(float(fcn.cached_nll))

            s = minimize(fcn.nll_grad, np.array(x0), method=method, jac=True, bounds=bnds, callback=callback,
                         options={"disp": 1, "maxcor": 10000, "ftol": 1e-15, "maxiter": maxiter})
            xn = s.x
            ndf = s.x.shape[0]
            min_nll = s.fun
            success = s.success
        elif method in ["iminuit"]:
            from .fit import fit_minuit
            m = fit_minuit(fcn)
            return m
        else:
            raise Exception("unknown method")
        self.vm.set_all(xn)
        params = self.vm.get_all_dic()
        return FitResult(params, fcn, min_nll, ndf=ndf, success=success)

    def cal_error(self, params=None, batch=10000):
        if params is None:
            params = {}
        if hasattr(params, "params"):
            params = getattr(params, "params")
        fcn = self.get_fcn(batch=batch)
        t = time.time()
        # data_w,mcdata,weight=weights,batch=50000)
        nll, g, h = fcn.nll_grad_hessian(params)
        print("Time for calculating errors:", time.time() - t)
        # print(nll)
        # print([i.numpy() for i in g])
        # print(h.numpy())
        self.inv_he = np.linalg.pinv(h.numpy())
        np.save("error_matrix.npy", self.inv_he)
        # print("edm:",np.dot(np.dot(inv_he,np.array(g)),np.array(g)))
        return self.inv_he

    def get_params_error(self, params=None, batch=10000):
        if params is None:
            params = {}
        if hasattr(params, "params"):
            params = getattr(params, "params")
        self.inv_he = self.cal_error(params, batch=20000)
        diag_he = self.inv_he.diagonal()
        print("parameters order")
        print(self.vm.trainable_vars)
        print("error matrix:")
        print(self.inv_he)
        diag_he_s = np.sqrt(np.fabs(diag_he))
        print("correlation matrix:")
        print(self.inv_he /(diag_he_s[:, np.newaxis] * diag_he_s[np.newaxis, :]))
        hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
        print("hesse_error:", hesse_error)
        err = dict(zip(self.vm.trainable_vars, hesse_error))
        return err

    def get_params(self, trainable_only=True):
        _amps = self.get_amplitudes()
        return self.vm.get_all_dic(trainable_only)

    def set_params(self, params):
        _amps = self.get_amplitudes()
        if isinstance(params, str):
            with open(params) as f:
                params = yaml.safe_load(f)
        if hasattr(params, "params"):
            params = params.params
        if isinstance(params, dict):
            if "value" in params:
                params = params["value"]
        self.vm.set_all(params)


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


class PlotParams(dict):
    def __init__(self, plot_config, decay_struct):
        self.config = plot_config
        self.defaults_config = self.config.get("config", {}) #???
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
            bins = v.get("bins", self.defaults_config.get("bins", 50))
            yield {"name": "m_"+k, "display": display, "upper_ylim": upper_ylim,
                   "idx": ("particle", self.re_map.get(get_particle(k), get_particle(k)), "m"),
                   "legend": True, "range": xrange, "bins": bins,
                   "trans": trans, "units": "GeV"}

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
                yield {"name": validate_file_name(k+"_"+j), "display": display, "upper_ylim": upper_ylim,
                       "idx": ("decay", decay_chain, decay, part, "ang", theta),
                       "trans": trans, "bins": bins, "range": xrange}

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
