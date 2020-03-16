import yaml
import json
from tf_pwa.amp import get_particle, get_decay, DecayChain, DecayGroup, AmplitudeModel
from tf_pwa.particle import split_particle_type
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.model import Model, FCN
import re
import functools
import time
from scipy.interpolate import interp1d
from scipy.optimize import minimize, BFGS, basinhopping
import numpy as np
import matplotlib.pyplot as plt
from tf_pwa.data import data_index, data_shape


class ConfigLoader(object):
    """class for loading config.yml"""

    def __init__(self, file_name):
        self.config = self.load_config(file_name)
        self.particle_key_map = {
            "Par": "P",
            "m0": "mass",
            "g0": "width",
            "J": "J",
            "P": "P",
            "spins": "spins",
            "bw": "model",
            "model": "model"
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

    @staticmethod
    def load_config(file_name):
        with open(file_name) as f:
            ret = yaml.safe_load(f)
        return ret

    def get_data_file(self, idx):
        return self.config["data"][idx]

    def get_dat_order(self):
        order = self.config["data"].get("dat_order", None)
        if order is None:
            order = list(self.decay_struct.outs)
        else:
            order = [get_particle(str(i)) for i in order]
        return order

    def get_data(self, idx):
        files = self.get_data_file(idx)
        order = self.get_dat_order()
        data = prepare_data_from_decay(files, self.decay_struct, order)
        return data

    def get_all_data(self):
        datafile = ["data", "phsp", "bg"]
        return [self.get_data(i) for i in datafile]

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
            if k in key_map:
                ret[key_map[k]] = v
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
        return dec_chain

    def get_data_index(self, sub, name):
        dec = self.decay_struct.topology_structure()
        if sub == "mass":
            p = get_particle(name)
            return "particle", p, "m"
        if sub == "angle":
            de, de_i = None, None
            name_i = name.split("/")
            if len(name_i) > 1:
                _id = int(name_i[-1])
            else:
                _id = 0
            p = get_particle(name_i[0])
            for idx, i in enumerate(dec):
                for j in i:
                    if j.core == p:
                        de = j.core.decay[_id]
                    if j == de:
                        de_i = idx
            if de is None or de_i is None:
                raise ValueError("not found {}".format(name))
            return "decay", de_i, de, de.outs[0], "ang"
        raise ValueError("unknown sub {}".format(sub))

    @functools.lru_cache()
    def get_amplitude(self):
        decay_group = self.full_decay
        return AmplitudeModel(decay_group)

    @functools.lru_cache()
    def get_model(self):
        amp = self.get_amplitude()
        w_bkg = self.config["data"].get("bg_weight", 0.0)
        return Model(amp, w_bkg)

    def fit(self, data, phsp, bg=None, batch=65000, method="BFGS"):
        model = self.get_model()
        for i in self.full_decay:
            print(i)
        fcn = FCN(model, data, phsp, bg=bg, batch=batch)
        print("initial NLL: ", fcn({}))
        # fit configure
        args = {}
        args_name = model.Amp.vm.trainable_vars
        x0 = []
        bnds = []
        bounds_dict = {
            # "Zc(3900)p_mass":(4.1,4.22),
            # "Zc_4160_g:0":(0,None)
        }

        for i in model.Amp.trainable_variables:
            args[i.name] = i.numpy()
            x0.append(i.numpy())
            if i.name in bounds_dict:
                bnds.append(bounds_dict[i.name])
            else:
                bnds.append((None, None))
            args["error_" + i.name] = 0.1

        points = []
        nlls = []
        now = time.time()
        maxiter = 1000

        if method in ["BFGS", "CG", "Nelder-Mead"]:
            def callback(x):
                if np.fabs(x).sum() > 1e7:
                    x_p = dict(zip(args_name, x))
                    raise Exception("x too large: {}".format(x_p))
                points.append(model.Amp.vm.get_all_val())
                nlls.append(float(fcn.cached_nll))
                # if len(nlls) > maxiter:
                #    with open("fit_curve.json", "w") as f:
                #        json.dump({"points": points, "nlls": nlls}, f, indent=2)
                #    pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
                print(fcn.cached_nll)

            #bd = Bounds(bnds)
            fcn.model.Amp.vm.set_bound(bounds_dict)
            f_g = fcn.model.Amp.vm.trans_fcn_grad(fcn.nll_grad)
            s = minimize(f_g, np.array(fcn.model.Amp.vm.get_all_val(True)), method=method,
                         jac=True, callback=callback, options={"disp": 1, "gtol": 1e-4, "maxiter": maxiter})
            xn = fcn.model.Amp.vm.get_all_val()  # bd.get_y(s.x)
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
        else:
            raise Exception("unknown method")
        params = dict(zip(args_name, xn))
        return FitResult(params, fcn)

    def cal_error(self, params, data, phsp, bg=None, batch=10000):
        if hasattr(params, "params"):
            params = getattr(params, "params")
        fcn = FCN(self.get_model(), data, phsp, bg=bg, batch=batch)
        t = time.time()
        # data_w,mcdata,weight=weights,batch=50000)
        nll, g, h = fcn.nll_grad_hessian(params)
        print("Time for calculating errors:", time.time() - t)
        # print(nll)
        # print([i.numpy() for i in g])
        # print(h.numpy())
        inv_he = np.linalg.pinv(h.numpy())
        np.save("error_matrix.npy", inv_he)
        # print("edm:",np.dot(np.dot(inv_he,np.array(g)),np.array(g)))
        return inv_he

    def get_params_error(self, params, data, phsp, bg=None, batch=10000):
        inv_he = self.cal_error(params, data, phsp, bg, batch=20000)
        diag_he = inv_he.diagonal()
        hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
        print(hesse_error)
        model = self.get_model()
        err = dict(zip(model.Amp.vm.trainable_vars, hesse_error))
        return err

    def plot_partial_wave(self, params, data, phsp, bg=None, prefix="figure/"):
        amp = self.get_amplitude()
        w_bkg = self.config["data"].get("bg_weight", 1.0)
        print(w_bkg)
        with amp.temp_params(params):
            total_weight = amp(phsp)
            if bg is None:
                norm_frac = data_shape(data) / np.sum(total_weight)
            else:
                norm_frac = (data_shape(data) - w_bkg *
                             data_shape(bg)) / np.sum(total_weight)
            weights = amp.partial_weight(phsp)
            for name, idx, trans in self.get_plot_params():
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                data_i = trans(data_index(data, idx))
                phsp_i = trans(data_index(phsp, idx))
                if bg is not None:
                    bg_i = trans(data_index(bg, idx))
                    bg_weight = np.ones_like(bg_i)*w_bkg
                    ax.hist(bg_i, weights=bg_weight,
                             label="back ground", bins=50, histtype="stepfilled",alpha=0.5,color="grey")
                    ax.hist(np.concatenate([bg_i, phsp_i]),
                             weights=np.concatenate([bg_weight, total_weight*norm_frac]), histtype="step", label="total fit", bins=50)
                else:
                    ax.hist(phsp, weights=total_weight,
                             label="total fit", bins=50)
                # plt.hist(data_i, label="data", bins=50, histtype="step")
                for i, j in enumerate(weights):
                    x, y = hist_line(phsp_i, weights=j * norm_frac, bins=50)
                    ax.plot(x, y, label=self.get_chain_name(i), linestyle="solid", linewidth=1)
                data_x, data_y, data_err = hist_error(data_i, bins=50)
                ax.errorbar(data_x, data_y, yerr=data_err, fmt=".", zorder = -2, label="data")
                ax.legend()
                ax.set_title(name)
                fig.savefig(prefix+name, dpi=300)
                fig.savefig(prefix+name+".pdf", dpi=300)

    def get_plot_params(self):
        yield "m_BC", ("particle", get_particle("(B, C)"), "m"), lambda x: x
        yield "m_CD", ("particle", get_particle("(C, D)"), "m"), lambda x: x
        yield "m_BD", ("particle", get_particle("(B, D)"), "m"), lambda x: x
        # raise NotImplementedError

    def get_chain(self, i):
        decay_group = self.full_decay
        return list(decay_group)[i]

    def get_chain_name(self, i):
        chain = self.get_chain(i)
        combine = []
        for i in chain:
            if i.core == chain.top:
                combine = list(i.outs)
        names = []
        for i in combine:
            pro = self.particle_property[str(i)]
            names.append(pro.get("display", str(i)))
        return " ".join(names)


def hist_error(data, bins, xrange=None, kind="binomial"):
    data_hist = np.histogram(data, bins=50, range=xrange)
    #ax.hist(fd(data[idx].numpy()),range=xrange,bins=bins,histtype="step",label="data",zorder=99,color="black")
    data_y, data_x = data_hist[0:2]
    data_x = (data_x[:-1]+data_x[1:])/2
    if kind == "possion":
        data_err = np.sqrt(data_y)
    elif kind == "binomial":
        n = data.shape[0]
        p = data_y / n
        data_err = np.sqrt(p*(1-p)*n)
    else:
        raise ValueError("unknown error kind {}".format(kind))
    return data_x, data_y, data_err


def hist_line(data, weights, bins, xrange=None, inter=1, kind ="quadratic"):
    y, x = np.histogram(data, bins=bins, range=xrange, weights=weights)
    x = (x[:-1] + x[1:])/2
    if xrange is None:
        xrange = (np.min(data), np.max(data))
    func = interp1d(x, y, kind=kind)
    num = data.shape[0] * inter
    x_new = np.linspace(np.min(x), np.max(x), num=num, endpoint=True)
    y_new = func(x_new)
    return x_new, y_new


class FitResult(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model

    def save_as(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params, f, indent=2)
