import functools
import os
import re
import warnings

import numpy as np
import tensorflow as tf

from tf_pwa.amp import get_particle
from tf_pwa.amp.preprocess import create_preprocessor
from tf_pwa.cal_angle import (
    cal_angle_from_momentum,
    load_dat_file,
    parity_trans,
)
from tf_pwa.config import create_config, get_config, regist_config, temp_config
from tf_pwa.config_loader.decay_config import DecayConfig
from tf_pwa.data import (
    LazyCall,
    LazyFile,
    data_index,
    data_shape,
    data_split,
    data_to_numpy,
    data_to_tensor,
    load_data,
    save_data,
)

DATA_MODE = "data_mode"
regist_config(DATA_MODE, {})


def register_data_mode(name=None, f=None):
    """register a data mode

    :params name: mode name used in configuration
    :params f: Data Mode class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(DATA_MODE)
        if my_name in config:
            warnings.warn("Override mode {}".format(my_name))
        config[my_name] = g
        return g

    if f is None:
        return regist
    return regist(f)


def load_data_mode(dic, decay_struct, default_mode="multi", config=None):
    if dic is None:
        dic = {}
    mode = dic.get("format", None)
    if mode is None:
        mode = dic.get("mode", default_mode)
    return get_config(DATA_MODE)[mode](dic, decay_struct, config=config)


@register_data_mode("simple")
class SimpleData:
    def __init__(self, dic, decay_struct, config=None):
        self.decay_struct = decay_struct
        self.root_config = config
        self.dic = dic
        self.extra_var = {
            "weight": {"default": 1},
            "charge": {"key": "charge_conjugation", "default": 1},
        }
        if self.dic.get("model", "default") == "cfit":
            self.extra_var.update(
                {"bg_value": {"default": 1}, "eff_value": {"default": 1}}
            )
        self.extra_var.update(self.dic.get("extra_var", {}))
        self.cached_data = None
        chain_map = self.decay_struct.get_chains_map()
        self.re_map = {}
        for i in chain_map:
            for _, j in i.items():
                for k, v in j.items():
                    self.re_map[v] = k
        self.scale_list = self.dic.get("scale_list", ["bg"])
        self.lazy_call = self.dic.get("lazy_call", False)
        self.lazy_file = self.dic.get("lazy_file", False)
        cp_trans = self.dic.get("cp_trans", True)
        center_mass = self.dic.get("center_mass", False)
        r_boost = self.dic.get("r_boost", True)
        random_z = self.dic.get("random_z", True)
        align_ref = self.dic.get("align_ref", None)
        only_left_angle = self.dic.get("only_left_angle", False)
        preprocessor_model = self.dic.get("preprocessor", "default")
        no_p4 = self.dic.get("no_p4", False)
        no_angle = self.dic.get("no_angle", False)
        self.preprocessor = create_preprocessor(
            decay_struct,
            center_mass=center_mass,
            r_boost=r_boost,
            random_z=random_z,
            align_ref=align_ref,
            only_left_angle=only_left_angle,
            root_config=self.root_config,
            model=preprocessor_model,
            no_p4=no_p4,
            no_angle=no_angle,
            cp_trans=cp_trans,
        )

    def get_data_file(self, idx):
        if idx in self.dic:
            ret = self.dic[idx]
        else:
            ret = None
        return ret

    def get_dat_order(self, standard=False):
        order = self.dic.get("dat_order", None)
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

    def get_weight_sign(self, idx):
        negtive_idx = self.dic.get("negtive_idx", ["bg*"])
        weight_sign = 1
        for i in negtive_idx:
            if re.match(i, idx):
                weight_sign = -1
        return weight_sign

    def get_data(self, idx) -> dict:
        if self.cached_data is not None:
            data = self.cached_data.get(idx, None)
            if data is not None:
                return data
        files = self.get_data_file(idx)
        weights = self.dic.get(idx + "_weight", None)
        weight_sign = self.get_weight_sign(idx)
        charge = self.dic.get(idx + "_charge", None)
        ret = self.load_data(
            files, weight_sign=weight_sign, weight=weights, charge=charge
        )
        ret = self.process_scale(idx, ret)
        return ret

    def process_scale(self, idx, data):
        if idx in self.scale_list and self.dic.get("weight_scale", False):
            n_bg = data_shape(data)
            scale_factor = self.get_n_data() / n_bg
            data["weight"] = (
                data.get("weight", np.ones((n_bg,))) * scale_factor
            )
        return data

    def set_lazy_call(self, data, idx):
        if isinstance(data, LazyCall):
            name = idx
            cached_file = self.dic.get("cached_lazy_call", None)
            prefetch = self.dic.get("lazy_prefetch", -1)
            data.set_cached_file(cached_file, name)
            data.prefetch = prefetch

    def get_n_data(self):
        data = self.get_data("data")
        weight = data.get("weight", np.ones((data_shape(data),)))
        return np.sum(weight)

    def load_p4(self, fnames):
        particles = self.get_dat_order()
        mmap_mode = "r" if self.lazy_file else None
        p = load_dat_file(fnames, particles, mmap_mode=mmap_mode)
        return p

    def cal_angle(self, p4, **kwargs):
        if isinstance(p4, (list, tuple)):
            p4 = {k: v for k, v in zip(self.get_dat_order(), p4)}
        # charge = kwargs.get("charge_conjugation", None)
        # p4 = self.process_cp_trans(p4, charge)
        if self.lazy_call:
            if self.lazy_file:
                data = LazyCall(
                    self.preprocessor, LazyFile({"p4": p4, "extra": kwargs})
                )
            else:
                data = LazyCall(self.preprocessor, {"p4": p4, "extra": kwargs})
        else:
            data = self.preprocessor({"p4": p4, "extra": kwargs})
        return data

    def load_extra_var(self, n_data, **kwargs):
        extra_var = {}
        for k, v in self.extra_var.items():
            value = kwargs.get(k, None)
            if value is None:
                value = v.get("default", 1)
            if isinstance(value, (int, float)):
                value = np.ones((n_data,)) * value
            elif isinstance(value, (list, str)):
                value = self.load_weight_file(value)
                value = value[:n_data]
            else:
                raise NotImplemented
            extra_var[v.get("key", k)] = value
        return extra_var

    def load_data(self, files, weight_sign=1, **kwargs) -> dict:
        # print(files, weights)
        if files is None:
            return None
        order = self.get_dat_order()
        p4 = self.load_p4(files)
        n_data = data_shape(p4)
        extra_var = self.load_extra_var(n_data, **kwargs)
        extra_var["weight"] = weight_sign * extra_var["weight"]
        data = self.cal_angle(p4, **extra_var)
        for k, v in extra_var.items():
            data[k] = v
        return data

    def load_weight_file(self, weight_files):
        ret = []
        if isinstance(weight_files, list):
            for i in weight_files:
                if i.endswith(".npy"):
                    data = np.load(i).reshape((-1,))
                else:
                    data = np.loadtxt(i).reshape((-1,))
                ret.append(data)
        elif isinstance(weight_files, str):
            if weight_files.endswith(".npy"):
                data = np.load(weight_files).reshape((-1,))
            else:
                data = np.loadtxt(weight_files).reshape((-1,))
            ret.append(data)
        else:
            raise TypeError(
                "weight files must be string of list of strings, not {}".format(
                    type(weight_files)
                )
            )
        if len(ret) == 1:
            return ret[0]
        return np.concatenate(ret)

    def load_cached_data(self, file_name=None):
        if file_name is None:
            file_name = self.dic.get("cached_data", None)
        if file_name is not None and os.path.exists(file_name):
            if self.cached_data is None:
                self.cached_data = load_data(file_name)
                print("load cached_data {}".format(file_name))

    def save_cached_data(self, data, file_name=None):
        if file_name is None:
            file_name = self.dic.get("cached_data", None)
        if file_name is not None:
            if not os.path.exists(file_name):
                save_data(file_name, data)
                print("save cached_data {}".format(file_name))

    def get_all_data(self):
        datafile = ["data", "phsp", "bg", "inmc"]
        self.load_cached_data()
        data, phsp, bg, inmc = [self.get_data(i) for i in datafile]
        self.save_cached_data(dict(zip(datafile, [data, phsp, bg, inmc])))
        return data, phsp, bg, inmc

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

    def get_phsp_noeff(self):
        if "phsp_noeff" in self.dic:
            phsp_noeff = self.get_data("phsp_noeff")
            return phsp_noeff
        warnings.warn(
            "No data file as 'phsp_noeff', using the first 'phsp' file instead."
        )
        return self.get_data("phsp")

    def get_phsp_plot(self):
        if "phsp_plot" in self.dic:
            return self.get_data("phsp_plot")
        return self.get_data("phsp")

    def savetxt(self, file_name, data):
        if isinstance(data, dict):
            dat_order = self.get_dat_order()
            if "particle" in data:
                p4 = [
                    data_index(data, ("particle", i, "p")) for i in dat_order
                ]
            else:
                p4 = [data_index(data, i) for i in dat_order]
        elif isinstance(data, (tuple, list)):
            p4 = data
        else:
            raise ValueError("not support data")
        p4 = data_to_numpy(p4)
        p4 = np.stack(p4).transpose((1, 0, 2))
        if file_name.endswith("npy"):
            np.save(file_name, p4)
        else:
            np.savetxt(file_name, p4.reshape((-1, 4)))


@register_data_mode("multi")
class MultiData(SimpleData):
    def __init__(self, *args, **kwargs):
        super(MultiData, self).__init__(*args, **kwargs)
        self._Ngroup = 0

    def process_scale(self, idx, data):
        if idx in self.scale_list and self.dic.get("weight_scale", False):
            for i, data_i in enumerate(data):
                n_bg = data_shape(data_i)
                scale_factor = self.get_n_data()[i] / n_bg
                data_i["weight"] = (
                    data_i.get("weight", np.ones((n_bg,))) * scale_factor
                )
        return data

    def set_lazy_call(self, data, idx):
        for i, data_i in enumerate(data):
            super().set_lazy_call(data_i, "s{}{}".format(i, idx))

    def get_n_data(self):
        data = self.get_data("data")
        weight = [
            data_i.get("weight", np.ones((data_shape(data_i),)))
            for data_i in data
        ]
        return [np.sum(weight_i) for weight_i in weight]

    @functools.lru_cache()
    def get_data(self, idx) -> list:
        if self.cached_data is not None:
            data = self.cached_data.get(idx, None)
            if data is not None:
                return data
        files = self.get_data_file(idx)
        if files is None:
            return None
        if not isinstance(files[0], list):
            files = [files]
        weight_sign = self.get_weight_sign(idx)
        kwargs = [{} for i in range(len(files))]
        for k in self.extra_var:
            tmp = self.dic.get(idx + "_" + k, None)
            if tmp is None:
                tmp = [None] * len(kwargs)
            if isinstance(tmp, (float, int, str)):
                tmp = [tmp] * len(kwargs)
            if isinstance(tmp, list):
                for i in range(len(kwargs)):
                    kwargs[i][k] = tmp[i]
            else:
                raise NotImplementedError
        ret = [
            self.load_data(i, weight_sign=weight_sign, **k)
            for i, k in zip(files, kwargs)
        ]
        if self._Ngroup == 0:
            self._Ngroup = len(ret)
        elif idx != "phsp_noeff":
            assert self._Ngroup == len(ret), "not the same data group"
        ret = self.process_scale(idx, ret)
        self.set_lazy_call(ret, idx)
        return ret

    def get_phsp_noeff(self):
        if "phsp_noeff" in self.dic:
            phsp_noeff = self.get_data("phsp_noeff")
            assert len(phsp_noeff) == 1
            return phsp_noeff[0]
        warnings.warn(
            "No data file as 'phsp_noeff', using the first 'phsp' file instead."
        )
        return self.get_data("phsp")[0]
