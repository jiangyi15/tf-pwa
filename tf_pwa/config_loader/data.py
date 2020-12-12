import os
import warnings

import numpy as np

from tf_pwa.amp import get_particle
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.config import create_config, get_config, regist_config, temp_config
from tf_pwa.data import (
    data_index,
    data_shape,
    data_split,
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
            warnings.warn("Override mode {}", my_name)
        config[my_name] = g
        return g

    if f is None:
        return regist
    return regist(f)


def load_data_mode(dic, decay_struct, default_mode="multi"):
    if dic is None:
        dic = {}
    mode = dic.get("mode", default_mode)
    return get_config(DATA_MODE)[mode](dic, decay_struct)


@register_data_mode("simple")
class SimpleData:
    def __init__(self, dic, decay_struct):
        self.decay_struct = decay_struct
        self.dic = dic
        self.cached_data = None
        chain_map = self.decay_struct.get_chains_map()
        self.re_map = {}
        for i in chain_map:
            for _, j in i.items():
                for k, v in j.items():
                    self.re_map[v] = k

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
        negtive_idx = self.dic.get("negtive_idx", ["bg"])
        weight_sign = 1
        if idx in negtive_idx:
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
        return self.load_data(files, weights, weight_sign, charge)

    def load_data(
        self, files, weights=None, weights_sign=1, charge=None
    ) -> dict:
        # print(files, weights)
        if files is None:
            return None
        order = self.get_dat_order()
        center_mass = self.dic.get("center_mass", True)
        r_boost = self.dic.get("r_boost", False)
        random_z = self.dic.get("random_z", False)
        data = prepare_data_from_decay(
            files,
            self.decay_struct,
            order,
            center_mass=center_mass,
            r_boost=r_boost,
            random_z=random_z,
        )
        if weights is not None:
            if isinstance(weights, float):
                data["weight"] = np.array(
                    [weights * weights_sign] * data_shape(data)
                )
            elif isinstance(weights, str):  # weight files
                weight = self.load_weight_file(weights)
                data["weight"] = weight[: data_shape(data)] * weights_sign
            else:
                raise TypeError(
                    "weight format error: {}".format(type(weights))
                )
        if charge is not None:
            charges = self.load_weight_file(charge)
            data["charge_conjugation"] = charges[: data_shape(data)]
        else:
            data["charge_conjugation"] = np.ones((data_shape(data),))
        return data

    def load_weight_file(self, weight_files):
        ret = []
        if isinstance(weight_files, list):
            for i in weight_files:
                data = np.loadtxt(i).reshape((-1,))
                ret.append(data)
        elif isinstance(weight_files, str):
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


@register_data_mode("multi")
class MultiData(SimpleData):
    def __init__(self, *args, **kwargs):
        super(MultiData, self).__init__(*args, **kwargs)
        self._Ngroup = 0

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
        weights = self.dic.get(idx + "_weight", None)
        if weights is None:
            weights = [None] * len(files)
        elif isinstance(weights, float):
            weights = [weights] * len(files)
        weight_sign = self.get_weight_sign(idx)
        charge = self.dic.get(idx + "_charge", None)
        if charge is None:
            charge = [None] * len(files)
        elif not isinstance(charge[0], list):
            charge = [charge]
        ret = [
            self.load_data(i, j, weight_sign, k)
            for i, j, k in zip(files, weights, charge)
        ]
        if self._Ngroup == 0:
            self._Ngroup = len(ret)
        elif idx != "phsp_noeff":
            assert self._Ngroup == len(ret), "not the same data group"
        bg_value = self.dic.get(idx + "_bg_value", None)
        if bg_value is not None:
            if isinstance(bg_value, str):
                bg_value = [bg_value]
            for i, file_name in enumerate(bg_value):
                ret[i]["bg_value"] = np.reshape(np.loadtxt(file_name), (-1,))
        eff_value = self.dic.get(idx + "_eff_value", None)
        if eff_value is not None:
            if isinstance(eff_value, str):
                eff_value = [eff_value]
            for i, file_name in enumerate(eff_value):
                ret[i]["eff_value"] = np.reshape(np.loadtxt(file_name), (-1,))
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
