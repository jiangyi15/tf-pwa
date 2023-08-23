import numpy as np
import sympy
import tensorflow as tf

from tf_pwa.config_loader.data import MultiData, register_data_mode
from tf_pwa.data import data_mask
from tf_pwa.root_io import uproot, uproot_version


def build_matrix(order, matrix):
    if len(order) == 0:
        yield {}
    else:
        idx = order[0]
        if isinstance(idx, str):
            for k in matrix[idx]:
                tmp = {idx: k}
                for x in build_matrix(order[1:], matrix):
                    yield {**tmp, **x}
        elif isinstance(idx, list):
            for k in zip(*[matrix[i] for i in idx]):
                tmp = dict(zip(idx, k))
                for x in build_matrix(order[1:], matrix):
                    yield {**tmp, **x}
        else:
            raise TypeError(f"not supported type {type(idx)}")


def touch_var(name, data, var, size, default=1):
    for i, v, s in zip(data, var, size):
        if v is None:
            v = default
        if isinstance(v, (float, int)):
            i[name] = v * np.ones(s)
        else:
            i[name] = v
    return data


def custom_cond(x, dic, key=None):
    if key is None:
        key = list(dic.keys())
    if len(key) == 0:
        return np.zeros_like(x)
    return np.where(x == key[0], dic[key[0]], custom_cond(x, dic, key[1:]))


def cut_data(data):
    mask = data["weight"] != 0
    return data_mask(data, mask)


@register_data_mode("root_lhcb")
class RootData(MultiData):
    def create_data(self, p4, **kwargs):
        ret = self.cal_angle(p4, **kwargs)
        for k, v in kwargs.items():
            ret[k] = v
        return ret

    def get_data(self, idx):
        if uproot_version < 4:
            print("uproot < 4 is not support")
            return None
        if idx not in self.dic:
            return None
        p4 = self.get_p4(idx)
        n_data = [i.shape[0] for i in p4]
        p4 = [list(np.moveaxis(i, 1, 0)) for i in p4]
        weight = self.get_weight(idx)
        ret = [{"p4": i} for i in p4]
        # touch_var("weight", ret, weight, n_data)
        # print(idx, weight)
        # touch_var("charge_conjugation", ret, self.load_var(idx, "_charge"), n_data)
        for k, v in self.extra_var.items():
            touch_var(
                v.get("key", k),
                ret,
                self.load_var(idx, "_" + k),
                n_data,
                v.get("default", 1),
            )
        ret = [cut_data(i) for i in ret]
        ret = [self.create_data(**i) for i in ret]
        return ret

    def load_var(self, idx, tail):
        matrix = self.dic["matrix"]
        matrix_order = self.dic["matrix_order"]
        file_name = self.dic[idx]

        ret = []

        custom_function = {
            "float": lambda x: np.array(x).astype(np.float64),
            "int": lambda x: np.array(x).astype(np.int32),
            "cond": custom_cond,
        }

        for i, file_name_part in enumerate(
            build_matrix(matrix_order[:-2], matrix)
        ):
            expr = self.dic[idx + tail].format(**file_name_part)
            expr = sympy.simplify(expr)
            var = list(expr.free_symbols)
            tmp = {}
            custom_function["select"] = lambda x: x[i]
            with uproot.open(file_name.format(**file_name_part)) as t:
                for name in var:
                    b = t.get(str(name))
                    if b is None:
                        print("not found", name)
                        continue
                    tmp[str(name)] = b.array(library="np")
            ret.append(
                sympy.lambdify(var, expr, modules=[custom_function, "numpy"])(
                    **tmp
                )
            )
        return ret

    def get_weight(self, idx):
        return self.load_var(idx, "_weight")

    def get_p4(self, idx):
        matrix = self.dic["matrix"]
        matrix_order = self.dic["matrix_order"]
        file_name = self.dic[idx]
        p4_name = self.dic[idx + "_var"]
        scale = self.dic.get("unit_scale", 0.001)
        ret = []
        for file_name_part in build_matrix(matrix_order[:-2], matrix):
            tmp = []
            with uproot.open(file_name.format(**file_name_part)) as t:
                for pname in build_matrix(matrix_order[-3:], matrix):
                    tmp.append(
                        t.get(p4_name.format(**pname)).array(library="np")
                    )
            ret.append(
                scale * np.stack(tmp, axis=-1).reshape((-1, len(tmp) // 4, 4))
            )
        return ret
