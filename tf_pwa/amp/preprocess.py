import warnings

import tensorflow as tf

from tf_pwa.cal_angle import (
    CalAngleData,
    cal_angle_from_momentum,
    parity_trans,
)
from tf_pwa.config import create_config, get_config, regist_config, temp_config
from tf_pwa.data import HeavyCall, data_strip

PREPROCESSOR_MODEL = "preprocessor_model"
regist_config(PREPROCESSOR_MODEL, {})


def register_preprocessor(name=None, f=None):
    """register a data mode

    :params name: mode name used in configuration
    :params f: Data Mode class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(PREPROCESSOR_MODEL)
        if my_name in config:
            warnings.warn("Override mode {}".format(my_name))
        config[my_name] = g
        return g

    if f is None:
        return regist
    return regist(f)


def create_preprocessor(decay_group, **kwargs):
    mode = kwargs.get("model", "default")
    return get_config(PREPROCESSOR_MODEL)[mode](decay_group, **kwargs)


@register_preprocessor("default")
class BasePreProcessor(HeavyCall):
    def __init__(
        self, decay_struct, root_config=None, model="defualt", **kwargs
    ):
        self.decay_struct = decay_struct
        self.kwargs = kwargs
        self.model = model
        self.root_config = root_config

    def __call__(self, x):
        p4 = x["p4"]
        if self.kwargs.get("cp_trans", False):
            charges = x.get("extra", {}).get("charge_conjugation", None)
            p4 = {k: parity_trans(v, charges) for k, v in p4.items()}
        kwargs = {}
        for k in [
            "center_mass",
            "r_boost",
            "random_z",
            "align_ref",
            "only_left_angle",
        ]:
            if k in self.kwargs:
                kwargs[k] = self.kwargs[k]
        ret = cal_angle_from_momentum(p4, self.decay_struct, **kwargs)
        return ret


def list_to_tuple(data):
    if isinstance(data, list):
        return tuple([list_to_tuple(i) for i in data])
    return data


@register_preprocessor("cached_amp")
class CachedAmpPreProcessor(BasePreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp = self.root_config.get_amplitude()
        self.decay_group = self.amp.decay_group
        self.no_angle = self.kwargs.get("no_angle", False)
        self.no_p4 = self.kwargs.get("no_p4", False)

    def build_cached(self, x):
        from tf_pwa.experimental.build_amp import build_angle_amp_matrix

        x2 = super().__call__(x)
        for k, v in x["extra"].items():
            x2[k] = v  # {**x2, **x["extra"]}
        # print(x["c"])
        idx, c_amp = build_angle_amp_matrix(self.decay_group, x2)
        x2["cached_amp"] = list_to_tuple(c_amp)
        # print(x)
        return x2

    def strip_data(self, x):
        strip_var = []
        if self.no_angle:
            strip_var += ["ang", "aligned_angle"]
        if self.no_p4:
            strip_var += ["p"]
        if strip_var:
            x = data_strip(x, strip_var)
        return x

    def __call__(self, x):
        extra = x["extra"]
        x = self.build_cached(x)
        x = self.strip_data(x)
        for k in extra:
            del x[k]
        return x


@register_preprocessor("cached_shape")
class CachedShapePreProcessor(CachedAmpPreProcessor):
    def build_cached(self, x):
        from tf_pwa.experimental.build_amp import build_params_vector

        # old_chains_idx = self.decay_group.chains_idx
        cached_shape_idx = self.amp.get_cached_shape_idx()
        # used_chains_idx = [i for i in old_chains_idx if i not in cached_shape_idx]
        # self.decay_group.set_used_chains(used_chains_idx)
        x = super().build_cached(x)
        # self.decay_group.set_used_chains(old_chains_idx)

        old_cached_amp = list(x["cached_amp"])
        dec = self.decay_group

        used_chains = dec.chains_idx
        dec.set_used_chains(cached_shape_idx)
        with self.amp.temp_total_gls_one():
            pv = build_params_vector(dec, x)
        hij = []
        for k, i in zip(cached_shape_idx, pv):
            tmp = old_cached_amp[k]
            a = tf.reshape(i, [-1, i.shape[1]] + [1] * (len(tmp[0].shape) - 1))
            old_cached_amp[k] = a * tf.stack(tmp, axis=1)
        dec.set_used_chains(used_chains)
        x["cached_amp"] = list_to_tuple(old_cached_amp)
        return x


@register_preprocessor("cached_angle")
class CachedAnglePreProcessor(BasePreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp = self.root_config.get_amplitude()
        self.decay_group = self.amp.decay_group
        self.no_angle = self.kwargs.get("no_angle", False)
        self.no_p4 = self.kwargs.get("no_p4", False)

    def build_cached(self, x):
        x2 = super().__call__(x)
        for k, v in x["extra"].items():
            x2[k] = v  # {**x2, **x["extra"]}
        c_amp = self.decay_group.get_factor_angle_amp(x2)
        x2["cached_angle"] = list_to_tuple(c_amp)
        # print(x)
        return x2


@register_preprocessor("p4_directly")
class CachedAmpPreProcessor(BasePreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return {"p4": x["p4"]}
