import warnings

from tf_pwa.cal_angle import cal_angle_from_momentum
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
        for k, v in x.get("extra", {}).items():
            ret[k] = v
        return ret


def list_to_tuple(data):
    if isinstance(data, list):
        return tuple([list_to_tuple(i) for i in data])
    return data


@register_preprocessor("cached_amp")
class CachedAmpPreProcessor(BasePreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_group = self.root_config.get_amplitude().decay_group
        self.no_angle = self.kwargs.get("no_angle", False)
        self.no_p4 = self.kwargs.get("no_p4", False)

    def __call__(self, x):
        from tf_pwa.experimental.build_amp import build_angle_amp_matrix

        x = super().__call__(x)
        idx, c_amp = build_angle_amp_matrix(self.decay_group, x)
        x["cached_amp"] = list_to_tuple(c_amp)
        # print(x)

        strip_var = []
        if self.no_angle:
            strip_var += ["ang", "aligned_angle"]
        if self.no_p4:
            strip_var += ["p"]
        if strip_var:
            x = data_strip(x, strip_var)
        return x
