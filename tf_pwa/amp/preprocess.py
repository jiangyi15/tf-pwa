import warnings

from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import create_config, get_config, regist_config, temp_config
from tf_pwa.data import HeavyCall

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
        ret = cal_angle_from_momentum(p4, self.decay_struct, **self.kwargs)
        for k, v in x.get("extra", {}).items():
            ret[k] = v
        return ret


@register_preprocessor("cached_amp")
class CachedAmpPreProcessor(BasePreProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_group = self.root_config.get_amplitude().decay_group

    def __call__(self, x):
        from tf_pwa.experimental.build_amp import build_angle_amp_matrix

        x = super().__call__(x)
        idx, c_amp = build_angle_amp_matrix(self.decay_group, x)
        x["cached_amp"] = c_amp
        # print(x)
        return x
