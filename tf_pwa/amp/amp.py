import contextlib
import warnings

import tensorflow as tf

from tf_pwa.amp.core import Variable, variable_scope
from tf_pwa.config import create_config, get_config, regist_config, temp_config
from tf_pwa.data import LazyCall, data_shape, split_generator

AMP_MODEL = "amplitude_model"
regist_config(AMP_MODEL, {})


def register_amp_model(name=None, f=None):
    """register a data mode

    :params name: mode name used in configuration
    :params f: Data Mode class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(AMP_MODEL)
        if my_name in config:
            warnings.warn("Override mode {}".format(my_name))
        config[my_name] = g
        return g

    if f is None:
        return regist
    return regist(f)


def create_amplitude(decay_group, **kwargs):
    mode = kwargs.get("model", "default")
    return get_config(AMP_MODEL)[mode](decay_group, **kwargs)


class AbsPDF:
    def __init__(
        self,
        *args,
        name="",
        vm=None,
        polar=None,
        use_tf_function=False,
        no_id_cached=False,
        jit_compile=False,
        **kwargs
    ):
        self.name = name
        with variable_scope(vm) as vm:
            if polar is not None:
                vm.polar = polar
            self.init_params(name)
            self.vm = vm
        self.vm = vm
        self.no_id_cached = no_id_cached
        self.f_data = []
        if use_tf_function:
            from tf_pwa.experimental.wrap_function import WrapFun

            self.cached_fun = WrapFun(self.pdf, jit_compile=jit_compile)
        else:
            self.cached_fun = self.pdf
        self.extra_kwargs = kwargs

    def get_params(self, trainable_only=False):
        return self.vm.get_all_dic(trainable_only)

    def set_params(self, var):
        self.vm.set_all(var)

    @contextlib.contextmanager
    def temp_params(self, var):
        params = self.get_params()
        self.set_params(var)
        yield var
        self.set_params(params)

    @contextlib.contextmanager
    def mask_params(self, var):
        with self.vm.mask_params(var):
            yield

    @property
    def variables(self):
        return self.vm.variables

    @property
    def trainable_variables(self):
        return self.vm.trainable_variables

    def cached_available(self):
        return True

    def __call__(self, data, cached=False):
        if isinstance(data, LazyCall):
            data = data.eval()
        if id(data) in self.f_data or self.no_id_cached:
            if self.cached_available():  # decay_group.not_full:
                return self.cached_fun(data)
        else:
            self.f_data.append(id(data))
        ret = self.pdf(data)
        return ret


class BaseAmplitudeModel(AbsPDF):
    def __init__(self, decay_group, **kwargs):
        self.decay_group = decay_group
        super().__init__(**kwargs)
        res = decay_group.resonances
        self.used_res = res
        self.res = res

    def init_params(self, name=""):
        self.decay_group.init_params(name)

    def __del__(self):
        if hasattr(self, "cached_fun"):
            del self.cached_fun
        # super(AmplitudeModel, self).__del__()

    def cache_data(self, data, split=None, batch=None):
        for i in self.decay_group:
            for j in i.inner:
                print(j)
        if split is None and batch is None:
            return data
        else:
            n = data_shape(data)
            if batch is None:  # split个一组，共batch组
                batch = (n + split - 1) // split
            ret = list(split_generator(data, batch))
            return ret

    def set_used_res(self, res):
        self.decay_group.set_used_res(res)

    def set_used_chains(self, used_chains):
        self.decay_group.set_used_chains(used_chains)

    def partial_weight(self, data, combine=None):
        if isinstance(data, LazyCall):
            data = data.eval()
        if combine is None:
            combine = [[i] for i in range(len(self.decay_group.chains))]
        o_used_chains = self.decay_group.chains_idx
        weights = []
        for i in combine:
            self.decay_group.set_used_chains(i)
            weight = self.pdf(data)
            weights.append(weight)
        self.decay_group.set_used_chains(o_used_chains)
        return weights

    def partial_weight_interference(self, data):
        return self.decay_group.partial_weight_interference(data)

    def chains_particle(self):
        return self.decay_group.chains_particle()

    def cached_available(self):
        return not self.decay_group.not_full

    def pdf(self, data):
        ret = self.decay_group.sum_amp(data)
        return ret

    @contextlib.contextmanager
    def temp_total_gls_one(self):
        mask_params = []
        for i in self.decay_group:
            if hasattr(i, "total") and isinstance(i.total, Variable):
                mask_params.append(i.total)
            for j in i:
                if hasattr(j, "g_ls") and isinstance(j.g_ls, Variable):
                    mask_params.append(j.g_ls)
        tmp = {}
        for i in mask_params:
            tmp.update(i.params_one())
        with self.mask_params(tmp):
            yield


@register_amp_model("default")
class AmplitudeModel(BaseAmplitudeModel):
    def partial_weight(self, data, combine=None):
        if isinstance(data, LazyCall):
            data = data.eval()
        return self.decay_group.partial_weight(data, combine)


@register_amp_model("cached_amp")
class CachedAmpAmplitudeModel(BaseAmplitudeModel):
    def pdf(self, data):
        from tf_pwa.experimental.build_amp import build_params_vector

        n_data = data_shape(data)
        cached_data = data["cached_amp"]
        pv = build_params_vector(self.decay_group, data)
        partial_cached_data = [
            cached_data[i] for i in self.decay_group.chains_idx
        ]
        ret = []
        for idx, (i, j) in enumerate(zip(pv, partial_cached_data)):
            # print(j)
            # print(i.shape)
            a = tf.reshape(i, [-1, i.shape[1]] + [1] * (len(j[0].shape) - 1))
            ret.append(tf.reduce_sum(a * tf.stack(j, axis=1), axis=1))
        # print(ret)
        amp = tf.reduce_sum(ret, axis=0)
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        return tf.reduce_sum(amp2s, list(range(1, len(amp2s.shape))))


@register_amp_model("cached_shape")
class CachedShapeAmplitudeModel(BaseAmplitudeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_shape_idx = self.extra_kwargs.get("cached_shape_idx", None)

    def get_cached_shape_idx(self):
        if self.cached_shape_idx is not None:
            return self.cached_shape_idx
        ret = []
        for idx, decay_chain in enumerate(self.decay_group):
            for decay in decay_chain:
                if not decay.core.is_fixed_shape():
                    ret.append(idx)
        ret2 = [i for i in self.decay_group.chains_idx if i not in ret]
        self.cached_shape_idx = ret2
        print("cached shape idx", ret2)
        return ret2

    def pdf(self, data):
        from tf_pwa.experimental.build_amp import build_params_vector
        from tf_pwa.experimental.opt_int import build_params_vector as bv2

        n_data = data_shape(data)
        cached_data = data["cached_amp"]

        cached_shape_idx = self.get_cached_shape_idx()

        old_chains_idx = self.decay_group.chains_idx
        cached_shape_idx = self.get_cached_shape_idx()

        ret = []
        # amp parts without cached shape
        used_chains_idx = [
            i for i in old_chains_idx if i not in cached_shape_idx
        ]
        self.decay_group.set_used_chains(used_chains_idx)
        pv = build_params_vector(self.decay_group, data)
        partial_cached_data = [cached_data[i] for i in used_chains_idx]
        self.decay_group.set_used_chains(old_chains_idx)
        ret = []

        for idx, (i, j) in enumerate(zip(pv, partial_cached_data)):
            a = tf.reshape(i, [-1, i.shape[1]] + [1] * (len(j[0].shape) - 1))
            ret.append(tf.reduce_sum(a * tf.stack(j, axis=1), axis=1))

        # amp parts with cached shape
        cached_shape_idx2 = [
            i for i in cached_shape_idx if i in old_chains_idx
        ]
        partial_cached_data2 = [cached_data[i] for i in cached_shape_idx2]
        pv2 = bv2(self.decay_group, stack=False)
        pv2 = [pv2[i] for i in cached_shape_idx2]
        for idx, (i, j) in enumerate(zip(pv2, partial_cached_data2)):
            a = tf.reshape(i, [-1, i.shape[0]] + [1] * (len(j[0].shape) - 1))
            ret.append(tf.reduce_sum(a * j, axis=1))

        # print(ret)
        amp = tf.reduce_sum(ret, axis=0)
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        return tf.reduce_sum(amp2s, list(range(1, len(amp2s.shape))))


@register_amp_model("p4_directly")
class P4DirectlyAmplitudeModel(BaseAmplitudeModel):
    def cal_angle(self, p4):
        from tf_pwa.cal_angle import cal_angle_from_momentum

        ret = cal_angle_from_momentum(p4, self.decay_group)
        return ret

    def pdf(self, data):
        new_data = self.cal_angle(data["p4"])
        return self.decay_group.sum_amp({**new_data, **data})
