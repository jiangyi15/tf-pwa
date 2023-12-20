import numpy as np
import tensorflow as tf


def _wrap_struct(dic, first_none=True):
    if isinstance(dic, dict):
        return {
            k: _wrap_struct(dic[k], first_none) for k in sorted(dic.keys())
        }
    if isinstance(dic, list):
        return [_wrap_struct(v, first_none) for v in dic]
    if isinstance(dic, tuple):
        return tuple([_wrap_struct(v, first_none) for v in dic])
    if isinstance(dic, (tf.Tensor, np.ndarray)):
        shape = dic.shape
        if first_none:
            shape = (None, *shape[1:])
        return tf.TensorSpec(shape, dtype=dic.dtype)
    return dic


def _flatten(dic):
    if isinstance(dic, dict):
        for k in sorted(dic.keys()):
            yield from _flatten(dic[k])
    if isinstance(dic, (list, tuple)):
        for v in dic:
            yield from _flatten(v)
    if isinstance(dic, (tf.Tensor, np.ndarray, tf.TensorSpec)):
        yield dic


class Count:
    def __init__(self, idx=0):
        self.idx = 0

    def add(self, value=1):
        self.idx += value


def _nest(dic, value, idx=None):
    if idx is None:
        idx = Count(0)
    if isinstance(dic, dict):
        return {k: _nest(v, value, idx) for k, v in dic.items()}
    if isinstance(dic, list):
        return [_nest(v, value, idx) for v in dic]
    if isinstance(dic, tuple):
        return tuple([_nest(v, value, idx) for v in dic])
    if isinstance(dic, (tf.Tensor, np.ndarray, tf.TensorSpec)):
        idx.add()
        return value[(idx.idx - 1) % len(value)]
    return dic


class WrapFun:
    def __init__(self, f, jit_compile=False):
        self.f = f
        self.cached_f = {}
        self.struct = {}
        self.jit_compile = jit_compile

    def __call__(self, *args, **kwargs):

        new_x = list(_flatten((args, kwargs)))
        idx = len(new_x)

        if idx not in self.cached_f:
            self.struct[idx] = _wrap_struct((args, kwargs))

            def _g(*x):
                new_args, new_kwargs = _nest(self.struct[idx], x)
                return self.f(
                    *new_args, **new_kwargs
                )  # *new_args, **new_kwargs)

            _g2 = tf.function(_g, jit_compile=self.jit_compile)

            self.cached_f[idx] = _g2.get_concrete_function(
                *list(_flatten(self.struct[idx]))
            )
        new_x = [
            tf.convert_to_tensor(i) if not isinstance(i, tf.Tensor) else i
            for i in new_x
        ]
        return self.cached_f[idx](
            *new_x
        )  # *args, **kwargs) # _flatten((args, kwargs)))
