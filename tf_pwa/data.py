"""
module for describing data process.

All data structure is decaribing as nested combination of `dict` or `list` for `ndarray`.
Aata process is a transation from data structure to another data structure or typical `ndarray`.
Data cache can be implemented based on the dynamic features of `list` and `dict`.

The full data structure is
```

{
  "particle":{
    "A":{"p":...,"m":...}
    ...
  },
  "decay":[
    {
      "A->R1+B": {
        "R1": {
          "ang":  {
            "alpha":[...],
            "beta": [...],
            "gamma": [...]
          },
          "z": [[x1,y1,z1],...],
          "x": [[x2,y2,z2],...]
        },
        "B" : {...}
      },
      "R->C+D": {
        "C": {
          ...,
          "aligned_angle":{
            "alpha":[...],
            "beta":[...],
            "gamme":[...]
          }
        },
        "D": {...}
      },
    },
    {
      "A->R2+C": {...},
      "R2->B+D": {...}
    },
    ...
  ],
  "weight": [...]
}
```
"""

from pprint import pprint
import numpy as np
# import tensorflow as tf
# from pysnooper import  snoop

from .particle import BaseParticle, BaseDecay, DecayChain, DecayGroup
from .angle_tf import LorentzVector, EularAngle
from .tensorflow_wrapper import tf
from .config import get_config

try:
    from collections.abc import Iterable
except ImportError:  # python version < 3.7
    from collections import Iterable


def load_dat_file(fnames, particles, dtype=None, split=None, order=None, _force_list=False):
    """
    load *.dat file(s) for particles momentum.
    """
    n = len(particles)
    if dtype is None:
        dtype = get_config("dtype")

    if isinstance(fnames, str):
        fnames = [fnames]
    elif isinstance(fnames, Iterable):
        fnames = list(fnames)
    else:
        raise TypeError("fnames must be string or list of strings")

    datas = []
    sizes = []
    for fname in fnames:
        data = np.loadtxt(fname, dtype=dtype)
        sizes.append(data.shape[0])
        datas.append(data)

    if split is None:
        n_total = sum(sizes)
        if n_total % n != 0:
            raise ValueError("number of data find {}/{}".format(n_total, n))
        n_data = n_total // n
        split = [size // n_data for size in sizes]

    if order is None:
        order = (1, 0, 2)

    ret = {}
    idx = 0
    for size, data in zip(split, datas):
        data_1 = data.reshape((-1, size, 4))
        data_2 = data_1.transpose(order)
        for i in data_2:
            part = particles[idx]
            ret[part] = i
            idx += 1

    return ret


save_data = np.save


def load_data(*args, **kwargs):
    if "allow_pickle" not in kwargs:
        kwargs["allow_pickle"] = True
    data = np.load(*args, **kwargs)
    try:
        return data.item()
    except ValueError:
        return data


def data_split(dat, batch_size, axis=0):
    data_size = dat.shape[axis]
    if axis == 0:
        for i in range(0, data_size, batch_size):
            yield dat[i:min(i + batch_size, data_size)]
    elif axis == -1:
        for i in range(0, data_size, batch_size):
            yield dat[..., i:min(i + batch_size, data_size)]
    else:
        raise Exception("unsupport axis: {}".format(axis))


def data_generator(data, fun=data_split, args=(), kwargs=None):
    """
    split data generator.
    """
    kwargs = kwargs if kwargs is not None else {}

    def _gen(dat):
        if isinstance(dat, dict):
            ks, vs = [], []
            for k, v in dat.items():
                ks.append(k)
                vs.append(_gen(v))
            for s_data in zip(*vs):
                yield dict(zip(ks, s_data))
        elif isinstance(dat, list):
            vs = []
            for v in dat:
                vs.append(_gen(v))
            for s_data in zip(*vs):
                yield list(s_data)
        elif isinstance(dat, tuple):
            vs = []
            for v in dat:
                vs.append(_gen(v))
            for s_data in zip(*vs):
                yield s_data
        else:
            for i in fun(dat, *args, **kwargs):
                yield i

    return _gen(data)


def split_generator(data, batch_size, axis=0):
    return data_generator(data, fun=data_split, args=(batch_size,), kwargs={"axis": axis})


def data_map(data, fun, args=(), kwargs=None):
    kwargs = kwargs if kwargs is not None else {}

    def g_fun(*args1, **kwargs1):
        yield fun(*args1, **kwargs1)

    g = data_generator(data, fun=g_fun, args=args, kwargs=kwargs)
    return next(g)


def data_merge(data1, data2, axis=0):
    if isinstance(data1, dict):
        return {i: data_merge(data1[i], data2[i]) for i in set(list(data1)) & set(list(data2))}
    if isinstance(data2, list):
        return [data_merge(data, data2[i]) for i, data in enumerate(data1)]
    if isinstance(data1, tuple):
        return tuple([data_merge(data, data2[i]) for i, data in enumerate(data1)])
    m_data = tf.concat([data1, data2], axis=axis)
    return m_data


def data_shape(data, axis=0, all_list=False):
    def flatten(dat):
        ret = []

        def data_list(dat1):
            ret.append(dat1.shape)

        data_map(dat, data_list)
        return ret

    shapes = flatten(data)
    if all_list:
        return shapes
    return shapes[0][axis]


def data_to_numpy(dat):
    def to_numpy(data):
        if hasattr(data, "numpy"):
            return data.numpy()
        return data

    dat = data_map(dat, to_numpy)
    return dat


def data_to_tensor(dat):
    def to_tensor(data):
        return tf.convert_to_tensor(data)

    dat = data_map(dat, to_tensor)
    return dat


def flatten_dict_data(data, fun="{}/{}".format):
    def dict_gen(dat):
        return dat.items()

    def list_gen(dat):
        return enumerate(dat)

    if isinstance(data, (dict, list, tuple)):
        ret = {}
        gen_1 = dict_gen if isinstance(data, dict) else list_gen
        for i, data_i in gen_1(data):
            tmp = flatten_dict_data(data_i)
            if isinstance(tmp, (dict, list, tuple)):
                gen_2 = dict_gen if isinstance(tmp, dict) else list_gen
                for j, tmp_j in gen_2(tmp):
                    ret[fun(i, j)] = tmp_j
            else:
                ret[i] = tmp
        return ret
    return data
