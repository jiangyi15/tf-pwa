"""
module for describing data process.

All data structure is describing as nested combination of ``dict`` or ``list`` for ``ndarray``.
Data process is a translation from data structure to another data structure or typical ``ndarray``.
Data cache can be implemented based on the dynamic features of ``list`` and ``dict``.

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
            "gamma":[...]
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

import random
from pprint import pprint

import numpy as np

from .config import get_config
from .tensorflow_wrapper import tf

# import tensorflow as tf
# from pysnooper import  snoop


try:
    from collections.abc import Iterable
except ImportError:  # python version < 3.7
    from collections import Iterable


def set_random_seed(seed):
    """
    set random seed for random, numpy and tensorflow
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def load_dat_file(
    fnames, particles, dtype=None, split=None, order=None, _force_list=False
):
    """
    Load ``*.dat`` file(s) of 4-momenta of the final particles.

    :param fnames: String or list of strings. File names.
    :param particles: List of Particle. Final particles.
    :param dtype: Data type.
    :param split: sizes of each splited dat files
    :param order: transpose order

    :return: Dictionary of data indexed by Particle.
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
        if fname.endswith(".npz"):
            data = np.load(fname)["arr_0"]
        elif fname.endswith(".npy"):
            data = np.load(fname)
        else:
            data = np.loadtxt(fname, dtype=dtype)
        data = np.reshape(data, (-1, 4))
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


def save_data(file_name, obj, **kwargs):
    """Save structured data to files. The arguments will be passed to ``numpy.save()``."""
    return np.save(file_name, obj, **kwargs)


def save_dataz(file_name, obj, **kwargs):
    """Save compressed structured data to files. The arguments will be passed to ``numpy.save()``."""
    return np.savez(file_name, obj, **kwargs)


def load_data(file_name, **kwargs):
    """Load data file from save_data. The arguments will be passed to ``numpy.load()``."""
    if "allow_pickle" not in kwargs:
        kwargs["allow_pickle"] = True
    data = np.load(file_name, **kwargs)
    try:
        return data["arr_0"].item()
    except IndexError:
        try:
            return data.item()
        except ValueError:
            return data


def _data_split(dat, batch_size, axis=0):
    data_size = dat.shape[axis]
    if axis == 0:
        for i in range(0, data_size, batch_size):
            yield dat[i : min(i + batch_size, data_size)]
    elif axis == -1:
        for i in range(0, data_size, batch_size):
            yield dat[..., i : min(i + batch_size, data_size)]
    else:
        raise Exception("unsupported axis: {}".format(axis))


def data_generator(data, fun=_data_split, args=(), kwargs=None, MAX_ITER=1000):
    """Data generator: call ``fun`` to each ``data`` as a generator. The extra arguments will be passed to ``fun``."""
    kwargs = kwargs if kwargs is not None else {}

    def _gen(dat):
        if isinstance(dat, dict):
            if not dat:
                for i in range(MAX_ITER):
                    yield {}
            ks, vs = [], []
            for k, v in dat.items():
                ks.append(k)
                vs.append(_gen(v))
            for s_data in zip(*vs):
                yield dict(zip(ks, s_data))
        elif isinstance(dat, list):
            if not dat:
                for i in range(MAX_ITER):
                    yield []
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


def data_split(data, batch_size, axis=0):
    """
    Split ``data`` for ``batch_size`` each in ``axis``.

    :param data: structured data
    :param batch_size: Integer, data size for each split data
    :param axis: Integer, axis for split, [option]
    :return: a generator for split data

    >>> data = {"a": [np.array([1.0, 2.0]), np.array([3.0, 4.0])], "b": {"c": np.array([5.0, 6.0])}, "d": [], "e": {}}
    >>> for i, data_i in enumerate(data_split(data, 1)):
    ...     print(i, data_to_numpy(data_i))
    ...
    0 {'a': [array([1.]), array([3.])], 'b': {'c': array([5.])}, 'd': [], 'e': {}}
    1 {'a': [array([2.]), array([4.])], 'b': {'c': array([6.])}, 'd': [], 'e': {}}

    """
    return data_generator(
        data, fun=_data_split, args=(batch_size,), kwargs={"axis": axis}
    )


split_generator = data_split


def data_map(data, fun, args=(), kwargs=None):
    """Apply fun for each data. It returns the same structure."""
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(data, dict):
        return {k: data_map(v, fun, args, kwargs) for k, v in data.items()}
    if isinstance(data, list):
        return [data_map(data_i, fun, args, kwargs) for data_i in data]
    if isinstance(data, tuple):
        return tuple([data_map(data_i, fun, args, kwargs) for data_i in data])
    return fun(data, *args, **kwargs)


def data_struct(data):
    """get the structure of data, keys and shape"""
    if isinstance(data, dict):
        return {k: data_struct(v) for k, v in data.items()}
    if isinstance(data, list):
        return [data_struct(data_i) for data_i in data]
    if isinstance(data, tuple):
        return tuple([data_struct(data_i) for data_i in data])
    if hasattr(data, "shape"):
        return tuple(data.shape)
    return data


def data_mask(data, select):
    """
    This function using boolean mask to select data.

    :param data: data to select
    :param select: 1-d boolean array for selection
    :return: data after selection
    """
    ret = data_map(data, tf.boolean_mask, args=(select,))
    return ret


def data_cut(data, expr, var_map=None):
    """cut data with boolean expression

    :param data: data need to cut
    :param expr: cut expression
    :param var_map: variable map between parameters in expr and data, [option]

    :return: data after being cut,
    """
    var_map = var_map if isinstance(var_map, dict) else {}
    import sympy as sym

    expr_s = sym.sympify(expr)
    params = tuple(expr_s.free_symbols)
    args = [data_index(data, var_map.get(i.name, i.name)) for i in params]
    expr_f = sym.lambdify(params, expr, "tensorflow")
    mask = expr_f(*args)
    return data_mask(data, mask)


def data_merge(*data, axis=0):
    """This function merges data with the same structure."""
    if isinstance(data[0], dict):
        assert all([isinstance(i, dict) for i in data]), "not all type same"
        all_idx = [set(list(i)) for i in data]
        idx = set.intersection(*all_idx)
        return {i: data_merge(*[data_i[i] for data_i in data]) for i in idx}
    if isinstance(data[0], list):
        assert all([isinstance(i, list) for i in data]), "not all type same"
        return [data_merge(*data_i) for data_i in zip(*data)]
    if isinstance(data[0], tuple):
        assert all([isinstance(i, tuple) for i in data]), "not all type same"
        return tuple([data_merge(*data_i) for data_i in zip(*data)])
    m_data = tf.concat(data, axis=axis)
    return m_data


def data_shape(data, axis=0, all_list=False):
    """
    Get data size.

    :param data: Data array
    :param axis: Integer. ???
    :param all_list: Boolean. ???
    :return:
    """

    def flatten(dat):
        ret = []

        def data_list(dat1):
            if hasattr(dat1, "shape"):
                ret.append(dat1.shape)
            else:
                ret.append(())

        data_map(dat, data_list)
        return ret

    shapes = flatten(data)
    if all_list:
        return shapes
    return shapes[0][axis]


def data_to_numpy(dat):
    """Convert Tensor data to ``numpy.ndarray``."""

    def to_numpy(data):
        if hasattr(data, "numpy"):
            return data.numpy()
        return data

    dat = data_map(dat, to_numpy)
    return dat


def data_to_tensor(dat):
    """convert data to ``tensorflow.Tensor``."""

    def to_tensor(data):
        return tf.convert_to_tensor(data)

    dat = data_map(dat, to_tensor)
    return dat


def flatten_dict_data(data, fun="{}/{}".format):
    """Flatten data as dict with structure named as ``fun``."""

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


def data_index(data, key):
    """Indexing data for key or a list of keys."""

    def idx(data, i):
        if isinstance(i, int):
            return data[i]
        assert isinstance(data, dict)
        if i in data:
            return data[i]
        for k, v in data.items():
            if str(k) == str(i):
                return v
        raise ValueError("{} is not found".format(i))

    if isinstance(key, (list, tuple)):
        keys = list(key)
        if len(keys) > 1:
            return data_index(idx(data, keys[0]), keys[1:])
        return idx(data, keys[0])
    return idx(data, key)


def data_strip(data, keys):
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(data, dict):
        ret = {}
        for k, v in data.items():
            if k not in keys:
                ret[k] = data_strip(v, keys)
        return ret
    if isinstance(data, list):
        return [data_strip(data_i, keys) for data_i in data]
    if isinstance(data, tuple):
        return tuple([data_strip(data_i, keys) for data_i in data])
    return data


def check_nan(data, no_raise=False):
    """check if there is nan in data"""
    head_keys = []

    def _check_nan(dat, head):
        if isinstance(dat, dict):
            return {k: _check_nan(v, head + [k]) for k, v in dat.items()}
        if isinstance(dat, list):
            return [
                _check_nan(data_i, head + [i]) for i, data_i in enumerate(dat)
            ]
        if isinstance(dat, tuple):
            return tuple(
                [
                    data_struct(data_i, head + [i])
                    for i, data_i in enumerate(dat)
                ]
            )
        if np.any(tf.math.is_nan(dat)):
            if no_raise:
                return False
            raise ValueError("nan in data[{}]".format(head))
        return True

    return _check_nan(data, head_keys)
