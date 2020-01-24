"""
module for describing data file.
"""
import numpy as np
#import tensorflow as tf
#from pysnooper import  snoop

try:
    from collections.abc import Iterable
except ImportError: # python version < 3.7
    from collections import Iterable

def load_data(fnames, particles, split=None, order=None, _force_list=False):
    """
    load *.dat file(s) for particles momentum.
    """
    n = len(particles)

    if isinstance(fnames, str):
        fnames = [fnames]
    elif isinstance(fnames, Iterable):
        fnames = list(fnames)
    else:
        raise TypeError("fnames must be string or list of strings")

    datas = []
    sizes = []
    for fname in fnames:
        data = np.loadtxt(fname)
        sizes.append(data.shape[0])
        datas.append(data)

    if split is None:
        n_total = sum(sizes)
        if n_total % n != 0:
            raise ValueError("number of data find {}/{}".format(n_total, n))
        n_data = n_total // n
        split = [size//n_data for size in sizes]

    if order is None:
        order = (1, 0, 2)

    ret = {}
    idx = 0
    for size, data in zip(split, datas):
        data_1 = data.reshape((-1, size, 4))
        data_2 = data_1.transpose(order)
        for i in data_2:
            part = particles[idx]
            if isinstance(part, str):
                name = part
            elif hasattr(part, "name"):
                name = part.name
            else:
                name = idx
            ret[name] = i
            idx += 1

    return ret
