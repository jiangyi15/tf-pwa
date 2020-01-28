"""
module for describing data process.

All data structure is decaribing as nested combination of `dict` or `list` for `ndarray`.
Aata process is a transation from data structure to another data structure or typical `ndarray`.
Data cache can be implemented based on the dynamic features of `list` and `dict`.

"""
import numpy as np
#import tensorflow as tf
#from pysnooper import  snoop
from .particle import BaseParticle, BaseDecay, DecayChain
from .angle import EularAngle
from .tensorflow_wrapper import tf

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


# data process
def infer_momentum(p, decay_chain: DecayChain) -> dict:
    """
    {outs:momentum} => {top:{p:momentum},inner:{p:..},outs:{p:..}}
    """
    st = decay_chain.sorted_table()
    ret = {}
    for i in st:
        ps = []
        for j in st[i]:
            ps.append(p[j])
        ret[i] = {"p": np.sum(ps, 0)}
    return ret

def add_mass(data: dict, _decay_chain: DecayChain = None) -> dict:
    """
    {top:{p:momentum},inner:{p:..},outs:{p:..}} => {top:{p:momentum,m:mass},...}
    """
    for i in data:
        p = data[i]["p"]
        data[i]["m"] = np.sqrt(np.sum(np.array([1., -1., -1., -1.])*p*p, -1))
    return data

def Getp(M_0, M_1, M_2):
    M12S = M_1 + M_2
    M12D = M_1 - M_2
    p = (M_0 - M12S) * (M_0 + M12S) * (M_0 - M12D) * (M_0 + M12D)
    q = (p + tf.abs(p))/2 # if p is negative, which results from bad data, the return value is 0.0
    return tf.sqrt(q) / (2 * M_0)

def get_relativate_momentum(data: dict, decay: BaseDecay, m0=None, m1=None, m2=None):
    if m0 is None:
        m0 = data[decay.core]["m"]
    if m1 is None:
        m1 = data[decay.outs[0]]["m"]
    if m2 is None:
        m2 = data[decay.outs[1]]["m"]
    p = Getp(m0, m1, m2)
    return p

def test_process(fnames=None):
    a, b, c, d, r = [BaseParticle(i) for i in ["A", "B", "C", "D", "R"]]
    if fnames is None:
        p = {
            "B": [[1.0, 0.2, 0.3, 0.2]],
            "C": [[2.0, 0.1, 0.3, 0.4]],
            "D": [[3.0, 0.2, 0.5, 0.7]]
        }
    else:
        p = load_data(fnames, [b, c, d])
    st = {b: ["B"], c: ["C"], d: ["D"], a: ["B", "C", "D"], r: ["B", "D"]}
    dec = DecayChain.from_sorted_table(st)
    print(dec)
    data = infer_momentum(p, dec)
    data = add_mass(data, dec)
    print(data)
