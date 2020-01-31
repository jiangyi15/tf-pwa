"""
module for describing data process.

All data structure is decaribing as nested combination of `dict` or `list` for `ndarray`.
Aata process is a transation from data structure to another data structure or typical `ndarray`.
Data cache can be implemented based on the dynamic features of `list` and `dict`.

"""
import numpy as np
#import tensorflow as tf
#from pysnooper import  snoop
from .particle import BaseParticle, BaseDecay, DecayChain, DecayGroup
from .angle_tf import LorentzVector, EularAngle
from .tensorflow_wrapper import tf

try:
    from collections.abc import Iterable
except ImportError: # python version < 3.7
    from collections import Iterable

def load_dat_file(fnames, particles, split=None, order=None, _force_list=False):
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
            yield dat[i:min(i+batch_size, data_size)]
    elif axis == -1:
        for i in range(0, data_size, batch_size):
            yield dat[..., i:min(i+batch_size, data_size)]
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
        return [fun(*args1, **kwargs1)]
    g = data_generator(data, fun=g_fun, args=args, kwargs=kwargs)
    return next(g)

def data_merge(data1, data2, axis=0):
  def flatten(data):
    ret = []
    def data_list(dat):
      ret.append(dat)
    data_map(data, data_list)
    return ret
  def rebuild(data, ret):
    idx = -1
    def data_build(_dat):
      nonlocal idx
      idx += 1
      return ret[idx]
    data = data_map(data, data_build)
    return data
  f_data1 = flatten(data1)
  f_data2 = flatten(data2)
  m_data = [tf.concat(data_i, axis=axis) for data_i in zip(f_data1, f_data2)]
  return rebuild(data1, m_data)

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

def flatten_dict_data(data, fun="{}/{}".format):
  def dict_gen(data):
    return data.items()
  def list_gen(data):
    return enumerate(data)

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

# data process
def infer_momentum(p, decay_chain: DecayChain, center_mass=True) -> dict:
    """
    {outs:momentum} => {top:{p:momentum},inner:{p:..},outs:{p:..}}
    """
    p_outs = {}
    if center_mass:
        ps_top = []
        for i in decay_chain.outs:
            ps_top.append(p[i])
        p_top = tf.reduce_sum(ps_top, 0)
        for i in decay_chain.outs:
            p_outs[i] = LorentzVector.rest_vector(p_top, p[i])
    else:
        for i in decay_chain.outs:
            p_outs[i] = p[i]

    st = decay_chain.sorted_table()
    ret = {}
    for i in st:
        ps = []
        for j in st[i]:
            ps.append(p_outs[j])
        ret[i] = {"p": tf.reduce_sum(ps, 0)}
    return ret

def add_mass(data: dict, _decay_chain: DecayChain = None) -> dict:
    """
    {top:{p:momentum},inner:{p:..},outs:{p:..}} => {top:{p:momentum,m:mass},...}
    """
    for i in data:
        p = data[i]["p"]
        data[i]["m"] = LorentzVector.M(p)
    return data

def add_weight(data: dict, weight: float = 1.0) -> dict:
    """
    {top:{p:momentum},inner:{p:..},outs:{p:..}} => {top:{p:momentum,m:mass},...}
    """
    data_size = data_shape(data)
    weight = [1.0] * data_size
    data["weight"] = np.array(weight)
    return data

def cal_helicity_angle(data: dict, decay_chain: DecayChain = None) -> dict:
    """
    {top:{p:momentum},inner:{p:..},outs:{p:..}} => {top:{p:momentum,m:mass},...}
    """
    part_data = {}
    for i in decay_chain:
        part_data[i] = {}
        p_rest = data[i.core]["p"]
        part_data[i]["rest_p"] = {}
        for j in i.outs:
            pj = data[j]["p"]
            p = LorentzVector.rest_vector(p_rest, pj)
            part_data[i]["rest_p"][j] = p
    set_x = {decay_chain.top: np.array([[1.0, 0.0, 0.0]])}
    set_z = {decay_chain.top: np.array([[0.0, 0.0, 1.0]])}
    set_decay = list(decay_chain)
    while set_decay:
        extra_decay = []
        for i in set_decay:
            if i.core in set_x:
                for j in i.outs:
                    data[i][j] = {}
                    z2 = LorentzVector.vect(part_data[i]["rest_p"][j])
                    ang, x = EularAngle.angle_zx_z_getx(set_z[i.core], set_x[i.core], z2)
                    set_x[j] = x
                    set_z[j] = z2
                    data[i][j]["ang"] = ang
                    data[i][j]["x"] = x
                    data[i][j]["z"] = z2
            else:
                extra_decay.append(i)
        set_decay = extra_decay
    return data

def cal_angle(data: list, decay_group: DecayGroup) -> dict:
  for i in decay_group:
    data = cal_helicity_angle(data, i)
  decay_chain_struct = decay_group.topology_structure()
  set_x = {}
  # for a the top rest farme
  for decay_chain in decay_chain_struct:
    for decay in decay_chain:
      if decay.core == decay_group.top:
        for i in decay.outs:
          if (i not in set_x) and (i in decay_group.outs):
            set_x[i] = decay
  # or the first chain
  for i in decay_group.outs:
    if i not in set_x:
      decay_chain = next(iter(decay_chain_struct))
      for decay in decay_chain:
        for j in decay.outs:
          if i == j:
            set_x[i] = decay
  for decay_chain in decay_group:
    for decay in decay_chain:
      for i in decay.outs:
        if i in decay_group.outs:
          if decay != set_x[i]:
            x1 = data[decay][i]["x"]
            x2 = data[set_x[i]][i]["x"]
            z1 = data[decay][i]["z"]
            z2 = data[set_x[i]][i]["z"]
            ang = EularAngle.angle_zx_zx(z1, x1, z2, x2)
            data[decay][i]["aligned_angle"] = ang
  return data


def Getp(M_0, M_1, M_2):
    M12S = M_1 + M_2
    M12D = M_1 - M_2
    p = (M_0 - M12S) * (M_0 + M12S) * (M_0 - M12D) * (M_0 + M12D)
    q = (p + tf.abs(p))/2 # if p is negative, which results from bad data, the return value is 0.0
    return tf.sqrt(q) / (2 * M_0)

def add_relativate_momentum(data: dict, decay_chain: DecayChain):
    for decay in decay_chain:
        m0 = data[decay.core]["m"]
        m1 = data[decay.outs[0]]["m"]
        m2 = data[decay.outs[1]]["m"]
        p = Getp(m0, m1, m2)
        if not decay in data:
            data[decay] = {}
        data[decay]["|q|"] = p
    return data

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
    a, b, c, d = [BaseParticle(i) for i in ["A", "B", "C", "D"]]
    if fnames is None:
        p = {
            b: np.array([[1.0, 0.2, 0.3, 0.2]]),
            c: np.array([[2.0, 0.1, 0.3, 0.4]]),
            d: np.array([[3.0, 0.2, 0.5, 0.7]])
        }
    else:
        p = load_dat_file(fnames, [b, c, d])
    # st = {b: [b], c: [c], d: [d], a: [b, c, d], r: [b, d]}
    decs = DecayChain.from_particles(a, [b, c, d])
    data = {}
    for dec in decs:
      data_i = infer_momentum(p, dec)
      data_i = add_mass(data_i, dec)
      data_i = add_relativate_momentum(data_i, dec)
      for i in data_i:
        data[i] = data_i[i]
    data = cal_angle(data, DecayGroup(decs))
    data = add_weight(data)
    for i, j in enumerate(split_generator(data, 5000)):
        print("split:", i, "is", j)
    from pprint import pprint
    def to_numpy(data):
      if hasattr(data, "numpy"):
        return data.numpy()
      return data
    data = data_map(data, to_numpy)
    pprint(data)
    return data
