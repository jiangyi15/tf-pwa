import itertools
import tensorflow as tf
from tf_pwa.amp import get_particle, get_decay
from tf_pwa.data import data_split


def split_gls(dec_chain):
    gls = [i.get_ls_list() for i in dec_chain]
    ls_combination = list(itertools.product(*gls))
    for i in ls_combination:
        for gi, j in zip(i, dec_chain):
            j.set_ls([gi])
        yield i, dec_chain
    for j, g in zip(dec_chain, gls):
        j.set_ls(g)


def build_sum_amplitude(dg, dec_chain, data):
    cached = []
    for i, dc in split_gls(dec_chain):
        amp = dg.get_amp(data)
        int_mc = amp
        gls = dc.product_gls()
        cached.append(amp / gls)
    return cached


def build_int_matrix(dec, data, weight=None):
    hij = {}
    used_chains = dec.chains_idx
    for k, i in enumerate(dec):
        dec.set_used_chains([k])
        for j, amp in enumerate(build_sum_amplitude(dec, i, data)):
            hij[(i, j)] = amp
    dec.set_used_chains(used_chains)
    ret = []
    if weight is None:
        weight = data.get("weight", 1.0)
    index = list(hij.keys())
    weight = tf.cast(weight, hij[index[0]].dtype)
    n_lambda = len(hij[index[0]].shape) - 1
    weight = tf.reshape(weight, [-1]+[1]*n_lambda)
    for i in index:
        tmp = []
        for j in index:
            xij = hij[i] * tf.math.conj(hij[j])
            xij = tf.reduce_sum(weight * xij)
            tmp.append(xij)
        ret.append(tmp)
    return index, ret


def build_int_matrix_batch(dec, data, batch=65000):
    index, ret = None, []
    for i in data_split(data, batch):
        index, a = build_int_matrix(dec, i)
        ret.append(a)
    return index, tf.reduce_sum(ret, axis=0)


def build_params_vector(dec):
    ret = []
    for i in dec:
        factor = i.get_all_factor()
        a = gls_combine(factor)
        ret.append(a)
    return tf.concat(ret, axis=0)


def build_params_matrix(dec):
    pv = build_params_vector(dec)
    return pv[:,None] * tf.math.conj(pv)[None, :]


def gls_combine(fs):
    ret = fs[0]
    for i in fs[1:]:
        ret = tf.reshape(ret[:,None] * i[None, :],(-1,))
    return ret


def cached_int_mc(dec, data, batch=65000):
    a, int_matrix = build_int_matrix_batch(dec, data, batch)

    @tf.function
    def int_mc():
        pm = build_params_matrix(dec)
        ret = tf.reduce_sum(pm * int_matrix)
        return tf.math.real(ret)

    return int_mc

