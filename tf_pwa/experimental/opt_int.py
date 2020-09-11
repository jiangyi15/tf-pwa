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
    weight = data.get("weight", 1.0)
    for i, dc in split_gls(dec_chain):
        amp = dg.get_amp(data)
        int_mc = amp * tf.cast(weight, amp.dtype)
        gls = dc.product_gls()
        cached.append(amp / gls)
    return cached


def build_int_matrix(dec, data):
    hij = {}
    for k, i in enumerate(dec):
        dec.set_used_chains([k])
        for j, amp in enumerate(build_sum_amplitude(dec, i, data)):
            hij[(i, j)] = amp
    ret = []
    index = list(hij.keys())
    for i in index:
        tmp = []
        for j in index:
            xij = tf.reduce_sum(hij[i] * tf.math.conj(hij[j]))
            tmp.append(xij)
        ret.append(tmp)
    return index, ret


def build_int_matrix_batch(dec, data, batch=65000):
    index, ret = None, []
    for i in data_split(data, batch):
        index, a = build_int_matrix(dec, i)
        ret.append(a)
    return index, tf.reduce_sum(ret, axis=0)


def gls_combine():
    return None
