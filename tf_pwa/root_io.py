import uproot
import numpy as np
from .data import data_merge


def load_root_data(fnames):
    """load root file as dict"""
    if isinstance(fnames, str):
        fnames = [fnames]
    ret = {}
    root_file = [uproot.open(i) for i in fnames]
    keys = [i.allkeys() for i in root_file]
    common_keys = set()
    for i in keys:
        for j in i:
            pj = j.decode().split(";")[0]
            common_keys.add(pj)
    ret = {}
    for i in common_keys:
        data = []
        for j in root_file:
            data_i = load_Ttree(j.get(i))
            data.append(data_i)
        ret[i] = data_merge(*data)
    return ret


def load_Ttree(tree):
    """load TTree as dict"""
    ret = {}
    for i in tree.keys():
        arr = tree.get(i).array()
        if isinstance(arr, np.ndarray):
            ret[i.decode()] = arr
    return ret
