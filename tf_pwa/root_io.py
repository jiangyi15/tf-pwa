import numpy as np
from .data import data_merge

has_uproot = True
try:
    import uproot
except ImportError as e:
    has_uproot = False
    print(e, "you should install `uproot` correctly for using this module")


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


def save_dict_to_root(dic, file_name, tree_name=None):
    """
    This function stores data arrays in the form of a dictionary into a root file.
    It provides a convenient interface to ``uproot``.

    :param dic: Dictionary of data
    :param file_name: String
    :param tree_name: String. By default it's "tree".
    """
    if file_name[-5:] == ".root":
        file_name = file_name[:-5]
    if isinstance(dic, dict):
        dic = [dic]
    if tree_name is None:
        tree_name = "DataTree"
    Ndic = len(dic)
    if isinstance(tree_name, list):
        assert len(tree_name) == Ndic
    else:
        t = []
        for i in range(Ndic):
            t.append(tree_name + str(i))
        tree_name = t

    with uproot.recreate(file_name + ".root") as f:
        for d, t in zip(dic, tree_name):
            branch_type = {}
            branch_data = {}
            for i in d:
                j = (
                    i.replace("(", "_")
                    .replace(")", "_")
                    .replace(" ", "_")
                    .replace("*", "star")
                    .replace("+", "p")
                    .replace("-", "m")
                )
                branch_data[j] = np.array(d[i])
                branch_type[j] = branch_data[j].dtype.name
            f[t] = uproot.newtree(branch_type)
            f[t].extend(branch_data)
