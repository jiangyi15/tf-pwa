"""
This module provides some functions that may be useful in other modules.
"""
import json
import math
import time
import functools


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


has_yaml = True
try:
    import yaml
except ImportError:
    has_yaml = False


def _load_json_file(name):
    with open(name) as f:
        return json.load(f)


def _load_yaml_file(name):
    with open(name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_config_file(name):
    """
    Load config file such as **Resonances.yml**.

    :param name: File name. Either yml file or json file.
    :return: Dictionary read from the file.
    """
    if name.endswith("json"):
        return _load_json_file(name)
    if has_yaml:
        if name.endswith("yml"):
            return _load_yaml_file(name)
        return _load_yaml_file(name + ".yml")
    else:
        print("No yaml support, using json file")
        return _load_json_file(name + ".json")


def flatten_dict_data(data, fun="{}/{}".format):
    """
    Flatten nested dictionary data into one layer dictionary.

    :return: Dictionary
    """
    if isinstance(data, dict):
        ret = {}
        for i in data:
            tmp = flatten_dict_data(data[i])
            if isinstance(tmp, dict):
                for j in tmp:
                    ret[fun(i, j)] = tmp[j]
            else:
                ret[i] = tmp
        return ret
    else:
        return data


flatten_np_data = lambda data: flatten_dict_data(data, fun=lambda x, y: "{}{}".format(y, x[3:]))


def error_print(x, err=None):
    """
    It returns a format string "value +/- error". The precision is modified according to ``err``

    :param x: Value
    :param err: Error
    :return: String
    """
    if err is None:
        return ("{}").format(x)
    if err <= 0 or math.isnan(err):
        return ("{} ? {}").format(x, err)
    d = math.ceil(math.log10(err))
    b = 10 ** d
    b_err = err / b
    b_val = x / b
    if b_err < 0.355:  # 0.100 ~ 0.354
        dig = 2
    elif b_err < 0.950:  # 0.355 ~ 0.949
        dig = 1
    else:  # 0.950 ~ 0.999
        dig = 0
    err = round(b_err, dig) * b
    x = round(b_val, dig) * b
    d_p = dig - d
    if d_p > 0:
        return ("{0:.%df} +/- {1:.%df}" % (d_p, d_p)).format(x, err)
    return ("{0:.0f} +/- {1:.0f}").format(x, err)


def pprint(dicts):
    """
    Print dictionary using json format.
    """
    try:
        s = json.dumps(dicts, indent=2)
        print(s, flush=True)
    except:
        print(dicts, flush=True)


def print_dic(dic):
    """
    Another way to print dictionary.
    """
    if type(dic) == dict:
        for i in dic:
            print(i + " :\t", dic[i])
    else:
        print(dic)


def std_polar(rho, phi):
    """
    To standardize a polar variable. By standard form, it means :math:`\\rho>0, -\\pi<\\phi<\\pi`.

    :param rho: Real number
    :param phi: Real number
    :return: ``rho``, ``phi``
    """
    if rho < 0:
        rho = -rho
        phi += math.pi
    while phi < -math.pi:
        phi += 2 * math.pi
    while phi > math.pi:
        phi -= 2 * math.pi
    return rho, phi


def deep_iter(base, deep=1):
    for i in base:
        if deep == 1:
            yield [i]
        else:
            for j in deep_iter(base, deep - 1):
                yield [i] + j


def deep_ordered_iter(base, deep=1):
    ids = list(base)
    size = len(ids)
    for i in deep_ordered_range(size, deep):
        yield [ids[j] for j in i]


def deep_ordered_range(size, deep=1, start=0):
    for i in range(start, size):
        if deep <= 1:
            yield [i]
        else:
            for j in deep_ordered_range(size, deep - 1, i + 1):
                yield [i] + j


# from amplitude.py
def is_complex(x):
    """
    If **x** is of type ``complex``, it returns ``True``.
    """
    try:
        y = complex(x)
    except:
        return False
    return True


# from model.py
def array_split(data, batch=None):
    """Split a data array. **batch** is the number of data in a row."""
    if batch is None:
        return [data]
    ret = []
    n_data = data[0].shape[0]
    n_split = (n_data + batch - 1) // batch
    for i in range(n_split):
        tmp = []
        for data_i in data:
            tmp.append(data_i[i * batch:min(i * batch + batch, n_data)])
        ret.append(tmp)
    return ret


def time_print(f):
    """It provides a wrapper to print the time cost on a process."""
    @functools.wraps(f)
    def g(*args, **kwargs):
        now = time.time()
        ret = f(*args, **kwargs)
        print(f.__name__, " cost time:", time.time() - now)
        return ret

    return g


def std_periodic_var(p, mid=0., pi=math.pi):
    """
    Transform a periodic variable into its range.

    :param p: Value
    :param mid: The middle value
    :param pi: Half-range
    :return: The transformed value
    """
    twopi = 2 * pi
    while p <= mid - pi:
        p += twopi
    while p >= mid + pi:
        p -= twopi
    return p
