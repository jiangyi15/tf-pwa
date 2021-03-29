"""

Module for factor system.

```
A = a1 ( B x C x D) + a2 (E x F)
B = b1 B1 + b2 B2
```

is a tree structure
```
A -> [(a1, [(b1, B1), (b2, B2)], C, D), (a2, E, F)]
```

Each component is a path for root to a leaf.
```
(a1, b1),  (a1, b2),  (a2,)
```

We can add some options to change the possible combination. (TODO)

"""
import contextlib
import itertools

from tf_pwa.variable import _shape_func


def get_all_chain(a):
    for i in a:
        for j in get_prod_chain(i):
            yield j


def get_split_chain(a):
    for i in a:
        for j in get_prod_chain(i):
            yield from j


def get_prod_chain(i):
    ret = []
    for j in i:
        if isinstance(j, (list, tuple)):
            ret.append(list(get_split_chain(j)))
        else:
            ret.append([j])
    for i in itertools.product(*ret):
        yield i


def get_chain_name(chain):
    ret = []
    for i in chain:
        tmp = []
        if i.shape:
            if i.cplx:

                def fun(name, idx):
                    tmp.append((name + "r", name + "i"))

            else:

                def fun(name, idx):
                    tmp.append(name)

            _shape_func(fun, i.shape, i.name)
        else:
            if i.cplx:
                tmp.append((i.name + "r", i.name + "i"))
            else:
                tmp.append(i.name)
        ret.append(tmp)
    return itertools.product(*ret)


@contextlib.contextmanager
def temp_var(vm):
    params = vm.get_all_dic()
    yield vm
    vm.set_all(params)


def flatten_all(x):
    ret = []
    if isinstance(x, (tuple, list)):
        for i in x:
            ret += list(flatten_all(i))
        return ret
    return [x]


def get_all_partial_amp(amp, data, strip_part=[]):
    var = amp.decay_group.get_factor_variable()
    chains = list(get_all_chain(var))

    part = []
    for i in chains:
        part += list(get_chain_name(i))

    ret_part = []
    for j in strip_variable(part, strip_part):
        ret_part.append(j)
    all_var = flatten_all(ret_part)

    ret = []
    id_ = []
    for i in ret_part:
        ret.append(partial_amp(amp, data, all_var, flatten_all(i)))
        id_.append(get_id_variable(all_var, i))
    return id_, ret


def partial_amp(amp, data, all_va, need_va):
    with temp_var(amp.vm) as vm:
        others_va = set(all_va) - set(need_va)
        amp.vm.set_all({i: 0.0 for i in others_va})
        ret = amp(data)
    return ret


def strip_variable(var_all, part=[]):
    def match(x):
        for i in part:
            if i in x:
                return True
        return False

    cached_list = [frozenset()]

    for j in var_all:
        j = flatten_all(j)
        after_strip = [k for k in j if not match(k)]
        if frozenset(after_strip) in cached_list:
            continue
        yield after_strip
        cached_list.append(frozenset(after_strip))


def get_id_variable(all_var, var):
    var = flatten_all(var)
    ret = frozenset(set(var) - (set(all_var) - set(var)))
    if len(ret) == 0:
        return frozenset(var)
    return ret
