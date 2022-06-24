import copy

from tf_pwa.amp.core import get_particle_model_name
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import get_config
from tf_pwa.data import data_mask, data_merge, data_shape
from tf_pwa.generator import GenTest
from tf_pwa.particle import BaseParticle
from tf_pwa.phasespace import generate_phsp as generate_phsp_o
from tf_pwa.tensorflow_wrapper import tf

from .config_loader import ConfigLoader


@ConfigLoader.register_function()
def generate_toy(config, N=1000, force=True, max_N=100000):
    decay_group = config.get_decay()
    amp = config.get_amplitude()

    def gen(M):
        return generate_phsp(config, M)

    all_data = []
    n_gen = 0
    n_accept = 0
    n_total = 0
    test_N = 10 * N
    while N > n_accept:
        test_N = abs(min(max_N, test_N))
        data = single_sampling(gen, amp, test_N)
        n_gen = data_shape(data)
        n_total += test_N
        n_accept += n_gen
        test_N = int(1.01 * n_total / (n_accept + 1) * (N - n_accept))
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    return ret


def single_sampling(phsp, amp, N):
    data = phsp(N)
    weight = amp(data)
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * tf.reduce_max(weight) * 1.1 < weight
    data = data_mask(data, cut)
    return data


def gen_random_charge(N, random=True):
    if random:
        charge = (
            tf.cast(
                tf.random.uniform((N,)) > 0.5,
                get_config("dtype"),
            )
            * 2
            - 1
        )
    else:
        charge = tf.ones((N,), get_config("dtype"))
    return charge


@ConfigLoader.register_function()
def generate_toy2(
    config,
    N=1000,
    force=True,
    gen=None,
    gen_p=None,
    importance_f=None,
    max_N=100000,
    include_charge=False,
):
    """
    A more accurate method for generating toy data.

    :param N: number of events.
    :param force: if romove extra data generated.
    :param gen: optional function for generate phase space, the return value is same as config.get_data.
    :param gen_p:  optional function for generate phase space, the return value is dict as `{B: pb, C: pc, D: pd}`.
    :param max_N: max number of events for every try.

    """

    decay_group = config.get_decay()
    amp = config.get_amplitude()

    if gen is None:
        if gen_p is not None:

            def gen(N):
                p = gen_p(N)
                p = {
                    BaseParticle(k) if isinstance(k, str) else k: v
                    for k, v in p.items()
                }
                charge = gen_random_charge(N, include_charge)
                ret = config.data.cal_angle(p, charge=charge)
                ret["charge_conjugation"] = charge
                return ret  # # cal_angle_from_momentum(p, config.get_decay(False))

        else:

            def gen(M):
                return generate_phsp(config, M, include_charge=include_charge)

    if not hasattr(config, "max_amplitude"):
        config.max_amplitude = None
    max_weight = None
    if importance_f is None:
        max_weight = config.max_amplitude

    ret, status = multi_sampling(
        gen,
        amp,
        N,
        force=force,
        max_N=max_N,
        max_weight=max_weight,
        importance_f=importance_f,
    )

    if importance_f is None:
        config.max_amplitude = max_weight

    return ret


@ConfigLoader.register_function()
def generate_toy_p(
    config,
    N=1000,
    force=True,
    gen_p=None,
    importance_f=None,
    max_N=100000,
    include_charge=False,
):
    """
    generate toy data momentum.
    """
    if gen_p is None:
        gen_p = config.generate_phsp_p

    new_gen = gen_p
    fun = config.eval_amplitude
    if include_charge:
        new_gen = lambda N: {"p4": gen_p(N), "charge": gen_random_charge(N)}
        fun = lambda x: config.eval_amplitude(extra=x)

    if not hasattr(config, "max_amplitude"):
        config.max_amplitude = None
    max_weight = None
    if importance_f is None:
        max_weight = config.max_amplitude

    ret, status = multi_sampling(
        new_gen,
        fun,
        N,
        force=force,
        max_N=max_N,
        max_weight=max_weight,
        importance_f=importance_f,
    )

    if importance_f is None:
        config.max_amplitude = max_weight

    return ret


def multi_sampling(
    phsp, amp, N, max_N=100000, force=True, max_weight=None, importance_f=None
):
    a = GenTest(max_N)
    all_data = []
    for i in a.generate(N):
        data, new_max_weight = single_sampling2(
            phsp, amp, i, max_weight, importance_f
        )
        if max_weight is None:
            max_weight = new_max_weight
        if new_max_weight > max_weight and len(all_data) > 0:
            tmp = data_merge(*all_data)
            rnd = tf.random.uniform((data_shape(tmp),), dtype=max_weight.dtype)
            cut = (
                rnd * new_max_weight / max_weight < 1.0
            )  # .max_amplitude < 1.0
            tmp = data_mask(tmp, cut)
            all_data = [tmp]
            a.set_gen(data_shape(tmp))
        a.add_gen(data_shape(data))
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    status = (a, max_weight)

    return ret, status


def single_sampling2(phsp, amp, N, max_weight=None, importance_f=None):
    data = phsp(N)
    weight = amp(data)
    if importance_f is not None:
        weight = weight / importance_f(data)
    new_max_weight = tf.reduce_max(weight)
    if max_weight is None or max_weight < new_max_weight:
        max_weight = new_max_weight * 1.01
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max_weight < weight
    data = data_mask(data, cut)
    return data, max_weight


@ConfigLoader.register_function()
def generate_phsp_p(config, N=1000):
    decay_group = config.get_decay()

    m0, mi, idx = build_phsp_chain(decay_group)

    pi = generate_phsp_o(m0, mi, N=N)

    def loop_index(tree, idx):
        for i in idx:
            tree = tree[i]
        return tree

    return {k: loop_index(pi, idx[k]) for k in decay_group.outs}


@ConfigLoader.register_function()
def generate_phsp(config, N=1000, include_charge=False):
    p = generate_phsp_p(config, N)
    charge = gen_random_charge(N, include_charge)
    ret = config.data.cal_angle(p, charge=charge)
    if include_charge:
        ret["charge_conjugation"] = charge
    return ret


def build_phsp_chain(decay_group):
    struct = decay_group.topology_structure()
    inner_node = [set(i.inner) for i in struct]
    a = inner_node[0]
    for i in inner_node[1:]:
        a = a & i

    m0 = decay_group.top.get_mass()
    mi = [i.get_mass() for i in decay_group.outs]

    if any(i is None for i in [m0] + mi):
        raise ValueError("mass required to generate phase space")

    m0 = float(m0)
    mi = [float(i) for i in mi]

    if len(a) == 0:
        return m0, mi, {k: (v,) for v, k in enumerate(decay_group.outs)}

    # print(type(decay_group.get_particle("D")))
    # print([type(i) for i in decay_group[0].inner])

    ref_dec = decay_group[0]

    decay_map = struct[0].topology_map(ref_dec)
    # print(decay_map)
    nodes = []
    for i in a:
        if get_particle_model_name(decay_map[i]) == "one":
            nodes.append((i, float(decay_map[i].get_mass())))

    mi = dict(zip(decay_group.outs, mi))

    st = struct[0].sorted_table()
    mi, final_idx = build_phsp_chain_sorted(st, mi, nodes)
    return m0, mi, final_idx


def build_phsp_chain_sorted(st, final_mi, nodes):
    """
    {A: [B,C, D], R: [B,C]} + {R: M} => ((mr, (mb, mc)), md)
    """
    st = copy.deepcopy(st)
    for i in final_mi:
        del st[i]
    mass_table = final_mi.copy()
    final_idx = {}
    index_root_map = {}

    nodes = sorted(nodes, key=lambda x: x[1])  # lower mass
    nodes.append(("top", 1 + sum([i[1] for i in nodes])))
    st["top"] = [i for i in final_mi]

    decay_map = {}

    for pi, mi in nodes:
        sub_node = st[pi]
        assert all(
            i in mass_table for i in sub_node
        ), "unexcepted node {}".format(sub_node)
        mass_table[pi] = (mi, [mass_table[i] for i in sub_node])
        for k, i in enumerate(sub_node):
            final_idx[i] = (k,)
            decay_map[i] = pi
            del mass_table[i]
        new_idx = {}
        for k, v in final_idx.items():
            if k not in sub_node:
                new_idx[k] = (*final_idx[decay_map[k]], *final_idx[k])
                decay_map[k] = pi  # decay_map[decay_map[k]]
        final_idx.update(new_idx)

        for k, v in st.items():
            if k == pi:
                continue
            if all(i in v for i in sub_node):
                for n in sub_node:
                    st[k].remove(n)
                st[k].append(pi)

    return mass_table["top"][1], final_idx
