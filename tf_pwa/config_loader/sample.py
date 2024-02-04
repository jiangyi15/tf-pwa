import copy

from tf_pwa.amp.core import get_particle_model_name
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import get_config
from tf_pwa.data import data_mask, data_merge, data_shape
from tf_pwa.generator.generator import BaseGenerator, GenTest, multi_sampling
from tf_pwa.particle import BaseParticle
from tf_pwa.phasespace import ChainGenerator  # as generate_phsp_o
from tf_pwa.tensorflow_wrapper import tf

from .config_loader import ConfigLoader


@ConfigLoader.register_function()
def generate_toy_o(config, N=1000, force=True, max_N=100000):
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
def generate_toy2(config, *args, **kwargs):
    return generate_toy(config, *args, **kwargs)


@ConfigLoader.register_function()
def generate_toy(
    config,
    N=1000,
    force=True,
    gen=None,
    gen_p=None,
    importance_f=None,
    max_N=100000,
    include_charge=False,
    cal_phsp_max=False,
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

            p_gen = get_phsp_generator(config, include_charge=include_charge)
            if cal_phsp_max:
                p_gen.cal_max_weight()
            gen = p_gen.generate

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
    cal_phsp_max=False,
):
    """
    generate toy data momentum.
    """
    if gen_p is None:
        p_gen = config.get_phsp_p_generator()
        if cal_phsp_max:
            p_gen.cal_max_weight()
        gen_p = p_gen.generate

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


class AfterGenerator(BaseGenerator):
    def __init__(self, gen, f_after=lambda x: x):
        self.gen = gen
        self.f_after = f_after

    def generate(self, N):
        ret = self.gen.generate(N)
        return self.f_after(ret)

    def cal_max_weight(self):
        self.gen.cal_max_weight()


@ConfigLoader.register_function()
def get_phsp_p_generator(config, nodes=[]):
    decay_group = config.get_decay()

    m0, mi, idx = build_phsp_chain(decay_group)
    for node in nodes:
        (m0, mi), idx = perfer_node((m0, mi), idx, node)

    chain_gen = ChainGenerator(m0, mi)
    chain_gen.unpack_map = idx

    # pi = chain_gen.generate(N)

    def loop_index(tree, idx):
        for i in idx:
            tree = tree[i]
        return tree

    def f_after(pi):
        return {k: loop_index(pi, idx[k]) for k in decay_group.outs}

    return AfterGenerator(chain_gen, f_after)


@ConfigLoader.register_function()
def generate_phsp_p(config, N=1000, cal_max=False):
    gen = get_phsp_p_generator(config)
    if cal_max:
        gen.cal_max_weight()
    return gen.generate(N)


def create_cal_calangle(config, include_charge=False):
    def f_after(p):
        N = data_shape(p)
        charge = gen_random_charge(N, include_charge)
        ret = config.data.cal_angle(p, charge=charge)
        if include_charge:
            ret["charge_conjugation"] = charge
        return ret

    return f_after


@ConfigLoader.register_function()
def get_phsp_generator(config, include_charge=False, nodes=[]):
    gen_p = get_phsp_p_generator(config, nodes=nodes)
    f_after = create_cal_calangle(config, include_charge=include_charge)
    return AfterGenerator(gen_p, f_after)


@ConfigLoader.register_function()
def generate_phsp(config, N=1000, include_charge=False, cal_max=False):
    gen = get_phsp_generator(config, include_charge=include_charge)
    if cal_max:
        gen.cal_max_weight()
    return gen.generate(N)


def build_phsp_chain(decay_group):
    """
    find common decay those mother particle mass is fixed

    """
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
            if k not in sub_node and decay_map[k] in final_idx:
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


def perfer_node(struct, index, nodes):
    """
    reorder struct to make node exisits in PhaseGenerator
    """
    index2 = {str(k): v for k, v in index.items()}
    used_index = {}
    for i in nodes:
        used_index[str(i)] = index2[str(i)]
    min_index = 0
    for i in used_index.values():
        min_index = min(len(i), min_index)
    node_a = list(used_index.values())[0][: min_index - 1]
    assert all(i[: min_index - 1] == node_a for i in used_index.values())
    node_same_level = []
    for k, v in index2.items():
        if v[: min_index - 1] == node_a:
            node_same_level.append(k)
    node_head = [i for i in node_same_level if i not in used_index]
    node_tail = [i for i in node_same_level if i in used_index]
    all_node = node_head + node_tail
    new_order = dict(zip(all_node, range(len(all_node))))
    order_trans = {}
    for i in node_same_level:
        order_trans[index2[i][min_index]] = new_order[i]
    return trans_node_order(struct, index, order_trans, min_index)


def trans_node_order(struct, index, order_trans, level):
    ret_index = {}
    for k, v in index.items():
        if len(v) >= level:
            a = list(v)
            a[level] = order_trans[a[level]]
            ret_index[k] = tuple(a)
        else:
            ret_index[k] = v

    def create_new_struct(struct, index):
        if isinstance(struct, (list, tuple)):
            m0, mi = struct
            if index == 0:
                new_mi = [None] * len(mi)
                for i, v in enumerate(mi):
                    new_mi[order_trans[i]] = v
                return m0, new_mi
            return m0, [create_new_struct(i, index - 1) for i in mi]
        return struct

    ret_struct = create_new_struct(struct, level)
    return ret_struct, ret_index


def get_SDP_p_generator_legacy(config, node):
    from tf_pwa.generator.generator import ARGenerator
    from tf_pwa.generator.square_dalitz_plot import square_dalitz_cut

    gen_p = config.get_phsp_p_generator()

    decay_chain = config.get_decay(False).get_decay_chain(node)
    for i in decay_chain:
        if str(i.core) == str(node):
            particle_1 = i.outs[0]
            particle_2 = i.outs[1]
        if str(node) in [str(j) for j in i.outs]:
            particle_0 = i.core
            particle_3 = [j for j in i.outs if str(j) != str(node)][0]

    idx_table = decay_chain.sorted_table()

    def node_cut(p):
        p1 = sum([p[i] for i in idx_table[particle_1]])
        p2 = sum([p[i] for i in idx_table[particle_2]])
        p3 = sum([p[i] for i in idx_table[particle_3]])
        return square_dalitz_cut([p1, p2, p3])

    return ARGenerator(gen_p.generate, node_cut, max_weight=1.0)


@ConfigLoader.register_function()
def get_SDP_p_generator(config, node, legacy=True):
    if legacy:
        return get_SDP_p_generator_legacy(config, node)

    from tf_pwa.generator.square_dalitz_plot import SDPGenerator

    decay_chain = config.get_decay(False).get_decay_chain(node)
    decay_group = config.get_decay()

    struct = config.get_decay(False).topology_structure()

    ref_struct = None
    for i in struct:
        if i.topology_same(decay_chain):
            ref_struct = i
    inner_node = [set(i.inner) for i in struct]
    same_inner = inner_node[0]
    for i in inner_node[1:]:
        same_inner = same_inner & i

    for i in decay_chain:
        if str(i.core) == str(node):
            particle_1 = i.outs[0]
            particle_2 = i.outs[1]
        if str(node) in [str(j) for j in i.outs]:
            particle_0 = i.core
            particle_3 = [j for j in i.outs if str(j) != str(node)][0]

    force_inner_node = [
        i
        for i in [particle_1, particle_2, particle_3]
        if i not in decay_chain.outs
    ]
    if particle_0 != decay_chain.top:
        force_inner_node.append(particle_0)

    idx_table = decay_chain.sorted_table()
    decay_map = ref_struct.topology_map(decay_chain)
    st = ref_struct.sorted_table()
    for j in same_inner:
        for i in force_inner_node:
            if idx_table[i] != st[j]:
                force_inner_node.append(decay_map[i])

    nodes = [(i, float(i.get_mass())) for i in force_inner_node]
    mi = [i.get_mass() for i in decay_group.outs]
    mi = dict(zip(decay_chain.outs, mi))
    mi, final_idx = build_phsp_chain_sorted(idx_table, mi, nodes)
    m0 = float(decay_chain.top.get_mass())

    chain_gen = ChainGenerator(m0, mi)
    chain_gen.unpack_map = final_idx

    find_idx = tuple(final_idx[particle_1][1:])
    for i, j in enumerate(chain_gen.idxs):
        if j == find_idx:
            gen_idx = i
    old_phsp = chain_gen.gen[gen_idx]

    mi_order = [
        float(i.get_mass()) for i in [particle_1, particle_2, particle_3]
    ]

    new_order = []
    for i in old_phsp.m_mass:
        for idx, j in enumerate(mi_order):
            if abs(i - j) < 1e-6 and idx not in new_order:
                new_order.append(idx)

    sdp_gen = SDPGenerator(old_phsp.m0, mi_order, legacy=False)

    def reorder(pi):
        return [pi[i] for i in new_order]

    chain_gen.gen[gen_idx] = AfterGenerator(sdp_gen, reorder)

    def loop_index(tree, idx):
        for i in idx:
            tree = tree[i]
        return tree

    def f_after(pi):
        return {k: loop_index(pi, final_idx[k]) for k in decay_group.outs}

    return AfterGenerator(chain_gen, f_after)


@ConfigLoader.register_function()
def generate_SDP_p(config, node, N=1000, legacy=False):
    gen = get_SDP_p_generator(config, node, legacy=legacy)
    return gen.generate(N)


@ConfigLoader.register_function()
def get_SDP_generator(config, node, include_charge=False, legacy=True):
    gen_p = get_SDP_p_generator(config, node, legacy=legacy)
    f_after = create_cal_calangle(config, include_charge=include_charge)
    return AfterGenerator(gen_p, f_after)


@ConfigLoader.register_function()
def generate_SDP(config, node, N=1000, include_charge=False, legacy=True):
    gen = get_SDP_generator(
        config, node, include_charge=include_charge, legacy=legacy
    )
    return gen.generate(N)
