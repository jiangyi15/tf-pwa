import copy

from tf_pwa.amp.core import get_particle_model_name
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.data import data_mask, data_merge, data_shape
from tf_pwa.phasespace import generate_phsp as generate_phsp_o
from tf_pwa.tensorflow_wrapper import tf

from .config_loader import ConfigLoader


@ConfigLoader.register_function()
def sampling(config, N=1000, force=True):
    decay_group = config.get_decay()
    amp = config.get_amplitude()

    def gen(M):
        pi = generate_phsp(config, M)
        return cal_angle_from_momentum(pi, decay_group)

    all_data = []
    n_gen = 0
    test_N = 10 * N
    while N > n_gen:
        data = single_sampling(gen, amp, test_N)
        n_gen += data_shape(data)
        test_N = int(test_N * (N - n_gen + 10) / (n_gen + 10))
        print(n_gen)
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


@ConfigLoader.register_function()
def generate_phsp(config, N=1000):
    decay_group = config.get_decay()

    m0, mi, idx = build_phsp_chain(decay_group)

    pi = generate_phsp_o(m0, mi, N=N)

    def loop_index(tree, idx):
        for i in idx:
            tree = tree[i]
        return tree

    return {k: loop_index(pi, idx[k]) for k in decay_group.outs}


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
    # print(st)
    max_iter = 10
    while nodes and max_iter > 0:
        pi, mi = nodes.pop(0)
        sub_node = st[pi]
        max_iter -= 1
        if all(i in mass_table for i in sub_node):
            index_root_map[pi] = sub_node
            # the order make the following loop work
            mass_table[pi] = (mi, [mass_table[i] for i in sub_node])
            for k, i in enumerate(sub_node):
                final_idx[i] = (k,)
                del mass_table[i]
            for k, v in st.items():
                if k == pi:
                    continue
                if all(i in v for i in sub_node):
                    for n in sub_node:
                        st[k].remove(n)
                    st[k].append(pi)
        else:
            nodes.append((pi, mi))
    # print(mass_table)
    ret = []
    for k, i in enumerate(mass_table):
        if i in final_mi:
            final_idx[i] = (k,)
        else:
            for j in index_root_map[i]:
                final_idx[j] = (k, *final_idx[j])
        ret.append(mass_table[i])
    # assert False
    return ret, final_idx
