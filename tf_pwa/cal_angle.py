"""
This module provides functions which are useful when calculating the angular variables.


The full data structure provided is ::

    {
        "particle": {
            A: {"p": ..., "m": ...},
            (C, D): {"p": ..., "m": ...},
            ...
        },

        "decay":{
            [A->(C, D)+B, (C, D)->C+D]:
            {
                A->(C, D)+B: {
                    (C, D): {
                        "ang": {
                            "alpha":[...],
                            "beta": [...],
                            "gamma": [...]
                        },
                        "z": [[x1,y1,z1],...],
                        "x": [[x2,y2,z2],...]
                    },
                    B: {...}
                },

                (C, D)->C+D: {
                    C: {
                        ...,
                        "aligned_angle": {
                        "alpha": [...],
                        "beta": [...],
                        "gamma": [...]
                    }
                },
                    D: {...}
                },
                A->(B, D)+C: {...},
                (B, D)->B+D: {...}
            },
        ...
        }
    }


Inner nodes are named as tuple of particles.

"""

import numpy as np

from .angle import EulerAngle, LorentzVector, Vector3, _epsilon
from .data import load_dat_file, flatten_dict_data, data_shape, split_generator, data_to_numpy
from .tensorflow_wrapper import tf
from .particle import BaseDecay, BaseParticle, DecayChain, DecayGroup
from .config import get_config


def struct_momentum(p, center_mass=True) -> dict:
    """
    restructure momentum as dict
        {outs:momentum} => {outs:{p:momentum}}
    """
    ret = {}
    if center_mass:
        ps_top = []
        for i in p:
            ps_top.append(p[i])
        p_top = tf.reduce_sum(ps_top, 0)
        for i in p:
            ret[i] = {"p": LorentzVector.rest_vector(p_top, p[i])}
    else:
        for i in p:
            ret[i] = {"p": p[i]}
    return ret


# data process
def infer_momentum(data, decay_chain: DecayChain) -> dict:
    """
    infer momentum of all particles in the decay chain from outer particles momentum.
        {outs:{p:momentum}} => {top:{p:momentum},inner:{p:..},outs:{p:..}}
    """
    st = decay_chain.sorted_table()
    for i in st:
        if i in data:
            continue
        ps = []
        for j in st[i]:
            ps.append(data[j]["p"])
        data[i] = {"p": tf.reduce_sum(ps, 0)}
    return data


def add_mass(data: dict, _decay_chain: DecayChain = None) -> dict:
    """
    add particles mass array for data momentum.
        {top:{p:momentum},inner:{p:..},outs:{p:..}} => {top:{p:momentum,m:mass},...}
    """
    for i in data:
        if isinstance(i, BaseParticle):
            p = data[i]["p"]
            data[i]["m"] = LorentzVector.M(p)
    return data


def add_weight(data: dict, weight: float = 1.0) -> dict:
    """
    add inner data weights for data.
        {...} => {..., "weight": weights}
    """
    data_size = data_shape(data)
    weight = [1.0] * data_size
    data["weight"] = np.array(weight)
    return data


def cal_helicity_angle(data: dict, decay_chain: DecayChain,
                       base_z=np.array([[0.0, 0.0, 1.0]]),
                       base_x=np.array([[1.0, 0.0, 0.0]])) -> dict:
    """
    Calculate helicity angle for A -> B + C: :math:`\\theta_{B}^{A}, \\phi_{B}^{A}` from momentum.
    {A:{p:momentum},B:{p:...},C:{p:...}} =>
        {A->B+C:{B:{"ang":{"alpha":...,"beta":...,"gamma":...},"x":...,"z"},...}}
    """
    part_data = {}
    ret = {}
    # boost all in them mother rest frame
    for i in decay_chain:
        part_data[i] = {}
        p_rest = data[i.core]["p"]
        part_data[i]["rest_p"] = {}
        for j in i.outs:
            pj = data[j]["p"]
            p = LorentzVector.rest_vector(p_rest, pj)
            part_data[i]["rest_p"][j] = p
    # calculate angle and base x,z axis from mother particle rest frame momentum and base axis
    set_x = {decay_chain.top: base_x}
    set_z = {decay_chain.top: base_z}
    set_decay = list(decay_chain)
    while set_decay:
        extra_decay = []
        for i in set_decay:
            if i.core in set_x:
                ret[i] = {}
                for j in i.outs:
                    ret[i][j] = {}
                    z2 = LorentzVector.vect(part_data[i]["rest_p"][j])
                    ang, x = EulerAngle.angle_zx_z_getx(set_z[i.core], set_x[i.core], z2)
                    set_x[j] = x
                    set_z[j] = z2
                    ret[i][j]["ang"] = ang
                    ret[i][j]["x"] = x
                    ret[i][j]["z"] = z2
                if len(i.outs) == 3:
                    # Euler Angle for
                    zi = [LorentzVector.vect(part_data[i]["rest_p"][j]) for j in i.outs]
                    ret[i]["ang"], xi = EulerAngle.angle_zx_zzz_getx(set_z[i.core], set_x[i.core], zi)
                    for j, x, z in zip(i.outs, xi, zi):
                        ret[i][j] = {}
                        ret[i][j]["x"] = x
                        ret[i][j]["z"] = z
            else:
                extra_decay.append(i)
        set_decay = extra_decay
    return ret


def cal_angle_from_particle(data, decay_group: DecayGroup, using_topology=True, random_z=True):
    """
    Calculate helicity angle for particle momentum, add aligned angle.
    
    :params data: dict as {particle: {"p":...}}

    :return: Dictionary of data
    """
    if using_topology:
        decay_chain_struct = decay_group.topology_structure()
    else:
        decay_chain_struct = decay_group
    decay_data = {}

    # get base z axis
    p4 = data[decay_group.top]["p"]
    p3 = LorentzVector.vect(p4)
    base_z = np.array([[0.0, 0.0, 1.0]]) + np.zeros_like(p3)
    if random_z:
        p3_norm = Vector3.norm(p3)
        mask = np.expand_dims(p3_norm < 1e-5, -1)
        base_z = np.where(mask, base_z, p3)
    # calculate chain angle
    for i in decay_chain_struct:
        data_i = cal_helicity_angle(data, i, base_z=base_z)
        decay_data[i] = data_i

    # calculate aligned angle of final particles in each decay chain
    set_x = {}  # reference particles
    # for particle from a the top rest frame
    for idx, decay_chain in enumerate(decay_chain_struct):
        for decay in decay_chain:
            if decay.core == decay_group.top:
                for i in decay.outs:
                    if (i not in set_x) and (i in decay_group.outs):
                        set_x[i] = (decay_chain, decay)
    # or in the first chain
    for i in decay_group.outs:
        if i not in set_x:
            decay_chain = next(iter(decay_chain_struct))
            for decay in decay_chain:
                for j in decay.outs:
                    if i == j:
                        set_x[i] = (decay_chain, decay)
    for idx, decay_chain in enumerate(decay_chain_struct):
        for decay in decay_chain:
            part_data = decay_data[decay_chain][decay]
            for i in decay.outs:
                if i in decay_group.outs and decay != set_x[i][1]:
                    idx2, decay2 = set_x[i]
                    part_data2 = decay_data[idx2][decay2]
                    x1 = part_data[i]["x"]
                    x2 = part_data2[i]["x"]
                    z1 = part_data[i]["z"]
                    z2 = part_data2[i]["z"]
                    ang = EulerAngle.angle_zx_zx(z1, x1, z2, x2)
                    part_data[i]["aligned_angle"] = ang
    return decay_data


def cal_angle(data, decay_group: DecayGroup) -> dict:
    """
    Calculate final particles aligned angle from particle momentum.

    :return: Dictionary of data
    """
    for i in decay_group:
        data = cal_helicity_angle(data, i)
    decay_chain_struct = decay_group.topology_structure()
    set_x = {}
    # for a the top rest frame
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
                        ang = EulerAngle.angle_zx_zx(z1, x1, z2, x2)
                        data[decay][i]["aligned_angle"] = ang
    return data


def Getp(M_0, M_1, M_2):
    """
    Consider a two-body decay :math:`M_0\\rightarrow M_1M_2`. In the rest frame of :math:`M_0`, the momentum of
    :math:`M_1` and :math:`M_2` are definite.

    :param M_0: The invariant mass of :math:`M_0`
    :param M_1: The invariant mass of :math:`M_1`
    :param M_2: The invariant mass of :math:`M_2`
    :return: the momentum of :math:`M_1` (or :math:`M_2`)
    """
    M12S = M_1 + M_2
    M12D = M_1 - M_2
    p = (M_0 - M12S) * (M_0 + M12S) * (M_0 - M12D) * (M_0 + M12D)
    q = (p + tf.abs(p)) / 2  # if p is negative, which results from bad data, the return value is 0.0
    return tf.sqrt(q) / (2 * M_0)


def get_relative_momentum(data: dict, decay_chain: DecayChain):
    """
    add add rest frame momentum scalar from data momentum.
    {"particle": {A: {"m": ...}, ...}, "decay": {A->B+C: {...}, ...}
        => {"particle": {A: {"m": ...}, ...},"decay": {A->B+C:{...,"|q|": ...},...}
    """
    ret = {}
    for decay in decay_chain:
        m0 = data[decay.core]["m"]
        m1 = data[decay.outs[0]]["m"]
        m2 = data[decay.outs[1]]["m"]
        p = Getp(m0, m1, m2)
        ret["decay"][decay] = {}
        ret["decay"][decay]["|q|"] = p
    return ret


def prepare_data_from_decay(fnames, decs, particles=None, dtype=None, **kwargs):
    """
    Transform 4-momentum data in files for the amplitude model automatically via DecayGroup.

    :param fnames: File name(s).
    :param decs: DecayGroup
    :param particles: List of Particle. The final particles.
    :param dtype: Data type.
    :return: Dictionary
    """
    if dtype is None:
        dtype = get_config("dtype")
    if particles is None:
        particles = sorted(decs.outs)
    p = load_dat_file(fnames, particles, dtype=dtype)
    data = cal_angle_from_momentum(p, decs, **kwargs)
    return data


def prepare_data_from_dat_file(fnames):
    """
    [deprecated] angle for amplitude.py
    """
    a, b, c, d = [BaseParticle(i) for i in ["A", "B", "C", "D"]]
    bc, cd, bd = [BaseParticle(i) for i in ["BC", "CD", "BD"]]
    p = load_dat_file(fnames, [d, b, c])
    # st = {b: [b], c: [c], d: [d], a: [b, c, d], r: [b, d]}
    decs = DecayGroup([
        [BaseDecay(a, [bc, d]), BaseDecay(bc, [b, c])],
        [BaseDecay(a, [bd, c]), BaseDecay(bd, [b, d])],
        [BaseDecay(a, [cd, b]), BaseDecay(cd, [c, d])]
    ])
    # decs = DecayChain.from_particles(a, [d, b, c])
    data = cal_angle_from_momentum(p, decs)
    data = data_to_numpy(data)
    data = flatten_dict_data(data)
    return data


def cal_angle_from_momentum(p, decs: DecayGroup, using_topology=True, center_mass=False) -> dict:
    """
    Transform 4-momentum data in files for the amplitude model automatically via DecayGroup.

    :param p: 4-momentum data
    :param decs: DecayGroup
    :return: Dictionary of data
    """
    p = {i: p[i] for i in decs.outs}
    data_p = struct_momentum(p, center_mass=center_mass)
    if using_topology:
        decay_chain_struct = decs.topology_structure()
    else:
        decay_chain_struct = decs
    for dec in decay_chain_struct:
        data_p = infer_momentum(data_p, dec)
        # print(data_p)
        # exit()
        data_p = add_mass(data_p, dec)
    data_d = cal_angle_from_particle(data_p, decs, using_topology)
    data = {"particle": data_p, "decay": data_d}
    return data


def prepare_data_from_dat_file4(fnames):
    """
    [deprecated] angle for amplitude4.py
    """
    a, b, c, d, e, f = [BaseParticle(i) for i in "ABCDEF"]
    bc, cd, bd = [BaseParticle(i) for i in ["BC", "CD", "BD"]]
    p = load_dat_file(fnames, [d, b, c, e, f])
    p = {i: p[i] for i in [b, c, e, f]}
    # st = {b: [b], c: [c], d: [d], a: [b, c, d], r: [b, d]}
    BaseDecay(a, [bc, d])
    BaseDecay(bc, [b, c])
    BaseDecay(a, [cd, b])
    BaseDecay(cd, [c, d])
    BaseDecay(a, [bd, c])
    BaseDecay(bd, [b, d])
    BaseDecay(d, [e, f])
    decs = DecayGroup(a.chain_decay())
    # decs = DecayChain.from_particles(a, [d, b, c])
    data = cal_angle_from_momentum(p, decs)
    data = data_to_numpy(data)
    data = flatten_dict_data(data)
    return data

def get_keys(dic, key_path=''):
    """get_keys of nested dictionary
    """
    keys_list = []
    def get_keys(dic, key_path):
        if type(dic) == dict:
            for i in dic:
                get_keys(dic[i], key_path + "/" + str(i))
        else:
            keys_list.append(key_path)
    get_keys(dic, key_path)
    return keys_list

def get_key_content(dic, key_path):
    """get key content. E.g. get_key_content(data, '/particle/(B, C)/m')
    """
    keys = key_path.strip('/').split('/')
    def get_content(dic, keys):
        if len(keys) == 0:
            return dic
        for k in dic:
            if str(k) == keys[0]:
                ret = get_content(dic[k], keys[1:])
                break
        return ret
    return get_content(dic, keys)