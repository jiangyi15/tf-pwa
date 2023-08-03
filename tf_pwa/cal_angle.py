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
import itertools

import numpy as np

from .angle import SU2M, EulerAngle, LorentzVector, Vector3, _epsilon
from .config import get_config
from .data import (
    HeavyCall,
    LazyCall,
    data_index,
    data_merge,
    data_shape,
    data_strip,
    data_to_numpy,
    flatten_dict_data,
    load_dat_file,
    split_generator,
)
from .histogram import Hist1D
from .particle import (
    BaseDecay,
    BaseParticle,
    DecayChain,
    DecayGroup,
    cp_charge_group,
)
from .tensorflow_wrapper import tf


class CalAngleData(dict):
    def get_decay(self):
        return DecayGroup(list(self["decay"].keys()))

    def get_mass(self, name):
        return data_index(self, ("particle", name, "m"))

    def get_momentum(self, name):
        return data_index(self, ("particle", name, "p"))

    def get_weight(self):
        if "weight" in self:
            return self["weight"]
        return tf.ones(data_shape(self), dtype=get_config("dtype"))

    def get_angle(self, decay, p):
        """get hilicity angle of decay which product particle p"""
        if isinstance(decay, str):
            decay = self.get_decay().get_decay_chain(decay)
        dec = decay.standard_topology()
        dec_map = decay.topology_map()
        dec_i = decay[0]
        for i in decay:
            if str(p) in [str(j) for j in i.outs]:
                dec_i = i
                break
        p_name = data_index(dec_map, p)
        dec_name = dec_map[dec_i]
        return data_index(self, ("decay", dec, dec_name, p_name, "ang"))

    def mass_hist(self, name, bins="sqrt", **kwargs):
        data = data_to_numpy(self.get_mass(name))
        return Hist1D.histogram(data, bins=bins, **kwargs)

    def savetxt(
        self, file_name, order=None, cp_trans=False, save_charge=False
    ):
        if order is None:
            order = self.get_decay().outs
        pi = [data_to_numpy(self.get_momentum(i)) for i in order]
        if cp_trans:
            c = tf.cast(self["charge_conjugation"] > 0, tf.float64) * 2 - 1
            pi = [
                np.stack(
                    [i[:, 0], c * i[:, 1], c * i[:, 2], c * i[:, 3]], axis=-1
                )
                for i in pi
            ]
        pi = np.stack(pi).transpose((1, 0, 2)).reshape((-1, 4))
        np.savetxt(file_name, pi)
        if save_charge:
            np.savetxt(
                file_name[::-1].replace(".", ".c", 1)[::-1],
                self["charge_conjugation"],
            )


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
    weight = [weight] * data_size
    data["weight"] = np.array(weight)
    return data


def cal_chain_boost(data, decay_chain: DecayChain) -> dict:
    """
    calculate chain boost for a decay chain
    """
    part_data = {}
    core_decay_map = {}

    decay_set = list(decay_chain)
    particle_set = set(decay_chain.inner) | set(decay_chain.outs)
    while len(decay_set) > 0:
        tmp_decay_set = []
        for i in decay_set:
            if i.core == decay_chain.top:
                part_data[i] = {}
                p_rest = data[i.core]["p"]
                part_data[i]["rest_p"] = {}
                for j in i.outs:
                    core_decay_map[j] = i
                    pj = data[j]["p"]
                    p = LorentzVector.rest_vector(p_rest, pj)
                    part_data[i]["rest_p"][j] = p
                    particle_set.remove(j)
                for j in particle_set:
                    pj = data[j]["p"]
                    p = LorentzVector.rest_vector(p_rest, pj)
                    part_data[i]["rest_p"][j] = p
            elif i.core in core_decay_map:
                part_data[i] = {}
                p_rest = part_data[core_decay_map[i.core]]["rest_p"][i.core]
                part_data[i]["rest_p"] = {}
                for j in i.outs:
                    core_decay_map[j] = i
                    pj = part_data[core_decay_map[i.core]]["rest_p"][j]
                    p = LorentzVector.rest_vector(p_rest, pj)
                    part_data[i]["rest_p"][j] = p
                    particle_set.remove(j)
                for j in particle_set:
                    pj = part_data[core_decay_map[i.core]]["rest_p"][j]
                    p = LorentzVector.rest_vector(p_rest, pj)
                    part_data[i]["rest_p"][j] = p
            else:
                tmp_decay_set.append(i)
        decay_set = tmp_decay_set
    # from pprint import pprint
    # pprint(part_data)
    # exit()
    return part_data


def cal_single_boost(data, decay_chain: DecayChain) -> dict:
    part_data = {}
    # part_data = cal_chain_boost(data, decay_chain)
    for i in decay_chain:
        part_data[i] = {}
        p_rest = data[i.core]["p"]
        part_data[i]["rest_p"] = {}
        for j in i.outs:
            pj = data[j]["p"]
            p = LorentzVector.rest_vector(p_rest, pj)
            part_data[i]["rest_p"][j] = p
    return part_data


# from pysnooper import snoop
# @snoop()
def cal_helicity_angle(
    data: dict,
    decay_chain: DecayChain,
    base_z=np.array([0.0, 0.0, 1.0]),
    base_x=np.array([1.0, 0.0, 0.0]),
) -> dict:
    """
    Calculate helicity angle for A -> B + C: :math:`\\theta_{B}^{A}, \\phi_{B}^{A}` from momentum.

    from `{A:{p:momentum},B:{p:...},C:{p:...}}`

    to   `{A->B+C:{B:{"ang":{"alpha":...,"beta":...,"gamma":...},"x":...,"z"},...}}`
    """
    ret = {}
    # boost all in them mother rest frame

    # print(decay_chain, part_data)
    part_data = cal_chain_boost(data, decay_chain)
    # calculate angle and base x,z axis from mother particle rest frame momentum and base axis
    set_x = {decay_chain.top: base_x}
    set_z = {decay_chain.top: base_z}
    r_matrix = {}
    b_matrix = {}
    set_decay = list(decay_chain)
    while set_decay:
        extra_decay = []
        for i in set_decay:
            if i.core in set_x:
                ret[i] = {}
                bias = -np.pi
                for j in i.outs:
                    ret[i][j] = {}
                    p_rest = part_data[i]["rest_p"][j]
                    z2 = LorentzVector.vect(p_rest)
                    ang, x = EulerAngle.angle_zx_z_getx(
                        set_z[i.core], set_x[i.core], z2
                    )
                    set_x[j] = x
                    set_z[j] = z2
                    # set range to make sure opposite allow be - phi
                    ang["alpha"] = (ang["alpha"] - bias) % (2 * np.pi) + bias
                    bias -= np.pi
                    ret[i][j]["ang"] = ang
                    ret[i][j]["x"] = x
                    ret[i][j]["z"] = z2
                    Bp = SU2M.Boost_z_from_p(p_rest)
                    b_matrix[j] = Bp
                    r = SU2M.Rotation_y(ang["beta"]) * SU2M.Rotation_z(
                        ang["alpha"]
                    )
                    if i.core in r_matrix:
                        r_matrix[j] = r * b_matrix[i.core] * r_matrix[i.core]
                    else:
                        r_matrix[j] = r
                if len(i.outs) == 3:
                    # Euler Angle for
                    p_rest = [part_data[i]["rest_p"][j] for j in i.outs]
                    zi = [LorentzVector.vect(i) for i in p_rest]
                    ret[i]["ang"], xi = EulerAngle.angle_zx_zzz_getx(
                        set_z[i.core], set_x[i.core], zi
                    )
                    for j, x, z, p_rest_i in zip(i.outs, xi, zi, p_rest):
                        ret[i][j] = {}
                        ret[i][j]["x"] = x
                        ret[i][j]["z"] = z
                        Bp = SU2M.Boost_z_from_p(p_rest_i)
                        b_matrix[j] = Bp
                        r = SU2M.Rotation_y(ang["beta"]) * SU2M.Rotation_z(
                            ang["alpha"]
                        )
                        if i.core in r_matrix:
                            r_matrix[j] = (
                                r * b_matrix[i.core] * r_matrix[i.core]
                            )
                        else:
                            r_matrix[j] = r
            else:
                extra_decay.append(i)
        set_decay = extra_decay
    ret["r_matrix"] = r_matrix
    ret["b_matrix"] = b_matrix
    return ret


def aligned_angle_ref_rule1(decay_group, decay_chain_struct, decay_data):
    # calculate aligned angle of final particles in each decay chain
    set_x = {}  # reference particles
    ref_matrix = {}

    # for particle from a the top rest frame
    for idx, decay_chain in enumerate(decay_chain_struct):
        for decay in decay_chain:
            if decay.core == decay_group.top:
                for i in decay.outs:
                    if (i not in set_x) and (i in decay_group.outs):
                        part_data2 = decay_data[decay_chain][decay]
                        set_x[i] = (decay_chain, part_data2[i])
                        ref_matrix[i] = decay_chain
    # or in the first chain
    for i in decay_group.outs:
        if i not in set_x:
            decay_chain = next(iter(decay_chain_struct))
            for decay in decay_chain:
                for j in decay.outs:
                    if i == j:
                        part_data2 = decay_data[decay_chain][decay]
                        set_x[i] = (decay_chain, part_data2[i])
                        ref_matrix[i] = decay_chain

    ref_matrix_final = {}
    for i in decay_group.outs:
        ref_matrix_final[i] = {
            "b_matrix": decay_data[ref_matrix[i]]["b_matrix"][i],
            "r_matrix": decay_data[ref_matrix[i]]["r_matrix"][i],
        }

    return set_x, ref_matrix_final


def aligned_angle_ref_rule2(decay_group, decay_chain_struct, decay_data):
    # calculate aligned angle of final particles in each decay chain
    set_x = {}  # reference particles
    ref_matrix = {}

    ref_matrix_final = {}
    for i in decay_group.outs:
        set_x[i] = (
            None,
            {"x": np.array([[1.0, 0, 0]]), "z": np.array([[0.0, 0, 1]])},
        )
        ref_matrix_final[i] = {
            "b_matrix": SU2M([[1, 0], [0, 1]]),
            "r_matrix": SU2M([[1, 0], [0, 1]]),
        }

    return set_x, ref_matrix_final


def cal_angle_from_particle(
    data,
    decay_group: DecayGroup,
    using_topology=True,
    random_z=True,
    r_boost=True,
    final_rest=True,
    align_ref=None,  # "center_mass",
    only_left_angle=False,
):
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
    base_z = np.array([0.0, 0.0, 1.0]) + tf.zeros_like(p3)
    if random_z:
        p3_norm = Vector3.norm(p3)
        mask = tf.expand_dims(p3_norm < 1e-5, -1)
        base_z = tf.where(mask, base_z, p3)
    # calculate chain angle
    for i in decay_chain_struct:
        data_i = cal_helicity_angle(data, i, base_z=base_z)
        decay_data[i] = data_i
    if align_ref == "center_mass":
        set_x, ref_matrix_final = aligned_angle_ref_rule2(
            decay_group, decay_chain_struct, decay_data
        )
    else:
        set_x, ref_matrix_final = aligned_angle_ref_rule1(
            decay_group, decay_chain_struct, decay_data
        )

    for idx, decay_chain in enumerate(decay_chain_struct):
        for decay in decay_chain:
            part_data = decay_data[decay_chain][decay]
            for i in decay.outs:
                if i in decay_group.outs and decay_chain != set_x[i][0]:
                    if r_boost:
                        r_matrix = decay_data[decay_chain]["r_matrix"][i]
                        b_matrix = decay_data[decay_chain]["b_matrix"][i]
                        r_matrix_ref = ref_matrix_final[i][
                            "r_matrix"
                        ]  # decay_data[ref_matrix[i]]["r_matrix"][i]
                        b_matrix_ref = ref_matrix_final[i][
                            "b_matrix"
                        ]  # decay_data[ref_matrix[i]]["b_matrix"][i]
                        R = SU2M(r_matrix_ref["x"]) * SU2M.inv(r_matrix)
                        # print(R)
                        if final_rest:
                            R = (
                                SU2M(b_matrix_ref["x"])
                                * R
                                * SU2M.inv(b_matrix)
                            )
                        ang = R.get_euler_angle()
                    else:
                        idx2, part_data2 = set_x[i]
                        # part_data2 = decay_data[idx2][decay2]
                        x1 = part_data[i]["x"]
                        x2 = part_data2["x"]
                        z1 = part_data[i]["z"]
                        z2 = part_data2["z"]
                        ang = EulerAngle.angle_zx_zx(z1, x1, z2, x2)
                    # ang = AlignmentAngle.angle_px_px(z1, x1, z2, x2)
                    part_data[i]["aligned_angle"] = ang
    ret = data_strip(decay_data, ["r_matrix", "b_matrix", "x", "z"])
    if only_left_angle:
        for i in ret:
            for j in ret[i]:
                del ret[i][j][j.outs[1]]["ang"]
    return ret


cal_angle = cal_angle_from_particle


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
    q = (
        p + tf.abs(p)
    ) / 2  # if p is negative, which results from bad data, the return value is 0.0
    return tf.sqrt(q) / (2 * M_0)


def Getp2(M_0, M_1, M_2):
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
    return p / (4 * M_0 * M_0)


def get_relative_momentum(data: dict, decay_chain: DecayChain):
    """
    add add rest frame momentum scalar from data momentum.

    from `{"particle": {A: {"m": ...}, ...}, "decay": {A->B+C: {...}, ...}`

    to `{"particle": {A: {"m": ...}, ...},"decay": {A->B+C:{...,"|q|": ...},...}`
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


def add_relative_momentum(data: dict):
    """
    add add rest frame momentum scalar from data momentum.

    from `{"particle": {A: {"m": ...}, ...}, "decay": {A->B+C: {...}, ...}`

    to `{"particle": {A: {"m": ...}, ...},"decay": {[A->B+C,...]: {A->B+C:{...,"|q|": ...},...},...}`
    """
    data_p = data["particle"]
    for decay_chain in data["decay"]:
        for decay in decay_chain:
            m0 = data_p[decay.core]["m"]
            m1 = data_p[decay.outs[0]]["m"]
            m2 = data_p[decay.outs[1]]["m"]
            p2 = Getp2(m0, m1, m2)
            data["decay"][decay_chain][decay]["|q|2"] = p2
    return data


def parity_trans(p, charges):
    charges = charges[: p.shape[0], None]
    return tf.where(charges > 0, p, LorentzVector.neg(p))


def prepare_data_from_decay(
    fnames, decs, particles=None, dtype=None, charges=None, **kwargs
):
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
    if charges is not None:
        p = {k: parity_trans(v, charges) for k, v in p.items()}
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
    decs = DecayGroup(
        [
            [BaseDecay(a, [bc, d]), BaseDecay(bc, [b, c])],
            [BaseDecay(a, [bd, c]), BaseDecay(bd, [b, d])],
            [BaseDecay(a, [cd, b]), BaseDecay(cd, [c, d])],
        ]
    )
    # decs = DecayChain.from_particles(a, [d, b, c])
    data = cal_angle_from_momentum(p, decs)
    data = data_to_numpy(data)
    data = flatten_dict_data(data)
    return data


def get_chain_data(data, decay_chain=None):
    """
    get all independent data for a decay chain
    """
    if decay_chain is None:
        decay_chain = list(data["decay"].keys())[0]
    chain_data = data["decay"][decay_chain]
    ret = {"mass": {}, "costheta": {}, "phi": {}}
    for dec in chain_data.keys():
        ret["mass"][dec.core] = data["particle"][dec.core]["m"]
        out1 = dec.outs[0]
        ang = chain_data[dec][out1]["ang"]
        ret["costheta"][dec] = tf.cos(ang["beta"])
        ret["phi"][dec] = ang["alpha"]
    return ret


def cal_angle_from_momentum_base(
    p,
    decs: DecayGroup,
    using_topology=True,
    center_mass=False,
    r_boost=True,
    random_z=False,
    batch=65000,
    align_ref=None,
    only_left_angle=False,
) -> CalAngleData:
    """
    Transform 4-momentum data in files for the amplitude model automatically via DecayGroup.

    :param p: 4-momentum data
    :param decs: DecayGroup
    :return: Dictionary of data
    """
    if data_shape(p) is None:
        return cal_angle_from_momentum_single(
            p,
            decs,
            using_topology,
            center_mass,
            r_boost,
            random_z,
            align_ref=align_ref,
            only_left_angle=only_left_angle,
        )
    ret = []
    for i in split_generator(p, batch):
        ret.append(
            cal_angle_from_momentum_single(
                i,
                decs,
                using_topology,
                center_mass,
                r_boost,
                random_z,
                align_ref=align_ref,
                only_left_angle=only_left_angle,
            )
        )
    return data_merge(*ret)


def identical_particles_swap(id_particles):
    ret = []
    for i in id_particles:
        ret.append(list(itertools.permutations(i)))
    for i in itertools.product(*ret):
        yield i


def identical_particles_swap_p(p4, id_particles):
    old_order = tuple(tuple(i) for i in id_particles)
    for comb in identical_particles_swap(id_particles):
        all_keys = tuple(p4.keys())
        name_map = {str(k): k for k in all_keys}
        swap_map = {}
        for c, p_list in zip(comb, id_particles):
            for ci, pi in zip(c, p_list):
                swap_map[ci] = name_map[pi]
        new_order = tuple([swap_map.get(str(i), i) for i in all_keys])
        if comb == old_order:
            continue
        yield (new_order, comb), dict(zip(new_order, p4.values()))


def cp_swap_p(p4, finals, id_particles, cp_particles):
    cg = cp_charge_group([str(i) for i in finals], id_particles, cp_particles)
    ret = {}
    name_map = {str(i): i for i in p4.keys()}
    for a, b in cg:
        for i, j in zip(a, b):
            ret[name_map[j]] = LorentzVector.neg(p4[name_map[i]])
            if i != j:
                ret[name_map[i]] = LorentzVector.neg(p4[name_map[j]])
    return ret


def cal_angle_from_momentum_id_swap(
    p,
    decs: DecayGroup,
    using_topology=True,
    center_mass=False,
    r_boost=True,
    random_z=False,
    batch=65000,
    align_ref=None,
    only_left_angle=False,
) -> CalAngleData:
    ret = []
    id_particles = decs.identical_particles
    data = cal_angle_from_momentum_base(
        p,
        decs,
        using_topology,
        center_mass,
        r_boost,
        random_z,
        batch,
        align_ref=align_ref,
        only_left_angle=only_left_angle,
    )
    if id_particles is None or len(id_particles) == 0:
        return data
    else:
        data["id_swap"] = {}
        for i, pi in identical_particles_swap_p(p, id_particles):
            data["id_swap"][i] = cal_angle_from_momentum_base(
                pi,
                decs,
                using_topology,
                center_mass,
                r_boost,
                random_z,
                batch,
                align_ref=align_ref,
                only_left_angle=only_left_angle,
            )
        return data


def cal_angle_from_momentum(
    p,
    decs: DecayGroup,
    using_topology=True,
    center_mass=False,
    r_boost=True,
    random_z=False,
    batch=65000,
    align_ref=None,
    only_left_angle=False,
) -> CalAngleData:
    """
    Transform 4-momentum data in files for the amplitude model automatically via DecayGroup.

    :param p: 4-momentum data
    :param decs: DecayGroup
    :return: Dictionary of data
    """
    if isinstance(p, LazyCall):
        return LazyCall(
            HeavyCall(cal_angle_from_momentum),
            p,
            decs=decs,
            using_topology=using_topology,
            center_mass=center_mass,
            r_boost=r_boost,
            random_z=random_z,
            align_ref=align_ref,
            only_left_angle=only_left_angle,
            batch=batch,
        )
    ret = []
    id_particles = decs.identical_particles
    cp_particles = decs.cp_particles
    data = cal_angle_from_momentum_id_swap(
        p,
        decs,
        using_topology,
        center_mass,
        r_boost,
        random_z,
        batch,
        align_ref=align_ref,
        only_left_angle=only_left_angle,
    )
    if cp_particles is None or len(cp_particles) == 0:
        return data
    else:
        p2 = cp_swap_p(p, decs.outs, id_particles, cp_particles)
        data["cp_swap"] = cal_angle_from_momentum_id_swap(
            p2,
            decs,
            using_topology,
            center_mass,
            r_boost,
            random_z,
            batch,
            align_ref=align_ref,
            only_left_angle=only_left_angle,
        )
        return data


def cal_angle_from_momentum_single(
    p,
    decs: DecayGroup,
    using_topology=True,
    center_mass=False,
    r_boost=True,
    random_z=True,
    align_ref=None,
    only_left_angle=False,
) -> CalAngleData:
    """
    Transform 4-momentum data in files for the amplitude model automatically via DecayGroup.

    :param p: 4-momentum data
    :param decs: DecayGroup
    :return: Dictionary of data
    """
    p = {BaseParticle(k) if isinstance(k, str) else k: v for k, v in p.items()}
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
    data_d = cal_angle_from_particle(
        data_p,
        decs,
        using_topology,
        r_boost=r_boost,
        random_z=random_z,
        align_ref=align_ref,
        only_left_angle=only_left_angle,
    )
    data = {"particle": data_p, "decay": data_d}
    add_relative_momentum(data)
    return CalAngleData(data)


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


def get_keys(dic, key_path=""):
    """get_keys of nested dictionary

    >>> a = {"a": 1, "b": {"c": 2}}
    >>> get_keys(a)
    ['/a', '/b/c']

    """
    keys_list = []

    def get_keys(dic, key_path):
        if isinstance(dic, dict):
            for i in dic:
                get_keys(dic[i], key_path + "/" + str(i))
        else:
            keys_list.append(key_path)

    get_keys(dic, key_path)
    return keys_list


def get_key_content(dic, key_path):
    """get key content. E.g. get_key_content(data, '/particle/(B, C)/m')

    >>> data = {"particle": {"(B, C)": {"p": 0.1, "m": 1.0},"B":1.0}}
    >>> get_key_content(data, '/particle/(B, C)/m')
    1.0

    """
    keys = key_path.strip("/").split("/")

    def get_content(dic, keys):
        if len(keys) == 0:
            return dic
        for k in dic:
            if str(k) == keys[0]:
                ret = get_content(dic[k], keys[1:])
                break
        return ret

    return get_content(dic, keys)
