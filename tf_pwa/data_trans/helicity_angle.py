import numpy as np
import tensorflow as tf

from tf_pwa.amp import get_relative_p
from tf_pwa.angle import LorentzVector as lv


class HelicityAngle1:
    """simple implement for angle to monmentum trasform"""

    def __init__(self, decay_chain):
        self.decay_chain = decay_chain
        self.par = [i.outs[1] for i in self.decay_chain]
        self.par.append(list(self.decay_chain)[-1].outs[0])

    def generate_p(self, ms, costheta, phi):
        msp = [i.get_mass() for i in self.par[:-1]]
        return generate_p(ms, msp, costheta, phi)

    def generate_p2(self, ms, costheta, phi):
        all_ms = (
            [self.decay_chain[0].core.get_mass()]
            + ms
            + [self.decay_chain[-1].outs[0].get_mass()]
        )
        return self.generate_p(all_ms, costheta, phi)

    def generate_p_mass(self, name, m, random=False):
        """generate monmentum with M_name = m"""
        m = tf.convert_to_tensor(m, tf.float64)
        ms = [
            i.core.get_mass() if str(i.core) != str(name) else m
            for i in self.decay_chain
        ]
        ms.append(self.par[-1].get_mass())
        if random:
            costheta = [
                np.random.random(m.shape) * 2 - 1 for i in self.decay_chain
            ]
            phi = [
                np.random.random(m.shape) * 2 * np.pi for i in self.decay_chain
            ]
        else:
            costheta = [tf.zeros_like(m) for i in self.decay_chain]
            phi = [tf.zeros_like(m) for i in self.decay_chain]
        ret = self.generate_p(ms, costheta, phi)
        return dict(zip(self.par, ret))

    def get_phsp_factor(self, name, m):
        m = tf.convert_to_tensor(m, tf.float64)
        ms = [
            i.core.get_mass() if str(i.core) != str(name) else m
            for i in self.decay_chain
        ]
        ms.append(self.par[-1].get_mass())
        msp = [i.get_mass() for i in self.par[:-1]]
        n_decay = len(msp)
        ps = [get_relative_p(ms[i], ms[i + 1], msp[i]) for i in range(n_decay)]
        ret = tf.ones_like(m)
        for i in ps:
            ret = ret * tf.cast(i, ret.dtype)
        return ret


class HelicityAngle:
    """general implement for angle to monmentum trasform"""

    def __init__(self, decay_chain):
        self.decay_chain = decay_chain

    def get_all_mass(self, replace_mass):
        ms = {}
        for i in self.decay_chain:
            for j in [i.core] + list(i.outs):
                if j not in ms:
                    if str(j) in replace_mass:
                        ms[j] = tf.convert_to_tensor(
                            replace_mass[str(j)], tf.float64
                        )
                    else:
                        ms[j] = tf.convert_to_tensor(j.get_mass(), tf.float64)
        return ms

    def generate_p_mass(self, name, m, random=False):
        """generate monmentum with M_name = m"""
        m = tf.convert_to_tensor(m, tf.float64)
        ms = self.get_all_mass({name: m})
        data = {}

        for i in self.decay_chain:
            data[i] = {}
            data[i]["|p|"] = get_relative_p(
                ms[i.core], ms[i.outs[0]], ms[i.outs[1]]
            )
            if random:
                costheta = np.random.random(m.shape) * 2 - 1
                phi = np.random.random(m.shape) * 2 * np.pi
            else:
                costheta = tf.zeros_like(m)
                phi = tf.zeros_like(m)
            data[i][i.outs[0]] = {
                "angle": {"alpha": phi, "beta": tf.acos(costheta)}
            }
        ret = create_rotate_p_decay(self.decay_chain, ms, data)
        # ret = self.generate_p(ms, costheta, phi)
        return ret  # dict(zip(self.par, ret))

    def build_data(self, ms, costheta, phi):
        """generate monmentum with M_name = m"""
        data = {}
        for j, i in enumerate(self.decay_chain):
            data[i] = {}
            data[i]["|p|"] = get_relative_p(
                ms[i.core], ms[i.outs[0]], ms[i.outs[1]]
            )
            costheta_i = costheta[j]
            phi_i = phi[j]
            data[i][i.outs[0]] = {
                "angle": {"alpha": phi_i, "beta": tf.acos(costheta_i)}
            }
        ret = create_rotate_p_decay(self.decay_chain, ms, data)
        # ret = self.generate_p(ms, costheta, phi)
        return ret  # dict(zip(self.par, ret))

    def get_phsp_factor(self, name, m):
        m = tf.convert_to_tensor(m, tf.float64)
        ms = self.get_all_mass({name: m})
        ps = []
        for i in self.decay_chain:
            ps.append(get_relative_p(ms[i.core], ms[i.outs[0]], ms[i.outs[1]]))
        ret = tf.ones_like(m)
        for i in ps:
            ret = ret * tf.cast(i, ret.dtype)
        return ret

    def get_mass_range(self, name):
        name = str(name)
        low_bound = None
        high_bound = None
        for i in self.decay_chain:
            if str(i.core) == name:
                low_bound = sum([j.get_mass() for j in i.outs])
            if name in [str(j) for j in i.outs]:
                sum_mass = 0.0
                for j in i.outs:
                    if str(j) != name:
                        sum_mass = sum_mass + j.get_mass()
                high_bound = i.core.get_mass() - sum_mass
        return (low_bound, high_bound)

    def find_variable(self, dat):
        decay_chain = self.decay_chain.standard_topology()
        topo_map = decay_chain.topology_map(self.decay_chain)

        mi = {decay_chain.top: dat["particle"][decay_chain.top]["m"]}
        m2 = {i.outs[1]: dat["particle"][i.outs[1]]["m"] for i in decay_chain}
        m1 = {i.outs[0]: dat["particle"][i.outs[0]]["m"] for i in decay_chain}
        ang = [
            dat["decay"][decay_chain][i][i.outs[0]]["ang"] for i in decay_chain
        ]
        costheta = [tf.cos(i["beta"]) for i in ang]
        phi = [i["alpha"] for i in ang]
        ms = {**mi, **m1, **m2}
        ms = {topo_map[k]: v for k, v in ms.items()}
        return ms, costheta, phi

    def cal_angle(self, p4):
        from tf_pwa.cal_angle import DecayGroup, cal_angle_from_momentum

        decay_group = DecayGroup([self.decay_chain])
        return cal_angle_from_momentum(p4, decay_group)

    def mass_linspace(self, name, N):
        x_min, x_max = self.get_mass_range(name)
        return np.linspace(x_min + 1e-10, x_max - 1e-10, N)


def normal(p):
    return p / tf.expand_dims(tf.sqrt(tf.reduce_sum(p**2, axis=-1)), axis=-1)


def create_rotate_p(ps, ms, costheta, phi):
    px = np.array([[1, 0, 0]])
    pz = np.array([[0, 0, 1]])
    py = np.array([[0, 1, 0]])
    ret = []
    for p, m, c, i in zip(ps, ms, costheta, phi):
        px_o, py_o, pz_o = px, py, pz
        E = tf.sqrt(m * m + p * p)
        s = tf.sqrt(1 - c**2)
        vx = p * s * tf.cos(i)
        vy = p * s * tf.sin(i)
        vz = p * c
        p_new = (
            tf.expand_dims(vx, axis=-1) * px
            + tf.expand_dims(vy, axis=-1) * py
            + tf.expand_dims(vz, axis=-1) * pz
        )
        E = tf.ones_like(p_new[..., 0]) * E
        # print("p2", vx**2+vy**2+vz**2, np.sum(p_new**2, axis=-1), p**2)
        ret.append(tf.concat([tf.expand_dims(E, axis=-1), p_new], axis=-1))
        pz = normal(p_new)
        py = normal(
            -px_o * tf.expand_dims(tf.sin(i), axis=-1)
            + py_o * tf.expand_dims(np.cos(i), axis=-1)
        )
        px = np.cross(py, pz)
    return ret


def create_rotate_p_decay(decay_chain, mass, data):
    px = np.array([[1, 0, 0]])
    pz = np.array([[0, 0, 1]])
    py = np.array([[0, 1, 0]])
    monmentum_in_rest = {}
    axis_map = {decay_chain.top: [px, py, pz]}
    for _, dec in decay_chain.depth_first():
        p, c, i = (
            data[dec]["|p|"],
            tf.cos(data[dec][dec.outs[0]]["angle"]["beta"]),
            data[dec][dec.outs[0]]["angle"]["alpha"],
        )
        px, py, pz = axis_map[dec.core]
        m1 = mass[dec.outs[0]]
        E1 = tf.sqrt(m1 * m1 + p * p)
        m2 = mass[dec.outs[1]]
        E2 = tf.sqrt(m2 * m2 + p * p)
        s = tf.sqrt(1 - c**2)
        vx = p * s * tf.cos(i)
        vy = p * s * tf.sin(i)
        vz = p * c
        p_new = (
            tf.expand_dims(vx, axis=-1) * px
            + tf.expand_dims(vy, axis=-1) * py
            + tf.expand_dims(vz, axis=-1) * pz
        )
        E1 = tf.ones_like(p_new[..., 0]) * E1
        E2 = tf.ones_like(p_new[..., 0]) * E2
        p1 = tf.concat([tf.expand_dims(E1, axis=-1), p_new], axis=-1)
        p2 = tf.concat([tf.expand_dims(E2, axis=-1), -p_new], axis=-1)
        monmentum_in_rest[dec.outs[0]] = p1
        monmentum_in_rest[dec.outs[1]] = p2
        px_o, py_o, pz_o = px, py, pz
        pz = normal(p_new)
        py = normal(
            -px_o * tf.expand_dims(tf.sin(i), axis=-1)
            + py_o * tf.expand_dims(np.cos(i), axis=-1)
        )
        px = np.cross(py, pz)
        axis_map[dec.outs[0]] = [px, py, pz]
        axis_map[dec.outs[1]] = [px, -py, -pz]
    monmentum = {}
    for _, dec in list(decay_chain.depth_first())[::-1]:
        monmentum[dec.core] = {}
        for i in dec.outs:
            if i in monmentum:
                p_boost = monmentum_in_rest[i]
                for j in monmentum[i]:
                    monmentum[dec.core][j] = lv.boost(
                        monmentum[i][j], lv.boost_vector(p_boost)
                    )
            else:
                monmentum[dec.core][i] = monmentum_in_rest[i]
    return monmentum[decay_chain.top]


def lorentz_neg(pc):
    return tf.concat([pc[..., 0:1], -pc[..., 1:]], axis=-1)


def generate_p(ms, msp, costheta, phi):
    """
    ms(0) -> ms(1) + msp(0), costheta(0), phi(0)
    ms(1) -> ms(2) + msp(1), costheta(1), phi(1)
    ...
    ms(n) -> ms(n+1) + msp(n), costheta(n), phi(n)

    """
    assert len(ms) == len(msp) + 1
    assert len(msp) == len(costheta)
    assert len(msp) == len(phi)
    n_decay = len(msp)

    ms = [tf.convert_to_tensor(i, dtype=tf.float64) for i in ms]
    msp = [tf.convert_to_tensor(i, dtype=tf.float64) for i in msp]
    costheta = [tf.convert_to_tensor(i, dtype=tf.float64) for i in costheta]
    phi = [tf.convert_to_tensor(i, dtype=tf.float64) for i in phi]

    ps = [get_relative_p(ms[i], ms[i + 1], msp[i]) for i in range(n_decay)]
    p_gen = create_rotate_p(ps, ms[1:], costheta, phi)

    p_gen2 = []
    for i in range(n_decay):
        pa = ps[n_decay - i - 1]
        Ea2 = tf.sqrt(msp[n_decay - i - 1] ** 2 + pa**2) + tf.zeros_like(
            p_gen[n_decay - i - 1][..., 0]
        )
        pa2 = tf.concat(
            [tf.expand_dims(Ea2, axis=-1), -p_gen[n_decay - i - 1][..., 1:]],
            axis=-1,
        )
        p_gen2.append(pa2)

    ret = [p_gen[-1]]
    for i in range(n_decay - 1):
        boost_p = p_gen[n_decay - i - 2]
        pa = lv.boost(p_gen2[i], lv.boost_vector(boost_p))
        ret = [lv.boost(j, lv.boost_vector(boost_p)) for j in ret]
        ret.append(pa)
    ret.append(p_gen2[-1])
    return ret[::-1]
