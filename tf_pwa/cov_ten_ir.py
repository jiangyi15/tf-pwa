import functools

import numpy as np
import tensorflow as tf

from tf_pwa.angle import LorentzVector as lv
from tf_pwa.angle import Vector3 as v3


@functools.lru_cache()
def normal_factor(L):
    from sympy.physics.quantum.cg import CG

    if L < 2:
        return (-1) ** L
    ret = (-1) ** L
    for k in range(1, L):
        ret *= float(CG(k, 0, 1, 0, k + 1, 0).doit().evalf())
    return ret


def xyzToangle(pxyz):
    px, py, pz = tf.unstack(pxyz, axis=-1)
    pxy = px * px + py * py
    theta = tf.where(pxy < 1e-16, tf.zeros_like(pz), tf.math.acos(pz))
    phi = (tf.math.atan2(py, px)) % (2 * np.pi)
    phi = tf.where(pxy < 1e-16, tf.zeros_like(pz), phi)
    theta = tf.where(tf.math.is_nan(theta), tf.zeros_like(theta), theta)
    phi = tf.where(tf.math.is_nan(phi), tf.zeros_like(phi), phi)
    return theta, phi


@functools.lru_cache()
def cg_in_amp0ls(s1, lens1, s2, lens2, s, lens, S, L):
    from sympy.physics.quantum.cg import CG

    ret = np.zeros((lens, lens1, lens2))
    for i in range(lens):
        for is1 in range(lens1):
            for is2 in range(lens2):
                sig1 = is1 - s1
                sig2 = is2 - s2
                sigma = i - s
                m = sigma - sig1 - sig2
                if abs(sig1 + sig2) > S or abs(sigma - sig1 - sig2) > L:
                    ret[i, is1, is2] = 0
                else:
                    ret[i, is1, is2] = float(
                        (
                            CG(s1, sig1, s2, sig2, S, sig1 + sig2)
                            * CG(
                                S,
                                sig1 + sig2,
                                L,
                                sigma - sig1 - sig2,
                                s,
                                sigma,
                            )
                        )
                        .doit()
                        .evalf()
                    )
    return ret


def sphericalHarmonic(l, theta, phi):
    from tf_pwa.dfun import get_D_matrix_lambda

    angle = {"alpha": phi, "beta": theta, "gamma": tf.zeros_like(theta)}
    d = np.sqrt((2 * l + 1) / 4 / np.pi) * get_D_matrix_lambda(
        angle, l, tuple(list(range(-l, l + 1))), (0,), (0,)
    )
    ret = tf.reshape(d, (-1, 2 * l + 1))
    return ret


@functools.lru_cache()
def delta_idx_in_amp0ls(s1, lens1, s2, lens2, s, lens, l):
    ret = np.zeros((lens, lens1, lens2), dtype=np.int32)
    for i in range(lens):
        for is1 in range(lens1):
            for is2 in range(lens2):
                sig1 = is1 - s1
                sig2 = is2 - s2
                sigma = i - s
                m = sigma - sig1 - sig2
                idx = int(m + l)
                if idx < 0 or idx >= 2 * l + 1:
                    idx = 2 * l + 1
                ret[i, is1, is2] = idx
    return ret


def amp0ls(s1, lens1, s2, lens2, s, lens, theta, phi, S, L):
    cg_matrix = cg_in_amp0ls(s1, lens1, s2, lens2, s, lens, S, L)
    idx = delta_idx_in_amp0ls(s1, lens1, s2, lens2, s, lens, L)
    sh = sphericalHarmonic(L, theta, -phi)
    sh = tf.pad(sh, [[0, 0], [0, 1]], "CONSTANT")
    result = cg_matrix * tf.reshape(
        tf.gather(sh, tf.reshape(idx, (-1,)), axis=-1), (-1, *cg_matrix.shape)
    )
    return result


def MasslessTransAngle(p1, p2):
    p1xyz = lv.vect(p1)
    p2xyz = lv.vect(p2)
    q1 = tf.sqrt(v3.norm2(p1xyz))
    q2 = tf.sqrt(v3.norm2(p2xyz))
    bias = tf.where(q1 < 1e-16, tf.ones_like(q1), tf.zeros_like(q1))[..., None]
    p1xyz = p1xyz + bias * np.array([0.0, 0.0, 1.0])
    q1n = tf.sqrt(v3.norm2(p1xyz))
    n1 = -p1xyz / q1n[..., None]
    n2 = p2xyz / q2[..., None]
    psi_1 = tf.math.atan2(n1[..., 1], n1[..., 0]) % (2 * np.pi)
    n1x, n1y, n1z = tf.unstack(n1, axis=-1)
    n2x, n2y, n2z = tf.unstack(n2, axis=-1)
    beta2 = v3.norm2(lv.boost_vector(p1))
    gamma = 1 / tf.sqrt(1 - beta2)
    beta = tf.sqrt(beta2)
    c3 = v3.dot(n1, n2)
    delta1 = n1z * (gamma - 1) + n2z * gamma * beta
    delta2 = n2z + n1z * (c3 * (gamma - 1) + gamma * beta)
    delta = tf.sqrt(
        (1 - n2z * n2z) * ((gamma + c3 * gamma * beta) ** 2 - delta2 * delta2)
    )
    c4 = n1x * n2y - n2x * n1y
    psi_2 = -tf.math.atan2(
        delta1 * c4 / delta,
        ((n2z * c3 - n1z) * delta1 + (1 + c3 * beta) * gamma * (1 - n2z * n2z))
        / delta,
    )
    psi = tf.where(
        (1 - n2z * n2z) <= 1e-16,
        tf.where((1 - n1z * n1z) <= 1e-16, tf.zeros_like(psi_1), psi_1),
        tf.where(delta <= 1e-16, tf.zeros_like(psi_2), psi_2),
    )
    return psi


def MassiveTransAngle(p1, p2):
    p1xyz = p1[..., -3:]
    p2xyz = p2[..., -3:]
    nhat = v3.cross_unit(p1xyz, p2xyz)
    theta, phi = xyzToangle(nhat)
    m1 = tf.sqrt(tf.abs(lv.M2(p1)))
    m2 = tf.sqrt(tf.abs(lv.M2(p2)))
    gamma1 = p1[:, 0] / tf.where(m1 < 1e-16, tf.ones_like(m1), m1)
    gamma2 = p2[:, 0] / tf.where(m2 < 1e-16, tf.ones_like(m2), m2)
    cosx = -v3.cos_theta(p1xyz, p2xyz)
    har = v3.dot(p1xyz, p2xyz)
    cospsi_num = (1 - cosx * cosx) * (gamma1 - 1) * (gamma2 - 1)
    cospsi_dom = 1 - har / (m1 * m2) + gamma1 * gamma2
    cospsi = 1 - cospsi_num / tf.where(
        tf.abs(cospsi_dom) < 1e-16, tf.ones_like(cospsi_dom), cospsi_dom
    )
    psi = tf.math.acos(tf.clip_by_value(cospsi, -1, 1))
    psi = tf.where(tf.math.is_nan(psi), tf.zeros_like(psi), psi)
    return theta, phi, psi


def wigerDx(j, alpha, beta, gamma):
    from tf_pwa.dfun import D_matrix_conj

    return D_matrix_conj(-alpha, beta, -gamma, j)


def PWFA(p1, m1_zero, s1, p2, m2_zero, s2, s, S, L):
    lens1 = int(2 * s1 + 1)
    lens2 = int(2 * s2 + 1)
    lens = int(2 * s + 1)
    p1xyz = p1[..., -3:]
    p2xyz = p2[..., -3:]
    # double m1 = (mu1 <= 1e-16) ? mu1 : (std::sqrt(p1(0)*p1(0)-p1xyz.dot(p1xyz)));
    # double m2 = (mu2 <= 1e-16) ? mu2 : (std::sqrt(p2(0)*p2(0)-p2xyz.dot(p2xyz)));
    m1 = lv.M(p1)
    m2 = lv.M(p2)
    p0 = p1 + p2
    p12 = lv.Dot(p1, p2)
    m = lv.M(p0)
    lam = (m * m + m1 * m1 - m2 * m2 + 2 * m * p1[:, 0]) / (
        2.0 * m * (m + p1[:, 0] + p2[:, 0])
    )
    qs = tf.sqrt(
        m**4 + (m1 * m1 - m2 * m2) ** 2 - 2 * m * m * (m1 * m1 + m2 * m2)
    ) / (2.0 * m)
    ns = (p1xyz - (p1xyz + p2xyz) * lam[:, None]) / qs[:, None]
    theta, phi = xyzToangle(ns)
    result = amp0ls(s1, lens1, s2, lens2, s, lens, theta, phi, S, L)
    # MatrixXd wigerdj1 = wigerdjx(s1,lens1);
    # MatrixXd wigerdj2 = wigerdjx(s2,lens2);
    # MatrixXcd trans1 = MatrixXcd::Zero(lens1, lens1);
    # MatrixXcd trans2 = MatrixXcd::Zero(lens2, lens2);
    if m1_zero:
        psix1 = MasslessTransAngle(p0, p1)
        trans1 = wigerDx(lens1 - 1, phi, theta, -psix1)
    else:
        thetay1, phiy1, psiy1 = MassiveTransAngle(p0, p1)
        trans1 = tf.matmul(
            wigerDx(lens1 - 1, phiy1, thetay1, psiy1),
            wigerDx(lens1 - 1, tf.zeros_like(phiy1), -thetay1, -phiy1),
        )
    if m2_zero:
        psix2 = MasslessTransAngle(p0, p2)
        trans2 = wigerDx(lens2 - 1, np.pi + phi, np.pi - theta, -psix2)
    else:
        thetay2, phiy2, psiy2 = MassiveTransAngle(p0, p2)
        trans2 = tf.matmul(
            wigerDx(lens2 - 1, phiy2, thetay2, psiy2),
            wigerDx(lens2 - 1, tf.zeros_like(phiy2), -thetay2, -phiy2),
        )
    factor = qs**L / np.sqrt(2 * s + 1)
    a = tf.einsum(
        "...ijk,...jl,...km->...ilm",
        tf.cast(result, trans1.dtype),
        trans1,
        trans2,
    )
    return a * tf.cast(factor[..., None, None, None], trans1.dtype)

    # Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten1(trans1.data(), trans1.rows(), trans1.cols());
    # Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 2>> ten2(trans2.data(), trans2.rows(), trans2.cols());
    # Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    # Eigen::Tensor<std::complex<double>, 3> resultx = result.contract(ten1, product_dims);
    # Eigen::Tensor<std::complex<double>, 3> resulty = resultx.contract(ten2, product_dims);
    # return  factor*resulty;


# 三粒子分波振幅中线性独立的LS组合的数目: i=0 代表三个粒子均有质量的情况; i=1 代表粒子2无质量的情况; i=2 \
# 代表粒子2和3均无质量的情况; i=3 代表初末态三粒子均无质量的情况。


def NumSL0(s1, s2, s3):
    if s1 <= (s2 - s3):
        return (2 * s1 + 1) * (2 * s3 + 1)
    if s1 <= (s3 - s2):
        return (2 * s1 + 1) * (2 * s2 + 1)
    if s1 >= (s2 + s3):
        return (2 * s2 + 1) * (2 * s3 + 1)
    return (
        -(s1**2 + s2**2 + s3**2)
        + 2 * (s1 * s2 + s2 * s3 + s1 * s3)
        + s1
        + s2
        + s3
        + 1
    )


def NumSL1(s1, s2, s3):
    if s2 == 0:
        return NumSL0(s1, 0, s3)
    if s1 < (s2 - s3):
        return 0
    if s1 <= (s3 - s2):
        return 2 * (2 * s1 + 1)
    if s1 >= (s2 + s3):
        2 * (2 * s3 + 1)
    return 2 * (s1 - s2 + s3 + 1)


def NumSL2(s1, s2, s3):
    if s3 == 0:
        return NumSL1(s1, s2, 0)
    if s2 == 0:
        return NumSL1(s1, s3, 0)
    if s1 < abs(s2 - s3):
        return 0
    if s1 >= (s2 + s3):
        return 4
    return 2


def NumSL3(s1, s2, s3):
    if s1 == 0:
        return NumSL2(0, s2, s3)
    if (s1 - abs(s3 - s2) == 0) or s1 - s2 - s3 == 0:
        return 2
    return 0


def force_int(f):
    def _f(*args, **kwargs):
        ret = f(*args, **kwargs)
        return int(ret)

    return _f


@force_int
def NumSL(s1, s2, s3, i):
    if i == 0:
        return NumSL0(s1, s2, s3)
    if i == 1:
        return NumSL1(s1, s2, s3)
    if i == 2:
        return NumSL2(s1, s2, s3)
    return NumSL3(s1, s2, s3)


# 在选择线性独立的LS组合时, 衡量特定LS组合是否被优先考虑的权重函数


def FS(s1, s2, s3, S):
    return -(s2 + s3 + 1) * abs(S - s1) + S


def FL(s1, s2, s3, S, L):
    return -2 * (s2 + s3 + 1) ** 2 * abs(L - abs(S - s1) - 1 / 2)


def F_Sigma(s1, s2, s3, S, L):
    from sympy.physics.quantum.cg import CG

    if (
        CG(L, 0, S, s2 - s3, s1, s2 - s3).doit() != 0
        or CG(L, 0, S, s2 + s3, s1, s2 + s3).doit() != 0
    ):
        return 0
    return -2 * (s2 + s3 + 1) ** 2 * (s1 + s2 + s3)


def WFunc1(s1, s2, s3, S, L):
    return FS(s1, s2, s3, S)


def WFunc2(s1, s2, s3, S, L):
    return FS(s1, s2, s3, S) + FL(s1, s2, s3, S, L) + F_Sigma(s1, s2, s3, S, L)


def _srange(a, b=None):
    if b is None:
        a, b = 0, a
    x = a
    while x < b:
        yield x
        x += 1


def SCombLS(s1, s2, s3, i):
    """给出一组线性独立且完备的LS组合: i=0 代表三个粒子均有质量的情况; i=1 代表粒子2无质量的情况; i=2
    代表粒子2和3均无质量的情况; i=3 代表初末态三粒子均无质量的情况。输出结果是一个集合,其元素为二元数组 {S,L}"""
    if i == 0 or (i == 1 and s2 == 0):
        res = []
        for S in _srange(abs(s2 - s3), s2 + s3 + 1):
            for L in _srange((abs(s1 - S)), s1 + S + 1):
                res.append((int(L), S))
    else:
        com = []
        for S in _srange(abs(s2 - s3), s2 + s3 + 1):
            for L in _srange(abs(s1 - S), s1 + S + 1):
                if i == 1:
                    com.append((int(L), S, WFunc1(s1, s2, s3, S, L)))
                else:
                    com.append((int(L), S, WFunc2(s1, s2, s3, S, L)))
        even = sorted(
            list(filter(lambda x: x[0] % 2 == 0, com)), key=lambda x: -x[2]
        )
        odd = sorted(
            list(filter(lambda x: x[0] % 2 == 1, com)), key=lambda x: -x[2]
        )
        leven = len(even)
        lodd = len(odd)
        if leven >= lodd:
            list1 = even
            list2 = odd
            lmin = lodd
        else:
            list1 = odd
            list2 = even
            lmin = leven
        list_all = []
        for k in range(1 if lmin == 0 else 2 * lmin):
            if k % 2 == 0:
                list_all.append(list1[k // 2])
            else:
                list_all.append(list2[(k - 1) // 2])
        list_all = list_all[: NumSL(s1, s2, s3, i)]
        res = [(i[0], i[1]) for i in list_all]
    return res


def ls_selector_weight(decay, all_ls):
    p1 = decay.core
    p2 = decay.outs[0]
    p3 = decay.outs[1]

    idx = 0
    if p2.get_mass() == 0:
        idx += 1
    if p3.get_mass() == 0:
        idx += 1
    if p1.get_mass() == 0:
        idx += 1
    if p3.get_mass() == 0:
        independent_ls = SCombLS(p1.J, p3.J, p2.J, idx)
    else:
        independent_ls = SCombLS(p1.J, p2.J, p3.J, idx)
    ret = []
    for i in all_ls:
        found = False
        for j in independent_ls:
            if j[0] - i[0] == 0 and j[1] - i[1] == 0.0:
                found = True
        if found:
            ret.append(i)
    return ret
