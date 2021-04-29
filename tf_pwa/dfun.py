"""
This module provides functions to calculate the Wigner D-function.

:math:`D^{j}_{m_1,m_2}(\\alpha,\\beta,\\gamma)=e^{-im_1\\alpha}d^{j}_{m_1,m_2}(\\beta)e^{-im_2\\gamma}`,
where the expression of the Wigner d-function is

.. math:: d^{j}_{m_1,m_2}(\\beta) = \\sum_{l=0}^{2j} w_{l}^{(j,m_1,m_2)}\\sin^{l}(\\frac{\\beta}{2}) \\cos^{2j-l}(\\frac{\\beta}{2}),

where the weight :math:`w_{l}^{(j,m_1,m_2)}` in each term satisfies

.. math:: w^{(j,m_1,m_2)}_{l} = (-1)^{m_1-m_2+k}\\frac{\\sqrt{(j+m_1)!(j-m_1)!(j+m_2)!(j-m_2)!}}{(j-m_1-k)!(j+m_2-k)!(m_1-m_2+k)!k!}

when :math:`k=\\frac{l+m_2-m_1}{2} \\in [\\max(0,m_2-m_1),\\min(j-m_1,j+m_2)]`, and otherwise
:math:`w^{(j,m_1,m_2)}_{l} = 0`.

"""

import functools
import math

import numpy as np

from .tensorflow_wrapper import tf
import nvtx.plugins.tf as tf_nvtx

def _spin_int(x):
    if isinstance(x, int):
        return x
    return int(x + 0.1)


@functools.lru_cache()
def _tuple_delta_D_trans(j, la, lb, lc):
    ln = _spin_int(2 * j + 1)
    s = np.zeros(shape=(ln, ln, len(la), len(lb), len(lc)))
    for i_a, la_i in enumerate(la):
        for i_b, lb_i in enumerate(lb):
            for i_c, lc_i in enumerate(lc):
                delta = lb_i - lc_i
                if abs(delta) <= j:
                    idx = la_i + j, delta + j, i_a, i_b, i_c
                    s[tuple(map(_spin_int, idx))] = 1.0
    return s


def delta_D_trans(j, la, lb, lc):
    """
    The decay from particle *a* to *b* and *c* requires :math:`|l_b-l_c|\\leqslant j`

    (ja,ja) -> (ja,jb,jc)???
    """
    la, lb, lc = map(tuple, (la, lb, lc))
    ret = _tuple_delta_D_trans(j, la, lb, lc)
    return ret


def delta_D_index(j, la, lb, lc):
    la, lb, lc = map(tuple, (la, lb, lc))
    ret = _tuple_delta_D_index(j, la, lb, lc)
    return ret


@functools.lru_cache()
def _tuple_delta_D_index(j, la, lb, lc):
    ln = _spin_int(2 * j + 1)
    ret = []
    max_idx = ln * ln
    for i_a, la_i in enumerate(la):
        for i_b                                                                                   , lb_i in enumerate(lb):
            for i_c, lc_i in enumerate(lc):
                delta = lb_i - lc_i
                if abs(delta) <= j:
                    ret.append(_spin_int((la_i + j) * ln + delta + j))
                else:
                    ret.append(max_idx)
    return ret


def Dfun_delta(d, ja, la, lb, lc=(0,)):
    """
    The decay from particle *a* to *b* and *c* requires :math:`|l_b-l_c|\\leqslant j`

    :math:`D_{ma,mb-mc} = \\delta[(m1,m2)->(ma, mb,mc))] D_{m1,m2}`
    """
    t = delta_D_trans(ja, la, lb, lc)
    ln = _spin_int(2 * ja + 1)
    t_trans = tf.reshape(t, (ln * ln, len(la) * len(lb) * len(lc)))

    t_cast = tf.cast(t_trans, d.dtype)
    # print(d[0])

    d = tf.reshape(d, (-1, ln * ln))

    ret = tf.matmul(d, t_cast)
    return tf.reshape(ret, (-1, len(la), len(lb), len(lc)))


def Dfun_delta_v2(d, ja, la, lb, lc=(0,)):
    """
    The decay from particle *a* to *b* and *c* requires :math:`|l_b-l_c|\\leqslant j`

    :math:`D_{ma,mb-mc} = \\delta[(m1,m2)->(ma, mb,mc))] D_{m1,m2}`
    """
    idx = delta_D_index(ja, la, lb, lc)
    ln = _spin_int(2 * ja + 1)
    # print(d[0])

    d = tf.reshape(d, (-1, ln * ln))
    over_d = tf.pad(d, [[0, 0], [0, 1]], mode="CONSTANT")
    # print(d.shape, over_d.shape)
    # zeros = tf.zeros((d.shape[0], 1), dtype=d.dtype)

    # over_d = tf.concat([d, zeros], axis=-1)
    ret = tf.gather(over_d, idx, axis=-1)
    return tf.reshape(ret, (-1, len(la), len(lb), len(lc)))


@functools.lru_cache()
def small_d_weight(j):  # the prefactor in the d-function of β
    """
    For a certain j, the weight coefficient with index (:math:`m_1,m_2,l`) is
    :math:`w^{(j,m_1,m_2)}_{l} = (-1)^{m_1-m_2+k}\\frac{\\sqrt{(j+m_1)!(j-m_1)!(j+m_2)!(j-m_2)!}}{(j-m_1-k)!(j+m_2-k)!(m_1-m_2+k)!k!}`,
    and :math:`l` is an integer ranging from 0 to :math:`2j`.

    :param j: Integer :math:`2j` in the formula???
    :return: Of the shape (**j** +1, **j** +1, **j** +1). The indices correspond to (:math:`l,m_1,m_2`)
    """
    ret = np.zeros(shape=(j + 1, j + 1, j + 1))

    def f(x):
        return math.factorial(x >> 1)

    for m in range(-j, j + 1, 2):
        for n in range(-j, j + 1, 2):
            for k in range(max(0, n - m), min(j - m, j + n) + 1, 2):
                l = (2 * k + (m - n)) // 2
                sign = (-1) ** ((k + m - n) // 2)
                tmp = sign * math.sqrt(
                    1.0 * f(j + m) * f(j - m) * f(j + n) * f(j - n)
                )
                tmp /= f(j - m - k) * f(j + n - k) * f(k + m - n) * f(k)
                ret[l][(m + j) // 2][(n + j) // 2] = tmp
    return ret


def small_d_matrix(theta, j):
    """
    The matrix element of :math:`d^{j}(\\theta)` is
    :math:`d^{j}_{m_1,m_2}(\\theta) = \\sum_{l=0}^{2j} w_{l}^{(j,m_1,m_2)}\\sin^{l}(\\frac{\\theta}{2}) \\cos^{2j-l}(\\frac{\\theta}{2})`

    :param theta: Array :math:`\\theta` in the formula
    :param j: Integer :math:`2j` in the formula???
    :return: The d-matrices array. Same length as theta
    """
    theta, small_id = tf_nvtx.ops.start(theta, "small_d_matrix")
    a = tf.reshape(tf.range(0, j + 1, 1), (1, -1))

    half_theta = 0.5 * theta

    sintheta = tf.reshape(tf.sin(half_theta), (-1, 1))
    costheta = tf.reshape(tf.cos(half_theta), (-1, 1))

    a = tf.cast(a, dtype=sintheta.dtype)
    s = tf.pow(sintheta, a)
    c = tf.pow(costheta, j - a)
    sc = s * c
    w = small_d_weight(j)

    w = tf.cast(w, sc.dtype)
    w = tf.reshape(w, (j + 1, (j + 1) * (j + 1)))
    ret = tf.matmul(sc, w)
    # ret = tf.einsum("il,lab->iab", sc, w)

    ret = tf.reshape(ret, (-1, j + 1, j + 1))
    ret = tf_nvtx.ops.end(ret, small_id)
    return ret


def exp_i(theta, mi):
    """
    :math:`e^{im\\theta}`

    :param theta: Array :math:`\\theta` in the formula
    :param mi: Integer or half-integer :math:`m` in the formula\\
    :return: Array of tf.complex. Same length as **theta**
    """
    theta_i = tf.reshape(theta, (-1, 1))
    mi = tf.cast(mi, dtype=theta.dtype)
    m_theta = mi * theta_i
    zeros = tf.zeros_like(m_theta)
    im_theta = tf.complex(zeros, m_theta)
    exp_theta = tf.exp(im_theta)
    return exp_theta

import tf_pwa_op

small_d_matrix = tf_nvtx.ops.trace("small_d_matrix")(tf_pwa_op.small_d)

@tf_nvtx.ops.trace("D_matrix_conj")
def D_matrix_conj(alpha, beta, gamma, j):
    """
    The conjugated D-matrix element with indices (:math:`m_1,m_2`) is

    .. math::
        D^{j}_{m_1,m_2}(\\alpha, \\beta, \\gamma)^\\star =
                        e^{i m_1 \\alpha} d^{j}_{m_1,m_2}(\\beta) e^{i m_2 \\gamma}

    :param alpha: Array
    :param beta: Array
    :param gamma: Array
    :param j: Integer :math:`2j` in the formula
    :return: Array of the conjugated D-matrices. Same shape as **alpha**, **beta**, and **gamma**
    """
    m = tf.reshape(np.arange(-j / 2, j / 2 + 1, 1), (1, -1))
    d = small_d_matrix(beta, j)
    expi_alpha = tf.reshape(exp_i(alpha, m), (-1, j + 1, 1))
    expi_gamma = tf.reshape(exp_i(gamma, m), (-1, 1, j + 1))
    expi_gamma = tf.cast(expi_gamma, dtype=expi_alpha.dtype)
    dc = tf.complex(d, tf.zeros_like(d))
    ret = tf.cast(expi_alpha * expi_gamma, dc.dtype) * dc
    return ret


def get_D_matrix_for_angle(angle, j, cached=True):
    """
    Interface to *D_matrix_conj()*

    :param angle: Dict of angle data {"alpha","beta","gamma"}
    :param j: Integer :math:`2j` in the formula
    :param cached: Haven't been used???
    :return: Array of the conjugated D-matrices. Same length as the angle data
    """
    alpha = angle["alpha"]
    beta = angle["beta"]
    gamma = angle["gamma"]
    name = "D_matrix_{}".format(j)
    if cached:
        if name not in angle:
            angle[name] = D_matrix_conj(alpha, beta, gamma, j)
        return angle[name]
    return D_matrix_conj(alpha, beta, gamma, j)


def get_D_matrix_lambda(angle, ja, la, lb, lc=None):
    """
    Get the D-matrix element

    :param angle: Dict of angle data {"alpha","beta","gamma"}
    :param ja:
    :param la:
    :param lb:
    :param lc:
    :return:
    """
    beta, d_id = tf_nvtx.ops.start(angle["beta"], "delta D_matrix")
    angle["beta"] = beta
    d = get_D_matrix_for_angle(angle, _spin_int(2 * ja))
    if lc is None:
        ret =  tf.reshape(
            Dfun_delta_v2(d, ja, la, lb, (0,)), (-1, len(la), len(lb))
        )
    else:
        ret = Dfun_delta_v2(d, ja, la, lb, lc)
    ret = tf_nvtx.ops.end(ret, d_id)
    return ret


def get_D_matrix_lambda(angle, ja, la, lb, lc=None):
    """
    Get the D-matrix element

    :param angle: 
    """
    alpha = angle["alpha"]
    beta = angle["beta"]
    gamma = angle["gamma"]
    beta, d_id = tf_nvtx.ops.start(beta, "delta D_matrix")
    if lc is None:
        ret = tf.reshape(
            tf_pwa_op.delta_D(alpha, beta, gamma, ja, la, lb, (0,)), (-1, len(la), len(lb))
        )
    else:
        ret = tf_pwa_op.delta_D(alpha, beta, gamma, ja, la, lb, lc)
    ret = tf_nvtx.ops.end(ret, d_id)
    return ret



