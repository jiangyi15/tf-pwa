import functools

import numpy as np
import tensorflow as tf

from tf_pwa.angle import LorentzVector as lv
from tf_pwa.angle import Vector3 as v3


def _half2(s):
    return int(round(s * 2))


def _size(s):
    return _half2(s) + 1


def _srange(s):
    for i in range(_size(s)):
        yield i - s


def _srange_inv(s):
    for i in range(_size(s)):
        yield -(i - s)


def _S(s):
    from sympy import S

    return S(_half2(s)) / 2


def _dim(l, r):
    if l == r:
        return _size(l) ** 2
    return 2 * _size(l) * _size(r)


def DFunc(s, Psi, Theta, Phi):
    """维格纳 D - 矩阵"""
    from tf_pwa.dfun import D_matrix_conj

    return D_matrix_conj(-Psi, Theta, -Phi, _half2(s))


def CGlrs(l, r, s):
    """
    CG coeffs l + r -> s, order [l, r, s]
    """
    from sympy.physics.quantum.cg import CG

    ret = np.zeros((_size(l), _size(r), _size(s)))
    for i_l, li in enumerate(_srange(l)):  # -s + 02 s + 1
        for i_r, ri in enumerate(_srange(r)):
            for i_s, si in enumerate(_srange(s)):
                ret[i_l, i_r, i_s] = float(
                    CG(_S(l), _S(li), _S(r), _S(ri), _S(s), _S(si))
                    .doit()
                    .evalf()
                )
    return ret[::-1, ::-1, ::-1]


def CGslr(l, r, s):
    """
    CG coeffs l + r -> s, order [l, r, s]
    """
    from sympy.physics.quantum.cg import CG

    ret = np.zeros((_size(s), _size(l), _size(r)))
    for i_l, li in enumerate(_srange(l)):  # -s + 02 s + 1
        for i_r, ri in enumerate(_srange(r)):
            for i_s, si in enumerate(_srange(s)):
                ret[i_s, i_l, i_r] = float(
                    CG(_S(l), _S(li), _S(r), _S(ri), _S(s), _S(si))
                    .doit()
                    .evalf()
                )
    return ret[::-1, ::-1, ::-1]


def CCGlrs(l1, l2, r1, r2, l, r, s):
    cg1 = CGlrs(l1, l2, l)
    cg2 = CGlrs(r1, r2, r)
    cg3 = CGlrs(l, r, s)
    return np.einsum("abc,def,cfg->abdeg", cg1, cg2, cg3)


_sqrt2 = np.sqrt(2)
# 洛伦兹群基本表示下手征表象 [m] 和时空表象 [\[Mu]] 之间的相似变换矩阵
X_mu_m = np.array(
    [
        [0 + 0j, 1 / _sqrt2, -1 / _sqrt2, 0],
        [1 / _sqrt2, 0, 0, -1 / _sqrt2],
        [1j / _sqrt2, 0, 0, 1j / _sqrt2],
        [0, -1 / _sqrt2, -1 / _sqrt2, 0],
    ]
)

X_m_mu = np.linalg.inv(X_mu_m)


def C_sigma_sigmabar(s, p):
    """SU2 群的从原表示的复共轭表示到原表示的相似变换矩阵"""
    from sympy import pi
    from sympy.physics.wigner import wigner_d_small

    ret = wigner_d_small(_S(s), p * pi)
    return np.array(ret.evalf(), dtype=np.float64)[::-1, ::-1]


def Metric(l, r):
    if l == r:
        C_sigma_sigmabar(l, 1)[:, None, :, None] * C_sigma_sigmabar(r, 1)[
            None, :, None, :
        ]
    else:
        a = (
            C_sigma_sigmabar(l, 1)[:, None, :, None]
            * C_sigma_sigmabar(r, 1)[None, :, None, :]
        )
        zeros = np.zeros_like(a)
        return np.concatenate(
            [
                np.concatenate([a, zeros], axis=0),
                np.concatenate([zeros, a], axis=0),
            ],
            axis=1,
        )


def u_m_sigma(l, r, s):
    ret = CGlrs(l, r, s)
    return np.reshape(ret, (-1, ret.shape[-1]))


def uu_m_sigma(l1, r1, l2, r2, l, r, s):
    ret = CCGlrs(l1, l2, r1, r2, l, r, s)
    ret = np.transpose(ret, (0, 2, 1, 3, 4))
    return np.reshape(ret, (-1, ret.shape[-1]))


def Uabba(A, B):
    delta = np.zeros((A * B, A * B, A, B))
    for m1 in range(A * B):
        for m2 in range(A * B):
            for a in range(A):
                for b in range(B):
                    if m1 == a * B + b and m2 == b * A + a:
                        delta[m1, m2, a, b] = 1
    delta = np.reshape(delta, (A * B, A * B, -1))
    return np.sum(delta, axis=-1)


def DirectSum(u1, u2, n=1):
    if n == 1:
        zeros = np.zeros((u2.shape[0], u1.shape[1]))
        return np.concatenate([u1, zeros], axis=0)
    elif n == 2:
        zeros = np.zeros((u1.shape[0], u2.shape[1]))
        return np.concatenate([zeros, u2], axis=0)
    else:
        raise ValueError("n not in 1,2")


def U_m_sigma(l, r, s, n=1):
    if l == r:
        return u_m_sigma(l, r, s)
    else:
        return DirectSum(u_m_sigma(l, r, s), u_m_sigma(r, l, s), n)


def Ubar_sigma_m(l, r, s, n):
    return U_m_sigma(l, r, s, (2 * n) % 3).T


def UU_m_sigma(l1, r1, l2, r2, l, r, s, n1, n2):
    if l1 == r1 and l2 == r2:
        return uu_m_sigma(l1, r1, l2, r2, l, r, s)
    elif l2 == r2:
        return DirectSum(
            uu_m_sigma(l1, r1, l2, r2, l, r, s),
            uu_m_sigma(r1, l1, l2, r2, r, l, s),
            n1,
        )
    elif l1 == r1:
        a = Uabba((2 * l1 + 1) * (2 * r1 + 1), 2 * (2 * l2 + 1) * (2 * r2 + 1))
        b = DirectSum(
            uu_m_sigma(l2, r2, l1, r1, l, r, s),
            uu_m_sigma(r2, l2, l1, r1, r, l, s),
            n2,
        )
        return np.dot(a, b)
    else:
        a = Uabba((2 * l1 + 1) * (2 * r1 + 1), 2 * (2 * l2 + 1) * (2 * r2 + 1))
        b = DirectSum(
            uu_m_sigma(l2, r2, l1, r1, l, r, s),
            uu_m_sigma(r2, l2, l1, r1, l, r, s),
            n2,
        )
        c = DirectSum(
            uu_m_sigma(l2, r2, r1, l1, r, l, s),
            uu_m_sigma(r2, l2, r1, l1, r, l, s),
            n2,
        )
        return DirectSum(np.dot(a, b), np.dot(a, c), n1)


def UUbar_sigma_m(l1, r1, l2, r2, l, r, s, n1, n2):
    ret = UU_m_sigma(l1, r1, l2, r2, l, r, s, (2 * n1) % 3, (2 * n2) % 3)
    return ret.T


def Pmm1m2(l, r, n, l1, r1, l2, r2, l12, r12, n1, n2, s):
    a = U_m_sigma(l, r, s, n1)
    b = UUbar_sigma_m(l1, r1, l2, r2, l12, r12, s, n1, n2)
    ret = np.dot(a, b)
    return np.reshape(ret, (_dim(l, r), _dim(l1, r1), _dim(l2, r2)))


def tLGen(q_mu, L, lib=np):
    p = Pmm1m2(
        L / 2,
        L / 2,
        1,
        (L - 1) / 2,
        (L - 1) / 2,
        1 / 2,
        1 / 2,
        L / 2,
        L / 2,
        1,
        1,
        L,
    )
    if hasattr(q_mu, "astype"):
        q_mu = q_mu.astype(X_m_mu.dtype)
    else:
        q_mu = tf.cast(q_mu, X_m_mu.dtype)
    return lib.einsum("abc,cd,...d->...ab", p, X_m_mu, q_mu)


def tmL(q_mu, L, lib=np):
    if L == 0:
        return lib.ones_like(q_mu[..., :1])
    res = tLGen(q_mu, 1)[..., :, 0]
    for i in range(2, L + 1):
        res = lib.einsum("...ab,...b->...a", tLGen(q_mu, i, lib=lib), res)
    return res


def t_sigma_L(q_Mu, L, lib=np):
    q_Sigma = lib.einsum(
        "ab,bc,...c->...a", Ubar_sigma_m(1 / 2, 1 / 2, 1, 1), X_m_mu, q_Mu
    )
    res = lib.ones_like(q_Sigma[..., 0:1])
    for i in range(1, L + 1):
        res = lib.einsum(
            "abc,...c,...b->...a", CGslr(i - 1, 1, i), q_Sigma, res
        )
    return res


def _slr(s, m=1):
    if _half2(s) % 2 == 0:
        return [s, s / 2, s / 2]
    else:
        return [s, (2 * s + 1) / 4, (2 * s - 1) / 4]


def mass2(p_mu):
    return np.sum(p_mu**2 * np.array([1, -1, -1, -1]), axis=-1)


def LorentzTrans(p_mu):
    x0, x1, x2, x3 = np.moveaxis(p_mu / np.sqrt(mass2(p_mu)), -1, 0)
    res = np.stack(
        [
            x0,
            x1,
            x2,
            x3,
            x1,
            x1 * x1 / (1 + x0) + 1,
            x1 * x2 / (1 + x0),
            (x1 * x3) / (1 + x0),
            x2,
            (x1 * x2) / (1 + x0),
            x2**2 / (1 + x0) + 1,
            (x2 * x3) / (1 + x0),
            x3,
            (x1 * x3) / (1 + x0),
            (x2 * x3) / (1 + x0),
            1 + x3**2 / (1 + x0),
        ],
        axis=-1,
    ).reshape((-1, 4, 4))
    return res


def _decomp(l1, r1, l2, r2):
    for l in range(_half2(abs(l1 - l2)), _half2(l1 + l2 + 1), 2):
        for s in range(_half2(abs(r1 - r2)), _half2(r1 + r2 + 1), 2):
            yield l / 2, s / 2


def Flatten(a, idxs=None, keep=False):
    if idxs is None:
        if keep:
            return tf.reshape(a, (a.shape[0], -1))
        return tf.reshape(a, (-1,))
    else:
        new_idx = []
        sizes = []
        idx_bias = 1
        if keep:
            sizes.append(a.shape[0])
            idx_bias = 0
        for i in idxs:
            new_idx += [j - idx_bias for j in i]
            tmp = 1
            for j in i:
                tmp = tmp * a.shape[j - idx_bias]
            sizes.append(tmp)
        ret = tf.transpose(a, new_idx)
        ret = tf.reshape(a, sizes)


def U_zeta_sigma(H, P):
    """
    螺旋度为 H 的无质量粒子的协变和逆变自旋波函数, 其中 P = 1, -1 代表宇称
    """
    dim = 2 * H + 1
    a = np.zeros(dim)
    a[-1] = 1
    a = np.diag(a)
    b = np.zeros(dim)
    b[0] = 1
    b = np.diag(b)
    res = DirectSum(a, b, 1)
    res += P * DirectSum(a, b, 2)
    # print(res)
    # res = DirectSum(
    #        DiagonalMatrix(Table(KroneckerDelta(l, dim), [l, 1, dim])),
    #        DiagonalMatrix(Table(KroneckerDelta(r, 1), [r, 1, dim])), nH);
    return res


def Ubar_sigma_zeta(H, nH):
    return U_zeta_sigma(H, nH).T


def SCRep(s):
    if int(s * 2) % 2 == 0:
        return s / 2, s / 2
    else:
        return (2 * s + 1) / 4, (2 * s - 1) / 4


def SCRep(s, m=1):
    if m == 0:
        return s, 0
    if int(s * 2) % 2 == 0:
        return s / 2, s / 2
    else:
        return (2 * s + 1) / 4, (2 * s - 1) / 4


# (*根据质量 m , 自旋 s 和标记 id 输出相应的协变和逆变自旋波函数, 其中 id=1,-1 \
# 对于有质量粒子和无质量粒子分别代表宇称和螺旋度*)
def SWF(s, id_, m_zero=False):
    l, r = SCRep(s)
    if s == 0:
        return np.array([[1]])
    if m_zero:
        if id_ == 1:
            ret = U_zeta_sigma(s, 1)
        else:
            ret = U_zeta_sigma(s, 2)
    else:
        if s % 1 == 0:
            ret = u_m_sigma(l, r, s)
        else:
            ret = (
                1
                / _sqrt2
                * (U_m_sigma(l, r, s, 1) + id_ * U_m_sigma(l, r, s, 2))
            )
    return ret


def SWFbar(s, id_, m_zero=False):
    l, r = SCRep(s)
    if s == 0:
        return np.array([[1]])
    else:
        if m_zero:
            if id_ == 1:
                ret = Ubar_sigma_zeta(s, 2)
            else:
                ret = Ubar_sigma_zeta(s, 1)
        else:
            if s % 1 == 0:
                ret = u_m_sigma(l, r, s).T
            else:
                ret = (1 / _sqrt2) * (
                    id_ * Ubar_sigma_m(l, r, s, 1) + Ubar_sigma_m(l, r, s, 2)
                )
    return ret


def SPT(s, id0, s1, id1, s2, id2, m0_zero=False, m1_zero=False, m2_zero=False):
    # return Flatten(dyad(SWF(s, id0, m0_zero), SWFbar(s1, id1, m1_zero),
    #  SWFbar(s2, id2, m2_zero)), [[1], [4], [6], [3, 5, 2]]) @ Flatten(CGlrs(s1, s2, s));
    f0 = SWF(s, id0, m0_zero)
    f1 = SWFbar(s1, id1, m1_zero)
    f2 = SWFbar(s2, id2, m2_zero)
    cg = CGlrs(s1, s2, s)
    return np.einsum("ms,lrs,lp,rq->mpq", f0, cg, f1, f2)


# (*任意不可约表示 (l,r) 下的洛伦兹变换*)
def LorentzRotation(l, r, Theta, Phi):
    zeros = tf.zeros_like(Theta)
    a = DFunc(l, -Phi, -Theta, zeros)
    b = a if l == r else DFunc(r, -Phi, -Theta, zeros)
    return tf.reshape(
        a[..., :, None, :, None] * b[..., None, :, None, :],
        (*a.shape[:-2], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]),
    )


def LorentzBoost3(l, r, CurlyTheta):
    zeros = tf.zeros_like(CurlyTheta)
    i_curly_theta = tf.complex(zeros, CurlyTheta)
    a = DFunc(l, i_curly_theta, zeros, zeros)
    b = DFunc(r, -i_curly_theta, zeros, zeros)
    return tf.reshape(
        a[..., :, None, :, None] * b[..., None, :, None, :],
        (*a.shape[:-2], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]),
    )


def LorentzBoost(l, r, CurlyTheta, Theta, Phi):
    a = LorentzRotation(l, r, Theta, Phi)
    b = LorentzBoost3(l, r, CurlyTheta)
    c = LorentzRotation(l, r, Theta, -Phi)
    return tf.einsum("...ab,...bc,...dc->...ad", a, b, c)


# (*任意自共轭表示 [l,r] 下的洛伦兹 BOOST 和转动*)
def LorentzBoostSC(l, r, CurlyTheta, Theta, Phi, lib=None):
    if l == r:
        return LorentzBoost(l, r, CurlyTheta, Theta, Phi)
    else:
        a = LorentzBoost(l, r, CurlyTheta, Theta, Phi)
        b = a if l == r else LorentzBoost(r, l, CurlyTheta, Theta, Phi)
        c1 = tf.zeros((*a.shape[:-2], b.shape[-2], a.shape[-1]), a.dtype)
        c2 = tf.zeros((*a.shape[:-2], a.shape[-2], b.shape[-1]), a.dtype)
        d1 = tf.concat([a, c1], axis=-2)
        d2 = tf.concat([c2, b], axis=-2)
        return tf.concat([d1, d2], axis=-1)


def LorentzRotationSC(l, r, Theta, Phi, lib=None):
    if l == r:
        return LorentzRotation(l, r, Theta, Phi)
    else:
        a = LorentzRotation(l, r, Theta, Phi)
        b = LorentzRotation(r, l, Theta, Phi)
        c1 = tf.zeros((*a.shape[:-2], b.shape[-2], a.shape[-1]), a.dtype)
        c2 = tf.zeros((*a.shape[:-2], a.shape[-2], b.shape[-1]), a.dtype)
        d1 = tf.concat([a, c1], axis=-2)
        d2 = tf.concat([c2, b], axis=-2)
        return tf.concat([d1, d2], axis=-1)


def LorentzInvSC_m_zero(p_Mu, s):
    x = p_Mu / p_Mu[..., 0:1]
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    zeros = tf.zeros_like(x1)
    CurlyTheta = zeros
    Theta = tf.where((x1 == 0) & (x2 == 0), zeros, tf.math.acos(x3))
    Phi = tf.where(
        (x1 == 0) & (x2 == 0),
        zeros,
        tf.where(
            x2 >= 0,
            tf.math.acos(x1 / (tf.sqrt(1 - x3**2))),
            2 * np.pi - tf.math.acos(x1 / (tf.sqrt(1 - x3**2))),
        ),
    )
    l, r = s, 0
    return LorentzRotationSC(l, r, Theta, Phi)


def LorentzInvSC_m_nzero(p_Mu, s):
    _epsilon = 1e-10
    pp = tf.reduce_sum(p_Mu * p_Mu * np.array([1, -1, -1, -1]), axis=-1)
    x = p_Mu / tf.sqrt(pp[..., None])
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

    zeros = tf.zeros_like(x1)
    cut2 = (tf.abs(x1) < _epsilon) & (tf.abs(x2) < _epsilon)
    cut1 = cut2 & (tf.abs(x3) < _epsilon)
    CurlyTheta = tf.where(cut1, zeros, tf.math.acosh(x0))
    Theta = tf.where(cut1, zeros, tf.math.acos(x3 / tf.sqrt(x0**2 - 1)))
    Phi = np.where(
        cut1,
        0,
        np.where(
            cut2,
            0,
            np.where(
                x2 > 0,
                tf.math.acos(x1 / (tf.sqrt(x0**2 - x3**2 - 1))),
                2 * np.pi
                - tf.math.acos(x1 / (tf.sqrt(x0**2 - x3**2 - 1))),
            ),
        ),
    )
    l, r = SCRep(s)
    # print(x0, x1, x2, x3)
    # print("theta: ", CurlyTheta, Theta, Phi)
    return LorentzBoostSC(l, r, CurlyTheta, Theta, Phi)


def LorentzInvSC(p_Mu, s, m_zero=True):
    if m_zero:
        return LorentzInvSC_m_zero(p_Mu, s)
    return LorentzInvSC_m_nzero(p_Mu, s)


def LorentzISO2(l, r, CurlyTheta, Theta, Phi):
    return tf.matmul(
        LorentzRotation(l, r, Theta, Phi), LorentzBoost3(l, r, CurlyTheta)
    )


def LorentzISO2SC(l, r, CurlyTheta, Theta, Phi):
    if l == r:
        return LorentzISO2(l, r, CurlyTheta, Theta, Phi)
    else:
        raise NotImplemented


# ArrayFlatten[{{LorentzISO2[l,
# r, {\[CurlyTheta], \[Theta], \[Phi]}], 0}, {0,
# LorentzISO2[r, l, {\[CurlyTheta], \[Theta], \[Phi]}]}}]];


def create_proj2(
    s, id0, s1, id1, s2, id2, S, L, m0_zero=False, m1_zero=False, m2_zero=False
):
    id0 = 1 if id0 is None else id0
    id1 = 1 if id1 is None else id1
    id2 = 1 if id2 is None else id2

    swf1 = SWF(s1, id1, m1_zero)
    swf2 = SWF(s2, id2, m2_zero)
    PsSL = SPT(s, id0, S, id1 * id2, L, 1, m0_zero=m0_zero)
    PSs1s2 = SPT(
        S, id1 * id2, s1, id1, s2, id2, m1_zero=m1_zero, m2_zero=m2_zero
    )
    CapitalGamma_L = np.einsum(
        "sSL,Spq->spqL", PsSL, PSs1s2
    )  # (PsSL @ tLq) @ PSs1s2;
    a0 = SWFbar(s, id0, m0_zero)
    # print((s, s1,s2), PsSL.shape,  PSs1s2.shape, CapitalGamma_L.shape,  swf1.shape, swf2.shape)
    if m1_zero:
        if m2_zero:
            Amp_L = np.einsum(
                "ms,spqL,Pi,Qj->LmijpPqQ", a0, CapitalGamma_L, swf1, swf2
            )
        else:
            Amp_L = np.einsum(
                "ms,spqL,Pi,qj->LmijpP", a0, CapitalGamma_L, swf1, swf2
            )
    else:
        if m2_zero:
            Amp_L = np.einsum(
                "ms,spqL,pi,Qj->LmijqQ", a0, CapitalGamma_L, swf1, swf2
            )
        else:
            Amp_L = np.einsum(
                "ms,spqL,pi,qj->Lmij", a0, CapitalGamma_L, swf1, swf2
            )
    return Amp_L


def create_proj3(
    s, id0, s1, id1, s2, id2, S, L, m0_zero=False, m1_zero=False, m2_zero=False
):
    id0 = 1 if id0 is None else id0
    id1 = 1 if id1 is None else id1
    id2 = 1 if id2 is None else id2

    swf1 = SWF(s1, id1, m1_zero)
    swf2 = SWF(s2, id2, m2_zero)
    PsSL = SPT(s, id0, S, id1 * id2, L, 1, m0_zero=m0_zero)
    PSs1s2 = SPT(
        S, id1 * id2, s1, id1, s2, id2, m1_zero=m1_zero, m2_zero=m2_zero
    )
    CapitalGamma_L = np.einsum(
        "sSL,Spq->spqL", PsSL, PSs1s2
    )  # (PsSL @ tLq) @ PSs1s2;
    a0 = SWFbar(s, id0, m0_zero)
    # print((s, s1,s2), PsSL.shape,  PSs1s2.shape, CapitalGamma_L.shape,  swf1.shape, swf2.shape)
    Amp_L = np.einsum(
        "ms,spqL,Pi,Qj->LmijpPqQ", a0, CapitalGamma_L, swf1, swf2
    )
    return Amp_L


def helicityPWA(
    s,
    id0,
    p1_Mu,
    s1,
    id1,
    p2_Mu,
    s2,
    id2,
    S,
    L,
    m0_zero=False,
    m1_zero=False,
    m2_zero=False,
):
    if m1_zero:
        swf1 = LorentzInvSC(p1_Mu, s1) @ SWF(s1, id1, m1_zero)
    else:
        swf1 = SWF(s1, id1, m1_zero)
    if m2_zero:
        swf2 = np.einsum(
            "...ab,bc->...ac", LorentzInvSC(p2_Mu, s2), SWF(s2, id2, m2_zero)
        )  # LorentzInvSC(p2_Mu, s2) @ SWF(s2, id2, m2_zero)
    else:
        swf2 = SWF(s2, id2, m2_zero)
    p = p1_Mu + p2_Mu
    from tf_pwa.angle import LorentzVector as lv

    p1s_Mu = lv.rest_vector(
        p, p1_Mu
    )  # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p1\[Mu];
    p2s_Mu = lv.rest_vector(
        p, p2_Mu
    )  # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p2\[Mu];
    qs_Mu = p1s_Mu - p2s_Mu
    PsSL = SPT(s, id0, S, id1 * id2, L, 1, m0_zero=m0_zero)
    tLq = tmL(qs_Mu, L)
    PSs1s2 = SPT(
        S, id1 * id2, s1, id1, s2, id2, m1_zero=m1_zero, m2_zero=m2_zero
    )
    CapitalGamma = np.einsum(
        "sSL,...L,Spq->...spq", PsSL, tLq, PSs1s2
    )  # (PsSL @ tLq) @ PSs1s2;
    a0 = SWFbar(s, id0, m0_zero)
    Amp = np.einsum(
        "ms,...spq,...pi,...qj->...mij", a0, CapitalGamma, swf1, swf2
    )  # SWFbar(m, s, id0) @ Flatten(CapitalGamma, [[1], [2, 3]]) @  Flatten(dyad(swf1, swf2), [[1, 3], [2], [4]]);
    # res = Amp // Simplify
    return Amp


def cal_amp(j0, j1, j2, l, s, p1, p2, coeff_s=[1], coeff_ls=[1]):
    slr0 = _slr(j0)
    slr1 = _slr(j1)
    slr2 = _slr(j2)

    Ubar0 = Ubar_sigma_m(slr0[1], slr0[2], slr0[0], 1)
    U1 = U_m_sigma(slr1[1], slr1[2], slr1[0], 1)
    U2 = U_m_sigma(slr2[1], slr2[2], slr2[0], 1)

    slrs = _slr(s)
    slrl = _slr(l)

    def Mstar_012(p1star, p2star):
        PS12 = 0
        for c, (chi_l, chi_r) in zip(
            coeff_s, _decomp(slr1[1], slr1[2], slr2[1], slr2[2])
        ):
            # print("s12", c, list(_decomp(slr1[1], slr1[2], slr2[1], slr2[2])), (chi_l, chi_r), Pmm1m2(slrs[1], slrs[2], 1, slr1[1], slr1[2], slr2[1], slr2[2], chi_l, chi_r, 1, 1, slrs[0]))
            PS12 = PS12 + c * Pmm1m2(
                slrs[1],
                slrs[2],
                1,
                slr1[1],
                slr1[2],
                slr2[1],
                slr2[2],
                chi_l,
                chi_r,
                1,
                1,
                slrs[0],
            )
        P0SL = 0
        for c, (chi_l, chi_r) in zip(
            coeff_ls, _decomp(slrs[1], slrs[2], slrl[1], slrl[2])
        ):
            # print("sl",c, list(_decomp(slrs[1], slrs[2], slrl[1], slrl[2])),  (chi_l, chi_r), Pmm1m2(slr0[1], slr0[2], 1, slrs[1], slrs[2], slrl[1], slrl[2], chi_l, chi_r, 1, 1, slr0[0]))
            P0SL = P0SL + c * Pmm1m2(
                slr0[1],
                slr0[2],
                1,
                slrs[1],
                slrs[2],
                slrl[1],
                slrl[2],
                chi_l,
                chi_r,
                1,
                1,
                slr0[0],
            )
        tL = tmL(p1star - p2star, l)
        # Gamma_012 = np.einsum("asl,...l,sbc->...abc", P0SL , tL ,PS12)
        # return np.einsum("ax,...xyz,yb,zc->...abc", Ubar0, Gamma_012, U1, U2)
        # return np.einsum("xa,asl,...l,sbc,by,cz->...xyz",Ubar0, P0SL , tL ,PS12, U1, U2 )
        proj = np.einsum("xa,asl,sbc,by,cz->lxyz", Ubar0, P0SL, PS12, U1, U2)
        return np.einsum("...l,lxyz->...xyz", tL, proj)

    g_munu = np.array([1, -1, -1, -1])
    p0 = p1 + p2
    trans = LorentzTrans(g_munu * p0)
    return Mstar_012(
        np.einsum("...ab,...b->...a", trans, p1),
        np.einsum("...ab,...b->...a", trans, p2),
    )


def create_proj(j0, j1, j2, l, s, coeff_s=[1], coeff_ls=[1]):
    slr0 = _slr(j0)
    slr1 = _slr(j1)
    slr2 = _slr(j2)

    Ubar0 = Ubar_sigma_m(slr0[1], slr0[2], slr0[0], 1)
    U1 = U_m_sigma(slr1[1], slr1[2], slr1[0], 1)
    U2 = U_m_sigma(slr2[1], slr2[2], slr2[0], 1)

    slrs = _slr(s)
    slrl = _slr(l)
    PS12 = 0
    for c, (chi_l, chi_r) in zip(
        coeff_s, _decomp(slr1[1], slr1[2], slr2[1], slr2[2])
    ):
        # print("s12", c, list(_decomp(slr1[1], slr1[2], slr2[1], slr2[2])), (chi_l, chi_r), Pmm1m2(slrs[1], slrs[2], 1, slr1[1], slr1[2], slr2[1], slr2[2], chi_l, chi_r, 1, 1, slrs[0]))
        PS12 = PS12 + c * Pmm1m2(
            slrs[1],
            slrs[2],
            1,
            slr1[1],
            slr1[2],
            slr2[1],
            slr2[2],
            chi_l,
            chi_r,
            1,
            1,
            slrs[0],
        )
    P0SL = 0
    for c, (chi_l, chi_r) in zip(
        coeff_ls, _decomp(slrs[1], slrs[2], slrl[1], slrl[2])
    ):
        # print("sl",c, list(_decomp(slrs[1], slrs[2], slrl[1], slrl[2])),  (chi_l, chi_r), Pmm1m2(slr0[1], slr0[2], 1, slrs[1], slrs[2], slrl[1], slrl[2], chi_l, chi_r, 1, 1, slr0[0]))
        P0SL = P0SL + c * Pmm1m2(
            slr0[1],
            slr0[2],
            1,
            slrs[1],
            slrs[2],
            slrl[1],
            slrl[2],
            chi_l,
            chi_r,
            1,
            1,
            slr0[0],
        )
    # Gamma_012 = np.einsum("asl,...l,sbc->...abc", P0SL , tL ,PS12)
    # return np.einsum("ax,...xyz,yb,zc->...abc", Ubar0, Gamma_012, U1, U2)
    # return np.einsum("xa,asl,...l,sbc,by,cz->...xyz",Ubar0, P0SL , tL ,PS12, U1, U2 )
    proj = np.einsum("xa,asl,sbc,by,cz->lxyz", Ubar0, P0SL, PS12, U1, U2)
    return proj


class WrapTF:
    def arccosh(x):
        return tf.math.acosh(x)

    def arccos(x):
        return tf.math.acos(x)

    def arctan2(y, x):
        return tf.math.atan2(y, x)

    def where(x, a, b):
        return tf.where(x, a, b)

    def sqrt(x):
        return tf.sqrt(x)

    def zeros_like(x):
        return tf.zeros_like(x)

    def log(x):
        return tf.math.log(x)

    def conjugate(x):
        return tf.math.conj(x)

    def einsum(*x):
        return tf.einsum(*x)


def ExRotationPara(mm, pMu, lib=np):
    if lib == "tf":
        lib = WrapTF
    px, py, pz = pMu[..., 1], pMu[..., 2], pMu[..., 3]
    r = lib.sqrt(px * px + py * py + pz * pz)
    Theta = lib.where(r < 1e-6, lib.zeros_like(pz), lib.arccos(pz / r))
    Phi = lib.arctan2(py, px)
    # {x0, x1, x2, x3, \[Theta], \[Phi], res},
    # if mm == 0:
    # x = p_Mu/p_Mu[...,0:1];
    # else:
    # x = lib.where(
    # lib.abs(mm - p_Mu[...,0]**2, 1e-6)
    # p_Mu/p_Mu[...,0:1], p_Mu/np.sqrt(p_Mu[...,0:1]**2 - mm[...,None])
    # if x1 == 0 and x2 == 0:
    # Theta, Phi = [0, 0]
    # else:
    # Theta = np.arccos(x3)
    # np.atan2()
    # if x2 >= 0:
    # Phi = np.arccos(x1/(np.sqrt(1 - x3**2)))
    # else:
    # Phi =  2 * np.pi - np.arccos(x1/np.sqrt(1 - x3**2))
    return Theta, Phi


def ExBoostPara(mm, p_Mu, lib=np):
    if lib == "tf":
        lib = WrapTF
    # {x0, \[CurlyTheta], \[Theta], \[Phi], res},
    CurlyTheta = lib.where(
        mm < 1e-6,
        lib.log(p_Mu[..., 0]),
        lib.arccosh(p_Mu[..., 0] / lib.sqrt(mm)),
    )
    Theta, Phi = ExRotationPara(mm, p_Mu, lib=lib)
    return CurlyTheta, Theta, Phi


"""根据粒子的质量平方 mm 和四动量 p\\[Mu]
给出任意自共轭表示[l,r]下任意参考系(pi,pf)中的振幅与质心系(ki,pfs)振幅之间的转换关系"""


def FrameTransSC(l, r, mmi, pi_Mu, mmf, pf_Mu, pfs_Mu, m_zero=False, lib=np):
    if lib == "tf":
        lib = WrapTF
    paraiInv = ExBoostPara(mmi, lv.neg(pi_Mu), lib=lib)
    paraf = ExBoostPara(mmf, pf_Mu, lib=lib)
    parafsInv = ExBoostPara(mmf, lv.neg(pfs_Mu), lib=lib)
    TransD = lib.einsum(
        "...ij,...jk,...kl->...il",
        LorentzBoostSC(l, r, *parafsInv),
        LorentzBoostSC(l, r, *paraiInv),
        LorentzBoostSC(l, r, *paraf),
    )
    if not m_zero:
        res = TransD
    else:
        parafs = ExBoostPara(mmf, pfs_Mu, lib=lib)
        res = lib.einsum(
            "...ij,...jk->...ik",
            TransD,
            LorentzRotationSC(l, r, *parafs[1:], lib=lib),
        )
        # res = lib.einsum(
        #    "...ij,...ik,...kl->...jl",
        #    lib.conjugate(LorentzRotationSC(l, r, *parafs[1:], lib=lib)),
        #    TransD,
        #    LorentzRotationSC(l, r, *paraf[1:], lib=lib),
        # )
    return res


def SO3SWF(m, s, P):
    # {l, r, res},
    l, r = SCRep(s, m)
    if l == r:
        res = u_m_sigma(l, r, s)
    else:
        res = (
            1
            / np.sqrt(2)
            * (U_m_sigma(l, r, s, 1) + P * U_m_sigma(l, r, s, 2))
        )
    return res


def SO3SWFbar(m, s, P):
    l, r = SCRep(s, m)
    if l == r:
        res = u_m_sigma(l, r, s).T
    else:
        res = (
            1
            / np.sqrt(2)
            * (P * Ubar_sigma_m(l, r, s, 1) + Ubar_sigma_m(l, r, s, 2))
        )
    return res


def CouplingLS(m, s, P, m1, s1, P1, m2, s2, P2, L, S, t_Sigma):
    return np.einsum(
        "al,gd,hf,ghk,...j,jkl->...adf",
        SO3SWF(m, s, P),
        SO3SWFbar(m1, s1, P1),
        SO3SWFbar(m2, s2, P2),
        CGlrs(s1, s2, S),
        t_Sigma,
        CGlrs(L, S, s),
    )


def CouplingLS_proj(m, s, P, m1, s1, P1, m2, s2, P2, L, S):
    return np.einsum(
        "al,gd,hf,ghk,jkl->adfj",
        SO3SWF(m, s, P),
        SO3SWFbar(m1, s1, P1),
        SO3SWFbar(m2, s2, P2),
        CGlrs(s1, s2, S),
        CGlrs(L, S, s),
    )


# Flatten[dyad[SO3SWF[m, s, P], SO3SWFbar[m1, s1, P1],
# SO3SWFbar[m2, s2, P2]], {{1}, {4}, {6}, {3, 5, 2}}] .
# Flatten[CGlrs[s1, s2, S] . (t\[Sigma] . CGlrs[L, S, s])];


def covtenPWA(
    p_Mu,
    s,
    id0,
    p1_Mu,
    s1,
    id1,
    p2_Mu,
    s2,
    id2,
    S,
    L,
):
    mm = lv.M2(p_Mu)
    mm1 = lv.M2(p1_Mu)
    mm2 = lv.M2(p2_Mu)
    # swf1, swf2, qs\[Mu],  p1s\[Mu], p2s\[Mu], paraInv, para1, para2, l1, r1, l2, r2,  tLq, [CapitalGamma], Amp, res
    p1s_Mu = lv.rest_vector(
        p_Mu, p1_Mu
    )  # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p1\[Mu];
    p2s_Mu = lv.rest_vector(
        p_Mu, p2_Mu
    )  # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p2\[Mu];
    # print(p1s_Mu, p2s_Mu)
    paraInv = ExBoostPara(mm, lv.neg(p_Mu))  # g\[Mu]\[Nu] . p\[Mu]];
    para1 = ExBoostPara(mm1, p1_Mu)  # ExBoostPara[mm1, p1\[Mu]];
    para2 = ExBoostPara(mm2, p2_Mu)
    print(paraInv)
    # print(paraInv, para1, para2)
    l1, r1 = SCRep(s1, mm1)
    l2, r2 = SCRep(s2, mm2)
    # print(LorentzBoostSC(l1, r1, *paraInv))
    if mm1 == 0:
        swf1 = np.einsum(
            "...ab,...bc,...cd->...ad",
            LorentzBoostSC(l1, r1, *paraInv),
            LorentzISO2SC(l1, r1, *para1),
            U_zeta_sigma(s1, id1),
        )
    else:
        swf1 = np.einsum(
            "...ab,...bc,...cd->...ad",
            LorentzBoostSC(l1, r1, *paraInv),
            LorentzBoostSC(l1, r1, *para1),
            SO3SWF(mm1, s1, id1),
        )
    # print("swf1", LorentzBoostSC(l1, r1, *para1), swf1)
    if mm2 == 0:
        swf2 = np.einsum(
            "...ab,...bc,...cd->...ad",
            LorentzBoostSC(l2, r2, *paraInv),
            LorentzISO2SC(l2, r2, *para2),
            U_zeta_sigma(s2, id2),
        )
    else:
        swf2 = np.einsum(
            "...ab,...bc,...cd->...ad",
            LorentzBoostSC(l2, r2, *paraInv),
            LorentzBoostSC(l2, r2, *para2),
            SO3SWF(mm2, s2, id2),
        )
    # print("swf2", LorentzBoostSC(l2, r2, *para2), swf2)
    qs = p1s_Mu - p2s_Mu
    tLq = t_sigma_L(qs, L)
    # print(tLq)
    CapitalGamma = CouplingLS(
        mm, s, id0, mm1, s1, id1, mm2, s2, id2, L, S, tLq
    )
    # print(CapitalGamma)
    # print(SO3SWFbar(mm, s, id0))
    Amp = np.einsum(
        "...ab,...bcd,...ce,...df->...aef",
        SO3SWFbar(mm, s, id0),
        CapitalGamma,
        swf1,
        swf2,
    )  # SO3SWFbar(mm, s, id0) . Flatten[\[CapitalGamma], {{1}, {2, 3}}] . Flatten[dyad[swf1, swf2], {{1, 3}, {2}, {4}}];
    res = Amp
    return res


def create_proj4(
    s, id0, s1, id1, s2, id2, S, L, m1_zero=False, m2_zero=False, m0_zero=False
):
    mm = 0 if m0_zero else 1  # lv.M2(p_Mu)
    mm1 = 0 if m1_zero else 1  # lv.M2(p1_Mu)
    mm2 = 0 if m2_zero else 1  #  lv.M2(p_Mu)
    # swf1, swf2, qs\[Mu],  p1s\[Mu], p2s\[Mu], paraInv, para1, para2, l1, r1, l2, r2,  tLq, [CapitalGamma], Amp, res
    # p1s_Mu = lv.rest_vector(p_Mu, p1_Mu) # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p1\[Mu];
    # p2s_Mu = lv.rest_vector(p_Mu, p1_Mu) # LorentzTrans[g\[Mu]\[Nu] . p\[Mu]] . p2\[Mu];
    # paraInv = ExBoostPara(mm, lv.neg(p_Mu)) # g\[Mu]\[Nu] . p\[Mu]];
    # para1 = ExBoostPara(mm1, p1_Mu) # ExBoostPara[mm1, p1\[Mu]];
    #  para2 = ExBoostPara(mm2, p2_Mu);
    l1, r1 = SCRep(s1, mm1)
    l2, r2 = SCRep(s2, mm2)
    if mm1 == 0:
        # swf1 = np.einsum("...ab,...bc,...cd->...ad", LorentzBoostSC(l1, r1, *paraInv), LorentzISO2SC(l1, r1, *para1), U_zeta_sigma(s1, id1%3))
        swf1 = U_zeta_sigma(s1, id1)
    else:
        # swf1 = np.einsum("...ab,...bc,...cd->...ad", LorentzBoostSC(l1, r1, *paraInv), LorentzBoostSC(l1, r1, *para1), SO3SWF(mm1, s1, id1))
        swf1 = SO3SWF(mm1, s1, id1)
    if mm2 == 0:
        # swf2 = np.einsum("...ab,...bc,...cd->...ad", LorentzBoostSC(l2, r2, *paraInv), LorentzISO2SC(l2, r2, *para2) , U_zeta_sigma(s2, id2 % 3))
        swf2 = U_zeta_sigma(s2, id2)
    else:
        # swf2 = np.einsum("...ab,...bc,...cd->...ad", LorentzBoostSC(l2, r2, *paraInv), LorentzISO2SC(l2, r2, *para2) , U_zeta_sigma(s2, id2 % 3))
        swf2 = SO3SWF(mm2, s2, id2)
    # qs = p1s - p2s
    #  tLq = t_sigma_L(qs, L);
    CapitalGamma = CouplingLS_proj(
        mm, s, id0, mm1, s1, id1, mm2, s2, id2, L, S
    )
    # Amp = np.einsum("...ab,...bcd,...ce,...df->...aef", SO3SWFbar(mm, s, id0), CapitalGamma, swf1, swf2)
    Amp = np.einsum("ab,bcdj->acdj", SO3SWFbar(mm, s, id0), CapitalGamma)
    # SO3SWFbar(mm, s, id0) . Flatten[\[CapitalGamma], {{1}, {2, 3}}] . Flatten[dyad[swf1, swf2], {{1, 3}, {2}, {4}}];
    res = Amp, swf1, swf2
    return res


def helicityPWA(p_Mu, s, id0, p1_Mu, s1, id1, p2_Mu, s2, id2, S, L):
    """协变张量方案下的三粒子分波振幅, 其中 id=1,-1 代表粒子的内禀宇称"""
    mm = lv.M2(p_Mu)
    mm1 = lv.M2(p1_Mu)
    mm2 = lv.M2(p2_Mu)
    p1s_Mu = lv.rest_vector(p_Mu, p1_Mu)
    p2s_Mu = lv.rest_vector(p_Mu, p2_Mu)
    l1, r1 = SCRep(s1, mm1)
    l2, r2 = SCRep(s2, mm2)
    TransD1 = FrameTransSC(l1, r1, mm, p_Mu, mm1, p1_Mu, p1s_Mu)
    TransD2 = FrameTransSC(l2, r2, mm, p_Mu, mm2, p2_Mu, p2s_Mu)
    # print(TransD1)
    # print(TransD2)
    if mm1 == 0:
        swf1 = np.dot(TransD1, U_zeta_sigma(s1, id1))
    else:
        swf1 = np.dot(TransD1, SO3SWF(mm1, s1, id1))
    if mm1 == 0:
        swf2 = np.dot(TransD2, U_zeta_sigma(s2, id2))
    else:
        swf2 = np.dot(TransD2, SO3SWF(mm2, s2, id2))
    qs_Mu = p1s_Mu - p2s_Mu
    tLq = t_sigma_L(qs_Mu, L)
    CapitalGamma = CouplingLS(
        mm, s, id0, mm1, s1, id1, mm2, s2, id2, L, S, tLq
    )
    Amp = np.einsum(
        "...ab,...bcd,...ce,...df->...aef",
        SO3SWFbar(mm, s, id0),
        CapitalGamma,
        swf1,
        swf2,
    )
    return Amp


def create_proj5(
    s, id0, s1, id1, s2, id2, S, L, m1_zero=False, m2_zero=False, m0_zero=False
):
    mm = 0 if m0_zero else 1  # lv.M2(p_Mu)
    mm1 = 0 if m1_zero else 1  # lv.M2(p1_Mu)
    mm2 = 0 if m2_zero else 1  #  lv.M2(p_Mu)
    l1, r1 = SCRep(s1, mm1)
    l2, r2 = SCRep(s2, mm2)
    if mm1 == 0:
        swf1 = U_zeta_sigma(s1, id1)
    else:
        swf1 = SO3SWF(mm1, s1, id1)
    if mm1 == 0:
        swf2 = U_zeta_sigma(s2, id2)
    else:
        swf2 = SO3SWF(mm2, s2, id2)
    CapitalGamma = CouplingLS_proj(
        mm, s, id0, mm1, s1, id1, mm2, s2, id2, L, S
    )
    Amp = np.einsum("ab,bcdj->acdj", SO3SWFbar(mm, s, id0), CapitalGamma)
    return Amp, swf1, swf2


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
