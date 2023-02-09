import numpy as np
import tensorflow as tf


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


def CCGlrs(l1, l2, r1, r2, l, r, s):
    cg1 = CGlrs(l1, l2, l)
    cg2 = CGlrs(r1, r2, r)
    cg3 = CGlrs(l, r, s)
    return np.einsum("abc,def,cfg->abdeg", cg1, cg2, cg3)


_sqrt2 = np.sqrt(2)
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
    return lib.einsum("abc,cd,...d->...ab", p, X_m_mu, q_mu)


def tmL(q_mu, L, lib=np):
    if L == 0:
        return np.array([1.0])
    res = tLGen(q_mu, 1)[..., :, 0]
    for i in range(2, L + 1):
        res = lib.einsum("...ab,...b->...a", tLGen(q_mu, i, lib=lib), res)
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
