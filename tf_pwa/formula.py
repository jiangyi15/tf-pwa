import sympy

from tf_pwa.breit_wigner import get_bprime_coeff


def Bprime_polynomial(l, z):
    coeff = {
        0: [1],
        1: [1, 1],
        2: [1, 3, 9],
        3: [1, 6, 45, 225.0],
        4: [1, 10, 135, 1575, 11025],
        5: [1, 15, 315, 6300, 99225, 893025],
    }
    l = int(l + 0.01)
    if l not in coeff:
        coeff[l] = [int(i) for i in get_bprime_coeff(l)]
    ret = sum([coeff[l][::-1][i] * z**i for i in range(l + 1)])
    return ret


def get_relative_p(ma, mb, mc):
    return (
        sympy.sqrt((ma**2 - (mb + mc) ** 2) * (ma**2 - (mb - mc) ** 2))
        / 2
        / ma
    )


def get_relative_p2(ma, mb, mc):
    return (
        ((ma**2 - (mb + mc) ** 2) * (ma**2 - (mb - mc) ** 2)) / 4 / ma / ma
    )


def BW_dom(m, m0, g0):
    return m0 * m0 - m * m - sympy.I * m0 * g0


def BWR_dom(m, m0, g0, l, m1, m2, d=3.0):
    delta = m0 * m0 - m * m
    p = get_relative_p(m, m1, m2)
    p0 = get_relative_p(m0, m1, m2)
    bf = Bprime_polynomial(l, (p0 * d) ** 2) / Bprime_polynomial(
        l, (p * d) ** 2
    )
    gamma = m0 * g0 * (p / p0) ** (2 * l + 1) * (m0 / m) * bf
    return delta - sympy.I * gamma


def BWR_coupling_dom(m, m0, g0, l, m1, m2, d=3.0):
    delta = m0 * m0 - m * m
    p = get_relative_p(m, m1, m2)
    bf = Bprime_polynomial(l, 1) / Bprime_polynomial(l, (p * d) ** 2)
    gamma = m0 * g0 * (p) ** (2 * l + 1) / m * bf
    return delta - sympy.I * gamma


def BWR_LS_dom(m, m0, g0, thetas, ls, m1, m2, d=3.0, fix_bug1=False):
    delta = m0 * m0 - m * m
    p = get_relative_p2(m, m1, m2)
    p0 = get_relative_p2(m0, m1, m2)

    def bf_f(l):
        bf = Bprime_polynomial(l, p0 * d**2) / Bprime_polynomial(
            l, p * d**2
        )
        return (p / p0) ** l * bf

    g_head = sympy.I * m0 * g0 * m / m0 * sympy.sqrt(p / p0)
    if fix_bug1:
        g_head = sympy.I * m0 * g0 * m0 / m * sympy.sqrt(p / p0)
    if len(thetas) == 0:
        return delta - g_head * bf_f(ls[0])
    else:
        g1 = sympy.cos(thetas[0])
        gs = [g1]
        tmp = 1
        for i in range(len(thetas)):
            a = tmp * sympy.sin(thetas[i])
            if i == len(thetas) - 1:
                gs.append(a)
            else:
                gs.append(a * sympy.cos(thetas[i + 1]))
                tmp = tmp * sympy.sin(thetas[i])
        gamma = sum([j * j * bf_f(ls[i]) for i, j in enumerate(gs)])
        return delta - g_head * gamma


def _flatten(x):
    ret = []
    for i in x:
        if isinstance(i, (list, tuple)):
            ret += _flatten(i)
        else:
            ret.append(i)
    return ret


def create_complex_root_sympy_tfop(f, var, x, x0, epsilon=1e-12, prec=50):
    import tensorflow as tf

    f_var = _flatten(var)

    def solve_f(y):
        return sympy.nsolve(f.subs(dict(zip(f_var, y))), x, x0)

    @tf.custom_gradient
    def real_f(y):

        y0 = [float(i) for i in y]
        z0 = solve_f(y0)

        def _grad(dg):
            g = []
            for i in range(len(f_var)):
                y0[i] += epsilon
                fu = solve_f(y0)
                y0[i] -= 2 * epsilon
                fd = solve_f(y0)
                y0[i] += epsilon
                df = (fu - fd) / (2 * epsilon)
                g.append([float(sympy.re(df)), float(sympy.im(df))])
            # print(dg, tf.cast(tf.stack(g) ,dg.dtype), tf.reduce_sum(tf.cast(tf.stack(g) ,dg.dtype) * dg, axis=-1))
            return tf.reduce_sum(tf.cast(tf.stack(g), dg.dtype) * dg, axis=-1)

        a, b = sympy.re(z0), sympy.im(z0)
        # print(tf.cast(tf.stack([float(a), float(b)]), y.dtype))
        return tf.cast(tf.stack([float(a), float(b)]), y.dtype), _grad

    def _f(*y):
        y = _flatten(y)
        z = real_f(tf.stack(y))
        return tf.complex(z[0], z[1])

    return _f
