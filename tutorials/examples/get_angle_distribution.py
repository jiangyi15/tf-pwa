from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy import Symbol, simplify, trigsimp
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import WignerD

from tf_pwa.config_loader import ConfigLoader


def spin_int(i):
    """use it for half integer in spins"""
    return sym.S(int(abs(i) * 2 + 0.001)) / 2 * sym.sign(i)


def get_decay_part(decay, ls, lambda_list, symbol_list):
    """
    .. math::
        \\sqrt{\\frac{ 2 l + 1 }{ 2 j_a + 1 }}
        \\langle j_b, j_c, \\lambda_b, - \\lambda_c | s, \\lambda_b - \\lambda_c \\rangle
        \\langle l, s, 0, \\lambda_b - \\lambda_c | j_a, \\lambda_b - \\lambda_c \\rangle

    .. math::
        D_{\\lambda_a, \\lambda_b - \\lambda_c}^{J_{A}*} (\\phi, \\theta, 0)

    """
    a = decay.core
    b, c = decay.outs
    l, s = ls
    lambda_list = {k: spin_int(v) for k, v in lambda_list.items()}
    delta = lambda_list[b] - lambda_list[c]
    if abs(delta) > a.J:
        return 0
    d_part = WignerD(
        spin_int(a.J),
        lambda_list[a],
        delta,
        symbol_list[decay]["alpha"],
        symbol_list[decay]["beta"],
        0,
    )
    # print(lambda_list)
    cg_part = sym.sqrt((2 * l + 1)) / sym.sqrt((2 * spin_int(a.J) + 1))
    cg_part = cg_part * CG(l, 0, spin_int(s), delta, spin_int(a.J), delta)
    cg_part = cg_part * CG(
        spin_int(b.J),
        lambda_list[b],
        spin_int(c.J),
        -lambda_list[c],
        spin_int(s),
        delta,
    )
    return simplify(d_part.conjugate() * cg_part)


def get_angle_distrubution_single(decay_chain, ls_list):
    ret = {}
    symbol_list = {
        decay: {
            "alpha": Symbol(f"phi{i}", real=True),
            "beta": Symbol(f"theta{i}", real=True),
        }
        for i, decay in enumerate(decay_chain)
    }
    out_particle = [decay_chain.top] + decay_chain.outs
    out_helicity = [i.spins for i in out_particle]
    inner_helicity = [i.spins for i in decay_chain.inner]
    for i in product(*out_helicity):
        lambda_list_out = dict(zip(out_particle, i))
        ret_part = 0
        for j in product(*inner_helicity):
            lambda_list = dict(zip(decay_chain.inner, j))
            lambda_list.update(lambda_list_out)
            tmp = 1
            for decay, ls in zip(decay_chain, ls_list):
                tmp = tmp * get_decay_part(decay, ls, lambda_list, symbol_list)
            ret_part = ret_part + tmp
        ret[i] = ret_part.expand(complex=True)
    return ret


def get_angle_distrubution(decay_chain):
    ls_com = []
    for i in decay_chain:
        ls_com.append(i.get_ls_list())

    ret = {}
    for i in product(*ls_com):
        ret[i] = get_angle_distrubution_single(decay_chain, i)
    return ret


def get_projection(f_theta, var):
    var = Symbol(var, real=True)
    args = f_theta.free_symbols
    inte_args = [i for i in args if i != var]
    inte_params = []
    for i in inte_args:
        if str(i).startswith("theta"):
            inte_params.append((i, 0, sym.pi))
            # d cos(theta) = sin(theta) d theta
            f_theta = f_theta * sym.sin(i)
        else:
            inte_params.append((i, -sym.pi, sym.pi))
    # projection in theta
    if len(inte_args) > 0:
        f_theta1 = sym.integrate(f_theta, *inte_params)
    else:
        f_theta1 = f_theta

    if str(var).startswith("theta"):
        normal = sym.integrate(f_theta1 * sym.sin(var), (var, 0, sym.pi))
    else:
        var_range = (-sym.pi, sym.pi)
        normal = sym.integrate(f_theta1, (var, -sym.pi, sym.pi))

    f_theta1 = simplify(f_theta1 / normal)
    return f_theta1


def plot_theta(f_name, f_theta, var):
    """ plot phi and cos theta """
    f_theta1 = get_projection(f_theta, var)
    var = Symbol(var, real=True)

    if str(var).startswith("theta"):
        var_range = (0, np.pi)
        trans = np.cos
    else:
        var_range = (-np.pi, np.pi)
        trans = lambda x: x

    print(f_theta1)

    f = sym.lambdify((var,), f_theta1.evalf(), "numpy")
    theta = np.linspace(var_range[0], var_range[1], 1000)
    x = trans(theta)
    y = f(theta) + np.zeros_like(x)

    plt.clf()
    plt.title("${}$".format(sym.latex(f_theta1)))
    plt.plot(x, y)
    plt.xlim((np.min(x), np.max(x)))
    plt.ylim((0, None))
    plt.savefig(f_name)


def main():
    sym.init_printing()
    config = ConfigLoader("config.yml")
    decay_group = config.get_decay()
    decay_chain = list(decay_group)[0]
    print(decay_chain)
    angle = get_angle_distrubution(decay_chain)
    ret = []
    for k, v in angle.items():
        for k2, v2 in v.items():
            f_theta = simplify(v2 * v2.conjugate())
            print("ls: ", k)
            print("  lambda:", k2)
            print("    :", f_theta)
            ret.append(f_theta)
            # break

    f_theta = ret[0]
    plot_theta("costheta.png", f_theta, "theta1")
    plot_theta("costheta2.png", f_theta, "theta2")
    plot_theta("phi2.png", f_theta, "phi2")


if __name__ == "__main__":
    main()
