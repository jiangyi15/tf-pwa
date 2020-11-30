from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy import Symbol, simplify, trigsimp
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import WignerD

from tf_pwa.config_loader import ConfigLoader


def get_decay_part(decay, ls, lambda_list, symbol_list):
    a = decay.core
    b, c = decay.outs
    l, s = ls
    delta = lambda_list[b] - lambda_list[c]
    if abs(delta) > a.J:
        return 0
    d_part = WignerD(
        a.J,
        lambda_list[a],
        delta,
        symbol_list[decay]["alpha"],
        symbol_list[decay]["beta"],
        0,
    )
    cg_part = (
        sym.sqrt((2 * l + 1))
        / sym.sqrt((2 * a.J + 1))
        * CG(l, 0, s, delta, a.J, delta)
    )
    cg_part = cg_part * CG(b.J, lambda_list[b], c.J, -lambda_list[c], s, delta)
    return d_part * cg_part


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
        ret[i] = simplify(ret_part).expand(complex=True)
    return ret


def get_angle_distrubution(decay_chain):
    ls_com = []
    for i in decay_chain:
        ls_com.append(i.get_ls_list())

    ret = {}
    for i in product(*ls_com):
        ret[i] = get_angle_distrubution_single(decay_chain, i)
    return ret


def plot_theta(f_name, f_theta, var, trans=lambda x: x):
    var = Symbol(var, real=True)
    if str(var).startswith("theta"):
        var_range = (0, sym.pi)
    else:
        var_range = (-sym.pi, sym.pi)
    args = f_theta.free_symbols
    inte_args = [i for i in args if i != var]
    inte_params = []
    for i in inte_args:
        if str(i).startswith("theta"):
            inte_params.append((i, 0, sym.pi))
        else:
            inte_params.append((i, -sym.pi, sym.pi))
    if len(inte_args) > 0:
        f_theta1 = sym.integrate(f_theta, *inte_params)
    else:
        f_theta1 = f_theta
    normal = sym.integrate(f_theta1, (var, var_range[0], var_range[1]))
    f_theta1 = simplify(f_theta1 / normal)
    print(f_theta1)
    f = sym.lambdify((var,), f_theta1.evalf(), "numpy")
    theta = np.linspace(float(var_range[0]), float(var_range[1]), 1000)
    x = trans(theta)
    plt.clf()
    plt.title("${}$".format(sym.latex(f_theta1)))
    plt.plot(x, f(theta) + np.zeros_like(x))
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
            print(k, k2, f_theta)
            ret.append(f_theta)

    f_theta = ret[1]
    plot_theta("costheta.png", f_theta, "theta1", np.cos)
    plot_theta("costheta2.png", f_theta, "theta2", np.cos)
    plot_theta("phi2.png", f_theta, "phi2")


if __name__ == "__main__":
    main()
