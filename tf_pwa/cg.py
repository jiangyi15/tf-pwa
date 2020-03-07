"""This module provides the function **cg_coef()** to calculate the Clebsch-Gordan coefficients :math:`\\langle
j_1m_1j_2m_2|JM\\rangle`.

This function has interface to `SymPy <https://www.sympy.org/en/index.html>`_ functions if it's installed correctly.
Otherwise, it will depend on the input file **tf_pwa/cg_table.json**.
"""

import json
import os

has_sympy = True
try:
    from sympy.physics.quantum.cg import CG
except ImportError:
    has_sympy = False


def cg_coef(jb, jc, mb, mc, ja, ma):
    """
    It returns the CG coefficient :math:`\\langle j_bm_bj_cm_c|j_am_a\\rangle`, as in a decay from particle *a* to *b*
    and *c*. It will either call **sympy.physics.quantum.cg()** or **get_cg_coef()**.
    """
    if has_sympy:
        return CG(jb, mb, jc, mc, ja, ma).doit().evalf()
    else:
        return get_cg_coef(jb, jc, mb, mc, ja, ma)


_dirname = os.path.dirname(os.path.abspath(__file__))

with open(_dirname + "/cg_table.json") as f:
    cg_table = json.load(f)


def get_cg_coef(j1, j2, m1, m2, j, m):
    """
    If SymPy is not installed, **cg_coef()** will call this function and derive the CG coefficient by searching into
    **tf_pwa/cg_table.json**.

    In fact, **tf_pwa/cg_table.json** only stores some of the coefficients, the others will be
    obtained by this function using some symmetry properties of the CG table.

    .. note:: **tf_pwa/cg_table.json** only contains the cases where :math:`j_1,j_2\\leqslant4`, but this should be enough for most cases in PWA.
    """
    assert (m1 + m2 == m)
    assert (j1 >= 0)
    assert (j2 >= 0)
    assert (j >= 0)
    if j1 == 0 or j2 == 0:
        return 1.0
    sign = 1
    if j1 < j2:
        if (j1 + j2 - j) % 2 == 1:
            sign = -1
        j1, j2 = j2, j1
        m1, m2 = m2, m1

    def find_cg_table(j1, j2, m1, m2, j, m):
        try:
            return cg_table[str(j1)][str(j2)][str(m1)][str(m2)][str(j)][str(m)]
        except:
            return 0.0
    return sign * find_cg_table(j1, j2, m1, m2, j, m)
