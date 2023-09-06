import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.polynomial import hermite
from scipy.special import gamma, gammainc


def integrate(n, a, b):
    """
    2 int_a^b x^n exp(-x^2) dx = igamma((n+1)/2,b^2) + (-1)^n igamma((n+1)/2,b^2)
    """

    r = gammainc((n + 1) / 2, b**2)
    l = gammainc((n + 1) / 2, a**2)

    y = r + l * (-1) ** n
    ret = y * gamma((n + 1) / 2) / 2
    return ret


def get_inte(n, a, b):
    inte = []
    for i in range(0, 2 * n):
        inte.append(integrate(i, a, b))

    inte = np.stack(inte, axis=-1)
    return inte


def get_coeff(inte):
    coeff = []
    n = (inte.shape[-1] + 1) // 2
    for i in range(n + 1):
        mat = []
        for j in range(n + 1):
            if i == j:
                continue
            mat.append(inte[..., j : j + n])
        mat = np.stack(mat, axis=-2)
        ci = np.linalg.det(mat) * (-1) ** (i + n - 1)
        coeff.append(ci)
    coeff = np.stack(coeff, axis=-1)
    coeff = coeff / np.mean(abs(coeff), axis=-1)[..., None]
    return coeff


def get_weight(point, inte):
    x = np.linspace(0, point.shape[-1] - 1, point.shape[-1])
    ax = point[..., None] ** x

    aa = np.linalg.inv(ax)
    bb = inte[..., : point.shape[-1]]
    ai = np.einsum("...ba,...b->...a", aa, bb)
    return ai


def gauss_point(n, a, b):
    inte = get_inte(n, a, b)
    # print(inte)
    coeff = get_coeff(inte)
    # coeff2 = hermite.poly2herm(coeff)
    # print(coeff)
    # print(coeff2)
    point = []
    for i in coeff:
        point.append(poly.polyroots(i))  # hermite.hermroots(coeff2)
    # print(poly.polyroots(coeff))
    point = np.stack(point, axis=0)
    weight = get_weight(point, inte)
    weight = weight  # / np.sum(weight)
    return point, weight
    # x = poly.polyroots(coeff)


if __name__ == "__main__":
    a, b = gauss_point(15, np.array([-10]), np.array([10]))
    c, d = hermite.hermgauss(15)
    print(a, c)
    print(b, d)
