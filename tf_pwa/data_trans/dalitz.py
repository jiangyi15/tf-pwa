import numpy as np
import tensorflow as tf
from tensorflow import sqrt


class Dalitz:
    def __init__(self, m0, m1, m2, m3):
        self.m0 = m0
        self.mi = [m1, m2, m3]

    def generate_p(self, m12, m23):
        """generate monmentum for dalitz variable"""
        return generate_p(m12, m23, self.m0, *self.mi)


def _generate_fun0(m12, m23, m0, m1, m2, m3):
    """solved by sympy"""
    x0 = m0**2
    x1 = m1**2
    x2 = -m23 + x0 + x1
    x3 = 1 / m0
    x4 = x3 / 2
    x5 = m3**2
    x6 = m0**4
    x7 = m1**4
    x8 = m23**2
    x9 = m23 * x0
    x10 = m23 * x1
    x11 = x0 * x1
    x12 = -2 * x10 + x7 + x8
    x13 = 2 * m0 * m1
    x14 = 1 / ((-x13 + x2) * (x13 + x2))
    x15 = m12 * m23
    x16 = m12 * x0
    x17 = x0 * x5
    x18 = x1 * x5
    x19 = m12 * x1
    x20 = m23 * x5
    x21 = m2**2
    return [
        x2 * x4,
        x4 * (m12 + m23 - x1 - x5),
        x4 * (-m12 + x0 + x5),
        x3
        * (x10 + x11 - x6 / 2 - x7 / 2 - x8 / 2 + x9)
        * sqrt(1 / (-2 * x11 + x12 + x6 - 2 * x9)),
        sqrt(x14)
        * x4
        * (-2 * x0 * x21 - x11 + x12 + x15 + x16 + x17 + x18 - x19 - x20 - x9),
        -sqrt(
            x14
            * (
                -(m12**2) * m23
                + m12 * x10
                + m12 * x18
                - m12 * x8
                + m12 * x9
                - m2**4 * x0
                - m3**4 * x1
                - x1 * x9
                + x10 * x5
                + x11 * x21
                + x11 * x5
                + x15 * x21
                + x15 * x5
                + x16 * x21
                - x16 * x5
                + x17 * x21
                + x18 * x21
                - x19 * x21
                - x20 * x21
                - x21 * x6
                + x21 * x9
                - x5 * x7
            )
        ),
    ]


def generate_p(m12, m23, m0, m1, m2, m3):
    """generate monmentum by dalitz variable m12, m23"""
    E1, E2, E3, pa, pb, pc = _generate_fun0(m12, m23, m0, m1, m2, m3)
    zero = tf.zeros_like(E1)
    p1 = tf.stack([E1, pa, zero, zero], axis=-1)
    p2 = tf.stack([E2, pb, pc, zero], axis=-1)
    p3 = tf.stack([E3, -pa - pb, -pc, zero], axis=-1)
    return p1, p2, p3
