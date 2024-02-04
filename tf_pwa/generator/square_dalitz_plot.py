import numpy as np
import tensorflow as tf

from tf_pwa.angle import LorentzVector
from tf_pwa.generator import BaseGenerator


def square_dalitz_variables(p):
    """

    .. math::
        m' = \\frac{1}{\\pi}\\cos^{-1} \\left(2 \\frac{m_{12}-m_{min}}{m_{max}-m_{min}} - 1\\right)

    .. math::
        \\theta' = \\frac{1}{\\pi} \\theta_{12}

    """
    p1, p2, p3 = p

    m0 = LorentzVector.M(p1 + p2 + p3)
    m1 = LorentzVector.M(p1)
    m2 = LorentzVector.M(p2)
    m3 = LorentzVector.M(p3)

    m12 = LorentzVector.M(p1 + p2)
    m23 = LorentzVector.M(p2 + p3)
    m13 = LorentzVector.M(p1 + p3)

    m12norm = 2 * ((m12 - (m1 + m2)) / (m0 - (m1 + m2 + m3))) - 1
    mPrime = tf.math.acos(m12norm) / np.pi

    p1st = tf.sqrt(
        (-m12 * m12 - m1 * m1 + m2 * m2) ** 2 - 4 * m12 * m12 * m1 * m1
    ) / (2 * m12)
    p3st = tf.sqrt(
        (-m12 * m12 + m0 * m0 - m3 * m3) ** 2 - 4 * m12 * m12 * m3 * m3
    ) / (2 * m12)
    p1p3 = (
        m12 * m12 * (m23 * m23 - m13 * m13)
        - (m2 * m2 - m1 * m1) * (m0 * m0 - m3 * m3)
    ) / (4 * m12 * m12)

    thPrime = tf.math.acos(p1p3 / (p1st * p3st)) / np.pi
    return mPrime, thPrime, p1st, p3st


def square_dalitz_cut(p):
    """Copy from EvtGen old version

    .. math::
        |J| = 4 p q m_{12} \\frac{\\partial m_{12}}{\\partial m'} \\frac{\\partial \\cos\\theta_{12}}{\\partial \\theta'}
    .. math::
        \\frac{\\partial m_{12}}{\\partial m'} = -\\frac{\\pi}{2} \\sin (\\pi m') (m_{12}^{max} - m_{12}^{min})
    .. math::
        \\frac{\\partial \\cos\\theta_{12}}{\\partial \\theta'} = -\\pi \\sin (\\pi \\theta')

    """

    p1, p2, p3 = p

    m0 = LorentzVector.M(p1 + p2 + p3)
    m1 = LorentzVector.M(p1)
    m2 = LorentzVector.M(p2)
    m3 = LorentzVector.M(p3)

    m12 = LorentzVector.M(p1 + p2)

    mPrime, thPrime, p1st, p3st = square_dalitz_variables(p)
    jacobian = (
        2
        * np.pi**2
        * tf.sin(np.pi * mPrime)
        * tf.sin(np.pi * thPrime)
        * p1st
        * p3st
        * m12
        * (m0 - (m1 + m2 + m3))
    )

    prob = 1 / jacobian

    return tf.where(prob < 1.0, prob, tf.ones_like(prob))


def generate_SDP(m0, mi, N=1000, legacy=True):
    """generate square dalitz plot ditribution for 1,2

    The legacy mode will include a cut off in the threshold.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.phasespace import generate_square_dalitz12, square_dalitz_variables
        >>> p1, p2, p3 = generate_square_dalitz12(5.0, [2.0, 1.8, 0.5], 1000, legacy=True)
        >>> p12, p22, p32 = generate_square_dalitz12(5.0, [2.0, 1.8, 0.5], 1000, legacy=False)
        >>> mp, thetap, *_ = square_dalitz_variables([p1, p2, p3])
        >>> mp2, thetap2, *_ = square_dalitz_variables([p12, p22, p32])
        >>> _ = plt.subplot(1,2,1)
        >>> _ = plt.hist(mp.numpy(), range=(0,1), label="legacy", alpha=0.5)
        >>> _ = plt.hist(mp2.numpy(), range=(0,1), label="real", alpha=0.5)
        >>> _ = plt.xlabel("m'")
        >>> _ = plt.subplot(1,2,2)
        >>> _ = plt.hist(thetap.numpy(), range=(0,1), label="legacy", alpha=0.5)
        >>> _ = plt.hist(thetap2.numpy(), range=(0,1), label="real", alpha=0.5)
        >>> _ = plt.xlabel("$\\\\theta$'")
        >>> _ = plt.legend()

    """
    assert len(mi) == 3, "only support 3-body decay"

    if legacy:
        from tf_pwa.generator.generator import multi_sampling
        from tf_pwa.phasespace import PhaseSpaceGenerator

        gen = PhaseSpaceGenerator(m0, mi)
        ret, _ = multi_sampling(
            gen.generate, square_dalitz_cut, N=N, max_weight=1, display=False
        )
    else:
        rnd = tf.random.uniform((N,), dtype="float64")
        m12 = 0.5 * (tf.cos(np.pi * rnd) + 1) * (m0 - sum(mi)) + mi[0] + mi[1]
        theta1 = tf.random.uniform((N,), dtype="float64") * np.pi
        costheta0 = tf.random.uniform((N,), dtype="float64") * 2 - 1
        phi0 = tf.random.uniform((N,), dtype="float64") * np.pi * 2
        phi1 = tf.random.uniform((N,), dtype="float64") * np.pi * 2

        from tf_pwa.data_trans.helicity_angle import generate_p

        ret = generate_p(
            [m0, m12, mi[0]],
            [mi[2], mi[1]],
            [costheta0, tf.cos(theta1)],
            [phi0, phi1],
        )
        ret = ret[::-1]
    return ret


class SDPGenerator(BaseGenerator):
    def __init__(self, m0, mi, legacy=True):
        self.m0 = m0
        self.mi = mi
        self.legacy = legacy

    def generate(self, N):
        """

        >>> from tf_pwa.generator.square_dalitz_plot import SDPGenerator
        >>> gen = SDPGenerator(3.0, [1.0, 0.5, 0.1])
        >>> p1, p2, p3 = gen.generate(100)

        """
        return generate_SDP(self.m0, self.mi, N, legacy=self.legacy)
