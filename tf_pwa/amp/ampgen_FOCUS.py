import numpy as np
import tensorflow as tf

from tf_pwa.amp import Particle, register_particle

from .ampgen_pipi_swave import constructKMatrix, phsp_FOCUS, pol, poleConfig


@register_particle("Kpi_Swave")
class KPiSwaveKmatrix(Particle):
    """

    Kpi S wave model from AmpGen (https://github.com/GooFit/AmpGen/blob/master/src/Lineshapes/FOCUS.cpp).

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> ax = plot_particle_model("Kpi_Swave")
        >>> ax = plot_particle_model("Kpi_Swave", params={"lineshape_modifier": "KEta"}, axis=ax)
        >>> ax = plot_particle_model("Kpi_Swave", params={"lineshape_modifier": "I32"}, axis=ax)
        >>> _ = ax[1].legend(["Kpi", "KEta", "I32"])

    """

    def __init__(self, name, **kwargs):
        self.lineshape_modifier = "Kpi"
        super().__init__(name, **kwargs)

    def __call__(self, m, **kwargs):
        s = m * m
        ret = FOCUS_fun(s, self.lineshape_modifier)
        return ret

    def get_amp(self, data, data_c, **kwargs):
        m = data["m"]
        return self(m)


def FOCUS_fun(s, lineshapeModifier="Kpi"):
    I = 1.0j
    sInGeV = s
    mK = 0.493677  # ParticlePropertiesList::get( "K+" )->mass() ;
    mPi = 0.13957018  # ParticlePropertiesList::get( "pi+" )->mass() ;
    mEtap = 0.95766  # # ParticlePropertiesList::get( "eta'(958)0" )->mass();
    sNorm = mK * mK + mPi * mPi
    s12 = 0.23
    s32 = 0.27
    I12_adler = (sInGeV - s12) / sNorm
    I32_adler = (sInGeV - s32) / sNorm

    poleConfigs = [poleConfig(np.array(1.7919 + 0j), [0.31072, -0.02323])]

    rho1 = phsp_FOCUS(sInGeV, mK, mPi)
    rho2 = phsp_FOCUS(sInGeV, mK, mEtap)
    X = (sInGeV / sNorm) - 1

    kMatrix = constructKMatrix(sInGeV, 2, poleConfigs)
    scattPart = [
        pol(X, [0.79299, -0.15099, 0.00811]),
        pol(X, [0.15040, -0.038266, 0.0022596]),
        pol(X, [0.15040, -0.038266, 0.0022596]),
        pol(X, [0.17054, -0.0219, 0.00085655]),
    ]
    scattPart = tf.reshape(tf.stack(scattPart, axis=-1), kMatrix.shape)

    kMatrix = kMatrix + scattPart

    I12_adler = tf.cast(I12_adler, kMatrix.dtype)
    I32_adler = tf.cast(I32_adler, kMatrix.dtype)

    K11 = I12_adler * kMatrix[..., 0, 0]
    K12 = I12_adler * kMatrix[..., 0, 1]
    K22 = I12_adler * kMatrix[..., 1, 1]
    K32 = I32_adler * pol(X, [-0.22147, 0.026637, -0.00092057])

    detK = K11 * K22 - K12 * K12

    del_ = 1 - rho1 * rho2 * detK - I * (rho1 * K11 + rho2 * K22)

    T11 = 1.0 - I * rho2 * K22
    T22 = 1.0 - I * rho1 * K11
    T12 = I * rho2 * K12

    T32 = 1 / (1 - I * K32 * rho1)
    if lineshapeModifier == "Kpi":
        return (K11 - I * rho2 * detK) / del_
    elif lineshapeModifier == "KEta":
        return K12 / del_
    elif lineshapeModifier == "I32":
        return T32
    else:
        print("P-vector component : ", lineshapeModifier, " is not recognised")
        return tf.complex(tf.ones_like(s), tf.zeros_like(s))
