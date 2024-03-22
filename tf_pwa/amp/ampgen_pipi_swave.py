import numpy as np
import tensorflow as tf

from tf_pwa.amp import Particle, register_particle


@register_particle("pipi_Swave")
class PiPiSwaveKmatrix(Particle):
    """

    pipi S wave model from AmpGen (https://github.com/GooFit/AmpGen/blob/master/src/Lineshapes/kMatrix.cpp).
    using the parameters from `DtoKKpipi_v2.opt` (https://github.com/GooFit/AmpGen/blob/master/options/DtoKKpipi_v2.opt)

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> ax = plot_particle_model("pipi_Swave", params={"all_tokens": ["pole.0"]})
        >>> ax = plot_particle_model("pipi_Swave", params={"all_tokens": ["poleKK.1"]}, axis=ax)
        >>> ax = plot_particle_model("pipi_Swave", params={"all_tokens": ["prodKK.0"]}, axis=ax)
        >>> ax = plot_particle_model("pipi_Swave", params={"all_tokens": ["prod.1"]}, axis=ax)
        >>> _ = ax[1].legend(["pole.0", "poleKK.1", "prodKK.0", "prod.1"])

    """

    def __init__(self, name, **kwargs):
        self.all_tokens = []
        for i in range(5):
            self.all_tokens.append(f"pole.{i}")
        for i in range(5):
            self.all_tokens.append(f"prod.{i}")
        self.kmatrix_params = {}
        self.particleName = "PiPi00"
        super().__init__(name, **kwargs)

    def init_params(self):
        super().init_params()
        self.all_var = []
        for tokens in self.all_tokens:
            self.all_var.append(self.add_var(tokens, is_complex=True))
        self.all_var[0].fixed(1.0)

    def get_params_vecter(self):
        return tf.stack([i() for i in self.all_var], axis=-1)

    def __call__(self, m, **kwargs):
        s = m * m
        params_vector = self.get_params_vecter()
        ret = kMatrix_fun(
            s, self.all_tokens, self.kmatrix_params, self.particleName
        )
        return tf.reduce_sum(ret * params_vector, axis=-1)

    def get_amp(self, data, data_c, **kwargs):
        m = data["m"]
        return self(m)


def to_matrix(x):
    return tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)


def d_prod(a, b):
    return tf.expand_dims(a, axis=-1) * tf.expand_dims(b, axis=-2)


def pol(x, p):
    ret = tf.math.polyval([tf.cast(i, x.dtype) for i in p], x)
    return tf.complex(ret, tf.zeros_like(ret))
    # F = tf.zeros_like(x)
    # L = 1
    # for ip in p:
    # F = F + ip * L;
    # L = L * x;
    # return tf.complex(F, tf.zeros_like(F))


def complex_sqrt(x):
    return tf.sqrt(tf.complex(x, tf.zeros_like(x)))


def phsp_twoBody(s, m0, m1):
    return complex_sqrt(1.0 - (m0 + m1) * (m0 + m1) / s)


def phsp_fourPi(s):
    """
    Parameterisation of the 4pi phase space taken from `Laura` (https://laura.hepforge.org/  or Ref. https://arxiv.org/abs/1711.09854)
    """
    mPiPlus = 0.139570

    rho_4pi = pol(
        s, [0.00051, -0.01933, 0.13851, -0.20840, -0.29744, 0.13655, 1.07885]
    )

    return tf.where(s > 1, phsp_twoBody(s, 2 * mPiPlus, 2 * mPiPlus), rho_4pi)


def phsp_FOCUS(s, m0, m1):
    mp = m0 + m1
    mm = m0 - m1
    return complex_sqrt((1.0 - mp * mp / s) * (1.0 - mm * mm / s))


def gFromGamma(m, gamma, rho):
    return tf.sqrt(m * gamma / rho)


def getPropagator(kMatrix, phaseSpace):
    nChannels = kMatrix.shape[-1]
    Id = tf.eye(nChannels, dtype=kMatrix.dtype)
    T = Id - 1j * kMatrix * tf.expand_dims(phaseSpace, axis=-2)
    return tf.linalg.inv(T)


def constructKMatrix(this_s, nChannels, poleConfigs):
    kMatrix = tf.zeros((nChannels, nChannels), dtype=tf.complex128)
    for pole in poleConfigs:
        num = d_prod(pole.couplings, pole.couplings)
        dom = pole.s - tf.cast(this_s, pole.s.dtype)
        term = tf.cast(num, dom.dtype) / to_matrix(dom)
        kMatrix = kMatrix + term
    return kMatrix


class poleConfig:
    def __init__(self, s, couplings=None):
        self.s = s
        if couplings is None:
            self._couplings = []
        else:
            self._couplings = couplings

    def add(self, x):
        self._couplings.append(x)

    @property
    def couplings(self):
        return tf.stack(self._couplings, axis=-1)


def kMatrix_fun(
    s, all_tokens=["pole.0"], new_params={}, particleName="PiPi00"
):
    sInGeV = s
    nPoles = 5
    nChannels = 5
    channels = ["pipi", "KK", "4pi", "EtaEta", "EtapEta"]

    mPiPlus = 0.139570
    mKPlus = 0.493677
    mEta = 0.547862
    mEtap = 0.967780

    all_tokens = [i.split(".") for i in all_tokens]

    params = default_params.copy()
    params.update(new_params)

    def Parameter(name, value=0.0):
        return params.get(name, value)

    def paramVector(name, n):
        return tf.stack([Parameter(name + str(i)) for i in range(n)])

    sA0 = Parameter("sA0", -0.15)
    sA = Parameter("sA", 1.0)
    s0_prod = Parameter(particleName + "_s0_prod", -0.07)
    s0_scatt = Parameter("s0_scatt", -3.92637)
    fScatt = paramVector("f_scatt", nChannels)
    poleConfigs = []
    addImaginaryMass = Parameter("kMatrix::fp", True)

    for pole in range(1, nPoles + 1):
        stub = "IS_p" + str(pole) + "_"
        mass = Parameter(stub + "mass")
        # add a tiny imaginary part to the mass to avoid floating point errors //
        if addImaginaryMass:
            p = poleConfig(
                tf.cast(mass * mass, dtype=tf.complex128) + (1.0e-6j)
            )
        else:
            p = poleConfig(tf.cast(mass * mass, dtype=tf.complex128))
        for ch in range(nChannels):
            p.add(Parameter(stub + channels[ch]))
        poleConfigs.append(p)

    phaseSpace = tf.stack(
        [
            phsp_twoBody(sInGeV, mPiPlus, mPiPlus),
            phsp_twoBody(sInGeV, mKPlus, mKPlus),
            phsp_fourPi(sInGeV),
            phsp_twoBody(sInGeV, mEta, mEta),
            phsp_twoBody(sInGeV, mEta, mEtap),
        ],
        axis=-1,
    )

    kMatrix = constructKMatrix(sInGeV, nChannels, poleConfigs)

    scattPart = tf.zeros((nChannels, nChannels))
    slow_term = (1 - s0_scatt) / (s - s0_scatt)
    fScatt_scale = fScatt * np.array([0.5] + [1.0] * (nChannels - 1))
    scattPart = tf.expand_dims(fScatt_scale, axis=-1) + tf.expand_dims(
        fScatt_scale, axis=-2
    )
    scattPart = tf.cast(scattPart, slow_term.dtype) * to_matrix(slow_term)

    kMatrix = kMatrix + tf.cast(scattPart, kMatrix.dtype)
    # kMatrix.imposeSymmetry(0,1);

    adlerTerm = (
        (1.0 - sA0) * (sInGeV - sA * mPiPlus * mPiPlus / 2) / (sInGeV - sA0)
    )

    kMatrix = tf.cast(to_matrix(adlerTerm), kMatrix.dtype) * kMatrix
    F = getPropagator(kMatrix, phaseSpace)

    all_ret = []
    for tokens in all_tokens:
        if tokens[0] == "scatt":
            i = int(tokens[1])
            j = int(tokens[2])
            M = tf.mathmul(F, kMatrix)
            ret_i = M[..., j, i]
        elif tokens[0].startswith("pole"):
            if len(tokens[0]) == 4:
                idx = 0
            else:
                idx = channels.index(tokens[0][4:])
            pTerm = int(tokens[1])
            pole = poleConfigs[pTerm]
            M = tf.reduce_sum(
                F[..., idx, :] * tf.cast(pole.couplings, F.dtype), axis=-1
            )
            ret_i = M / (pole.s - tf.cast(sInGeV, pole.s.dtype))
        elif tokens[0].startswith("prod"):
            if len(tokens[0]) == 4:
                idx = 0
            else:
                idx = channels.index(tokens[0][4:])
            pTerm = int(tokens[1])
            pd = (1 - s0_prod) / (sInGeV - s0_prod)
            ret_i = F[..., idx, pTerm] * tf.cast(pd, F.dtype)
        else:
            print(
                "Modifier not found: , expecting one of {scatt, pole, poleKK, prod, prodKK}"
            )
            ret_i = tf.complex(tf.zeros_like(s), tf.zeros_like(s))
        all_ret.append(ret_i)
    return tf.stack(all_ret, axis=-1)


# from DtoKKpipi_v2.opt
params_str = """
KK00_s0_prod	2	-0.165753	0
KK10_s0_prod	2	-0.165753	0
PiPi00_s0_prod	2	-0.165753	0
PiPi20_s0_prod	2	-0.165753	0

IS_p1_4pi	2	0	0
IS_p1_EtaEta	2	-0.39899	0
IS_p1_EtapEta	2	-0.34639	0
IS_p1_KK	2	-0.55377	0
IS_p1_mass	2	0.651	0
IS_p1_pipi	2	0.22889	0
IS_p2_4pi	2	0	0
IS_p2_EtaEta	2	0.39065	0
IS_p2_EtapEta	2	0.31503	0
IS_p2_KK	2	0.55095	0
IS_p2_mass	2	1.2036	0
IS_p2_pipi	2	0.94128	0
IS_p3_4pi	2	0.55639	0
IS_p3_EtaEta	2	0.1834	0
IS_p3_EtapEta	2	0.18681	0
IS_p3_KK	2	0.23888	0
IS_p3_mass	2	1.55817	0
IS_p3_pipi	2	0.36856	0
IS_p4_4pi	2	0.85679	0
IS_p4_EtaEta	2	0.19906	0
IS_p4_EtapEta	2	-0.00984	0
IS_p4_KK	2	0.40907	0
IS_p4_mass	2	1.21	0
IS_p4_pipi	2	0.3365	0
IS_p5_4pi	2	-0.79658	0
IS_p5_EtaEta	2	-0.00355	0
IS_p5_EtapEta	2	0.22358	0
IS_p5_KK	2	-0.17558	0
IS_p5_mass	2	1.82206	0
IS_p5_pipi	2	0.18171	0
f_scatt0	2	0.23399	0
f_scatt1	2	0.15044	0
f_scatt2	2	-0.20545	0
f_scatt3	2	0.32825	0
f_scatt4	2	0.35412	0
s0_prod	2	-1	0
s0_scatt	2	-3.92637	0
sA	2	1	0
sA0	2	-0.15	0
"""

default_params = {}
for i in params_str.splitlines():
    if len(i.strip()) == 0:
        continue
    name = i.strip().split("\t")
    default_params[name[0]] = float(name[2])
