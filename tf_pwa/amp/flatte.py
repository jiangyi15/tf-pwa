from tf_pwa.tensorflow_wrapper import tf

from .core import Particle, Variable, register_particle


def cal_monentum(m, ma, mb):
    mabp = ma + mb
    mabm = ma - mb
    s = m * m
    p2 = (s - mabp * mabp) * (s - mabm * mabm) / 4 / s
    zeros = tf.zeros_like(s)
    p_p = tf.complex(tf.sqrt(tf.abs(p2)), zeros)
    p_m = tf.complex(zeros, tf.sqrt(tf.abs(p2)))
    return tf.where(p2 > 0, p_p, p_m)


@register_particle("Flatte")
class ParticleFlatte(Particle):
    """

    Flatte like formula

.. math::

    R(m) = \\frac{1}{m_0^2 - m^2 + i m_0 (\\sum_{i}  g_i \\frac{q_i}{m})}

.. math::

    q_i = \\begin{cases}
    \\frac{\\sqrt{(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) >= 0 \\\\
    \\frac{i\\sqrt{|(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)|}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0 \\\\
    \\end{cases}

.. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("Flatte", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("Flatte", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "mass": 0.7}, {"R_BC_g_0": -0.3,"R_BC_g_1": -0.2})
        >>> _ = plt.legend(["$g_i$", "$-g_i$"])


Required input arguments `mass_list: [[m11, m12], [m21, m22]]` for :math:`m_{i,1}, m_{i,2}`.

    """

    def __init__(self, *args, mass_list=None, im_sign=1, **kwargs):
        super().__init__(*args, **kwargs)
        if mass_list is None:
            raise ValueError("required mass_list: [[a, b], [mc, md]]")
        self.mass_list = mass_list
        self.g_value = []
        self.float_list = list(kwargs.get("float", []))
        self.im_sign = im_sign

    def init_params(self):
        self.d = 3.0
        if self.mass is None:
            self.mass = self.add_var("mass", fix=True)
            # print("$$$$$",self.mass)
        else:
            if not isinstance(self.mass, Variable):
                if "m" in self.float_list:
                    self.mass = self.add_var(
                        "mass", value=self.mass, fix=False
                    )
                else:
                    self.mass = self.add_var("mass", value=self.mass, fix=True)
        self.g_value = []
        for i, mab in enumerate(self.mass_list):
            name = f"g_{i}"
            if hasattr(self, name):
                if name in self.float_list:
                    self.g_value.append(
                        self.add_var(
                            f"g_{i}", value=getattr(self, name), fix=False
                        )
                    )
                else:
                    self.g_value.append(
                        self.add_var(
                            f"g_{i}", value=getattr(self, name), fix=True
                        )
                    )
            else:
                self.g_value.append(self.add_var(f"g_{i}"))

    def __call__(self, m):
        return self.get_amp({"m": m})

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        mass = self.get_mass()
        zeros = tf.zeros_like(m)
        delta_s = mass * mass - m * m
        m_c = mass / m
        rhos = []
        for i, mab in enumerate(self.mass_list):
            ma, mb = mab
            pi = cal_monentum(m, ma, mb)
            # print(pi)
            m_rho_i = pi * tf.complex(zeros, self.g_value[i]() * m_c)
            rhos.append(m_rho_i)
        rho = self.im_sign * sum(rhos)
        re = delta_s + tf.math.real(rho)
        im = tf.math.imag(rho)
        d = re * re + im * im
        ret = tf.complex(re / d, -im / d)
        return ret


@register_particle("FlatteC")
class ParticleFlatteC(ParticleFlatte):
    """

    Flatte like formula

.. math::

    R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 (\\sum_{i}  g_i \\frac{q_i}{m})}

.. math::

    q_i = \\begin{cases}
    \\frac{\\sqrt{(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) >= 0 \\\\
    \\frac{i\\sqrt{|(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)|}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0 \\\\
    \\end{cases}

Required input arguments `mass_list: [[m11, m12], [m21, m22]]` for :math:`m_{i,1}, m_{i,2}`.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("FlatteC", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})

    """

    def __init__(self, *args, im_sign=-1, **kwargs):
        super().__init__(*args, im_sign=im_sign, **kwargs)


@register_particle("FlatteGen")
class ParticleFlateGen(ParticleFlatte):
    """

    More General Flatte like formula

.. math::

    R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 [\\sum_{i}  g_i \\frac{q_i}{m} \\times \\frac{m_0}{|q_{i0}|} \\times \\frac{|q_i|^{2l_i}}{|q_{i0}|^{2l_i}} B_{l_i}'^2(|q_i|,|q_{i0}|,d)]}

.. math::

    q_i = \\begin{cases}
    \\frac{\\sqrt{(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) >= 0 \\\\
    \\frac{i\\sqrt{|(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)|}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0 \\\\
    \\end{cases}

Required input arguments `mass_list: [[m11, m12], [m21, m22]]` for :math:`m_{i,1}, m_{i,2}`. And addition arguments `l_list: [l1, l2]` for :math:`l_i`


  `has_bprime=False` to remove :math:`B_{l_i}'^2(|q_i|,|q_{i0}|,d)`.

  `cut_phsp=True` to set :math:`q_i = 0` when :math:`(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0`

The plot use parameters :math:`m_0=0.7, m_{0,1}=m_{0,2}=0.1, m_{1,1}=m_{1,2}=0.3, g_0=0.3,g_1=0.2,l_0=0,l_1=1`.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7, "has_bprime": False}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7, "cut_phsp": True}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plt.legend(["all l=0", "has_bprime=False", "cut_phsp=True", "normal"])


  `no_m0=True` to set :math:`i m_0 => i` in the width part.

  `no_q0=True` to remove :math:`\\frac{m_0}{|q_{i0}|}=0` and set others :math:`q_{i0}=1`.


    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7, "no_q0": True}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7, "no_m0": True}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plot_particle_model("FlatteGen", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 1], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})
        >>> _ = plt.legend(["no_q0=True", "no_m0=True", "normal"])

    """

    def __init__(
        self,
        *args,
        im_sign=-1,
        l_list=None,
        has_bprime=True,
        no_m0=False,
        no_q0=False,
        cut_phsp=False,
        **kwargs
    ):
        super().__init__(*args, im_sign=im_sign, **kwargs)
        if l_list is None:
            l_list = [0] * len(self.mass_list)
        self.l_list = l_list
        self.has_bprime = has_bprime
        self.no_m0 = no_m0
        self.no_q0 = no_q0
        self.cut_phsp = cut_phsp

    def get_coeff(self):
        return [i() for i in self.g_value]

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        mass = self.get_mass()
        zeros = tf.zeros_like(m)
        delta_s = mass * mass - m * m
        if self.no_m0:
            m_c = 1 / m
        else:
            m_c = mass / m
        rhos = []
        gi = self.get_coeff()
        for i, (mab, l) in enumerate(zip(self.mass_list, self.l_list)):
            ma, mb = mab
            pi = cal_monentum(m, ma, mb)
            pi0 = cal_monentum(mass, ma, mb)
            m_rho_i = pi * tf.complex(zeros, gi[i] * m_c)
            if self.no_q0:
                pi0 = tf.ones_like(pi0)
            else:
                m_rho_i = m_rho_i * tf.complex(
                    mass / tf.abs(pi0), tf.zeros_like(mass)
                )
            if l != 0:
                m_rho_i = m_rho_i * tf.complex(
                    tf.abs(pi / pi0) ** (2 * l), zeros
                )
            if self.has_bprime:
                from tf_pwa.breit_wigner import Bprime_q2

                bf = (
                    Bprime_q2(l, tf.abs(pi) ** 2, tf.abs(pi0) ** 2, self.d)
                    ** 2
                )
                m_rho_i = m_rho_i * tf.complex(bf, zeros)
            if self.cut_phsp:
                m_rho_i = tf.where(
                    m < ma + mb, tf.zeros_like(m_rho_i), m_rho_i
                )
            rhos.append(m_rho_i)
        rho = self.im_sign * sum(rhos)
        re = delta_s + tf.math.real(rho)
        im = tf.math.imag(rho)
        d = re * re + im * im
        ret = tf.complex(re / d, -im / d)
        return ret


@register_particle("Flatte2")
class ParticleFlate2(ParticleFlateGen):
    """

    General Flatte like formula.

.. math::

    R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 [\\sum_{i}  \\color{red}{g_i^2}\\color{black} \\frac{q_i}{m} \\times \\frac{m_0}{|q_{i0}|} \\times \\frac{|q_i|^{2l_i}}{|q_{i0}|^{2l_i}} B_{l_i}'^2(|q_i|,|q_{i0}|,d)]}

.. math::

    q_i = \\begin{cases}
    \\frac{\\sqrt{(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) >= 0 \\\\
    \\frac{i\\sqrt{|(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)|}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0 \\\\
    \\end{cases}

It has the same options as `FlatteGen`.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("Flatte2", {"mass_list": [[0.1, 0.1], [0.3,0.3]], "l_list": [0, 0], "mass": 0.7}, {"R_BC_g_0": 0.3,"R_BC_g_1": 0.2})

    """

    def get_coeff(self):
        return [i() ** 2 for i in self.g_value]
