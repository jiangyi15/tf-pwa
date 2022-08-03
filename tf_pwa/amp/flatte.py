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

    R(m) = \\frac{1}{m_0^2 - m^2 + m_0 (\\sum_{i}  g_i \\frac{q_i}{m})}

.. math::

    q_i = \\begin{cases}
    \\frac{\\sqrt{(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) >= 0 \\\\
    \\frac{i\\sqrt{|(m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2)|}}{2m} & (m^2-(m_{i,1}+m_{i,2})^2)(m^2-(m_{i,1}-m_{i,2})^2) < 0 \\\\
    \\end{cases}

Required input arguments `mass_list: [[m11, m12], [m21, m22]]` for :math:`m_{i,1}, m_{i,2}`.

    """

    def __init__(self, *args, mass_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        if mass_list is None:
            raise ValueError("required mass_list: [[a, b], [mc, md]]")
        self.mass_list = mass_list
        self.g_value = []
        self.float_list = list(kwargs.get("float", []))

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
        rho = sum(rhos)
        re = delta_s + tf.math.real(rho)
        im = tf.math.imag(rho)
        d = re * re + im * im
        ret = tf.complex(re / d, -im / d)
        return ret
