import math

import tensorflow as tf

from tf_pwa.data_trans.helicity_angle import HelicityAngle
from tf_pwa.experimental import build_amp

from .config_loader import ConfigLoader


class ParticleFunction:
    def __init__(self, config, name, d_norm=False):
        self.decay_group = config.get_amplitude().decay_group
        self.decay_chain = self.decay_group.get_decay_chain(name)
        self.ha = HelicityAngle(self.decay_chain)
        self.norm_factor = 1
        if d_norm:
            for i in self.decay_chain:
                self.norm_factor /= math.sqrt(2 * i.core.J + 1)
        self.idx = list(self.decay_group).index(self.decay_chain)
        self.name = name
        self.config = config

    def __call__(self, m, random=False):
        p = self.ha.generate_p_mass(self.name, m, random=random)
        data = self.config.data.cal_angle(p)
        a = build_amp.build_params_vector(self.decay_group, data)
        ret = a[self.idx]
        return self.norm_factor * ret

    def mass_range(self):
        return self.ha.get_mass_range(self.name)

    def mass_linspace(self, N):
        return self.ha.mass_linspace(self.name, N)

    def phsp_factor(self, m):
        pf = self.ha.get_phsp_factor(self.name, m)
        return pf

    def phsp_fractor(self, m):
        print("phsp_fractor is wrong word, using phsp_factor instead!")
        return self.phsp_factor(m)

    def amp2s(self, m):
        return tf.abs(self(m)) ** 2

    def density(self, m):
        return tf.reduce_sum(self.amp2s(m), axis=-1) * self.phsp_factor(m)


@ConfigLoader.register_function()
def get_particle_function(config, name, d_norm=False):
    return ParticleFunction(config, name, d_norm=d_norm)
