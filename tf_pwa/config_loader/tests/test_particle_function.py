import numpy as np
import tensorflow as tf

from tf_pwa.config_loader.particle_function import ParticleFunction
from tf_pwa.tests.test_full import gen_toy, toy_config


def test_particle(toy_config):
    f = toy_config.get_particle_function("R_BC", d_norm=False)
    m = np.array([4.1, 4.15, 4.16, 4.17, 4.2])
    assert f(m).shape == (5, 6)
    g = toy_config.get_particle_function("R_BC", d_norm=True)
    g.phsp_fractor(m)
    g.density(m)
    assert g(m).shape == (5, 6)
    m_min, m_max = g.mass_range()
    m = g.mass_linspace(1000)
    g.density(m)
