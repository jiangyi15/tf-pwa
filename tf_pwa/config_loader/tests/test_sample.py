from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_shape
from tf_pwa.tests.test_full import gen_toy, toy_config


def test_generate_phsp(toy_config):
    data = toy_config.sampling(1000, force=False)
    assert data_shape(data) >= 1000
    data = toy_config.sampling(1000)
    assert data_shape(data) == 1000
