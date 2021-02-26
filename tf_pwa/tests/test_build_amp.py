from tf_pwa.experimental import build_amp

from .test_full import gen_toy, toy_config


def test_build_amp(toy_config):
    amp = toy_config.get_amplitude()
    dg = toy_config.get_decay()
    data = toy_config.get_data("data")[0]

    amp2s = build_amp.cached_amp2s(dg, data)
    print(amp2s())

    _amp = build_amp.cached_amp(
        dg, data, matrix_method=build_amp.build_amp_matrix
    )
    print(_amp.python_function())


def test_build_amp2s(toy_config):
    # assert False

    amp = toy_config.get_amplitude()
    dg = toy_config.get_decay()
    data = toy_config.get_data("data")[0]
    amp2s = build_amp.build_amp2s(dg)
    _, cached_data = build_amp.build_angle_amp_matrix(dg, data)
    amp2s.python_function(data, cached_data)


def test_as_dict(toy_config):
    amp = toy_config.get_amplitude()
    dg = toy_config.get_decay()
    data = toy_config.get_data("data")[0]
    idx, cached_data = build_amp.build_angle_amp_matrix(dg, data)
    dic = build_amp.amp_matrix_as_dict(idx, cached_data)
    assert len(dic) == len(cached_data)
