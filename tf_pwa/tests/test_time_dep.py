import os

import numpy as np
import tensorflow as tf
import yaml

from tf_pwa.amp.time_dep import fix_cp_params, fix_cp_params_aabar
from tf_pwa.config_loader import ConfigLoader

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_time_dep():
    config = ConfigLoader(f"{this_dir}/config_time_dep.yml")
    config.set_params({"A_gamma": 1 / 1.5, "A_delta_m": 0.5, "A_poqi": -0.77})
    amp = config.get_amplitude()

    config2 = ConfigLoader(f"{this_dir}/config_time_dep2.yml")
    config2.set_params(config.get_params())
    amp2 = config2.get_amplitude()

    phsp = config.generate_phsp(10)

    assert np.allclose(amp(phsp).numpy(), amp2(phsp).numpy())


def test_time_dep_cp():
    config = ConfigLoader(f"{this_dir}/config_time_dep.yml")
    fix_cp_params(config, ["R_BD"], ["R_CD"])
    config.set_params({"A_gamma": 1 / 1.5, "A_delta_m": 0.5, "A_poqi": -0.77})
    amp = config.get_amplitude()

    config2 = ConfigLoader(f"{this_dir}/config_time_dep2.yml")
    fix_cp_params(config2, ["R_BD"], ["R_CD"])
    config2.set_params(config.get_params())
    amp2 = config2.get_amplitude()

    config4 = ConfigLoader(f"{this_dir}/config_time_dep4.yml")
    fix_cp_params_aabar(
        config4, ["R_BD", "R_CD", "R_BC"], ["R_BDb", "R_CDb", "R_BCb"]
    )
    config4.set_params(
        {k: v for k, v in config.get_params().items() if "lsb" not in k}
    )
    amp4 = config4.get_amplitude()

    config_cp = ConfigLoader(f"{this_dir}/config_time_dep3.yml")
    config_cp.set_params(
        {k: v for k, v in config.get_params().items() if "lsb" not in k}
    )
    amp_cp = config_cp.get_amplitude()

    phsp = config_cp.generate_phsp(10)

    a = amp_cp(phsp).numpy()
    amp_cp.use_p_pbar_time = False
    a2 = amp_cp(phsp).numpy()

    phsp2 = phsp.copy()
    del phsp2["cp_swap"]
    b = amp(phsp2).numpy()
    c = amp2(phsp2).numpy()
    d = amp4(phsp2).numpy()

    assert np.allclose(a, b)
    assert np.allclose(a, c)
    assert np.allclose(a, d)
    assert np.allclose(a, a2)


def test_time_dep_cp_conv():

    with open(f"{this_dir}/config_time_dep2.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["extra_var"]["time_sigma"] = {"default": 0.01}
    config_dic["data"]["amp_model"] = "time_dep_params_conv"
    config2 = ConfigLoader(config_dic)
    fix_cp_params(config2, ["R_BD"], ["R_CD"])
    amp2 = config2.get_amplitude()

    with open(f"{this_dir}/config_time_dep3.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["extra_var"]["time_sigma"] = {"default": 0.01}
    config_dic["data"]["amp_model"] = "time_dep_cp_conv"
    config_cp = ConfigLoader(config_dic)

    config_cp.set_params(
        {k: v for k, v in config2.get_params().items() if "lsb" not in k}
    )

    amp_cp = config_cp.get_amplitude()

    phsp = config_cp.generate_phsp(10)

    a = amp_cp(phsp).numpy()
    phsp2 = phsp.copy()
    del phsp2["cp_swap"]
    c = amp2(phsp2).numpy()

    assert np.allclose(a, c)


def test_time_dep_fs():
    with open(f"{this_dir}/config_time_dep2.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["amp_model"] = "time_dep_params_fs"
    config2 = ConfigLoader(config_dic)
    fix_cp_params(config2, ["R_BD"], ["R_CD"])
    amp2 = config2.get_amplitude()

    with open(f"{this_dir}/config_time_dep3.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["amp_model"] = "time_dep_cp_fs"
    config_cp = ConfigLoader(config_dic)

    config_cp.set_params(
        {k: v for k, v in config2.get_params().items() if "lsb" not in k}
    )

    amp_cp = config_cp.get_amplitude()

    phsp = config_cp.generate_phsp(10)

    a = amp_cp(phsp).numpy()
    phsp2 = phsp.copy()
    del phsp2["cp_swap"]
    c = amp2(phsp2).numpy()

    assert np.allclose(a, c)

    var_name = ["A_delta_m", "A_delta_gamma"]
    var = [amp_cp.vm.variables[i] for i in var_name]

    def f():
        y = tf.reduce_sum(amp_cp(phsp))
        return y

    with tf.GradientTape() as tape:
        y = f()
    grad = tape.gradient(y, var)

    params = amp_cp.get_params()
    delta = 0.001
    for idx, name in enumerate(var_name):
        amp_cp.set_params({name: params[name] + delta})
        y1 = f()
        amp_cp.set_params({name: params[name] - delta})
        y2 = f()
        assert abs((y1 - y2) / delta / 2 - grad[idx]) < 1e-4
        amp_cp.set_params(params)


def test_time_dep_flavour_tag():
    with open(f"{this_dir}/config_time_dep_ft.yml") as f:
        config_dic = yaml.full_load(f)
    config = ConfigLoader(config_dic)
    amp = config.get_amplitude()
    phsp = config.generate_phsp(10)

    a = amp(phsp).numpy()


def test_time_dep_flavour_tag_linear():
    with open(f"{this_dir}/config_time_dep_ft.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["amp_model"]["flavour_tag_mix"]["taggers"] = [
        {"model": "flavour_tag_linear"}
    ]
    config = ConfigLoader(config_dic)
    amp = config.get_amplitude()
    phsp = config.generate_phsp(10)

    a = amp(phsp).numpy()
