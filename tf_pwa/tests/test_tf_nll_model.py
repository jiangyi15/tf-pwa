"""Tests for the compiled nll model ``default_tf``."""

import os

import numpy as np
import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.tests.test_full import gen_toy

this_dir = os.path.dirname(os.path.abspath(__file__))


def load_config(name, model):
    with open(os.path.join(this_dir, name)) as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["model"] = model
    config = ConfigLoader(config_dic)
    config.set_params(os.path.join(this_dir, "exp_params.json"))
    return config


def test_default_tf_consistency(gen_toy):
    # use a small batch to force multiple batches for both data and MC
    batch = 400
    ref = load_config("config_toy.yml", "default").get_fcn(batch=batch)
    new = load_config("config_toy.yml", "default_tf").get_fcn(batch=batch)
    nll_ref, grad_ref = ref.nll_grad()
    nll_new, grad_new = new.nll_grad()
    assert np.allclose(nll_ref, nll_new)
    assert np.allclose(np.asarray(grad_ref), np.asarray(grad_new))


def test_default_tf_lazy_nll(gen_toy):
    ref = load_config("config_lazycall.yml", "default").get_fcn(batch=400)
    new = load_config("config_lazycall.yml", "default_tf").get_fcn(batch=400)
    assert np.allclose(float(new.get_nll()), float(ref.get_nll()))


def test_fit_default_tf(gen_toy):
    config = load_config("config_lazycall.yml", "default_tf")
    results = config.fit(print_init_nll=False)
    assert np.allclose(results.min_nll, -204.9468493307786)
