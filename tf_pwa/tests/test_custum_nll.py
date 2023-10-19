import os

import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.model import custom
from tf_pwa.tests.test_full import gen_toy, toy_config

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_simple_model(gen_toy, toy_config):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["model"] = "simple"
    config = ConfigLoader(dic)
    fcn = config.get_fcn(batch=600)
    print(fcn())
    print(fcn.nll_grad())
    print(fcn.nll_grad_hessian(batch=600))


def test_simplechi2_model(gen_toy, toy_config):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["model"] = "simple_chi2"
    dic["data"]["extended"] = True
    config = ConfigLoader(dic)
    fcn = config.get_fcn(batch=600)
    print(fcn())
    print(fcn.nll_grad())
    print(fcn.nll_grad_hessian(batch=600))
