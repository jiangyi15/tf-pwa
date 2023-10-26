import yaml

from tf_pwa.tests.test_full import ConfigLoader, gen_toy, this_dir


def test_poisson(gen_toy):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["data_weight_smear"] = {"name": "Poisson"}
    config = ConfigLoader(dic)
    data = config.get_data("data")[0]


def test_dirichlet(gen_toy):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["data_weight_smear"] = {"name": "Dirichlet"}
    config = ConfigLoader(dic)
    data = config.get_data("data")[0]


def test_gamma(gen_toy):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["data_weight_smear"] = {"name": "Gamma"}
    config = ConfigLoader(dic)
    data = config.get_data("data")[0]
