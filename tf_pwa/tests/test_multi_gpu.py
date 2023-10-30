import yaml

from tf_pwa.tests.test_full import ConfigLoader, gen_toy, this_dir


def test_multigpu(gen_toy):
    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["multi_gpu"] = True
    dic["data"]["model"] = "simple"
    config = ConfigLoader(dic)
    fcn = config.get_fcn()
    fcn.nll_grad()
    data = config.get_data("data")[0]
    fcn.model._fast_nll_part_grad((data, data["weight"]), None)
    return config
