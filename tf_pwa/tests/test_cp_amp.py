import yaml

from tf_pwa.config_loader import ConfigLoader

from .test_full import gen_toy, this_dir


def test_cp_decay():
    with open(f"{this_dir}/config_toy.yml") as f:
        config_data = yaml.full_load(f)
    config_data["decay_chain"] = {"$all": {"is_cp": True}}
    config = ConfigLoader(config_data)

    amp = config.get_amplitude()
    data = config.get_data("data")[0]
    amp(data)
