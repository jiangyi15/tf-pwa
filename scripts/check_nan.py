"""
Scripts for checking nan in data.

"""

import tensorflow as tf

from tf_pwa.amp.core import (
    get_decay_model,
    get_particle_model,
    register_decay,
    register_particle,
)
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import check_nan


def override_model(name):

    model = get_particle_model(name)

    @register_particle(name)
    class NewModel(model):
        def get_amp(self, *args, **kwargs):
            ret = super().get_amp(*args, **kwargs)
            print("checking", self)
            check_nan(tf.math.real(ret))
            check_nan(tf.math.imag(ret))
            return ret


def override_decay_model(name):

    model = get_decay_model(name)

    @register_decay(name)
    class NewModel(model):
        def get_amp(self, *args, **kwargs):
            ret = super().get_amp(*args, **kwargs)
            print("checking", self)
            check_nan(ret)
            return ret


def main():
    import argparse

    parser = argparse.ArgumentParser(description="calculate fit fractions")
    parser.add_argument("-c", "--config", default="config.yml")
    results = parser.parse_args()

    for name in ["default"]:
        override_model(name)
    for name in ["default"]:
        override_decay_model(name)

    config = ConfigLoader(results.config)
    amp = config.get_amplitude()

    for name, i in zip(["data", "phsp", "bg"], config.get_all_data()):
        if i is None:
            continue
        for idx, j in enumerate(i):
            if j is None:
                continue
            print("checking", name, idx)
            check_nan(j)
            amp(j)
        print("no nan in ", name)
    print("no nan found")


if __name__ == "__main__":
    main()
