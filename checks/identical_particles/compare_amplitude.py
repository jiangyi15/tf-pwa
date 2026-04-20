import numpy as np
import tensorflow as tf

from tf_pwa.amp import cov_ten
from tf_pwa.angle import LorentzVector as lv
from tf_pwa.config_loader import ConfigLoader

config_hel = ConfigLoader("configs/config3.yml")

for i in config_hel.get_decay():
    for j in i:
        print(j, j.get_ls_list())

phsp = config_hel.generate_phsp_p(10000)

boost_vector = (np.random.random((10000, 3)) * 2 - 1) * 0.3

phsp = {k: lv.boost(v, boost_vector) for k, v in phsp.items()}


a1 = config_hel.eval_amplitude(phsp)
scale_a1 = tf.reduce_mean(a1)

for config_file in [
    "config3_constrains.yml",
    "config_covten3.yml",
    "config3_2.yml",
    "config3_3.yml",
]:
    config = ConfigLoader("configs/" + config_file)
    config.set_params(config_hel.get_params())
    ai = config.eval_amplitude(phsp)
    scale_ai = tf.reduce_mean(ai)

    chi2 = float(tf.reduce_sum(((a1 / scale_a1) / (ai / scale_ai) - 1) ** 2))
    status = "success" if chi2 < 1e-7 else "failed"
    print(
        config_file,
        "\t",
        status,
        "\t",
        chi2,
        "  \t",
        float(scale_ai / scale_a1),
    )
