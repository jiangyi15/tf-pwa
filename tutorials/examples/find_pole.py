import extra_amp
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import MultiConfig
from tf_pwa.data import data_to_numpy
from tf_pwa.utils import error_print

config = MultiConfig(["config.yml", "config_c.yml"], total_same=True)
config.set_params("final_params.json")
config.inv_he = np.load("error_matrix.npy")

ret = {}
with config.params_trans() as pm:
    for i in config.configs[1].get_decay():
        if "NR" in str(i):
            continue
        res = i[1].core
        p = res.solve_pole()
        ret[res] = [
            res.get_mass() * 1000,
            res.get_width() * 1000,
            tf.math.real(p) * 1000,
            tf.math.imag(p) * 1000,
        ]

a, b = data_to_numpy((ret, pm.get_error(ret)))

print("res  ", "\t", "pole(mass - iwidth/2)", "\t", "BW mass/(width/2)")
for i in a:
    if b[i][3] == 0:
        continue
    print(
        i,
        "\t",
        error_print(a[i][2], b[i][2]),
        "\t",
        error_print(a[i][0], b[i][0]),
    )
    print(
        i,
        "\t ",
        error_print(a[i][3], b[i][3]),
        "\t",
        error_print(a[i][1] / 2, b[i][1] / 2),
    )

for i in a:
    if b[i][3] != 0:
        continue
    print(
        i,
        "\t",
        error_print(a[i][2], b[i][2]),
        "\t",
        error_print(a[i][0], b[i][0]),
    )
    print(
        i,
        "\t ",
        error_print(a[i][3], b[i][3]),
        "\t",
        error_print(a[i][1] / 2, b[i][1] / 2),
    )
