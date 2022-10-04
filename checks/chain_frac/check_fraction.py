import numpy as np

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import error_print

config = ConfigLoader("config.yml")
config.set_params("a.json")

config3 = ConfigLoader("config3.yml")
config3.set_params("a3.json")


ret = {}
ret3 = {}

for i in range(100):
    print("iter: ", i)
    phsp = config.generate_phsp(100000)
    fit_frac, _ = config.cal_fitfractions(mcdata=phsp)
    for i in fit_frac:
        ret[i] = ret.get(i, []) + [fit_frac[i]]
    phsp = config3.generate_phsp(100000)
    fit_frac, _ = config3.cal_fitfractions(mcdata=phsp)
    for i in fit_frac:
        ret3[i] = ret3.get(i, []) + [fit_frac[i]]


for i in ret3:
    print("3body ", i, np.mean(ret3[i]), "+/-", np.std(ret3[i]))
    print("4body ", i, np.mean(ret[i]), "+/-", np.std(ret[i]))

for i in ret3:
    print(
        "pull ",
        i,
        (np.mean(ret3[i]) - np.mean(ret[i]))
        / np.sqrt(np.std(ret3[i]) ** 2 + np.std(ret[i]) ** 2),
    )
