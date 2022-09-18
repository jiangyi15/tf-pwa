import json
import os

import matplotlib.pyplot as plt

params = []

for i in os.listdir("results"):
    if i.endswith("json"):
        with open("results/" + i) as f:
            tmp = json.load(f)
        params.append(tmp)


with open("toy_params.json") as f:
    toy_params = json.load(f)


def plot_var(var_name="R_BC_width", var_value=0.1):
    x = [i["value"][var_name] for i in params]
    x_err = [i["error"][var_name] for i in params]
    plt.clf()
    plt.errorbar(range(len(x)), x, yerr=x_err, marker="o", linestyle="none")
    plt.axhline(y=var_value)
    plt.xlabel("iters")
    plt.ylabel(var_name)
    plt.savefig("params_" + var_name.replace(">", ".") + ".png")


plot_var("R_BC_width", toy_params["R_BC_width"])
plot_var("R_BC_mass", toy_params["R_BC_mass"])
plot_var("A->NR.DNR->B.C_total_0r", toy_params["A->NR.DNR->B.C_total_0r"])
plot_var("A->NR.DNR->B.C_total_0i", toy_params["A->NR.DNR->B.C_total_0i"])
