import json
import os

import matplotlib.pyplot as plt
import numpy as np

params = []

for i in os.listdir("results"):
    if i.endswith("json"):
        with open("results/" + i) as f:
            tmp = json.load(f)
        params.append(tmp)


with open("toy_params.json") as f:
    toy_params = json.load(f)


def plot_var(var_name="R_BC_width", var_value=0.1):
    x = np.array([i["value"][var_name] for i in params])
    x_err = np.array([i["error"][var_name] for i in params])
    plt.clf()
    N = len(x)
    x_mean = np.mean(x)
    x_mean_err = np.sqrt(np.sum(x_err**2)) / N
    plt.errorbar(
        [0], [x_mean], yerr=[x_mean_err], marker="o", linestyle="none"
    )
    plt.fill_between(
        [-1, N + 1],
        [x_mean - x_mean_err, x_mean - x_mean_err],
        [x_mean + x_mean_err, x_mean + x_mean_err],
        alpha=0.5,
        label="mean+/-sigma",
    )
    plt.errorbar(
        np.linspace(1, N, N),
        x,
        yerr=x_err,
        marker="o",
        linestyle="none",
        label="each iteration",
    )
    plt.axhline(y=var_value)
    plt.xlabel("iters")
    plt.ylabel(var_name)
    plt.legend()
    plt.savefig("params_" + var_name.replace(">", ".") + ".png")


plot_var("R_BC_width", toy_params["R_BC_width"])
plot_var("R_BC_mass", toy_params["R_BC_mass"])
plot_var("A->NR.DNR->B.C_total_0r", toy_params["A->NR.DNR->B.C_total_0r"])
plot_var("A->NR.DNR->B.C_total_0i", toy_params["A->NR.DNR->B.C_total_0i"])
