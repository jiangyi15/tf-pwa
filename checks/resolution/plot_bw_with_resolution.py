import matplotlib.pyplot as plt
import numpy as np
import yaml
from detector import trans_function
from sample import smear_function_table

# loda detector model parameters
with open("detector.yml") as f:
    detector_config = yaml.safe_load(f)


def bw(m, m0=1.8, g0=0.05):
    s = m * m
    m02 = m0 * m0
    return np.abs(g0 / (s - m02 - 1.0j * m0 * g0)) ** 2


def smear_function(m, N=100, method="legendre"):
    m_min = 0.1
    m_max = 1.9
    f = smear_function_table[method]
    ms = []
    ws = []
    for i in range(N):
        m2, w = f(m, m_min, m_max, i, N)
        ms.append(m2)
        ws.append(w)
    ms = np.stack(ms)
    ws = np.stack(ws)
    sum_ws = np.sum(ws, axis=0)
    ws = np.where(sum_ws[None, :] != 0, ws, 1e-6)
    sum_ws = np.where(sum_ws == 0, 1.0, sum_ws)
    ws = ws / sum_ws
    fm = np.sum(bw(ms) * ws, axis=0)
    return fm


def main():
    x = np.linspace(1.3, 1.9, 1000)
    N = 10

    # plt.plot(x, smear_function(x, N=N, method="random"), label="random")
    plt.plot(x, bw(x - detector_config["bias"]), label="origin")
    plt.plot(x, smear_function(x, N=N, method="legendre"), label="legendre")
    plt.plot(x, smear_function(x, N=N, method="linear"), label="linear")
    plt.plot(x, smear_function(x, N=N, method="hermite"), label="hermite")
    plt.plot(x, smear_function(x, N=N, method="hermite2"), label="hermite2")
    plt.plot(
        x, smear_function(x, N=N, method="gauss_interp"), label="gauss_interp"
    )
    plt.plot(
        x, smear_function(x, N=1000, method="linear"), label="ref", ls="--"
    )
    plt.legend()
    plt.xlabel("mass")
    plt.savefig("bw_with_resolution.png")


if __name__ == "__main__":
    main()
