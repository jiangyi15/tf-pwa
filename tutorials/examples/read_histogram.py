import argparse

import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy import interpolate


def mass(x):
    return x[:, 0] ** 2 - x[:, 1] ** 2 - x[:, 2] ** 2 - x[:, 3] ** 2


def get_histogram_function(file_name, branch):
    """get histogram function from root file in branch"""

    def fill_bound(x):
        x[0] = x[1] - (x[2] - x[1])
        x[-1] = x[-2] + (x[-2] - x[-3])
        return x

    # "data/EffMap_B0toD0Dspi_Run2.root"
    with uproot.open(file_name) as f:
        bg = f.get(branch)  # "RegDalitzEfficiency")

        counts, edges = bg.allnumpy()
    x, y = edges[0]  # D0barpi, Dspi
    x = fill_bound(x)
    y = fill_bound(y)
    x = (x[:-1] + x[1:]) / 2
    y = (y[:-1] + y[1:]) / 2
    f = interpolate.RectBivariateSpline(x, y, counts)
    return f, x, y


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default="data/phsp.dat")
    parser.add_argument("-o", "--out_file", default=None)
    parser.add_argument("-r", "--root_file", default="data/BG_Hist.root")
    parser.add_argument(
        "-b", "--branch_name", default="BackgroundDistribution"
    )

    args = parser.parse_args()

    p1, p2, p3 = (
        np.loadtxt(args.input_file).reshape((-1, 3, 4)).transpose((1, 0, 2))
    )

    print("m0: ", np.sqrt(mass(p1 + p2 + p3)))
    print("m1: ", np.sqrt(mass(p1)))
    print("m2: ", np.sqrt(mass(p2)))
    print("m3: ", np.sqrt(mass(p3)))

    m13 = mass(p1 + p3)
    m23 = mass(p2 + p3)

    f, x, y = get_histogram_function(args.root_file, args.branch_name)
    xx, yy = np.meshgrid(x, y)

    if args.out_file is None:
        out_file = "{}_value.dat".format(args.input_file[:-4])
    else:
        out_file = args.out_file
    np.savetxt(out_file, f.ev(m13, m23))

    plt.contourf(xx, yy, f(x, y), vmin=1e-6)
    plt.colorbar()
    plt.xlabel("$M^2_{13}$")
    plt.ylabel("$M^2_{23}$")
    plt.savefig("bg_histogram.png")


if __name__ == "__main__":
    main()
