import time

import matplotlib.pyplot as plt
import numpy as np

from tf_pwa.generator.interp_nd import InterpND


def test_gen_2d():
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 2 * np.pi, 101)
    z = x[:, None] * np.sin(y) + 1
    # z = np.array([[0.5,0],[0, 2], [0,1]])
    interp = InterpND([x, y], z)

    a = time.time()
    ret = interp.generate(20000)
    print(time.time() - a)

    plt.hist2d(ret[:, 0], ret[:, 1], bins=200)
    plt.savefig("interp_nd_b.png")
    plt.clf()

    X, Y = np.meshgrid(x, y)
    z = interp([X.flatten(), Y.flatten()])
    plt.imshow(z.reshape(101, 101), origin="lower")
    plt.savefig("interp_nd_b2.png")


def test_gen_1d():
    y = np.linspace(0, 2 * np.pi, 10000)
    z = np.sin(y) + 1
    interp = InterpND([y], z)
    a = time.time()
    ret = interp.generate(20000)
    print(time.time() - a)

    plt.hist(ret[:, 0], bins=1000)
    z = interp([y]) * 20000 / 1000  #  /( 2*np.pi )
    plt.plot(y, z)
    plt.savefig("interp_nd_b3.png")
