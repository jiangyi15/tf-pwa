import matplotlib.pyplot as plt
import numpy as np

from tf_pwa.amp.kmatrix_simple import HelicityDecay, KmatrixSimple, Particle


def test_KmatixSimple():
    a = KmatrixSimple(
        name="a", J=1, P=1, mass_list=[3.87, 4.0]
    )  # , decay_list=[[1.8, 1.8], [1.9,2.0]], l_list=[0, 1])
    b = Particle("b", mass=2.0, J=1, P=-1)
    c = Particle("c", mass=1.8, J=0, P=-1)
    HelicityDecay(a, [b, c])
    a.init_params()
    print(a.mi[0].vm.variables)
    m = np.linspace(3.8, 4.05, 1000)
    amp = a.get_ls_amp(m).numpy()
    plt.plot(m, np.real(amp[:, 0]))
    plt.plot(m, np.real(amp[:, 1]))
    plt.plot(m, np.imag(amp[:, 0]))
    plt.plot(m, np.imag(amp[:, 1]))
    plt.plot(m, np.abs(amp[:, 0]) ** 2)
    plt.plot(m, np.abs(amp[:, 1]) ** 2)
    plt.savefig("kmatrix_1pp2.png")
