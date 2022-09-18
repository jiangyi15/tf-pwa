import matplotlib.pyplot as plt

from tf_pwa.experimental import factor_system as fs

from .test_full import gen_toy, toy_config


def test_partial_amp(toy_config):
    amp = toy_config.get_amplitude()
    phsp = toy_config.get_data("phsp")[0]
    tw = amp(phsp)
    phsp = toy_config.generate_phsp(100000)
    tw = amp(phsp)
    id_, pw = fs.get_all_partial_amp(amp, phsp, ["g_ls"])
    mas = phsp.get_mass("(B, C)").numpy()
    plt.clf()
    plt.hist(mas, 100, weights=tw.numpy(), histtype="step")
    for name, i in zip(id_, pw):
        plt.hist(mas, 100, weights=i.numpy(), histtype="step", label=str(name))
    plt.yscale("log")
    plt.legend()
    plt.ylim((0.1, None))
    plt.savefig("factor_system.png")
