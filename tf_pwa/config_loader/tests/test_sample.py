import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.config_loader.sample import build_phsp_chain_sorted
from tf_pwa.data import data_shape
from tf_pwa.tests.test_full import gen_toy, toy_config


def test_generate_phsp(toy_config):
    data = toy_config.generate_toy(1000, force=False)
    assert data_shape(data) >= 1000
    data = toy_config.generate_toy(1000)
    assert data_shape(data) == 1000
    data = toy_config.generate_toy2(1000)
    assert data_shape(data) == 1000
    data = toy_config.generate_toy2(1000, gen=toy_config.generate_phsp)
    assert data_shape(data) == 1000
    data = toy_config.generate_toy2(
        1000, gen_p=toy_config.generate_phsp_p, include_charge=True
    )
    data.savetxt(
        "toy_data/data_c.dat",
        toy_config.get_dat_order(),
        cp_trans=True,
        save_charge=True,
    )
    assert data_shape(data) == 1000
    data = toy_config.generate_toy2(1000, gen_p=toy_config.generate_phsp_p)
    assert data_shape(data) == 1000
    gen_p2 = lambda N: {
        str(k): v for k, v in toy_config.generate_phsp_p(N).items()
    }
    data = toy_config.generate_toy2(1000, gen_p=gen_p2)
    assert data_shape(data) == 1000
    data = toy_config.generate_toy_p(1000)
    assert data_shape(data) == 1000
    data = toy_config.generate_toy_p(1000, include_charge=True)
    assert data_shape(data) == 1000


config_text = """
decay:
    A: [[R1, B], [R2, C]]
    R1: [C, D]
    R2: [B, D]
    D: [E, F]
    E: [G, H]

particle:
    $top:
        A: {m0: 2.0}
    $finals:
        B: {m0: 0.3}
        C: {m0: 0.3}
        F: {m0: 0.3}
        G: {m0: 0.1}
        H: {m0: 0.1}
    E: {m0: 0.3, J: 0, P: 1, model: one}
    R1: {m0: 1.4, g0: 0.05, J: 0, P: 1}
    R2: {m0: 1.3, g0: 0.03, J: 0, P: 1}
    D: {m0: 0.8, model: one}

"""


def test_chain_phsp():
    dic = yaml.full_load(config_text)
    config = ConfigLoader(dic)
    data = config.generate_toy(10)
    data.mass_hist("(C, F, G, H)").draw()
    import matplotlib.pyplot as plt

    plt.savefig("chain_phsp.png")


config_text2 = """
decay:
    A: [[R1, B, p_break: True], [R2, C, p_break: True]]
    R1: [C, D]
    R2: [B, D]
    D: [E, F]

particle:
    $top:
        A: {m0: 2.0}
    $finals:
        B: {m0: 0.3}
        C: {m0: 0.3}
        F: {m0: 0.3}
        E: {m0: 0.3}
    R1: {m0: 1.4, g0: 0.05, J: 1, P: -1}
    R2: {m0: 1.3, g0: 0.03, J: 1, P: -1}
    D: {m0: 0.8, g0: 0.05, J: 1, P: -1, model: BW}

"""


def test_importance_f():
    dic = yaml.full_load(config_text2)
    config = ConfigLoader(dic)
    from tf_pwa.generator.breit_wigner import BWGenerator
    from tf_pwa.phasespace import PhaseSpaceGenerator

    bw = BWGenerator(0.8, 0.05, 0.6, 1.4)
    phsp = PhaseSpaceGenerator(2.0, [0.3, 0.3, 0.3, 0.3])
    phsp.mass_generator[0] = bw

    def gen_p(N):
        ret = phsp.generate(N)
        return dict(zip("BCEF", ret))

    def importance_f(data):
        m = data.get_mass("(E, F)").numpy()
        return bw(m)

    data = config.generate_toy2(100, gen_p=gen_p, importance_f=importance_f)

    import matplotlib.pyplot as plt

    plt.clf()
    data.mass_hist("(E, F)").draw()
    plt.savefig("importance_f.png")
