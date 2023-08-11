import numpy as np

from tf_pwa.amp import Decay, DecayChain, DecayGroup, Particle
from tf_pwa.config_loader.data import load_data_mode
from tf_pwa.config_loader.decay_config import DecayConfig
from tf_pwa.tests.test_full import gen_toy


def test_load_data(gen_toy):
    p = [Particle(f"name:{i}") for i in range(5)]
    dec = DecayGroup(
        [DecayChain([Decay(p[0], [p[1], p[2]]), Decay(p[1], [p[3], p[4]])])]
    )

    # config = dec.as_config()
    # config["data"] = {"decay_chain_cut": []}

    # dec = DecayConfig(config)

    data_file = {
        "data": "toy_data/data.dat",
        "phsp": "toy_data/phsp.dat",
        "bg": "toy_data/bg.dat",
        "bg_weight": 0.1,
        "weight_scale": True,
    }
    data = load_data_mode(data_file, dec, "multi")
    w1 = data.get_data("bg")[0]["weight"]

    data = load_data_mode(data_file, dec, "simple")
    w2 = data.get_data("bg")["weight"]
    assert np.allclose(w1, w2)

    data_file["weight_scale"] = False
    data = load_data_mode(data_file, dec, "multi")
    w1 = data.get_data("bg")[0]["weight"]

    data = load_data_mode(data_file, dec, "simple")
    w2 = data.get_data("bg")["weight"]
    assert np.allclose(w1, w2)

    data = load_data_mode({**data_file, "cp_trans": True}, dec, "simple")
    charge = np.random.random(w2.shape) > 0.5
    np.savetxt("toy_data/test_charge.dat", charge)
    p1 = data.load_data("toy_data/bg.dat", charge="toy_data/test_charge.dat")
    assert np.sum(p1["charge_conjugation"] - charge) == 0

    data.savetxt("toy_data/test_save.dat", p1)
