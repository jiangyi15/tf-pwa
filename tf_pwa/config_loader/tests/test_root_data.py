import os

import numpy as np
import pytest

from tf_pwa import root_io
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import PhaseSpaceGenerator

this_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def create_fake_data():
    a = PhaseSpaceGenerator(5280.0, [1800.0, 1800.0, 500.0])
    p = a.generate(100)
    p = [i.numpy() for i in p]

    dic = {}
    for idx_i, i in enumerate(["D1", "D2", "K"]):
        for idx_j, j in enumerate(["E", "PX", "PY", "PZ"]):
            dic[f"{i}_{j}"] = p[idx_i][..., idx_j]
        for idx_j, j in enumerate(["P_E", "P_X", "P_Y", "P_Z"]):
            dic[f"{i}_TRUE{j}"] = p[idx_i][..., idx_j]

    dic["year"] = np.array([2011, 2012, 2015, 2016, 2017] * 20)
    dic["Effcorr"] = np.random.random(100) + 0.1
    dic["B_M"] = np.random.normal(size=100) * 30 + 5280
    dic["B_Q"] = (np.random.random(100) > 0.5).astype(np.int32) * 2 - 1

    files = [
        "./toy_data/Data_run1.root",
        "./toy_data/Data_run2.root",
        "./toy_data/MC_run1.root",
        "./toy_data/MC_run2.root",
    ]
    os.makedirs("./toy_data/", exist_ok=True)
    for f in files:
        root_io.save_dict_to_root(dic, f, "DecayTree")


def test_root_data(create_fake_data):
    config = ConfigLoader(f"{this_dir}/config_root_data.yml")
    data = config.get_all_data()
