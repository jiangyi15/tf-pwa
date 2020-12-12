import os

import matplotlib
import numpy as np
import pytest
import tensorflow as tf

from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import build_amp

matplotlib.use("agg")


this_dir = os.path.dirname(os.path.abspath(__file__))


def generate_phspMC(Nmc):
    """Generate PhaseSpace MC of size Nmc and save it as txt file"""
    # masses of mother particle A and daughters BCD
    mA = 4.6
    mB = 2.00698
    mC = 2.01028
    mD = 0.13957
    # a2bcd is a [3*Nmc, 4] array, which are the momenta of BCD in the rest frame of A
    a2bcd = gen_mc(mA, [mB, mC, mD], Nmc)
    return a2bcd


def generate_toy_from_phspMC(Ndata, mc_file, data_file):
    """Generate toy using PhaseSpace MC from mc_file"""
    config = ConfigLoader(f"{this_dir}/config_toy.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    amp = config.get_amplitude()
    data = gen_data(amp, Ndata=Ndata, mcfile=mc_file, genfile=data_file)
    return data


@pytest.fixture
def gen_toy():
    if not os.path.exists("toy_data"):
        os.mkdir("toy_data")
    phsp = generate_phspMC(1000)
    np.savetxt("toy_data/PHSP.dat", phsp)
    generate_toy_from_phspMC(100, "toy_data/PHSP.dat", "toy_data/data.dat")
    bg = generate_phspMC(100)
    data = np.loadtxt("toy_data/data.dat")
    np.savetxt("toy_data/data.dat", np.concatenate([data, bg[:30, :]]))
    np.savetxt("toy_data/bg.dat", bg)
    np.savetxt("toy_data/data_bg_value.dat", np.ones((100 + 10,)))
    np.savetxt("toy_data/data_eff_value.dat", np.ones((100 + 10,)))
    np.savetxt("toy_data/phsp_bg_value.dat", np.ones((1000,)))
    np.savetxt("toy_data/phsp_eff_value.dat", np.ones((1000,)))


@pytest.fixture
def toy_config(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_toy.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    return config


def test_build_angle_amplitude(toy_config):
    data = toy_config.get_data("data")
    dec = toy_config.get_amplitude().decay_group
    amp_dict = build_amp.build_angle_amp_matrix(dec, data[0])
    assert len(amp_dict) == 3


@pytest.fixture
def toy_config2(gen_toy, fit_result):
    config = MultiConfig(
        [f"{this_dir}/config_toy.yml", f"{this_dir}/config_toy2.yml"]
    )
    config.set_params("toy_data/final_params.json")
    return config


@pytest.fixture
def fit_result(toy_config):
    return toy_config.fit()


def test_cfit(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_cfit.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})


def test_fit(toy_config, fit_result):
    toy_config.plot_partial_wave(prefix="toy_data/figure", save_root=True)
    toy_config.plot_partial_wave(
        prefix="toy_data/figure", plot_pull=True, single_legend=True
    )
    toy_config.get_params_error(fit_result)
    fit_result.save_as("toy_data/final_params.json")
    toy_config.cal_fitfractions()


def test_cal_chi2(toy_config, fit_result):
    toy_config.cal_chi2(bins=[[2, 2]] * 2, mass=["R_BD", "R_CD"])


def test_cal_signal_yields(toy_config, fit_result):
    toy_config.cal_signal_yields()


def test_fit_combine(toy_config2):
    toy_config2.fit()
    toy_config2.get_params_error()
