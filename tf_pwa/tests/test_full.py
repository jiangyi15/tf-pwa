import os
import time

import matplotlib
import numpy as np
import pytest
import tensorflow as tf
import yaml

from tf_pwa import set_random_seed
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import build_amp
from tf_pwa.utils import save_frac_csv

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
    data = gen_data(
        amp,
        Ndata=Ndata,
        mcfile=mc_file,
        genfile=data_file,
        particles=config.get_dat_order(),
    )
    return data


@pytest.fixture
def gen_toy():
    set_random_seed(1)
    if not os.path.exists("toy_data"):
        os.mkdir("toy_data")
    phsp = generate_phspMC(10000)
    np.savetxt("toy_data/PHSP.dat", phsp)
    generate_toy_from_phspMC(1000, "toy_data/PHSP.dat", "toy_data/data.dat")
    bg = generate_phspMC(1000)
    data = np.loadtxt("toy_data/data.dat")
    np.savetxt("toy_data/data.dat", np.concatenate([data, bg[:300, :]]))
    np.savetxt("toy_data/bg.dat", bg)
    np.savetxt("toy_data/data_bg_value.dat", np.ones((1000 + 100,)))
    np.savetxt("toy_data/data_eff_value.dat", np.ones((1000 + 100,)))
    np.savetxt("toy_data/phsp_bg_value.dat", np.ones((10000,)))
    np.savetxt("toy_data/phsp_eff_value.dat", np.ones((10000,)))


@pytest.fixture
def toy_npy(gen_toy):
    for i in ["data", "bg", "PHSP"]:
        data = np.loadtxt(f"toy_data/{i}.dat")
        np.save(f"toy_data/{i}_npy.npy", data)


@pytest.fixture
def toy_config(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_toy.yml")
    config.set_params(f"{this_dir}/exp_params.json")
    return config


@pytest.fixture
def toy_config_extended(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_extended.yml")
    config.set_params(f"{this_dir}/exp_params.json")
    return config


@pytest.fixture
def toy_config_npy(toy_npy):
    config = ConfigLoader(f"{this_dir}/config_toy_npy.yml")
    config.set_params(f"{this_dir}/exp_params.json")
    return config


@pytest.fixture
def toy_config_lazy(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_lazycall.yml")
    config.set_params(f"{this_dir}/exp_params.json")
    return config


def test_build_angle_amplitude(toy_config):
    data = toy_config.get_data("data")
    dec = toy_config.get_amplitude().decay_group
    amp_dict = build_amp.build_angle_amp_matrix(dec, data[0])
    assert len(amp_dict[1]) == 3


@pytest.fixture
def toy_config2(gen_toy, fit_result):
    config = MultiConfig(
        [f"{this_dir}/config_toy.yml", f"{this_dir}/config_toy2.yml"]
    )
    config.set_params(f"{this_dir}/exp_params.json")
    return config


@pytest.fixture
def toy_config3(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_toy3.yml")
    config.set_params(f"{this_dir}/exp_params.json")
    return config


@pytest.fixture
def fit_result(toy_config):
    ret = toy_config.fit()
    assert np.allclose(ret.min_nll, -204.9468493307786)
    return ret


def test_save_model(toy_config):
    config = ConfigLoader(f"{this_dir}/config_toy.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    config.save_tensorflow_model("toy_data/model")


def test_cfit(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_cfit.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})
    config.plot_partial_wave()
    amp = config.get_amplitude()
    config.plot_partial_wave(
        prefix="toy_data/figure/s", linestyle_file="toy_data/a.yml"
    )
    config.plot_partial_wave(
        prefix="toy_data/figure/ss",
        linestyle_file="toy_data/a.yml",
        ref_amp=amp,
    )
    config.plot_partial_wave(
        prefix="toy_data/figure/s2",
        linestyle_file="toy_data/a.yml",
        chains_id_method="res",
    )
    config.get_plotter().save_all_frame(prefix="toy_data/figure/s3", idx=0)
    plotter = config.get_plotter("toy_data/a.yml", use_weighted=True)
    plotter.smooth = True
    plotter.add_ref_amp(config.get_amplitude())
    with plotter.old_style():
        plotter.save_all_frame(prefix="toy_data/figure/s4", plot_pull=True)
    plotter.forzen_style()
    plotter.style.save()
    plotter.plot_var(amp)


def test_precached(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_precached.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})
    config.plot_partial_wave(prefix="toy_data/figure/s5")


def test_precached2(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_precached2.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})


def test_precached3(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_precached3.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})


def test_cfit_cached(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_cfit_cached.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})


def test_extended(toy_config_extended):
    fcn = toy_config_extended.get_fcn()
    fcn({})
    fcn.nll_grad()
    toy_config_extended.cal_signal_yields()


def test_cfit_extended(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_cfit_extended.yml")
    config.set_params(f"{this_dir}/gen_params.json")
    fcn = config.get_fcn()
    fcn({})
    fcn.nll_grad({})
    fcn.nll_grad_hessian({})


def test_cfit_lazy_call(gen_toy):
    with open(f"{this_dir}/config_cfit.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["lazy_call"] = True
    config = ConfigLoader(config_dic)
    config.set_params(f"{this_dir}/exp_params.json")
    fcn = config.get_fcn()
    fcn.nll_grad()
    config.plot_partial_wave(prefix="toy_data/figure/c2")


def test_fit_lazy_call(gen_toy):
    with open(f"{this_dir}/config_toy.yml") as f:
        config_dic = yaml.full_load(f)
    config_dic["data"]["lazy_call"] = True
    config = ConfigLoader(config_dic)
    config.set_params(f"{this_dir}/exp_params.json")
    results = config.fit(print_init_nll=False)
    assert np.allclose(results.min_nll, -204.9468493307786)
    fcn = config.get_fcn()
    fcn.nll_grad()
    config.plot_partial_wave(prefix="toy_data/figure/s2")
    config.get_plotter().save_all_frame(prefix="toy_data/figure/s3")
    config.cal_fitfractions()


def test_cfit_resolution(gen_toy):
    with open(f"{this_dir}/config_rec.yml") as f:
        config_dic = yaml.full_load(f)
    config = ConfigLoader(config_dic)
    config.set_params(f"{this_dir}/exp_params.json")
    fcn = config.get_fcn()
    fcn.nll_grad()
    config.plot_partial_wave(prefix="toy_data/figure/c3")


def test_constrains(gen_toy):
    config = ConfigLoader(f"{this_dir}/config_cfit.yml")
    var_name = "A->R_CD.B_g_ls_1r"
    config.config["constrains"]["init_params"] = {var_name: 1.0}

    @config.register_extra_constrains("init_params")
    def float_var(amp, params=None):
        amp.set_params(params)

    config.register_extra_constrains("init_params2", float_var)

    amp = config.get_amplitude()
    assert amp.get_params()[var_name] == 1.0


def test_fit(toy_config, fit_result):
    toy_config.plot_partial_wave(
        prefix="toy_data/figure/no_pull", plot_pull=False
    )
    toy_config.plot_partial_wave(
        prefix="toy_data/figure/has_pull", plot_pull=True
    )
    toy_config.plot_partial_wave(prefix="toy_data/figure", save_root=True)
    toy_config.plot_partial_wave(
        prefix="toy_data/figure", plot_pull=True, single_legend=True
    )
    toy_config.plot_partial_wave(
        prefix="toy_data/figure/s_res",
        smooth=False,
        bin_scale=1,
        res=["R_BC", ["R_BD", "R_CD"]],
    )
    toy_config.plot_partial_wave(prefix="toy_data/figure", color_first=False)
    toy_config.get_params_error(fit_result)
    toy_config.get_params_error(fit_result, method="3-point")
    toy_config.get_params_error(
        fit_result, correct_params=["A->R_CD.BR_CD->C.D_total_0i"]
    )
    fit_result.save_as("toy_data/final_params.json")
    fit_frac, frac_err = toy_config.cal_fitfractions()
    fit_frac, frac_err = toy_config.cal_fitfractions(method="new")
    save_frac_csv("toy_data/fit_frac.csv", fit_frac)

    with toy_config.params_trans() as pt:
        a = pt["A->R_BC.D_g_ls_1r"]
        b = pt["A->R_BC.D_g_ls_1i"]
        alpha = 2 * a / (1 + a * a + b + b)
    alpha_err = pt.get_error(alpha)
    with toy_config.params_trans() as pt:
        a = pt["A->R_BC.D_g_ls_1r"]
        b = pt["A->R_BC.D_g_ls_1i"]
        x = a + b
        y = a - b
    xy_err = pt.get_error_matrix([x, y])
    with toy_config.params_trans() as pt:
        a = pt["A->R_BC.D_g_ls_1r"]
        with pt.mask_params({"A->R_BC.D_g_ls_1i": 0.0}):
            b = pt["A->R_BC.D_g_ls_1i"]
        x = a + b
        y = a - b
    xy_err2 = pt.get_error({"a": [x, y]})

    # mask params for fit fraction
    amp = toy_config.get_amplitude()
    phsp = toy_config.get_phsp_noeff()
    int_mc = amp.vm.batch_sum_var(amp, phsp)
    ys = []
    for i in toy_config.get_decay():
        mask_params = {}
        for j in toy_config.get_decay():
            if i != j:
                mask_params[str(j.total) + "_0r"] = 0
        with toy_config.mask_params(mask_params):
            ys.append(amp.vm.batch_sum_var(amp, phsp))
        with amp.mask_params(mask_params):
            pass

    with toy_config.params_trans() as pt:
        y = [i() for i in ys]
        frac = [i / int_mc() for i in y]
    pt.get_error(frac)

    for i in amp.factor_iteration():
        pass

    toy_config.attach_fix_params_error({"R_BC_mass": 0.01})


def test_bacth_sum(toy_config, fit_result):
    toy_config.get_params_error(fit_result)
    res = list(range(len(list(toy_config.get_decay()))))
    fit_frac, fit_frac_err = toy_config.cal_fitfractions(res=res)
    amp = toy_config.get_amplitude()
    phsp = toy_config.get_phsp_noeff()
    int_mc = toy_config.batch_sum_var(amp, phsp, batch=5000)
    ys = []
    for i in toy_config.get_decay():
        mask_params = {}
        for j in toy_config.get_decay():
            if i != j:
                mask_params[str(j.total) + "_0r"] = 0
        with toy_config.mask_params(mask_params):
            ys.append(toy_config.batch_sum_var(amp, phsp))

    with toy_config.params_trans() as pt:
        y = [i() for i in ys]
        frac = [i / int_mc() for i in y]
    frac_err = pt.get_error(frac)
    assert np.allclose(frac, [fit_frac[str(i)] for i in res])
    assert np.allclose(frac_err, [fit_frac_err[str(i)] for i in res])


def test_lazycall(toy_config_lazy):
    results = toy_config_lazy.fit(batch=100000)
    assert np.allclose(results.min_nll, -204.9468493307786)
    toy_config_lazy.plot_partial_wave(
        prefix="toy_data/figure_lazy", batch=100000
    )


def test_cal_chi2(toy_config, fit_result):
    toy_config.cal_chi2(bins=[[2, 2]] * 2, mass=["R_BD", "R_CD"])


def test_cal_signal_yields(toy_config, fit_result):
    toy_config.cal_signal_yields()


def test_fit_combine(toy_config2):
    results = toy_config2.fit()
    print(results)
    assert np.allclose(results.min_nll, -204.9468493307786 * 2)
    toy_config2.get_params_error()
    print(toy_config2.get_params())


def test_mix_likelihood(toy_config3):
    results = toy_config3.fit(maxiter=1)


def test_cp_particles():
    config = ConfigLoader(f"{this_dir}/config_self_cp.yml")
    phsp = config.generate_phsp(100)
    config.get_amplitude()(phsp)


def test_plot_2dpull(toy_config):
    import matplotlib.pyplot as plt

    toy_config.plot_adaptive_2dpull("m_R_BC**2", "m_R_CD**2")
    with pytest.raises(AssertionError):
        a, b = toy_config.get_dalitz_boundary("R_BC", "R_BC")
    a, b = toy_config.get_dalitz_boundary("R_BC", "R_CD")
    plt.plot(a, b, color="red")
    plt.savefig("adptive_2d.png")


def test_lazy_file(toy_config_npy):
    fcn = toy_config_npy.get_fcn()
    fcn.nll_grad()


def test_factor_hel():
    config = ConfigLoader(f"{this_dir}/config_hel.yml")
    phsp = config.generate_phsp(10)
    amp = config.get_amplitude()
    amp(phsp)
    amp.get_amp_list_part(phsp)
    amp.decay_group.get_factor()


def test_simple_model(toy_config):
    from tf_pwa.model import custom

    with open(f"{this_dir}/config_toy.yml") as f:
        dic = yaml.full_load(f)
    dic["data"]["model"] = "simple"
    config = ConfigLoader(dic)
    fcn = config.get_fcn()
    print(fcn())
    print(fcn.nll_grad())
