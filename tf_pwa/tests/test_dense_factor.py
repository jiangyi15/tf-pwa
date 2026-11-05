import numpy as np

import tf_pwa.amp  # noqa: F401  (register models)
import tf_pwa.amp.interpolation  # noqa: F401
import tf_pwa.experimental.dense_factor  # noqa: F401  (install dense plugin)
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import generate_phsp


def _config(particle, top=None, decay_opts=None):
    if top is None:
        top = {"J": 0, "P": 1, "mass": 4.6}
    r_decay = ["B", "D"]
    if decay_opts is not None:
        r_decay = ["B", "D", decay_opts]
    return {
        "data": {"format": "simple", "random_z": False, "r_boost": False},
        "decay": {"A": [["R", "C"]], "R": r_decay},
        "particle": {
            "$top": {"A": top},
            "$finals": {
                "B": {"J": 0, "P": 1, "mass": 1.0},
                "C": {"J": 0, "P": 1, "mass": 0.5},
                "D": {"J": 0, "P": 1, "mass": 0.4},
            },
            "R": particle,
        },
    }


def _compare(
    particle,
    force_cpu=False,
    top=None,
    decay_opts=None,
    amp_model="dense_factor",
):
    cfg = _config(particle, top=top, decay_opts=decay_opts)
    preprocessor = (
        {"dense_factor": {"force_cpu": True}} if force_cpu else "dense_factor"
    )
    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": preprocessor,
            "amp_model": amp_model,
        },
    }

    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    p4 = {"B": p_b, "D": p_d, "C": p_c}

    c1 = ConfigLoader(cfg)
    c2 = ConfigLoader(cfg2)
    data1 = c1.data.cal_angle(p4)
    data2 = c2.data.cal_angle(p4)

    a1 = c1.get_amplitude()
    a2 = c2.get_amplitude()
    a2.set_params(a1.get_params())

    for amp in (a1, a2):
        params = amp.get_params()
        amp.set_params(
            {
                k: (1.0 if k.endswith("_total_0r") else 0.0)
                for k in params
                if "_total_0" in k
            }
        )

    x1 = a1(data1).numpy()
    x2 = a2(data2).numpy()
    assert np.allclose(x1, x2, rtol=1e-6, atol=1e-8)


def test_dense_factor_bw():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        }
    )


def test_dense_factor_one():
    _compare({"J": 0, "P": 1, "model": "one"})


def test_dense_factor_force_cpu():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        },
        force_cpu=True,
    )


def test_dense_factor_polarization():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        },
        top={"J": 1, "P": -1, "mass": 4.6, "polarization": "vector"},
    )


def test_dense_factor_no_q0():
    _compare(
        {
            "J": 1,
            "P": -1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BWR",
        },
        top={"J": 1, "P": -1, "mass": 4.6},
        decay_opts={"no_q0": True},
    )


def test_dense_factor_below_threshold_finite():
    cfg = _config(
        {
            "J": 1,
            "P": -1,
            "m0": 1.0,
            "g0": 0.1,
            "model": "BWR",
        },
        top={"J": 1, "P": -1, "mass": 4.6},
    )
    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    config = ConfigLoader(cfg2)
    data = config.data.cal_angle({"B": p_b, "D": p_d, "C": p_c})
    amp = config.get_amplitude()
    values = amp(data).numpy()
    assert np.all(np.isfinite(values))


def test_dense_factor_bwr():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BWR",
            "running_width": True,
        }
    )


def test_dense_factor_flatte():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "model": "Flatte",
            "mass_list": [[1.0, 0.4], [0.5, 0.5]],
        }
    )


def test_dense_factor_flatte_c():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "model": "FlatteC",
            "mass_list": [[1.0, 0.4], [0.5, 0.5]],
        }
    )


def test_dense_factor_gs():
    _compare(
        {
            "J": 1,
            "P": -1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "GS_rho",
        }
    )


def test_dense_factor_gs_fixed_width():
    _compare(
        {
            "J": 1,
            "P": -1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "GS_rho",
            "running_width": False,
        }
    )


def test_dense_factor_bw_width_norm():
    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "width_norm": True,
        }
    )


def test_dense_factor_bwr_width_norm():
    _compare(
        {
            "J": 1,
            "P": -1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BWR",
            "width_norm": True,
        },
        top={"J": 1, "P": -1, "mass": 4.6},
    )


def test_dense_factor_complex64():
    from tf_pwa.config import get_config, set_config

    old = get_config("complex_dtype")
    set_config("complex_dtype", "complex64")
    try:
        _compare(
            {
                "J": 0,
                "P": 1,
                "m0": 2.0,
                "g0": 0.1,
                "model": "BWR",
            }
        )
    finally:
        set_config("complex_dtype", old)


def test_dense_factor_spline():
    _compare(
        {
            "J": 0,
            "P": 1,
            "model": "spline_c_idx",
            "min_m": 1.0,
            "max_m": 3.0,
            "interp_N": 8,
            "with_bound": True,
        }
    )


def test_decay_amplitude_terms():
    import tensorflow as tf

    from tf_pwa.amp.core import rename_data_dict

    cfg = _config(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        }
    )
    cfg["particle"]["$top"]["A"] = {"J": 1, "P": -1, "mass": 4.6}
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    config = ConfigLoader(cfg)
    data = config.data.cal_angle({"B": p_b, "D": p_d, "C": p_c})
    dg = config.get_amplitude().decay_group

    used = tuple(dg.chains[i] for i in dg.chains_idx)
    chain_maps = dg.get_chains_map(used)
    for chain in used:
        for chains in chain_maps:
            if str(chain) in [str(i) for i in chains]:
                maps = chains[chain]
                break
        data_c = rename_data_dict(
            data["decay"][chain.standard_topology()], maps
        )
        data_p = rename_data_dict(data["particle"], maps)
        for dec in chain:
            basis = dec.get_amplitude_basis(data_c[dec], data_p)["basis"]
            coeff = tf.stack(dec.get_amplitude_coeff()["coeff"])
            shape = [1, -1] + [1] * (len(basis.shape) - 2)
            recon = tf.reduce_sum(tf.reshape(coeff, shape) * basis, axis=1)
            ref = dec.get_amp(data_c[dec], data_p)
            assert np.allclose(recon.numpy(), ref.numpy())


def test_dense_factor_unsupported():
    import pytest

    cfg = _config({"J": 0, "P": 1, "model": "exp"})
    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    p4 = {"B": p_b, "D": p_d, "C": p_c}
    c2 = ConfigLoader(cfg2)
    with pytest.raises(
        NotImplementedError,
        match=r"R .*does not implement get_term_basis",
    ):
        c2.data.cal_angle(p4)


def test_dense_factor_used_chains():
    cfg = {
        "data": {"format": "simple", "random_z": False, "r_boost": False},
        "decay": {
            "A": [["R1", "C"], ["R2", "D"]],
            "R1": ["B", "D"],
            "R2": ["B", "C"],
        },
        "particle": {
            "$top": {"A": {"J": 1, "P": -1, "mass": 4.6}},
            "$finals": {
                "B": {"J": 0, "P": 1, "mass": 1.0},
                "C": {"J": 0, "P": -1, "mass": 0.5},
                "D": {"J": 0, "P": -1, "mass": 0.4},
            },
            "R1": {
                "J": 1,
                "P": 1,
                "m0": 2.0,
                "g0": 0.1,
                "model": "BW",
                "running_width": False,
            },
            "R2": {
                "J": 1,
                "P": 1,
                "m0": 2.2,
                "g0": 0.15,
                "model": "BW",
                "running_width": False,
            },
        },
    }
    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }

    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    p4 = {"B": p_b, "D": p_d, "C": p_c}

    c1 = ConfigLoader(cfg)
    c2 = ConfigLoader(cfg2)
    data1 = c1.data.cal_angle(p4)
    data2 = c2.data.cal_angle(p4)

    a1 = c1.get_amplitude()
    a2 = c2.get_amplitude()
    a2.set_params(a1.get_params())

    for amp in (a1, a2):
        params = amp.get_params()
        amp.set_params(
            {
                k: (1.0 if k.endswith("_total_0r") else 0.0)
                for k in params
                if "_total_0" in k
            }
        )

    for used in ([0, 1], [0], [1]):
        a1.set_used_chains(used)
        a2.set_used_chains(used)
        x1 = a1(data1).numpy()
        x2 = a2(data2).numpy()
        assert np.allclose(x1, x2, rtol=1e-6, atol=1e-8)


def test_dense_factor_unsupported_symmetry():
    import pytest

    cfg = _config(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        }
    )
    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }
    config = ConfigLoader(cfg2)
    amp = config.get_amplitude()

    with pytest.raises(NotImplementedError, match="identical-particle"):
        amp.build_dense_terms({"id_swap": {"x": {}}})

    with pytest.raises(NotImplementedError, match="CP conjugation"):
        amp.build_dense_terms({"cp_swap": {}})


def test_dense_rejects_custom_chain_and_decay():
    import pytest

    from tf_pwa.amp.core import DecayChain, HelicityDecay

    cfg = _config(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        }
    )
    dg = ConfigLoader(cfg).get_amplitude().decay_group
    chain = dg.chains[0]
    decay = list(chain)[0]

    class _CustomChain(DecayChain):
        def get_amp(self, *args, **kwargs):
            raise AssertionError

    class _CustomDecay(HelicityDecay):
        def get_amp(self, *args, **kwargs):
            raise AssertionError

    chain.__class__ = _CustomChain
    with pytest.raises(
        NotImplementedError, match="does not implement get_amplitude_basis"
    ):
        chain.get_amplitude_basis({}, {})
    with pytest.raises(
        NotImplementedError, match="does not implement get_amplitude_coeff"
    ):
        chain.get_amplitude_coeff()

    decay.__class__ = _CustomDecay
    with pytest.raises(
        NotImplementedError, match="does not implement get_amplitude_basis"
    ):
        decay.get_amplitude_basis({}, {})
    with pytest.raises(
        NotImplementedError, match="does not implement get_amplitude_coeff"
    ):
        decay.get_amplitude_coeff()


def test_dense_particle_subclass_guard():
    import pytest

    from tf_pwa.amp.base import ParticleBW
    from tf_pwa.amp.core import register_particle

    @register_particle("BWNoOverrideTest")
    class _NoOverride(ParticleBW):
        pass

    @register_particle("BWOverrideTest")
    class _Override(ParticleBW):
        def get_amp(self, data, _data_c=None, **kwargs):
            return super().get_amp(data, _data_c, **kwargs)

    with pytest.raises(
        NotImplementedError,
        match="does not implement get_term_basis",
    ):
        _Override.get_term_basis(_Override.__new__(_Override), {"m": 1.0})

    basis = _NoOverride.get_term_basis(
        _NoOverride.__new__(_NoOverride), {"m": 1.0}
    )
    assert "den" in basis


def test_dense_factor_op_model():
    """The fused-op model must reproduce ``dense_factor`` exactly."""
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    _compare(
        {
            "J": 0,
            "P": 1,
            "m0": 2.0,
            "g0": 0.1,
            "model": "BW",
            "running_width": False,
        },
        force_cpu=True,
        amp_model="dense_factor_op",
    )
