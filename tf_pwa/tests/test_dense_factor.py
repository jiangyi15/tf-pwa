import os

import numpy as np
import pytest

import tf_pwa.amp  # noqa: F401  (register models)
import tf_pwa.amp.interpolation  # noqa: F401
import tf_pwa.experimental.dense_factor  # noqa: F401  (install dense plugin)
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import PhaseSpaceGenerator, generate_phsp


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
    # the preprocessor pulls the amplitude when the data is built, so an
    # unsupported term model fails as soon as the config is loaded
    with pytest.raises(
        NotImplementedError,
        match=r"R .*does not implement get_term_coeff",
    ):
        ConfigLoader(cfg2)


def test_dense_preprocessor_with_default_model():
    """Another amp_model still gets complete data from the preprocessor."""
    cfg = _config({"J": 0, "P": 1, "model": "one"})
    base = {**cfg, "data": {**cfg["data"], "amp_model": "default"}}
    dense = {
        **cfg,
        "data": {
            **base["data"],
            "preprocessor": "dense_factor",
        },
    }
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    p4 = {"B": p_b, "D": p_d, "C": p_c}

    c0 = ConfigLoader(base)
    c1 = ConfigLoader(dense)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    data1 = c1.data.cal_angle(p4)
    # the basis comes from the decay group, not from the amplitude model
    assert "F" in data1 and "B" in data1
    x0 = a0(c0.data.cal_angle(p4)).numpy()
    x1 = a1(data1).numpy()
    assert np.allclose(x0, x1, rtol=1e-10, atol=1e-12)


def test_group_dense_structure():
    """Attaching the plugin gives DecayGroup a lazy basis getter."""
    from tf_pwa.amp.core import DecayGroup

    cfg = _config({"J": 0, "P": 1, "model": "one"})
    cfg = {**cfg, "data": {**cfg["data"], "amp_model": "default"}}
    dg = ConfigLoader(cfg).full_decay

    assert hasattr(DecayGroup, "get_dense_structure")
    structure = dg.get_dense_structure()
    assert isinstance(structure, dict)
    assert structure["n_u"] > 0
    # the particle-derived entries are there before any event is seen
    assert structure["f_shape"] == (len(dg.top.spins),) + tuple(
        len(p.spins) for p in dg.outs
    )
    assert structure["cp"] == (1.0, ())
    assert dg.get_dense_structure() is structure  # built once, no amp involved


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


def test_dense_factor_op_two_lines():
    """Multi-line chains fall back to the dense path and stay exact."""
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
            "decay": {
                "B": [["R1", "X"]],
                "R1": [["R2", "Y"]],
                "R2": ["P", "Q"],
            },
            "particle": {
                "$top": {"B": {"J": 1, "P": -1, "mass": 5.0}},
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.30},
                    "Y": {"J": 0, "P": -1, "mass": 0.25},
                    "P": {"J": 0, "P": -1, "mass": 0.20},
                    "Q": {"J": 0, "P": -1, "mass": 0.20},
                },
                "R1": {"J": 1, "P": -1, "m0": 3.0, "g0": 0.10, "model": "BWR"},
                "R2": {"J": 1, "P": -1, "m0": 1.5, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.30, 0.25, 0.20, 0.20])
    p4 = dict(zip(["X", "Y", "P", "Q"], ps.generate(100)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    x0 = a0(d0).numpy()
    x1 = a1(d1).numpy()
    assert np.allclose(x0, x1, rtol=1e-10, atol=1e-12)

    # Gradients are covered by test_dense_factor_op_gradient (shared
    # offsets, padded lines, nu > block size); this config has no free
    # couplings, so a tape here would be empty.


def test_dense_factor_op_three_lines():
    """Three denominator lines: the general op has no line-count bound."""
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
            "decay": {
                "B": [["R1", "X"]],
                "R1": [["R2", "Y"]],
                "R2": [["R3", "Z"]],
                "R3": ["P", "Q"],
            },
            "particle": {
                "$top": {"B": {"J": 1, "P": -1, "mass": 6.0}},
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.30},
                    "Y": {"J": 0, "P": -1, "mass": 0.25},
                    "Z": {"J": 0, "P": -1, "mass": 0.20},
                    "P": {"J": 0, "P": -1, "mass": 0.20},
                    "Q": {"J": 0, "P": -1, "mass": 0.20},
                },
                "R1": {"J": 1, "P": -1, "m0": 4.0, "g0": 0.10, "model": "BWR"},
                "R2": {"J": 1, "P": -1, "m0": 3.0, "g0": 0.10, "model": "BWR"},
                "R3": {"J": 1, "P": -1, "m0": 2.0, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(6.0, [0.30, 0.25, 0.20, 0.20, 0.20])
    p4 = dict(zip(["X", "Y", "Z", "P", "Q"], ps.generate(100)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    # Gradients for the general path are covered by
    # test_dense_factor_op_gradient; this config has no free couplings.


def test_dense_factor_op_gradient():
    """dC and dg with shared parameter blocks, padded lines, and nu > 256."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import (
        dense_factor_a,
        dense_factor_an,
        is_available,
    )

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    rng = np.random.default_rng(0)

    def rel(a, b):
        return float(
            tf.reduce_max(tf.abs(a - b)) / (tf.reduce_max(tf.abs(b)) + 1e-30)
        )

    def check(nu, off, sz, nline, nparam, np=1):
        ne = 7
        f = rng.standard_normal((ne, np, nu)) + 1j * rng.standard_normal(
            (ne, np, nu)
        )
        bt = rng.standard_normal((ne, nparam)) + 1j * rng.standard_normal(
            (ne, nparam)
        )
        c = rng.standard_normal(nu) + 1j * rng.standard_normal(nu)
        # the denominator coefficients are real; the op refuses complex ones
        g = rng.standard_normal(nparam)
        with tf.device("/GPU:0"):
            tf_f, tf_bt, tf_c, tf_g = map(tf.constant, (f, bt, c, g))
            tf_off = tf.constant(off, tf.int32)
            tf_sz = tf.constant(sz, tf.int32)
        fn = dense_factor_a if nline == 1 else dense_factor_an
        with tf.GradientTape() as tape:
            tape.watch([tf_c, tf_g])
            a = fn(tf_f, tf_bt, tf_c, tf_g, tf_off, tf_sz, max_basis=max(sz))
            p = tf.reduce_sum(tf.math.real(a * tf.math.conj(a)), axis=-1)
            loss = -tf.reduce_sum(tf.math.log(p))
        dc, dg = tape.gradient(loss, [tf_c, tf_g])

        with tf.GradientTape() as ref_tape:
            ref_tape.watch([tf_c, tf_g])
            dens = []
            for u in range(nu):
                den = 1
                for l in range(nline):
                    o = off[u * nline + l]
                    s = sz[u * nline + l]
                    if s > 0:
                        den = den * tf.reduce_sum(
                            tf_bt[:, o : o + s]
                            * tf.cast(tf_g[None, o : o + s], tf_bt.dtype),
                            axis=1,
                        )
                dens.append(den)
            amp = 0
            for k in range(np):
                ak = 0
                for u in range(nu):
                    ak = ak + tf_f[:, k, u] * tf_c[u] / dens[u]
                amp = amp + tf.math.real(ak * tf.math.conj(ak))
            ref_loss = -tf.reduce_sum(tf.math.log(amp))
        ref_dc, ref_dg = ref_tape.gradient(ref_loss, [tf_c, tf_g])
        assert rel(dc, ref_dc) < 1e-10
        assert rel(dg, ref_dg) < 1e-10

    # two chains sharing one coefficient block (the n_u > n_chains case)
    check(2, [0, 0], [2, 2], 1, 2)
    # two lines, the second chain has only one
    check(2, [0, 2, 0, 0], [2, 2, 2, 0], 2, 4)
    check(2, [0, 2, 0, 2], [2, 2, 2, 2], 2, 4)
    # a single-line layout with an absent line: den must multiply by one
    check(2, [0, 2], [0, 2], 1, 4)
    # several lines with more than one projection
    check(3, [0, 2, 0, 2, 0, 4], [2, 2, 2, 2, 2, 2], 2, 6, np=2)
    # more chains than the block has threads
    check(300, [0] * 300, [2] * 300, 1, 2)
    # the register path spans four lane slots and the shared path takes over
    # above 128: 33 is the second slot, 128/129 the dispatch boundary
    check(33, [0] * 33, [2] * 33, 1, 2)
    check(128, [0] * 128, [2] * 128, 1, 2)
    check(129, [0] * 129, [2] * 129, 1, 2)
    check(2, [0, 2, 0, 0], [2, 2, 2, 0], 2, 4)


def test_dense_factor_op_shared_parameters():
    """Two LS terms share one chain's parameter block; gradients must sum."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
            "decay": {"B": [["R1", "X"]], "R1": ["P", "Q"]},
            "particle": {
                "$top": {"B": {"J": 2, "P": -1, "mass": 5.0}},
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.30},
                    "P": {"J": 0, "P": -1, "mass": 0.20},
                    "Q": {"J": 0, "P": -1, "mass": 0.20},
                },
                "R1": {"J": 1, "P": -1, "m0": 2.5, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.30, 0.20, 0.20])
    p4 = dict(zip(["X", "P", "Q"], ps.generate(100)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    assert len(v0) > 0
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    with tf.GradientTape() as t0:
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


def test_dense_factor_op_projections():
    """The projection axis is the product of helicity dims, not the top spin."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        import pytest

        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
            "decay": {"B": [["R", "X"]], "R": ["A", "C"]},
            "particle": {
                "$top": {
                    "B": {
                        "J": 1,
                        "P": -1,
                        "mass": 5.0,
                        "spins": [-1, 1],
                    }
                },
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.20},
                    "A": {"J": 1, "P": -1, "mass": 0.30},
                    "C": {"J": 1, "P": -1, "mass": 0.30},
                },
                "R": {"J": 1, "P": -1, "m0": 2.0, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.20, 0.30, 0.30])
    p4 = dict(zip(["X", "A", "C"], ps.generate(100)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    # 2 top spins x 3 x 3 helicities; the old op refused anything above 8
    assert d0["F"].shape[1] == 18
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    assert len(v0) > 0
    assert len(v0) == len(v1)
    with tf.GradientTape() as t0:
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model", ["dense_factor", "dense_factor_op"])
def test_dense_factor_drop_decay_particle(tmp_path, model):
    """The angle tree can be dropped, eagerly and through a lazy cache."""
    import tensorflow as tf

    if model != "dense_factor":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model, drop, lazy=False):
        pre = {"force_cpu": True, "drop_decay_particle": drop}
        data = {
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": model,
        }
        if model != "default":
            data["preprocessor"] = {"dense_factor": pre}
        if lazy:
            data["lazy_call"] = True
            data["cached_lazy_call"] = str(tmp_path) + "/"
        return {
            "data": data,
            "decay": {"B": [["R", "X"]], "R": ["A", "C"]},
            "particle": {
                "$top": {"B": {"J": 1, "P": -1, "mass": 5.0}},
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.20},
                    "A": {"J": 1, "P": -1, "mass": 0.30},
                    "C": {"J": 1, "P": -1, "mass": 0.30},
                },
                "R": {"J": 1, "P": -1, "m0": 2.0, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.20, 0.30, 0.30])
    p4 = dict(zip(["X", "A", "C"], ps.generate(200)))
    pmc = dict(zip(["X", "A", "C"], ps.generate(400)))

    def evaluate(c, d):
        phsp = c.data.cal_angle(pmc, data_type="phsp")
        if not isinstance(d, dict):
            phsp.as_dataset(128)
        fcn = c.get_fcn(all_data=([d], [phsp], None, None), batch=64)
        nll, grad = fcn.nll_grad()
        return float(nll), [np.asarray(g) for g in grad]

    # reference: the dense path, angle tree kept, eager
    c_ref = ConfigLoader(cfg("dense_factor", drop=False))
    params = c_ref.get_params()
    c_ref.set_params(params)
    l_ref, g_ref = evaluate(c_ref, c_ref.data.cal_angle(p4))

    for lazy in (False, True):
        c = ConfigLoader(cfg(model, drop=True, lazy=lazy))
        c.set_params(params)
        d = c.data.cal_angle(p4)
        if lazy:
            c.data.set_lazy_call(d, "s0")
            d.as_dataset(64)
            batches = list(d)
            assert len(batches) > 1
            for b in batches:
                assert "decay" not in b and "particle" not in b
                assert "F" in b and "B" in b
            assert list(tmp_path.glob("s0_64*"))
        else:
            assert "decay" not in d and "particle" not in d
            assert "F" in d and "B" in d
        nll, grad = evaluate(c, d)
        assert np.allclose(nll, l_ref, rtol=1e-10)
        assert len(grad) == len(g_ref)
        for x, y in zip(grad, g_ref):
            assert np.allclose(x, y, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model", ["dense_factor", "dense_factor_op"])
def test_dense_factor_store_complex64(tmp_path, model):
    """F and B can be stored in complex64; pdf casts them back."""
    import tensorflow as tf

    if model != "dense_factor":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model, low, lazy=False):
        pre = {"force_cpu": True, "store_complex64": low}
        data = {
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": model,
            "preprocessor": {"dense_factor": pre},
        }
        if lazy:
            data["lazy_call"] = True
            data["cached_lazy_call"] = str(tmp_path) + "/"
        return {
            "data": data,
            "decay": {"B": [["R", "X"]], "R": ["A", "C"]},
            "particle": {
                "$top": {"B": {"J": 1, "P": -1, "mass": 5.0}},
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.20},
                    "A": {"J": 1, "P": -1, "mass": 0.30},
                    "C": {"J": 1, "P": -1, "mass": 0.30},
                },
                "R": {"J": 1, "P": -1, "m0": 2.0, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.20, 0.30, 0.30])
    p4 = dict(zip(["X", "A", "C"], ps.generate(100)))

    c0 = ConfigLoader(cfg("dense_factor", False))
    params = c0.get_params()
    c0.set_params(params)
    d0 = c0.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    v0 = a0.vm.trainable_variables
    with tf.GradientTape() as t0:
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)

    c1 = ConfigLoader(cfg(model, True))
    c1.set_params(params)
    d1 = c1.data.cal_angle(p4)
    assert d1["F"].dtype == tf.complex64
    assert d1["B"].dtype == tf.complex64
    a1 = c1.get_amplitude()
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t1:
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    # nothing leaks the storage dtype into the result or the gradients
    assert l1.dtype == tf.float64
    assert all(g.dtype == v.dtype for g, v in zip(g1, v1))
    assert np.allclose(float(l1), float(l0), rtol=1e-5)
    for x, y in zip(g1, g0):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-5, atol=1e-5)

    # the cache keeps the compact dtype, which is the point of the option
    c2 = ConfigLoader(cfg(model, True, lazy=True))
    c2.set_params(params)
    d2 = c2.data.cal_angle(p4)
    c2.data.set_lazy_call(d2, "s0")
    d2.as_dataset(64)
    for b in d2:
        assert b["F"].dtype == tf.complex64
        assert b["B"].dtype == tf.complex64
    assert list(tmp_path.glob("s0_64*"))


def test_dense_factor_op_polarization():
    """The op returns A, so a polarized top uses sum_with_polarization."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
            "decay": {"B": [["R", "X"]], "R": ["A", "C"]},
            "particle": {
                "$top": {
                    "B": {
                        "J": 1,
                        "P": -1,
                        "mass": 5.0,
                        "polarization": "vector",
                    }
                },
                "$finals": {
                    "X": {"J": 0, "P": -1, "mass": 0.20},
                    "A": {"J": 1, "P": -1, "mass": 0.30},
                    "C": {"J": 1, "P": -1, "mass": 0.30},
                },
                "R": {"J": 1, "P": -1, "m0": 2.0, "g0": 0.10, "model": "BWR"},
            },
        }

    ps = PhaseSpaceGenerator(5.0, [0.20, 0.30, 0.30])
    p4 = dict(zip(["X", "A", "C"], ps.generate(100)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert a0.decay_group.polarization == "vector"
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t0:
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    assert len(g0) == len(g1)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model", ["dense_factor", "dense_factor_op"])
@pytest.mark.parametrize("spin", [0, 1])
def test_dense_factor_identical_particles(model, spin):
    """Two identical finals: the swapped amplitude is summed, as upstream.

    ``spin = 1`` makes the particle-axis transpose a real permutation and
    gives a non-trivial ``get_swap_factor``.
    """
    import tensorflow as tf

    if model != "dense_factor":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(amp_model):
        data = {
            "dat_order": ["B", "C", "D"],
            "center_mass": True,
            "identical_particles": [["C", "D"]],
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": amp_model,
        }
        if amp_model != "default":
            data["preprocessor"] = {"dense_factor": {}}
        return {
            "data": data,
            "decay": {"A": ["BC", "D"], "BC": ["B", "C"]},
            "particle": {
                "$top": "A",
                "$finals": ["B", "C", "D"],
                "A": {"J": 0.5, "P": -1, "mass": 3.0},
                "B": {"J": 0.5, "P": -1, "mass": 0.1},
                "C": {"J": spin, "P": -1, "mass": 0.1},
                "D": {"J": spin, "P": -1, "mass": 0.1},
                "BC": {
                    "J": 0.5,
                    "P": -1,
                    "mass": 2.0,
                    "width": 0.5,
                    "model": "BWR",
                },
            },
        }

    ps = PhaseSpaceGenerator(3.0, [0.1, 0.1, 0.1])
    p4 = dict(zip(["B", "C", "D"], ps.generate(50)))
    c0 = ConfigLoader(cfg("default"))
    c1 = ConfigLoader(cfg(model))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert len(d0["id_swap"]) == 1
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    v0 = a0.vm.variables
    v1 = a1.vm.variables
    if isinstance(v0, dict):
        v0, v1 = list(v0.values()), list(v1.values())
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    # the fused model differentiates trainable variables only (coeff_on_cpu),
    # so compare the pairs that carry a gradient on both sides
    pairs = [(x, y) for x, y in zip(g0, g1) if x is not None and y is not None]
    assert len(pairs) > 0
    for x, y in pairs:
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("model", ["dense_factor", "dense_factor_op"])
@pytest.mark.parametrize("spin", [0, 1])
def test_dense_factor_cp_particles(model, spin):
    """A CP pair: the conjugated kinematics are added with the C parities.

    ``spin = 1`` makes the CP particle permutation a real transpose and
    keeps the ``frac`` parity factor non-trivial.
    """
    import tensorflow as tf
    import yaml

    if model != "dense_factor":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)
    if spin:
        finals = base["particle"]["$finals"]
        finals["pip"] = {**finals["pip"], "J": spin}
        finals["pim"] = {**finals["pim"], "J": spin}

    def cfg(amp_model):
        data = {
            **base["data"],
            "format": "simple",
            "r_boost": False,
            "amp_model": amp_model,
        }
        if amp_model != "default":
            data["preprocessor"] = {"dense_factor": {}}
        return {**base, "data": data}

    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(30)))
    c0 = ConfigLoader(cfg("default"))
    c1 = ConfigLoader(cfg(model))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert "cp_swap" in d0
    assert a0.decay_group.cp_particles

    # the op model differentiates trainable_variables only (coeff_on_cpu),
    # so the comparison is over those
    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    assert len(v0) == len(v1) > 0
    if isinstance(v0, dict):
        v0, v1 = list(v0.values()), list(v1.values())
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


def test_dense_factor_merge():
    """Identical basis blocks are stored once and expanded by the index."""
    import tensorflow as tf

    waves = 3
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
    }
    decay = {"X": [[f"R{i}", "c"] for i in range(waves)]}
    for i in range(waves):
        decay[f"R{i}"] = ["a", "b"]
        part[f"R{i}"] = {
            "J": 1,
            "P": -1,
            "mass": 2.0 + 0.01 * i,
            "width": 0.1,
            "model": "BWR",
        }

    def cfg(model):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {model: {"force_cpu": True}},
            },
            "decay": decay,
            "particle": part,
        }

    ps = PhaseSpaceGenerator(5.0, [0.2, 0.2, 0.2])
    p4 = dict(zip(["a", "b", "c"], ps.generate(200)))
    c0 = ConfigLoader(cfg("dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_merge"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    # one wave-equivalent block, one set of den columns
    assert d0["F"].shape[2] == waves
    assert d1["F"].shape[2] == 1
    assert d0["B"].shape[1] == 3 * waves
    assert d1["B"].shape[1] == 3
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-12, atol=1e-14)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-12)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-10, atol=1e-12)


def test_dense_factor_op_merge():
    """The fused contraction works on the merged basis, expanded by index."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    waves = 3
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
    }
    decay = {"X": [[f"R{i}", "c"] for i in range(waves)]}
    for i in range(waves):
        decay[f"R{i}"] = ["a", "b"]
        part[f"R{i}"] = {
            "J": 1,
            "P": -1,
            "mass": 2.0 + 0.01 * i,
            "width": 0.1,
            "model": "BWR",
        }

    def cfg(model, pre):
        return {
            "data": {
                "format": "simple",
                "random_z": False,
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {pre: {"force_cpu": True}},
            },
            "decay": decay,
            "particle": part,
        }

    ps = PhaseSpaceGenerator(5.0, [0.2, 0.2, 0.2])
    p4 = dict(zip(["a", "b", "c"], ps.generate(200)))
    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    c1 = ConfigLoader(cfg("dense_factor_op_merge", "dense_factor_merge"))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert d1["F"].shape[2] == 1 and d1["B"].shape[1] == 3
    # the reduced basis goes to the kernel with its index; nothing expands
    assert a1.op_maps() == (a1.merge["map_u"], a1.merge["map_i"])
    from tf_pwa.experimental.dense_factor import DenseFactorModel

    assert type(a1).expand_block is DenseFactorModel.expand_block
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    g0 = t0.gradient(l0, v0)
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g1 = t1.gradient(l1, v1)
    assert np.allclose(float(l0), float(l1), rtol=1e-10)
    for x, y in zip(g0, g1):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


def test_dense_factor_merge_lazy(tmp_path):
    """The merged basis works under lazy_call, with a reduced cache."""
    import tensorflow as tf

    waves = 3
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
    }
    decay = {"X": [[f"R{i}", "c"] for i in range(waves)]}
    for i in range(waves):
        decay[f"R{i}"] = ["a", "b"]
        part[f"R{i}"] = {
            "J": 1,
            "P": -1,
            "mass": 2.0 + 0.1 * i,
            "width": 0.1,
            "model": "BWR",
        }

    def cfg(model, lazy=False):
        data = {
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": model,
        }
        pre = (
            "dense_factor" if model == "dense_factor" else "dense_factor_merge"
        )
        data["preprocessor"] = {pre: {"force_cpu": True}}
        if lazy:
            data["lazy_call"] = True
            data["cached_lazy_call"] = str(tmp_path) + "/"
        return {"data": data, "decay": decay, "particle": part}

    ps = PhaseSpaceGenerator(5.0, [0.2, 0.2, 0.2])
    p4 = dict(zip(["a", "b", "c"], ps.generate(200)))
    c0 = ConfigLoader(cfg("dense_factor"))
    a0 = c0.get_amplitude()
    ref = a0(c0.data.cal_angle(p4)).numpy()

    c1 = ConfigLoader(cfg("dense_factor_merge", lazy=True))
    d1 = c1.data.cal_angle(p4)
    c1.data.set_lazy_call(d1, "s0")
    d1.as_dataset(64)
    batches = list(d1)
    assert len(batches) > 1
    for b in batches:
        assert b["F"].shape[2] == 1 and b["B"].shape[1] == 3

    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())
    got = np.concatenate([a1(b).numpy() for b in batches])
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)
    assert list(tmp_path.glob("s0_64*"))
    saved = dict(tf.train.list_variables(str(tmp_path / "s0_64")))
    assert any(
        1 in shape or list(shape) == [64, 1] for shape in saved.values()
    )


def test_dense_factor_merge_unsupported_model_raises():
    """A model with no rational basis must fail when the config is loaded."""
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
        # no width: get_term_coeff raises at construction
        "R": {"J": 1, "P": -1, "mass": 2.0},
    }
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_merge",
            "preprocessor": {"dense_factor_merge": {}},
        },
        "decay": {"X": [["R", "c"]], "R": ["a", "b"]},
        "particle": part,
    }
    # the term model names the particle in its own message, and the merge
    # preprocessor builds the amplitude at construction, so this raises here
    # instead of leaving a husk that fails at the first batch
    with pytest.raises(NotImplementedError, match="no width"):
        ConfigLoader(cfg)


def test_merge_axis_groups_tolerance():
    """The grouping tolerates cancellation-limited columns, not tiny ones."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor import _axis_groups

    X = np.zeros((60, 4), np.complex128)
    X[:, 0] = 1e-20  # tiny but distinct from zero
    X[:, 1] = 0.0
    X[:, 2] = 25.0
    X[::7, 2] = (
        25.0 - 1e-9
    )  # a cancellation-limited difference, 4e-11 relative
    X[:, 3] = 25.0

    groups, reps = _axis_groups(tf.constant(X), 1, rtol=1e-12)
    assert groups[0] != groups[1]  # tiny stays apart from zero
    assert groups[2] != groups[3]  # 1e-12 is too tight for 4e-11
    groups, reps = _axis_groups(tf.constant(X), 1)  # the loose default
    assert groups[2] == groups[3]
    groups, reps = _axis_groups(tf.constant(X), 1, rtol=1e-10)
    assert groups[2] == groups[3]


def test_chain_layout_pads_absent_lines():
    """The layout holds one ``(off, sz)`` pair per line slot, not per line."""
    from tf_pwa.experimental.dense_factor_op import chain_layout

    structure = {
        "u_chain": [0, 0, 1],
        "den_sizes": [[2, 1], [3]],
        "param_offset": [[0, 2], [3]],
        "max_internal_lines": 2,
    }
    off, sz = chain_layout(structure)
    # chain 0 appears twice and has both lines; chain 1 has one line and is
    # padded, so its second slot is (0, 0)
    assert off.numpy().tolist() == [0, 2, 0, 2, 3, 0]
    assert sz.numpy().tolist() == [2, 1, 2, 1, 3, 0]
    assert len(off) == len(structure["u_chain"]) * 2


def test_dense_factor_op_retrace():
    """A retrace and a later eager call must not reuse graph constants."""
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tensorflow as tf

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    cfg = _config(
        {"J": 1, "P": -1, "mass": 1.5, "width": 0.1, "model": "BWR"},
        top={"J": 1, "P": -1, "mass": 4.6},
    )
    cfg = {**cfg, "data": {**cfg["data"], "amp_model": "dense_factor_op"}}
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()

    @tf.function
    def call(data):
        return amp(data)

    sizes = []
    for n in (10, 4):  # the second batch shape forces a retrace
        (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=n)
        data = c.data.cal_angle({"B": p_b, "D": p_d, "C": p_c})
        sizes.append(call(data).numpy().shape[0])

    # eager after tracing used to be the same failure
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=3)
    data = c.data.cal_angle({"B": p_b, "D": p_d, "C": p_c})
    sizes.append(amp(data).numpy().shape[0])
    assert sizes == [10, 4, 3]


def _one_wave():
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
        "R": {"J": 1, "P": -1, "mass": 2.0, "width": 0.1, "model": "BWR"},
    }
    return {"X": [["R", "c"]], "R": ["a", "b"]}, part


def test_dense_preprocessor_with_wrapper_model():
    """A wrapper amp_model shares the group but builds F/B in its own pdf."""
    decay, part = _one_wave()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": {"dense_factor": {}, "default": {}},
            "preprocessor": "dense_factor",
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    assert type(amp).__name__ == "ProdPDF"

    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    # the group builds F/B whatever the wrapper is; the dense component
    # reads them, the other components ignore them
    assert "F" in data
    assert amp(data).numpy().shape == (10,)


def test_dense_merge_preprocessor_with_default_model():
    """The default model gets the merged F/B built from the group."""
    decay, part = _one_wave()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "default",
            # forcing the init-time preparation exercises its guard
            "preprocessor": {"dense_factor_merge": {"merge_prepare": True}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    assert "F" in data and "B" in data
    assert isinstance(amp.merge, dict)
    assert amp(data).numpy().shape == (10,)


def test_dense_factor_merge_disabled_falls_back():
    """A sample that cannot be built disables merging, not the model."""
    import tf_pwa.experimental.dense_factor as df

    decay, part = _one_wave()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_merge",
            "preprocessor": {"dense_factor_merge": {}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    real = df.PhaseSpaceGenerator

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no sample")

    df.PhaseSpaceGenerator = _Boom
    try:
        with pytest.warns(UserWarning, match="disabled"):
            data = c.data.cal_angle(p4)
    finally:
        df.PhaseSpaceGenerator = real

    assert amp.merge is False
    # the blocks are stored at full width and expand_block passes them on
    assert amp(data).numpy().shape == (10,)


def test_dense_preprocessor_wrapper_keeps_angles():
    """drop_decay_particle must not strip data a wrapper still reads."""
    decay, part = _one_wave()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": {"dense_factor": {}, "default": {}},
            "preprocessor": {"dense_factor": {"drop_decay_particle": True}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    assert "particle" in data and "decay" in data
    assert amp(data).numpy().shape == (10,)


def _identical_waves(n=3):
    decay = {"X": [[f"R{i}", "c"] for i in range(n)]}
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 1, "P": -1, "mass": 5.0},
        "a": {"J": 0, "P": -1, "mass": 0.2},
        "b": {"J": 0, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
    }
    for i in range(n):
        decay[f"R{i}"] = ["a", "b"]
        part[f"R{i}"] = {
            "J": 1,
            "P": -1,
            "mass": 2.0 + 0.01 * i,
            "width": 0.1,
            "model": "BWR",
        }
    return decay, part


def test_dense_merge_preprocessor_with_plain_model_fails():
    """The reduced widths reach a model that cannot expand them: mismatch."""
    import tensorflow as tf

    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor",
            "preprocessor": {"dense_factor_merge": {}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    assert isinstance(amp.merge, dict)
    assert int(data["F"].shape[2]) == 1
    with pytest.raises(tf.errors.InvalidArgumentError):
        amp(data)


def test_dense_merge_preprocessor_wrapper_fails():
    """A wrapper cannot expand the reduced blocks either."""
    import tensorflow as tf

    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": {"dense_factor_merge": {}, "default": {}},
            "preprocessor": {"dense_factor_merge": {}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    with pytest.raises(tf.errors.InvalidArgumentError):
        amp(data)
