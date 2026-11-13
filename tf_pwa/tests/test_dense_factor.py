import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import tf_pwa.amp  # noqa: F401  (register models)
import tf_pwa.amp.interpolation  # noqa: F401
import tf_pwa.experimental.dense_factor  # noqa: F401  (install dense plugin)
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.experimental.dense_factor import _TERMS
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


def test_dense_factor_backend_standalone_import():
    """The registry works from a fresh interpreter, whatever the import order."""
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    code = (
        "import tf_pwa.experimental.dense_factor.backend as b\n"
        "print(b.__file__)\n"
        "print(b.pick_backend(None, ('cuda', 'tf')))\n"
        "print(b.pick_backend('index', ('missing', 'tf')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
    )
    lines = out.stdout.split()
    # the child must import this tree, not an installed copy
    assert lines[0].startswith(root) and "dense_factor/backend.py" in lines[0]
    assert lines[1:] == ["tf", "tf"]


def test_dense_factor_backend_missing_method(monkeypatch):
    """A reference table without the method fails with a pointed error.

    Importing a submodule runs the package ``__init__``, which registers the
    TF backend, so this state is only reachable through a mistake in that
    wiring; the error names the module to import rather than a bare
    ``KeyError``.
    """
    from tf_pwa.experimental.dense_factor import backend as b

    monkeypatch.delitem(b.BACKENDS["tf"]["functions"], "index")
    with pytest.raises(RuntimeError, match="dense_factor"):
        b.pick_backend("index", ("missing",))


def test_dense_factor_op_plain_model_uses_fused_op(monkeypatch):
    """``amp_model: dense_factor_op`` contracts through the fused op.

    A plain basis has ``record = None``, so this is the one method that
    dispatches on ``None``.  The wrong answer here is *silent*: without the
    entry the model still produces the same numbers through the dense
    reference, only ~4x slower, so no accuracy test can see it.  The spy is
    the regression test; the dense model on the same data must not call the
    op at all.
    """
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import yaml

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    def cfg(model):
        return {
            **base,
            "data": {
                **base["data"],
                "format": "simple",
                "r_boost": False,
                "amp_model": model,
                "preprocessor": {"dense_factor": {"force_cpu": True}},
            },
        }

    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))

    c = ConfigLoader(cfg("dense_factor_op"))
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()

    calls = []
    for name in ("dense_factor_a", "dense_factor_an"):
        original = getattr(op_mod, name)

        def spy(*args, _orig=original, _name=name, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(op_mod, name, spy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_op = amp(d).numpy()
    assert calls, "the plain op model did not use the fused op"
    # the spy records before delegating, so a raising op would still fill
    # `calls`; forbid the fallback and pin the variant this layout picks
    assert not [w for w in caught if "op failed" in str(w.message)]
    assert set(calls) == {"dense_factor_a"}

    c0 = ConfigLoader(cfg("dense_factor"))
    a0 = c0.get_amplitude()
    a0.set_params(amp.get_params())
    calls.clear()
    x0 = a0(c0.data.cal_angle(p4)).numpy()
    assert not calls, "the dense model must stay on the reference backend"
    assert np.allclose(x_op, x0, rtol=1e-8, atol=1e-10)


def test_dense_factor_op_model_without_layout_uses_reference(monkeypatch):
    """A fused model with no layout declines and runs the reference.

    Selection is uniform now: every backend gets the model-owned ``layout``
    dict, so the fused entry is the one that notices ``off is None``, reports
    once and returns the TF reference for that call instead of failing.
    """
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import yaml

    import tf_pwa.experimental.dense_factor as df
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    cfg = {
        **base,
        "data": {
            **base["data"],
            "format": "simple",
            "r_boost": False,
            "amp_model": "dense_factor_op",
            "preprocessor": {"dense_factor": {"force_cpu": True}},
        },
    }
    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(100)))
    c = ConfigLoader(cfg)
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    assert amp._layout is not None

    fused = amp(d).numpy()

    # the fused backend is still selected, and declines inside the entry
    monkeypatch.setattr(
        amp, "get_layout", lambda: dict(amp._layout, off=None, sz=None)
    )
    snapshot = set(df._OP_FAILURES)
    df._OP_FAILURES.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference = amp(d).numpy()
    finally:
        df._OP_FAILURES.clear()
        df._OP_FAILURES.update(snapshot)

    said = [
        str(w.message) for w in caught if "no fused layout" in str(w.message)
    ]
    assert len(said) == 1, said
    assert "using the reference contraction" in said[0]
    assert np.allclose(reference, fused, rtol=1e-8, atol=1e-10)


def test_dense_factor_layout_keys():
    """``get_layout`` is the model-owned dict every backend reads.

    The plain dense model has no fused layout (``off``/``sz`` are ``None``);
    the op model fills those two keys from the op package.  Both expose the
    same key set and cache the dict rather than rebuilding it per call.
    """
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor import LAYOUT_KEYS

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
    cfg = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }
    c = ConfigLoader(cfg)
    amp = c.get_amplitude()
    layout = amp.get_layout()
    assert tuple(sorted(layout)) == tuple(sorted(LAYOUT_KEYS))
    assert amp.get_layout() is layout
    assert layout["off"] is None and layout["sz"] is None
    assert isinstance(layout["n_line"], int)

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    cfg2 = {
        **cfg,
        "data": {
            **cfg["data"],
            "amp_model": "dense_factor_op",
            "preprocessor": {"dense_factor": {"force_cpu": True}},
        },
    }
    amp2 = ConfigLoader(cfg2).get_amplitude()
    layout2 = amp2.get_layout()
    assert tuple(sorted(layout2)) == tuple(sorted(LAYOUT_KEYS))
    assert amp2.get_layout() is layout2
    assert isinstance(layout2["off"], tf.Tensor)
    assert isinstance(layout2["sz"], tf.Tensor)
    assert layout2["off"].dtype.is_integer
    assert layout2["sz"].dtype.is_integer
    assert isinstance(layout2["n_line"], int)
    assert isinstance(layout2["max_basis"], int)


def test_dense_factor_strategy_has_no_model(monkeypatch):
    """The model is not threaded through the strategy any more.

    The layout travels as the dict argument; a strategy that carried the
    model would be a second, per-call owner of the same data.
    """
    from tf_pwa.experimental.dense_factor import _Compression, backend

    assert not hasattr(_Compression, "model")

    seen = []
    original = backend.get_op("tf", None)

    def spy(strategy, *args, **kwargs):
        seen.append(strategy)
        return original(strategy, *args, **kwargs)

    monkeypatch.setitem(backend.BACKENDS["tf"]["functions"], None, spy)

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
    cfg = {
        **cfg,
        "data": {
            **cfg["data"],
            "preprocessor": "dense_factor",
            "amp_model": "dense_factor",
        },
    }
    c = ConfigLoader(cfg)
    (p_b, p_d), p_c = generate_phsp(4.6, ((2.0, (1.0, 0.4)), 0.5), N=10)
    amp = c.get_amplitude()
    amp(c.data.cal_angle({"B": p_b, "D": p_d, "C": p_c}))

    assert seen
    for strategy in seen:
        assert getattr(strategy, "model", None) is None


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
def test_dense_factor_store_dtype(tmp_path, model):
    """F and B can be stored in complex64; pdf casts them back."""
    import tensorflow as tf

    if model != "dense_factor":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    def cfg(model, low, lazy=False):
        pre = {
            "force_cpu": True,
            "store_dtype": "complex64" if low else "complex128",
        }
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


def test_dense_factor_store_complex32():
    """F/B can be cached as fp16 halves; pdf reads them back as complex."""
    import tensorflow as tf

    cfg = {
        "data": {
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": "dense_factor",
            "preprocessor": {
                "dense_factor": {"force_cpu": True, "store_dtype": "complex32"}
            },
        },
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
    exact = {
        **cfg,
        "data": {**cfg["data"], "preprocessor": {"dense_factor": {}}},
    }
    ps = PhaseSpaceGenerator(5.0, [0.20, 0.30, 0.30])
    p4 = dict(zip(["X", "A", "C"], ps.generate(100)))

    c0 = ConfigLoader(exact)
    d0 = c0.data.cal_angle(p4)
    a0 = c0.get_amplitude()
    c1 = ConfigLoader(cfg)
    c1.set_params(c0.get_params())
    d1 = c1.data.cal_angle(p4)
    a1 = c1.get_amplitude()

    assert d1["F"].dtype == tf.float16 and d1["F"].shape[-1] == 2
    assert d1["B"].dtype == tf.float16 and d1["B"].shape[-1] == 2
    assert np.allclose(a1(d1).numpy(), a0(d0).numpy(), rtol=1e-2, atol=1e-3)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g0 = t0.gradient(l0, v0)
    g1 = t1.gradient(l1, v1)
    assert all(g.dtype == v.dtype for g, v in zip(g1, v1))
    for x, y in zip(g1, g0):
        assert np.allclose(x.numpy(), y.numpy(), rtol=2e-2, atol=2e-3)


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
    assert a1.merge["method"] == "index"
    assert a1.merge["map_u"].shape[0] == a1.merge["n_u"]
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
    # the record is keyed, not an attribute of whatever amp is configured
    from tf_pwa.experimental.dense_factor import get_merge_record

    assert isinstance(get_merge_record(c.data.preprocessor.merge_id), dict)
    assert not hasattr(amp, "merge")
    assert amp(data).numpy().shape == (10,)


@pytest.mark.parametrize(
    "model", ["dense_factor_merge", "dense_factor_op_merge"]
)
def test_dense_factor_merge_svd(model):
    """method: svd stores low-rank coefficients and rebuilds the basis."""
    if model == "dense_factor_op_merge":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    import tensorflow as tf

    decay, part = _identical_waves()

    def cfg(amp_model, pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": amp_model,
                "preprocessor": pre,
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    c1 = ConfigLoader(cfg(model, {"dense_factor_merge": {"method": "svd"}}))
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)

    assert a1.merge["method"] == "svd"
    assert d1["F"].shape == (100, a1.merge["rank_f"])
    assert d1["B"].shape[1] == a1.merge["rank_b"]
    assert np.allclose(a1(d1).numpy(), a0(d0).numpy(), rtol=1e-8, atol=1e-10)

    v0 = a0.vm.trainable_variables
    v1 = a1.vm.trainable_variables
    with tf.GradientTape() as t0:
        t0.watch(v0)
        l0 = -tf.reduce_sum(tf.math.log(a0(d0)))
    with tf.GradientTape() as t1:
        t1.watch(v1)
        l1 = -tf.reduce_sum(tf.math.log(a1(d1)))
    g0 = t0.gradient(l0, v0)
    g1 = t1.gradient(l1, v1)
    for x, y in zip(g1, g0):
        assert np.allclose(x.numpy(), y.numpy(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "model", ["dense_factor_merge", "dense_factor_op_merge"]
)
def test_dense_factor_merge_disabled_falls_back(model):
    """A sample that cannot be built disables compression, not the model."""
    # the preprocessor resolves the generator in its own module, which is
    # where the preprocessor lives now
    import tf_pwa.experimental.dense_factor.preprocess as df_pre

    if model == "dense_factor_op_merge":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _one_wave()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": model,
            # preparation at init is where the sample is generated; eager
            # data would otherwise be fitted from its first batch
            "preprocessor": {"dense_factor_merge": {"merge_prepare": True}},
        },
        "decay": decay,
        "particle": part,
    }

    real = df_pre.PhaseSpaceGenerator

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no sample")

    df_pre.PhaseSpaceGenerator = _Boom
    try:
        with pytest.warns(UserWarning, match="disabled"):
            c = ConfigLoader(cfg)
            amp = c.get_amplitude()
    finally:
        df_pre.PhaseSpaceGenerator = real

    assert amp.merge is False
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=10)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    # the blocks are stored at full width and expand_block passes them on
    F, B = data["F"], data["B"]
    assert int(F.shape[2]) == amp.decay_group.get_dense_structure()["n_u"]
    assert amp.expand_block(F, B)[0] is F
    assert amp(data).numpy().shape == (10,)


def test_dense_factor_no_merge_side_effects():
    """The plain preprocessor leaves no merge state on the model or data."""
    decay, part = _one_wave()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=20)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    for model in ("dense_factor", "dense_factor_op"):
        if model == "dense_factor_op":
            from tf_pwa.experimental.dense_factor_op import is_available

            if not is_available():
                continue
            import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

        cfg = {
            "data": {
                "format": "simple",
                "amp_model": model,
                "preprocessor": "dense_factor",
            },
            "decay": decay,
            "particle": part,
        }
        c = ConfigLoader(cfg)
        amp = c.get_amplitude()
        pre = c.data.preprocessor
        data = c.data.cal_angle(p4)

        # nothing merge-shaped on the model, the preprocessor or the basis
        assert not hasattr(amp, "merge") and not hasattr(amp, "compress")
        assert [k for k in vars(pre) if "merge" in k or "compress" in k] == []
        assert [
            k for k in amp.decay_group.get_dense_structure() if "merge" in k
        ] == []

        # the hook is a pure identity, not a gather
        F, B = data["F"], data["B"]
        out = amp.expand_block(F, B)
        assert out[0] is F and out[1] is B


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
    # the record lives in the shared space, not on the model, and the plain
    # model has no hook to expand the reduced widths
    from tf_pwa.experimental.dense_factor import get_merge_record

    assert not hasattr(amp, "merge")
    assert isinstance(get_merge_record(c.data.preprocessor.merge_id), dict)
    assert int(data["F"].shape[2]) == 1
    with pytest.raises(tf.errors.InvalidArgumentError):
        amp(data)


def test_dense_merge_record_reaches_prod_pdf_submodel():
    """The preprocessor sees a ``ProdPDF``; the dense sub-model still finds it.

    ``self.amp`` is the composed PDF returned by ``get_amplitude``, not the
    model that evaluates the reduced blocks, so the record cannot be an
    attribute on that object.
    """
    decay, part = _identical_waves()

    def cfg(amp_model):
        return {
            "data": {
                "format": "simple",
                "amp_model": amp_model,
                "preprocessor": {"dense_factor_merge": {}},
            },
            "decay": decay,
            "particle": part,
        }

    c_prod = ConfigLoader(cfg({"dense_factor_merge": {}, "default": {}}))
    c_one = ConfigLoader(cfg("dense_factor_merge"))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=30)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d_prod = c_prod.data.cal_angle(p4)
    d_one = c_one.data.cal_angle(p4)

    a_prod = c_prod.get_amplitude()
    sub = a_prod.pdfs["dense_factor_merge"]
    assert sub is not a_prod
    assert isinstance(sub.merge, dict)

    a_one = c_one.get_amplitude()
    a_one.set_params(sub.get_params())
    assert np.allclose(
        sub(d_prod).numpy(), a_one(d_one).numpy(), rtol=1e-8, atol=1e-10
    )
    # the wrapper as a whole still evaluates
    assert a_prod(d_prod).numpy().shape == (30,)


def test_dense_factor_merge_key_is_shared_by_both_sides():
    """``merge_key`` is an input option of the preprocessor and the model."""
    from tf_pwa.config import get_config
    from tf_pwa.experimental.dense_factor import (
        MERGE_RECORDS,
        get_merge_record,
        set_merge_record,
    )

    key = "test_dense_factor_merge_key"
    decay, part = _identical_waves()

    def cfg():
        return {
            "data": {
                "format": "simple",
                "amp_model": {"dense_factor_merge": {"merge_key": key}},
                "preprocessor": {"dense_factor_merge": {"merge_key": key}},
            },
            "decay": decay,
            "particle": part,
        }

    c = ConfigLoader(cfg())
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=20)
    data = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    amp = c.get_amplitude()
    assert c.data.preprocessor.merge_id == key
    assert amp.merge_id == key
    assert amp.merge is get_merge_record(key)
    assert amp(data).numpy().shape == (20,)

    # a second config on the same key reads the same record object
    other = ConfigLoader(cfg()).get_amplitude()
    assert other.merge is amp.merge

    # reusing the key with another method is rejected instead of mixing them
    bad = cfg()
    bad["data"]["preprocessor"]["dense_factor_merge"]["method"] = "svd"
    with pytest.raises(ValueError, match="already holds"):
        ConfigLoader(bad).data.cal_angle({"a": p_a, "b": p_b, "c": p_c})

    set_merge_record(key, None)
    assert get_merge_record(key) is None
    assert key not in get_config(MERGE_RECORDS)


def test_dense_factor_merge_svd_rank_invalid():
    """svd_rank must keep at least one direction."""
    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_merge",
            "preprocessor": {
                "dense_factor_merge": {"method": "svd", "svd_rank": 0}
            },
        },
        "decay": decay,
        "particle": part,
    }
    with pytest.raises(ValueError, match="svd_rank"):
        ConfigLoader(cfg).get_amplitude()


@pytest.mark.parametrize(
    "store_dtype,method,tol",
    [
        ("complex32", "index", 5e-3),
        # real coefficients are float16 at complex32: half the width of the
        # packed complex form, with fp16's own precision
        ("complex32", "real_index", 2e-3),
        ("complex32", "svd", 5e-3),
        ("complex32", "real_svd", 5e-3),
        ("complex64", "real_index", 1e-4),
        ("complex64", "svd", 1e-4),
        # real coefficients reconstruct with a float32 matmul, whose GPU
        # (TF32) floor is ~5e-4; complex64 does not take that kernel.  The
        # measured spread over eight unseeded draws is 1.2e-3..2.3e-3, so the
        # old 2e-3 was only the TF32 floor plus luck; use the complex32
        # real_svd row's tolerance (same mechanism, same floor)
        ("complex64", "real_svd", 5e-3),
    ],
)
def test_dense_factor_merge_in_complex64_build(store_dtype, method, tol):
    """The stored dtype and the SVD must follow complex_dtype: complex64."""
    import tensorflow as tf

    from tf_pwa.config import get_config, set_config

    decay, part = _identical_waves()
    old = get_config("complex_dtype")
    set_config("complex_dtype", "complex64")
    try:

        def cfg(model, pre):
            return {
                "data": {
                    "format": "simple",
                    "amp_model": model,
                    "preprocessor": pre,
                },
                "decay": decay,
                "particle": part,
            }

        c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
        c1 = ConfigLoader(
            cfg(
                "dense_factor_merge",
                {
                    "dense_factor_merge": {
                        "method": method,
                        "store_dtype": store_dtype,
                    }
                },
            )
        )
        a0, a1 = c0.get_amplitude(), c1.get_amplitude()
        a1.set_params(a0.get_params())
        (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
        p4 = {"a": p_a, "b": p_b, "c": p_c}
        d0 = c0.data.cal_angle(p4)
        d1 = c1.data.cal_angle(p4)
        x0 = a0(d0).numpy()
        x1 = a1(d1).numpy()
        assert x1.dtype == x0.dtype
        assert np.allclose(x1, x0, rtol=tol, atol=10 * tol)
        assert tf.reduce_all(tf.math.is_finite(x1))
    finally:
        set_config("complex_dtype", old)


def test_dense_factor_op_rejects_reduced_basis():
    """The unmerged op must not read past a reduced basis."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import (
        dense_factor_a,
        is_available,
    )

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    f = tf.zeros([4, 2, 1], tf.complex128)  # one stored chain
    bt = tf.zeros([4, 3], tf.complex128)  # three stored coefficients
    c = tf.zeros([3], tf.complex128)  # three full chains
    g = tf.zeros([9], tf.float64)  # nine full coefficients
    off = tf.zeros([3], tf.int32)
    sz = tf.constant([3, 3, 3], tf.int32)
    with pytest.raises(tf.errors.InvalidArgumentError, match="unmerged op"):
        dense_factor_a(f, bt, c, g, off, sz, max_basis=3)


def test_dense_factor_build_batch():
    """build_batch chunks the construction and gives the same basis."""
    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor",
                "preprocessor": {"dense_factor": pre},
            },
            "decay": decay,
            "particle": part,
        }

    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    c0 = ConfigLoader(cfg({}))
    c1 = ConfigLoader(cfg({"build_batch": 16}))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    assert d0["F"].shape == d1["F"].shape
    assert np.allclose(d0["F"], d1["F"], rtol=1e-12, atol=0)
    assert np.allclose(d0["B"], d1["B"], rtol=1e-12, atol=0)

    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-10, atol=1e-12)

    # a chunk larger than the event count is a single build
    c2 = ConfigLoader(cfg({"build_batch": 1000}))
    d2 = c2.data.cal_angle(p4)
    assert np.allclose(d0["F"], d2["F"], rtol=1e-12, atol=0)

    with pytest.raises(ValueError, match="build_batch"):
        ConfigLoader(cfg({"build_batch": 0}))


@pytest.mark.parametrize(
    "model", ["dense_factor_merge", "dense_factor_op_merge"]
)
@pytest.mark.parametrize(
    "method,merge_sample,build_batch",
    [
        ("index", 16, 16),
        ("real_index", 16, 16),
        ("svd", 16, 16),
        ("real_svd", 16, 16),
    ],
)
def test_dense_factor_merge_build_batch(
    model, method, merge_sample, build_batch
):
    """Chunked build+compression equals the unchunked result."""
    if model == "dense_factor_op_merge":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": model,
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    base = {"method": method, "merge_sample": merge_sample}
    c0 = ConfigLoader(cfg(base))
    c1 = ConfigLoader(cfg({**base, "build_batch": build_batch}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)

    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)

    if method == "index":
        # the same maps: the stored tensors are identical
        assert np.allclose(d0["F"], d1["F"], rtol=1e-10, atol=0)
        assert np.allclose(d0["B"], d1["B"], rtol=1e-10, atol=0)
    elif model == "dense_factor_merge":  # svd and real_svd
        # a different fit gives a different (equivalent) basis: compare the
        # rebuilt one; the fused model keeps the reduced basis on purpose
        F0, B0 = a0.expand_block(d0["F"], d0["B"])
        F1, B1 = a1.expand_block(d1["F"], d1["B"])
        assert np.allclose(F0, F1, rtol=1e-8, atol=1e-10)
        assert np.allclose(B0, B1, rtol=1e-8, atol=1e-10)


def test_dense_factor_merge_build_batch_too_small():
    """A chunk smaller than the fit sample is rejected, for any method."""
    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_merge",
            "preprocessor": {
                "dense_factor_merge": {
                    "method": "index",
                    "merge_sample": 32,
                    "build_batch": 16,
                }
            },
        },
        "decay": decay,
        "particle": part,
    }
    with pytest.raises(ValueError, match="build_batch"):
        ConfigLoader(cfg).get_amplitude()


def test_dense_factor_build_batch_lazy_rejected():
    """lazy_call already batches; an inner batch would double it."""
    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor",
            "lazy_call": True,
            "preprocessor": {"dense_factor": {"build_batch": 8}},
        },
        "decay": decay,
        "particle": part,
    }
    with pytest.raises(ValueError, match="lazy_call"):
        ConfigLoader(cfg).get_amplitude()


def test_dense_factor_merge_build_batch_svd_narrow_chunk_raises():
    """A chunk that only spans its own rows would truncate the basis.

    ``rank(F)`` here is bounded by the chunk size, so later chunks would be
    projected onto a basis that never saw their directions: fail loudly
    instead of silently changing the amplitude.
    """
    decay, part = _identical_waves()
    # a spin-2 top widens the basis past the chunk size
    part["X"]["J"] = 2

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    base = {"method": "svd", "merge_sample": 8}
    c = ConfigLoader(cfg({**base, "build_batch": 8}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    with pytest.raises(ValueError, match="first chunk"):
        c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})


def test_dense_factor_merge_build_batch_svd_rank():
    """An explicit rank below the chunk size is honoured, not guarded."""
    decay, part = _identical_waves()
    part["X"]["J"] = 2

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    base = {"method": "svd", "merge_sample": 8, "svd_rank": 4}
    c = ConfigLoader(cfg({**base, "build_batch": 8}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    amp = c.get_amplitude()
    # a truncated rank is an approximation chosen on the first chunk; it is
    # stored as asked and the model contracts it, no guard involved
    assert amp.merge["rank_f"] == 4
    assert int(d["F"].shape[-1]) == 4
    assert amp(d).numpy().shape == (50,)


@pytest.mark.parametrize(
    "model", ["dense_factor_merge", "dense_factor_op_merge"]
)
def test_dense_factor_merge_real_svd(model):
    """`real_svd` stores a real basis and reconstructs the same amplitude."""
    if model == "dense_factor_op_merge":
        from tf_pwa.experimental.dense_factor_op import is_available

        if not is_available():
            pytest.skip("dense_factor_op cannot be built on this machine")
        import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": model,
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg({"method": "svd"}))
    c1 = ConfigLoader(cfg({"method": "real_svd"}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert a1.merge["method"] == "real_svd"
    assert not a1.merge["Vf"].dtype.is_complex
    assert not d1["F"].dtype.is_complex
    # the wide F keeps its complex rank; B may double it when its surviving
    # direction is genuinely complex (its coefficients then double too)
    assert a1.merge["rank_f"] == a0.merge["rank_f"]
    assert a1.merge["rank_b"] >= a0.merge["rank_b"]
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)

    if model == "dense_factor_merge":
        # the fused model keeps the reduced basis on purpose
        F0, B0 = a0.expand_block(d0["F"], d0["B"])
        F1, B1 = a1.expand_block(d1["F"], d1["B"])
        assert np.allclose(F0, F1, rtol=1e-8, atol=1e-10)
        assert np.allclose(B0, B1, rtol=1e-8, atol=1e-10)

    # a reduced rank truncates both fits identically
    rank = 2
    c2 = ConfigLoader(cfg({"method": "svd", "svd_rank": rank}))
    c3 = ConfigLoader(cfg({"method": "real_svd", "svd_rank": rank}))
    d2 = c2.data.cal_angle(p4)
    d3 = c3.data.cal_angle(p4)
    a2, a3 = c2.get_amplitude(), c3.get_amplitude()
    a3.set_params(a2.get_params())
    assert a2.merge["rank_f"] == a3.merge["rank_f"] == rank
    x2, x3 = a2(d2).numpy(), a3(d3).numpy()
    assert np.allclose(x2, x3, rtol=1e-8, atol=1e-10)


def test_real_svd_strategy_full_rank_generic():
    """A generic complex matrix (M M^H not real) is still exact at full rank.

    Layers only stay equivalent under truncation while M M^H is real; the
    full-rank reconstruction must not depend on that.
    """
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor import (
        _expand_svd,
        _RealSvdCompression,
    )

    rng = np.random.default_rng(0)
    F = tf.constant(
        (rng.normal(size=(40, 3, 4)) + 1j * rng.normal(size=(40, 3, 4)))
    )
    B = tf.constant(rng.normal(size=(40, 5)))
    comp = _RealSvdCompression(rtol=0.0)
    record = comp.prepare(F, B)
    Fc, Bc = comp.compress(record, F, B)
    assert not Fc.dtype.is_complex and not Bc.dtype.is_complex
    Fh, Bh = _expand_svd(record, Fc, Bc)
    assert np.allclose(Fh.numpy(), F.numpy(), rtol=1e-10, atol=1e-12)
    assert np.allclose(Bh.numpy(), B.numpy(), rtol=1e-10, atol=1e-12)


def test_dense_factor_build_batch_drop_decay_particle():
    """A chunked build drops each chunk's angles before the next slice."""
    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor",
                "preprocessor": {"dense_factor": pre},
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg({"drop_decay_particle": True}))
    c1 = ConfigLoader(cfg({"drop_decay_particle": True, "build_batch": 8}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)

    for d in (d0, d1):
        assert "particle" not in d and "decay" not in d
        assert "F" in d and "B" in d
    assert np.allclose(d0["F"], d1["F"], rtol=1e-12, atol=0)
    assert np.allclose(d0["B"], d1["B"], rtol=1e-12, atol=0)


def test_dense_factor_build_batch_streams_the_input(monkeypatch):
    """The slices are transformed one at a time, not after a whole pass."""
    import tf_pwa.experimental.dense_factor as df
    from tf_pwa.data import data_shape

    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor",
            "preprocessor": {"dense_factor": {"build_batch": 8}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=50)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    sizes = []
    original = df.BasePreProcessor.call

    def spy(self, x, **kwargs):
        sizes.append(int(data_shape(x)))
        return original(self, x, **kwargs)

    monkeypatch.setattr(df.BasePreProcessor, "call", spy)
    d1 = c.data.cal_angle(p4)
    assert sizes == [8] * 6 + [2]

    # and the chunked result is the one-shot build
    cfg["data"]["preprocessor"] = {"dense_factor": {}}
    c0 = ConfigLoader(cfg)
    d0 = c0.data.cal_angle(p4)
    assert np.allclose(d0["F"], d1["F"], rtol=1e-12, atol=0)
    assert np.allclose(d0["B"], d1["B"], rtol=1e-12, atol=0)


def test_dense_factor_merge_real_index():
    """real_index stores the index columns with the zero halves packed away."""
    decay, part = _identical_waves()
    # a spin-2 top gives a wide basis whose stored columns include pure-real
    # ones; the J=1 three-wave basis stores only mixed columns
    part["X"]["J"] = 2

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg({"method": "index"}))
    c1 = ConfigLoader(cfg({"method": "real_index"}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())

    assert a1.merge["method"] == "real_index"
    assert not d1["F"].dtype.is_complex
    # fewer bytes than the complex storage of the same columns
    assert np.prod(d1["F"].shape) * 8 < np.prod(d0["F"].shape) * 16
    assert not d1["B"].dtype.is_complex
    assert np.prod(d1["B"].shape) * 8 < np.prod(d0["B"].shape) * 16
    # and the same numbers
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)
    F0, B0 = a0.expand_block(d0["F"], d0["B"])
    F1, B1 = a1.expand_block(d1["F"], d1["B"])
    assert np.allclose(F0, F1, rtol=1e-10, atol=1e-12)
    assert np.allclose(B0, B1, rtol=1e-10, atol=1e-12)


def test_dense_factor_real_index_op_matches_rebuild(monkeypatch):
    """The split op reads the stored real block exactly like the rebuild.

    ``method: real_index`` in the op model goes through
    ``DenseFactorFwdRi``/``GradRi`` on ``d["F"]``/``d["B"]`` directly;
    forcing ``use_op = False`` makes the backend hand the record to the TF
    reference, which expands the stored block itself.  Both see the same data
    and the same record, so the amplitude and the parameter gradients must
    agree.
    """
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor import _RealIndexCompression
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()
    # a spin-2 top stores pure-real columns as well as mixed ones
    part["X"]["J"] = 2
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": "real_index"}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    amp = c.get_amplitude()
    assert amp.merge["method"] == "real_index"
    assert not d["F"].dtype.is_complex and not d["B"].dtype.is_complex

    import tf_pwa.experimental.dense_factor_op as op_mod

    used = []
    original = op_mod.dense_factor_ri
    monkeypatch.setattr(
        op_mod,
        "dense_factor_ri",
        lambda *a, **k: used.append(1) or original(*a, **k),
    )
    a_op = amp(d).numpy()
    assert used, "the split op was not exercised"

    def loss_and_grad():
        vs = amp.vm.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vs)
            loss = -tf.reduce_sum(tf.math.log(amp(d)))
        return [g.numpy() for g in tape.gradient(loss, vs)]

    g_op = loss_and_grad()

    monkeypatch.setattr(_RealIndexCompression, "use_op", False)
    a_rb = amp(d).numpy()
    g_rb = loss_and_grad()
    monkeypatch.undo()

    assert np.allclose(a_op, a_rb, rtol=1e-10, atol=1e-10)
    assert len(g_op) == len(g_rb) and g_op
    for x, y in zip(g_op, g_rb):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("method", ["svd", "real_svd"])
def test_dense_factor_merge_sample_matches_angle_options(method):
    """A sample fit must use the data's angle conventions.

    ``r_boost: false`` changes the kinematics; a sample built with the default
    (r_boost true) has a different basis and the SVD truncates real
    directions, so the amplitude would differ from the uncompressed model.
    """
    decay, part = _identical_waves()

    def cfg(model, pre):
        return {
            "data": {
                "format": "simple",
                "r_boost": False,
                "amp_model": model,
                "preprocessor": pre,
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    c1 = ConfigLoader(
        cfg(
            "dense_factor_merge",
            {
                "dense_factor_merge": {
                    "method": method,
                    "merge_prepare": True,
                }
            },
        )
    )
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)


def test_dense_factor_backend_registry():
    """Every recorded method has a TF reference, and lookup always falls back.

    ``_amplitude`` dispatches on the flat registry: the TF table must cover
    every method the preprocessor can record (so a fused backend can always
    decline), ``prefer`` must be honoured in order, and a backend this
    process never registered must land on TF rather than raising.
    """
    from tf_pwa.experimental.dense_factor import _COMPRESSIONS, BACKENDS
    from tf_pwa.experimental.dense_factor.backend import (
        get_op,
        pick_backend,
        register_op,
        supports,
    )

    assert "tf" in BACKENDS
    assert set(_COMPRESSIONS) <= set(BACKENDS["tf"]["functions"])
    for method in _COMPRESSIONS:
        assert callable(get_op("tf", method))
    # the slot is just the function table: no capability flags
    assert set(BACKENDS["tf"]) == {"functions"}

    # an explicit preference is honoured, and a backend this process never
    # registered is skipped for the next name in ``prefer``
    assert pick_backend(None, ("tf",)) == "tf"
    assert pick_backend("svd", ("missing", "tf")) == "tf"
    assert pick_backend("index", ("missing", "tf")) == "tf"

    @register_op("other", "index")
    def _other_index(strategy, record, F, B, C, g, layout, prepared):
        return get_op("tf", "index")(
            strategy, record, F, B, C, g, layout, prepared
        )

    try:
        assert get_op("other", "index") is _other_index
        assert supports("other", "index") is True
        assert supports("other", "basis_index") is False
        assert pick_backend("index", ("other", "tf")) == "other"
        assert pick_backend("svd", ("other", "tf")) == "tf"
    finally:
        BACKENDS.pop("other", None)

    assert "other" not in BACKENDS


def test_dense_factor_backend_registers_only_overrides():
    """A backend registers only the methods it overrides.

    A method it leaves out stays absent, so ``supports`` declines it and
    ``pick_backend`` asks the reference instead of the backend that does not
    provide the contraction.
    """
    from tf_pwa.experimental.dense_factor import BACKENDS
    from tf_pwa.experimental.dense_factor.backend import (
        get_op,
        pick_backend,
        register_op,
        supports,
    )

    @register_op("only_one", "index")
    def _only_index(strategy, record, F, B, C, g, layout, prepared):
        return get_op("tf", "index")(
            strategy, record, F, B, C, g, layout, prepared
        )

    try:
        # one method registered: the other five are simply absent
        assert set(BACKENDS["only_one"]["functions"]) == {"index"}
        assert get_op("only_one", "index") is _only_index
        assert supports("only_one", "index") is True
        assert supports("only_one", "basis_index") is False
        assert pick_backend("index", ("only_one", "tf")) == "only_one"
        assert (
            pick_backend(
                "basis_index",
                ("only_one", "tf"),
            )
            == "tf"
        )
    finally:
        BACKENDS.pop("only_one", None)

    assert "only_one" not in BACKENDS


def test_dense_factor_merge_basis_index():
    """basis_index drops columns that are short exact combinations."""
    import yaml

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    def cfg(method):
        return {
            **base,
            "data": {
                **base["data"],
                "format": "simple",
                "r_boost": False,
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": {"method": method}},
            },
        }

    c0 = ConfigLoader(cfg("index"))
    c1 = ConfigLoader(cfg("basis_index"))
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)

    assert a1.merge["method"] == "basis_index"
    ptr = a1.merge["f_ptr"].numpy()
    lens = np.diff(ptr)
    assert ptr[0] == 0
    assert ptr[-1] == len(a1.merge["f_col"]) == len(a1.merge["f_val"])
    assert a1.merge["f_width"] == len(ptr) - 1
    assert lens.min() >= 1  # every column has at least its own term
    assert lens.max() <= _TERMS
    # the unrolled reader takes one bound for both tensors
    assert a1.merge["terms"] == max(a1.merge["f_max"], a1.merge["b_max"])
    assert 1 <= a1.merge["terms"] <= _TERMS
    assert not d1["F"].dtype.is_complex
    # real halves: more columns, fewer bytes
    assert (
        np.prod(d1["F"].shape) * d1["F"].dtype.size
        < np.prod(d0["F"].shape) * d0["F"].dtype.size
    )
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)
    F0, B0 = a0.expand_block(d0["F"], d0["B"])
    F1, B1 = a1.expand_block(d1["F"], d1["B"])
    assert np.allclose(F0, F1, rtol=1e-8, atol=1e-10)
    assert np.allclose(B0, B1, rtol=1e-8, atol=1e-10)


def test_dense_factor_merge_basis_index_op_matches_rebuild(monkeypatch):
    """The sparse-basis op reads the stored block exactly like the rebuild.

    ``method: basis_index`` in the op model goes through
    ``DenseFactorFwdBi``/``GradBi`` on ``d["F"]``/``d["B"]`` directly;
    forcing ``use_op = False`` makes the backend hand the record to the TF
    reference, which rebuilds the complex basis and the dropped columns.  Both
    see the same data and the same record, so the amplitude and the parameter
    gradients must agree.
    """
    import tensorflow as tf
    import yaml

    from tf_pwa.experimental.dense_factor import _BasisIndexCompression
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    cfg = {
        **base,
        "data": {
            **base["data"],
            "format": "simple",
            "r_boost": False,
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": "basis_index"}},
        },
    }
    c = ConfigLoader(cfg)
    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    assert amp.merge["method"] == "basis_index"
    assert not d["F"].dtype.is_complex and not d["B"].dtype.is_complex

    import tf_pwa.experimental.dense_factor_op as op_mod

    used = []
    original = op_mod.dense_factor_bi
    monkeypatch.setattr(
        op_mod,
        "dense_factor_bi",
        lambda *a, **k: used.append(1) or original(*a, **k),
    )
    a_op = amp(d).numpy()
    assert used, "the sparse-basis op was not exercised"

    def loss_and_grad():
        vs = amp.vm.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vs)
            loss = -tf.reduce_sum(tf.math.log(amp(d)))
        return [g.numpy() for g in tape.gradient(loss, vs)]

    g_op = loss_and_grad()

    monkeypatch.setattr(_BasisIndexCompression, "use_op", False)
    a_rb = amp(d).numpy()
    g_rb = loss_and_grad()
    monkeypatch.undo()

    assert np.allclose(a_op, a_rb, rtol=1e-8, atol=1e-10)
    assert len(g_op) == len(g_rb) and g_op
    for x, y in zip(g_op, g_rb):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-10)


def test_dense_factor_basis_index_op_traced_uses_fused_op(monkeypatch):
    """The sparse-basis entry stays fused when the amplitude is traced.

    ``dense_factor_bi`` recovers its ``nproj`` template bound from
    ``f_ptr``.  Resolving it after the casts/reshapes made
    ``tf.get_static_value`` return ``None`` inside a ``tf.function``, so the
    op declined and every traced amplitude silently ran the reference.  The
    spy plus the absent fallback warning is the regression test.
    """
    import tensorflow as tf
    import yaml

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    cfg = {
        **base,
        "data": {
            **base["data"],
            "format": "simple",
            "r_boost": False,
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": "basis_index"}},
        },
    }
    c = ConfigLoader(cfg)
    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()

    used = []
    original = op_mod.dense_factor_bi
    monkeypatch.setattr(
        op_mod,
        "dense_factor_bi",
        lambda *a, **k: used.append(1) or original(*a, **k),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = tf.function(lambda: amp(d))().numpy()
    assert used, "the traced amplitude did not use the fused basis_index op"
    assert not [w for w in caught if "op failed" in str(w.message)]
    assert np.all(np.isfinite(a))


def _prepare_cfg(method):
    """A known-good op model for ``method``: the CP toy or the spin-2 toy."""
    if method == "basis_index":
        import yaml

        here = os.path.dirname(__file__)
        with open(os.path.join(here, "config_self_cp.yml")) as f:
            base = yaml.full_load(f)
        cfg = {
            **base,
            "data": {
                **base["data"],
                "format": "simple",
                "r_boost": False,
                "amp_model": "dense_factor_op_merge",
                "preprocessor": {
                    "dense_factor_merge": {"method": "basis_index"}
                },
            },
        }
        ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
        p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))
        return ConfigLoader(cfg), p4

    decay, part = _identical_waves()
    # the spin-2 top stores pure-real columns as well as mixed ones
    part["X"]["J"] = 2
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": "real_index"}},
        },
        "decay": decay,
        "particle": part,
    }
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    return ConfigLoader(cfg), {"a": p_a, "b": p_b, "c": p_c}


@pytest.mark.parametrize(
    "method,op_name",
    [("basis_index", "dense_factor_bi"), ("real_index", "dense_factor_ri")],
)
def test_dense_factor_op_prepare_static_inputs(monkeypatch, method, op_name):
    """The op inputs are derived once per record, not per call.

    ``nproj`` used to be recovered inside the op with ``tf.get_static_value``:
    a device sync in eager, and unavailable inside a ``tf.function``.  The
    record already holds ``np`` as an int, so the prepare derives it once and
    the model caches it per record; a fresh record (a refit) rebuilds it.
    """
    from tf_pwa.experimental.dense_factor import _compression_of
    from tf_pwa.experimental.dense_factor.backend import PREPARE
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    c, p4 = _prepare_cfg(method)
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    record = amp.merge
    assert record["method"] == method
    layout = amp.get_layout()

    key = ("cuda", method)
    assert key in PREPARE
    prepared = PREPARE[key](_compression_of(record), record, layout)
    assert prepared["nproj"] == record["np"]
    assert all(type(v) is int for v in prepared.values())
    if method == "basis_index":
        # ``n_line``/``max_basis`` are the positional layout arguments, so
        # they are not duplicated here
        assert set(prepared) == {"nproj", "terms"}
        assert prepared["terms"] == record["terms"]
        width = len(record["f_ptr"]) - 1
    else:
        assert set(prepared) == {"nproj"}
        width = record["f_re_map"].shape[0]
    # the prepare keeps the self-check ``_bi_nproj``/``_ri_nproj`` used to
    # make: the value drives a reshape, so it is pinned by the maps
    n_u = int(np.asarray(record["map_u"]).max()) + 1
    assert prepared["nproj"] * n_u == width

    # a stale ``np`` is rejected: the op then infers from the same maps
    bad = dict(record)
    bad["np"] = record["np"] + 1
    assert PREPARE[key](_compression_of(record), bad, layout) is None

    # the layout is a read-only mapping: a backend cannot rewrite it
    with pytest.raises(TypeError):
        layout["off"] = 1

    # the prepare runs once for two evaluations; a fresh record rebuilds it
    counts = []
    original_prepare = PREPARE[key]

    def counted(strategy, rec, lay):
        counts.append(1)
        return original_prepare(strategy, rec, lay)

    monkeypatch.setitem(PREPARE, key, counted)

    # the op must receive the value rather than infer it
    seen = []
    original_op = getattr(op_mod, op_name)

    def spy(*args, **kwargs):
        seen.append(kwargs.get("nproj"))
        return original_op(*args, **kwargs)

    monkeypatch.setattr(op_mod, op_name, spy)

    def _no_infer(*args, **kwargs):
        raise AssertionError("{} was reached".format(op_name))

    infer = "_bi_nproj" if method == "basis_index" else "_ri_nproj"
    monkeypatch.setattr(op_mod, infer, _no_infer)

    x1 = amp(d).numpy()
    x2 = amp(d).numpy()
    assert np.allclose(x1, x2, rtol=1e-12, atol=0)
    assert len(counts) == 1
    assert seen and set(seen) == {record["np"]}

    fresh = dict(record)
    monkeypatch.setattr(type(amp), "merge", property(lambda self: fresh))
    amp(d)
    assert len(counts) == 2
    assert seen[-1] == fresh["np"]


def test_dense_factor_op_prepare_terms_is_load_bearing(monkeypatch):
    """The entry reads ``terms`` from the prepare, not from the record.

    The record still carries a valid ``terms`` here, so a prepare that says
    ``terms = 0`` can only stop the fused op if the entry actually reads the
    prepared dict.
    """
    from tf_pwa.experimental.dense_factor.backend import PREPARE
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    c, p4 = _prepare_cfg("basis_index")
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    assert 1 <= amp.merge["terms"] <= _TERMS  # the record's own value is valid

    key = ("cuda", "basis_index")
    monkeypatch.setitem(
        PREPARE,
        key,
        lambda strategy, record, layout: {
            "nproj": int(record["np"]),
            "terms": 0,
        },
    )

    calls = []
    original = op_mod.dense_factor_bi

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(op_mod, "dense_factor_bi", spy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = amp(d).numpy()

    assert not calls, "the fused op ran despite the prepared terms"
    said = [
        str(w.message)
        for w in caught
        if "supports 1 to {}".format(_TERMS) in str(w.message)
    ]
    assert said, [str(w.message) for w in caught]
    assert "0-term transform" in said[0]
    assert np.all(np.isfinite(a))


def test_dense_factor_op_prepare_failure_falls_back_to_inference(monkeypatch):
    """A raising prepare warns once and lets the op infer, not fail.

    The prepare is a cache, not a precondition: a record the prepare cannot
    read must still evaluate, through the op's own ``_bi_nproj`` inference.
    """
    from tf_pwa.experimental.dense_factor.backend import PREPARE
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor as df
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    c, p4 = _prepare_cfg("basis_index")
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    ref = amp(d).numpy()

    key = ("cuda", "basis_index")
    calls = []

    def boom(strategy, record, layout):
        calls.append(1)
        raise RuntimeError("prepare boom")

    monkeypatch.setitem(PREPARE, key, boom)
    # the reference evaluation above filled the per-record cache
    amp._prepared.clear()

    saved = set(df._OP_FAILURES)
    df._OP_FAILURES.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = amp(d).numpy()
    finally:
        df._OP_FAILURES.clear()
        df._OP_FAILURES.update(saved)

    assert calls == [1]
    said = [str(w.message) for w in caught if "op failed" in str(w.message)]
    assert len(said) == 1 and "prepare boom" in said[0]
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_dense_factor_op_real_svd_failure_names_the_method(monkeypatch):
    """A fused failure is reported under ``real_svd``, not ``svd``.

    ``_run_expand`` serves both methods; the reference lookup and the
    warn-once key must use the record's own method.
    """
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor as df
    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": "real_svd"}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    amp = c.get_amplitude()

    def boom(*args, **kwargs):
        raise RuntimeError("op boom")

    monkeypatch.setattr(op_mod, "dense_factor_a", boom)

    saved = set(df._OP_FAILURES)
    df._OP_FAILURES.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = amp(d).numpy()
    finally:
        df._OP_FAILURES.clear()
        df._OP_FAILURES.update(saved)

    said = [str(w.message) for w in caught if "op failed" in str(w.message)]
    assert len(said) == 1 and "real_svd" in said[0]
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("method", ["svd", "real_svd"])
def test_dense_factor_op_svd_stays_fused(monkeypatch, method):
    """``svd``/``real_svd`` still contract through the fused plain op.

    ``_run_expand`` used to label both methods ``"svd"``; the numbers are the
    same on the reference fallback, so only a spy on the op pins the fused
    path.
    """
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()
    cfg = {
        "data": {
            "format": "simple",
            "amp_model": "dense_factor_op_merge",
            "preprocessor": {"dense_factor_merge": {"method": method}},
        },
        "decay": decay,
        "particle": part,
    }
    c = ConfigLoader(cfg)
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    amp = c.get_amplitude()
    assert amp.merge["method"] == method

    calls = []
    for name in ("dense_factor_a", "dense_factor_an"):
        original = getattr(op_mod, name)

        def spy(*args, _orig=original, _name=name, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(op_mod, name, spy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = amp(d).numpy()

    assert set(calls) == {"dense_factor_a"}
    assert not [w for w in caught if "op failed" in str(w.message)]
    assert np.all(np.isfinite(a))


@pytest.mark.parametrize("method", ["basis_index", "real_index"])
def test_dense_factor_op_stored_pair_warns_once(method):
    """A block the op cannot read is a loud decline, not a silent fallback.

    ``_stored_pair`` rejects a complex block (a mis-wired preprocessor) or a
    rank that is not the stored pair.  Returning ``None`` without a word
    disabled the fused path invisibly; the warn-once report names the reason
    and the call still falls back to the reference.
    """
    import tensorflow as tf

    import tf_pwa.experimental.dense_factor as df
    from tf_pwa.experimental.dense_factor import _compression_of, get_op
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    c, p4 = _prepare_cfg(method)
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    record = amp.merge
    layout = amp.get_layout()
    strategy = _compression_of(record)
    C, g, _ = amp._coefficients(d)

    # a complex view of the stored pair: the values are the same, the block
    # is not the real pair the kernel reads
    F = tf.cast(d["F"], tf.complex128)
    B = tf.cast(d["B"], tf.complex128)
    entry = get_op("cuda", method)
    reference = get_op("tf", method)

    snapshot = set(df._OP_FAILURES)
    df._OP_FAILURES.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = entry(strategy, record, F, B, C, g, layout, None)
            again = entry(strategy, record, F, B, C, g, layout, None)
        said = [
            str(w.message) for w in caught if "op failed" in str(w.message)
        ]
        assert len(said) == 1, said  # reported once, the second call dedupes
        assert "not the real pair" in said[0]
    finally:
        df._OP_FAILURES.clear()
        df._OP_FAILURES.update(snapshot)

    expected = reference(strategy, record, F, B, C, g, layout, None)
    assert got is not None and again is not None
    assert np.allclose(got.numpy(), again.numpy(), rtol=1e-12, atol=0)
    assert np.allclose(got.numpy(), expected.numpy(), rtol=1e-8, atol=1e-10)


def _wide_basis_model():
    """The J=2/spin-1 model whose transform needs three terms."""
    waves = 4
    decay = {"X": [[f"R{i}", "c"] for i in range(waves)]}
    part = {
        "$top": "X",
        "$finals": ["a", "b", "c"],
        "X": {"J": 2, "P": -1, "mass": 5.0},
        "a": {"J": 1, "P": -1, "mass": 0.2},
        "b": {"J": 1, "P": -1, "mass": 0.2},
        "c": {"J": 0, "P": -1, "mass": 0.2},
    }
    for i in range(waves):
        decay[f"R{i}"] = ["a", "b"]
        part[f"R{i}"] = {
            "J": 1,
            "P": -1,
            "mass": 2.0 + 0.03 * i,
            "width": 0.1,
            "model": "BWR",
        }
    return decay, part


def _wide_basis_cfg(decay, part, model, pre):
    return {
        "data": {
            "format": "simple",
            "amp_model": model,
            "preprocessor": pre,
        },
        "decay": decay,
        "particle": part,
    }


_STORED_OP = {
    "basis_index": ("_BasisIndexCompression", "dense_factor_bi"),
    "real_index": ("_RealIndexCompression", "dense_factor_ri"),
}


@pytest.mark.parametrize("method", ["basis_index", "real_index"])
@pytest.mark.parametrize(
    "store_dtype,tol",
    [
        # the stored block is real for both methods: float32 under complex64,
        # fp16 under complex32.  Tolerances as in the storage tests' real-block
        # rows, which keep the same kind of block.
        ("complex64", 1e-4),
        ("complex32", 2e-3),
    ],
)
def test_dense_factor_merge_stored_op_narrow_store(
    monkeypatch, method, store_dtype, tol
):
    """The fused op reads a float32/fp16 stored block in place.

    Both stored methods keep one *real* block and lower it with
    ``store_dtype``: ``basis_index`` as the sparse transform of the basis
    columns, ``real_index`` as the ``[Re | Im]`` merge.  The op takes that
    block with its own dtype and widens each value to double while reading
    instead of receiving an upcast copy.

    The test has to be able to fail, so it checks the three ways the path can
    silently fall back: every call must receive the *narrow* dtype (the spy
    records what the op got), a failed op must not be swallowed (the fallback
    reporter raises here), and the op's output must actually reach the
    amplitude (a 1e-3 bump on the op's result must move ``a(d)``).  Beyond
    that, amplitude and parameter gradients must agree with the rebuild path
    (same stored values), and the storage loss against the full-precision
    dense model must stay inside the tolerance the storage tests use.
    """
    import tensorflow as tf
    import yaml

    import tf_pwa.experimental.dense_factor as df
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    compression = getattr(df, _STORED_OP[method][0])
    op_name = _STORED_OP[method][1]
    want = tf.float32 if store_dtype == "complex64" else tf.float16

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "config_self_cp.yml")) as f:
        base = yaml.full_load(f)

    def cp_cfg():
        return {
            **base,
            "data": {
                **base["data"],
                "format": "simple",
                "r_boost": False,
                "amp_model": "dense_factor_op_merge",
                "preprocessor": {
                    "dense_factor_merge": {
                        "method": method,
                        "store_dtype": store_dtype,
                    }
                },
            },
        }

    c = ConfigLoader(cp_cfg())
    ps = PhaseSpaceGenerator(4.6, [0.139, 0.139, 3.0])
    p4 = dict(zip(["pip", "pim", "jpsi"], ps.generate(200)))
    d = c.data.cal_angle(p4)
    amp = c.get_amplitude()
    assert amp.merge["method"] == method
    # the stored blocks really are narrow, not upcast before the op
    assert d["F"].dtype == want and d["B"].dtype == want
    assert not d["F"].dtype.is_complex and not d["B"].dtype.is_complex

    original = getattr(op_mod, op_name)
    calls = []

    def spy(*args, **kwargs):
        # record before delegating, so a raising op still shows what it got
        calls.append((args[0].dtype, args[1].dtype))
        return original(*args, **kwargs)

    monkeypatch.setattr(op_mod, op_name, spy)
    a_op = amp(d).numpy()
    assert calls, "the fused %s op was not exercised" % method
    assert all(
        fs_dt == want and bs_dt == want for fs_dt, bs_dt in calls
    ), "the fused op got %r, expected %s for both blocks" % (
        sorted({str(dt) for pair in calls for dt in pair}),
        want.name,
    )

    # a failing op must not be swallowed: the backend warns and the model
    # then silently uses the reference, which is what hid the narrow path
    def no_fallback(*args, **kwargs):
        pytest.fail("the fused %s op failed: %r" % (method, args))

    monkeypatch.setattr(df, "_warn_op_failure", no_fallback)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a_checked = amp(d).numpy()
    assert not [w for w in caught if "op failed" in str(w.message).lower()]
    assert np.allclose(a_checked, a_op, rtol=1e-12, atol=1e-14)

    # and the op's result has to reach the amplitude: bump it by 1e-3
    # (the wrapper returns the amplitude tensor; the denominator rides along
    # inside the custom gradient, so there is nothing else to preserve)
    def bumped(*args, **kwargs):
        return original(*args, **kwargs) * (1.0 + 1e-3)

    monkeypatch.setattr(op_mod, op_name, bumped)
    a_bumped = amp(d).numpy()
    monkeypatch.setattr(op_mod, op_name, original)
    moved = np.abs(a_bumped - a_op).max() / np.abs(a_op).max()
    assert moved > 1e-4, "the op result does not reach a(d) (moved %g)" % moved

    def loss_and_grad(cfg_amp, data):
        vs = cfg_amp.vm.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vs)
            loss = -tf.reduce_sum(tf.math.log(cfg_amp(data)))
        return [g.numpy() for g in tape.gradient(loss, vs)]

    g_op = loss_and_grad(amp, d)

    # the same stored values through the rebuild path
    monkeypatch.setattr(compression, "use_op", False)
    a_rb = amp(d).numpy()
    g_rb = loss_and_grad(amp, d)
    monkeypatch.undo()

    assert np.allclose(a_op, a_rb, rtol=1e-8, atol=1e-10)
    assert len(g_op) == len(g_rb) and g_op
    for x, y in zip(g_op, g_rb):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-10)

    # storage loss against the full-precision dense model, on the same model
    # and data the storage tolerances were measured on
    decay, part = _identical_waves()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    def small_cfg(model, pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": model,
                "preprocessor": pre,
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(small_cfg("dense_factor", "dense_factor"))
    a0 = c0.get_amplitude()
    x0 = a0(c0.data.cal_angle(p4)).numpy()
    c1 = ConfigLoader(
        small_cfg(
            "dense_factor_op_merge",
            {
                "dense_factor_merge": {
                    "method": method,
                    "store_dtype": store_dtype,
                }
            },
        )
    )
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())
    d1 = c1.data.cal_angle(p4)
    assert d1["F"].dtype == want
    x1 = a1(d1).numpy()
    assert np.allclose(x1, x0, rtol=tol, atol=10 * tol)
    assert tf.reduce_all(tf.math.is_finite(x1))


@pytest.mark.parametrize("method", ["basis_index", "real_index"])
def test_dense_factor_merge_narrow_store_dense_model(method):
    """The dense merge model reads a whole fp16 stored block.

    No fused op on this path: the compression expands the stored block back
    to the coefficient dtype itself, so ``basis_index``/``real_index`` at
    ``store_dtype: complex32`` need their own coverage (the op tests exercise
    the kernel, not the expansion).
    """
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("no GPU")

    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _identical_waves()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}

    def cfg(model, pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": model,
                "preprocessor": pre,
            },
            "decay": decay,
            "particle": part,
        }

    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    a0 = c0.get_amplitude()
    x0 = a0(c0.data.cal_angle(p4)).numpy()

    c1 = ConfigLoader(
        cfg(
            "dense_factor_merge",
            {
                "dense_factor_merge": {
                    "method": method,
                    "store_dtype": "complex32",
                }
            },
        )
    )
    a1 = c1.get_amplitude()
    a1.set_params(a0.get_params())
    d1 = c1.data.cal_angle(p4)
    assert d1["F"].dtype == tf.float16 and d1["B"].dtype == tf.float16
    assert a1.merge["method"] == method
    x1 = a1(d1).numpy()
    # the real-block storage tolerance used for complex32
    assert np.allclose(x1, x0, rtol=2e-3, atol=2e-2)
    assert tf.reduce_all(tf.math.is_finite(x1))


def test_dense_factor_merge_basis_index_wide():
    """A wide J=2 basis fits three terms and keeps exactly its rank.

    The fit cap is four terms, so the columns that used to be promoted
    because they needed three are now represented by the transform.  The
    stored basis therefore reaches the numerical rank of the reduced F (25
    here) instead of keeping extra directions.
    """
    import tensorflow as tf

    decay, part = _wide_basis_model()

    def cfg(model, pre):
        return _wide_basis_cfg(decay, part, model, pre)

    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    c1 = ConfigLoader(
        cfg(
            "dense_factor_merge",
            {"dense_factor_merge": {"method": "basis_index"}},
        )
    )
    a0, a1 = c0.get_amplitude(), c1.get_amplitude()
    a1.set_params(a0.get_params())
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d0 = c0.data.cal_angle(p4)
    d1 = c1.data.cal_angle(p4)
    assert np.allclose(a0(d0).numpy(), a1(d1).numpy(), rtol=1e-8, atol=1e-10)

    record = a1.merge
    assert record["method"] == "basis_index"
    ptr = record["f_ptr"].numpy()
    lens = np.diff(ptr)
    assert ptr[0] == 0
    assert ptr[-1] == len(record["f_col"]) == len(record["f_val"])
    assert record["f_width"] == len(ptr) - 1
    assert lens.max() <= _TERMS
    # the numerical rank of the reduced F at 1e-8
    F = tf.gather(d0["F"], tf.constant(record["rep_u"], tf.int32), axis=2)
    F = tf.reshape(F, (tf.shape(F)[0], -1)).numpy()
    sv = np.linalg.svd(F, compute_uv=False)
    rank = int(np.sum(sv > 1e-8 * sv[0]))
    # stores exactly the numerical rank, whatever it is for this basis
    assert len(record["f_sel"]) == rank


def test_dense_factor_merge_basis_index_wide_op(monkeypatch):
    """The fused op reads the three-term transform exactly like the rebuild."""
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor import _BasisIndexCompression
    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")

    import tf_pwa.experimental.dense_factor_op as op_mod
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    decay, part = _wide_basis_model()

    def cfg(model, pre):
        return _wide_basis_cfg(decay, part, model, pre)

    c0 = ConfigLoader(cfg("dense_factor", "dense_factor"))
    c2 = ConfigLoader(
        cfg(
            "dense_factor_op_merge",
            {"dense_factor_merge": {"method": "basis_index"}},
        )
    )
    a0, a2 = c0.get_amplitude(), c2.get_amplitude()
    a2.set_params(a0.get_params())
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=100)
    p4 = {"a": p_a, "b": p_b, "c": p_c}
    d2 = c2.data.cal_angle(p4)
    assert a2.merge["method"] == "basis_index"
    ptr = a2.merge["f_ptr"].numpy()
    assert np.diff(ptr).max() == 3
    assert ptr[-1] == len(a2.merge["f_col"]) == len(a2.merge["f_val"])

    used = []
    original = op_mod.dense_factor_bi
    monkeypatch.setattr(
        op_mod,
        "dense_factor_bi",
        lambda *args, **kwargs: used.append(1) or original(*args, **kwargs),
    )

    def loss_and_grad():
        vs = a2.vm.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vs)
            loss = -tf.reduce_sum(tf.math.log(a2(d2)))
        return [g.numpy() for g in tape.gradient(loss, vs)]

    a_op = a2(d2).numpy()
    assert used, "the sparse-basis op was not exercised"
    g_op = loss_and_grad()

    monkeypatch.setattr(_BasisIndexCompression, "use_op", False)
    a_rb = a2(d2).numpy()
    g_rb = loss_and_grad()
    monkeypatch.undo()

    assert np.allclose(a_op, a_rb, rtol=1e-8, atol=1e-10)
    assert len(g_op) == len(g_rb) and g_op
    for x, y in zip(g_op, g_rb):
        assert np.allclose(x, y, rtol=1e-8, atol=1e-10)


def test_dense_factor_basis_index_graph():
    """The rebuild branches on a Python flag, so tf.function works."""
    import tensorflow as tf

    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    c = ConfigLoader(cfg({"method": "basis_index"}))
    amp = c.get_amplitude()
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=64)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    eager = amp(d).numpy()
    graph = tf.function(amp)(d).numpy()
    assert np.allclose(eager, graph, rtol=1e-10, atol=1e-12)


def _spin_one_swap():
    """A decay with two identical spin-1 finals, whose swap is not cosmetic."""

    def cfg(amp_model, pre):
        data = {
            "dat_order": ["B", "C", "D"],
            "center_mass": True,
            "format": "simple",
            "random_z": False,
            "r_boost": False,
            "amp_model": amp_model,
            "identical_particles": [["C", "D"]],
        }
        if pre is not None:
            data["preprocessor"] = pre
        return {
            "data": data,
            "decay": {"A": ["BC", "D"], "BC": ["B", "C"]},
            "particle": {
                "$top": "A",
                "$finals": ["B", "C", "D"],
                "A": {"J": 0.5, "P": -1, "mass": 3.0},
                "B": {"J": 0.5, "P": -1, "mass": 0.1},
                "C": {"J": 1, "P": -1, "mass": 0.1},
                "D": {"J": 1, "P": -1, "mass": 0.1},
                "BC": {
                    "J": 0.5,
                    "P": -1,
                    "mass": 2.0,
                    "width": 0.5,
                    "model": "BWR",
                },
            },
        }

    return cfg


def test_dense_merge_block_check_rejects_mismatch():
    """A block the record cannot describe is refused before it is stored."""
    import tensorflow as tf

    decay, part = _identical_waves()

    def cfg(pre):
        return {
            "data": {
                "format": "simple",
                "amp_model": "dense_factor_merge",
                "preprocessor": {"dense_factor_merge": pre},
            },
            "decay": decay,
            "particle": part,
        }

    c = ConfigLoader(cfg({"method": "index"}))
    (p_a, p_b), p_c = generate_phsp(5.0, ((2.0, (0.2, 0.2)), 0.2), N=64)
    d = c.data.cal_angle({"a": p_a, "b": p_b, "c": p_c})
    pre = c.data.preprocessor
    record = pre._existing_record()
    assert record
    # cal_angle stores the compressed block; the check runs on the expanded one
    F, B = pre.compression.expand(record, d["F"], d["B"])
    noise = tf.cast(tf.random.normal(tf.shape(F), dtype=tf.float64), F.dtype)
    with pytest.raises(ValueError, match="identical-particle"):
        pre._check_blocks(record, {"F": F, "B": B}, [{"F": F + noise, "B": B}])


def test_dense_merge_identical_spin_one_swap():
    """One record describes every block: the swap only changes the input.

    Every block is built in the same axis convention and the helicity
    transpose is applied to the amplitude, as ``get_amp2`` does, so the swap
    block shares the main block's column space (rank 12 of 144 for each, 12
    together) and one compression record is exact for all of them.  Storing
    the transposed ``F`` instead rotates the columns out of that convention
    and every per-column record then misreads the block.
    """
    import tensorflow as tf

    from tf_pwa.experimental.dense_factor_op import is_available

    if not is_available():
        pytest.skip("dense_factor_op cannot be built on this machine")
    import tf_pwa.experimental.dense_factor_op_model  # noqa: F401

    p4 = dict(
        zip(
            ["B", "C", "D"],
            PhaseSpaceGenerator(3.0, [0.1, 0.1, 0.1]).generate(50),
        )
    )
    build = _spin_one_swap()
    c0 = ConfigLoader(build("dense_factor", {"dense_factor": {}}))
    a0 = c0.get_amplitude()
    d0 = c0.data.cal_angle(p4)
    ref = a0(d0).numpy()

    key = list(d0["id_swap"])[0]
    main = d0["F"].numpy().reshape(len(ref), -1)
    swap = d0["id_swap"][key]["F"].numpy().reshape(len(ref), -1)
    rank = np.linalg.matrix_rank(main)
    assert rank < main.shape[1]
    assert np.linalg.matrix_rank(np.vstack([main, swap])) == rank

    for method in ("index", "real_index", "basis_index", "svd", "real_svd"):
        c = ConfigLoader(
            build(
                "dense_factor_op_merge",
                {"dense_factor_merge": {"method": method}},
            )
        )
        amp = c.get_amplitude()
        amp.set_params(c0.get_params())
        data = c.data.cal_angle(p4)
        out = amp(data).numpy()
        assert np.allclose(out, ref, rtol=1e-9, atol=1e-12), method
        # the transpose now lives in the graph too
        graph = tf.function(amp)(data).numpy()
        assert np.allclose(graph, out, rtol=1e-9, atol=1e-12), method
