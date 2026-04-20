import numpy as np

from tf_pwa.mass_cons import MassCons, mass_sq

# Particle masses (GeV)
M_D = 1.86966
M_K = 0.493677
M_B = 5.27943
MASS_TARGETS = np.array([M_D**2, M_D**2, M_K**2, M_B**2])
MOMENTUM_TARGET = np.array([M_B, 0.0, 0.0, 0.0])


def get_all_mass(data):
    """Compute masses for all particles and total."""
    n_particles = data.shape[1]
    masses = [mass_sq(data[:, i]) for i in range(n_particles)]
    masses.append(mass_sq(np.sum(data, axis=1)))
    return np.stack(masses, axis=-1)


def test_mass_constraints_satisfied():
    """Test that mass constraints are satisfied within tolerance."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3, mass_constraints=[(0, 1, 2)])

    result = model.do_constraints(data, MASS_TARGETS[:4])

    masses_sq = get_all_mass(result)
    fitted_masses = np.sqrt(masses_sq)

    # Check individual particle masses
    assert np.allclose(
        fitted_masses[:, 0], M_D, atol=1e-6
    ), f"D- mass: {fitted_masses[:, 0].mean()}"
    assert np.allclose(
        fitted_masses[:, 1], M_D, atol=1e-6
    ), f"D+ mass: {fitted_masses[:, 1].mean()}"
    assert np.allclose(
        fitted_masses[:, 2], M_K, atol=1e-6
    ), f"K+ mass: {fitted_masses[:, 2].mean()}"
    # Total mass
    assert np.allclose(
        fitted_masses[:, 3], M_B, atol=1e-6
    ), f"B mass: {fitted_masses[:, 3].mean()}"


def test_momentum_constraint():
    """Test that momentum constraint is satisfied."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3)  # No total mass constraint

    result = model.do_constraints(
        data, MASS_TARGETS[:3], momentum_target=MOMENTUM_TARGET
    )

    # Check total momentum
    total_p = np.sum(result, axis=1)
    assert np.allclose(
        total_p[:, 0], M_B, atol=1e-6
    ), f"E: {total_p[:, 0].mean()}"
    assert np.allclose(
        total_p[:, 1], 0.0, atol=1e-6
    ), f"px: {total_p[:, 1].mean()}"
    assert np.allclose(
        total_p[:, 2], 0.0, atol=1e-6
    ), f"py: {total_p[:, 2].mean()}"
    assert np.allclose(
        total_p[:, 3], 0.0, atol=1e-6
    ), f"pz: {total_p[:, 3].mean()}"


def test_per_event_momentum():
    """Test with per-event momentum targets."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3)

    # Different energy for each event
    momentum_target = np.zeros((n_events, 4))
    momentum_target[:, 0] = M_B + np.linspace(-0.1, 0.1, n_events)

    result = model.do_constraints(
        data, MASS_TARGETS[:3], momentum_target=momentum_target
    )

    total_p = np.sum(result, axis=1)
    assert np.allclose(
        total_p, momentum_target, atol=1e-6
    ), "Per-event momentum not satisfied"


def test_per_event_mass():
    """Test with per-event mass targets."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3, mass_constraints=[(0, 1, 2)])

    # Varying mass targets
    mass_targets_2d = np.zeros((n_events, 4))
    mass_targets_2d[:, 0] = (M_D + np.linspace(-0.01, 0.01, n_events)) ** 2
    mass_targets_2d[:, 1] = M_D**2
    mass_targets_2d[:, 2] = M_K**2
    mass_targets_2d[:, 3] = M_B**2

    result = model.do_constraints(data, mass_targets_2d)

    masses_sq = get_all_mass(result)
    for i in range(4):
        expected = np.sqrt(mass_targets_2d[:, i])
        fitted = np.sqrt(masses_sq[:, i])
        assert np.allclose(fitted, expected, atol=1e-6), f"Mass {i} mismatch"


def test_auto_mass_targets():
    """Test auto-computed mass targets converge to mean of input."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3, mass_constraints=[(0, 1, 2)])

    # Expected targets from input mean
    expected_masses = np.sqrt(get_all_mass(data).mean(axis=0))

    result = model.do_constraints(data)

    fitted_masses = np.sqrt(get_all_mass(result))
    for i in range(4):
        assert np.allclose(
            fitted_masses[:, i], expected_masses[i], atol=1e-6
        ), f"Mass {i}"


def test_no_extra_mass_constraints():
    """Test model without extra mass constraints (only individual particles)."""
    np.random.seed(42)
    n_events = 100
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3)  # No extra constraints

    result = model.do_constraints(data, MASS_TARGETS[:3])

    assert model.n_mass_constraints == 3
    masses_sq = get_all_mass(result)
    assert np.allclose(np.sqrt(masses_sq[:, 0]), M_D, atol=1e-6)
    assert np.allclose(np.sqrt(masses_sq[:, 1]), M_D, atol=1e-6)
    assert np.allclose(np.sqrt(masses_sq[:, 2]), M_K, atol=1e-6)
    # Total mass should NOT be constrained
    assert not np.allclose(np.sqrt(masses_sq[:, 3]), M_B, atol=1e-6)


def test_chunking():
    """Test that chunking gives same result as non-chunked."""
    np.random.seed(42)
    n_events = 5000
    data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
    model = MassCons(3, mass_constraints=[(0, 1, 2)])

    # Without chunking
    result_no_chunk = model.do_constraints(
        data, MASS_TARGETS, chunk_size=n_events + 1
    )

    # With chunking
    result_chunked = model.do_constraints(data, MASS_TARGETS, chunk_size=1000)

    assert np.allclose(
        result_no_chunk, result_chunked, atol=1e-10
    ), "Chunking gives different results"


def test_small_dataset():
    """Test with very small dataset."""
    np.random.seed(42)
    for n_events in [1, 2, 5]:
        data = np.random.randn(n_events, 3, 4) * 2 + np.array([5, 0, 0, 0])
        model = MassCons(3, mass_constraints=[(0, 1, 2)])
        result = model.do_constraints(data, MASS_TARGETS)
        assert result.shape == (n_events, 3, 4)
