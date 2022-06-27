import numpy as np
import pytest

from tf_pwa.phasespace import *


def test_phasespace():
    a = PhaseSpaceGenerator(10, [3, 2, 1])
    data = a.generate(100)
    assert len(data), 3
    for i in data:
        assert i.shape == (100, 4)

    assert np.allclose(LorentzVector.M(data[0]), [3] * 100)
    assert np.allclose(LorentzVector.M(data[1]), [2] * 100)
    assert np.allclose(LorentzVector.M(data[2]), [1] * 100)
    p_all = data[0] + data[1] + data[2]
    assert np.allclose(LorentzVector.M(p_all), [10] * 100)
    assert np.allclose(p_all[:, 1:3], np.zeros_like(p_all[:, 1:3]))


def test_generate_phsp():
    (a, b), c = generate_phsp(5.0, ((3.0, (1.0, 1.0)), 1.0))
    assert np.allclose(LorentzVector.M(a + b + c), 5.0)
    # assert np.sum(np.abs((a+b+c)[:,1:])) < 2e-5


def test_error():
    with pytest.raises(ValueError):
        PhaseSpaceGenerator(1, [0.3, 0.4, 0.5]).generate(10)


def test_uniform():
    a = UniformGenerator(1.0, 2.0)
    b = a.generate(10).numpy()
    assert np.all(b < 2.0)
    assert np.all(b >= 1.0)
