from tf_pwa.phasespace import *
import numpy as np


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
