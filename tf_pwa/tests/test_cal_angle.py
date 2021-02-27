import numpy as np

from tf_pwa.cal_angle import *
from tf_pwa.data import *


def test_process():
    a, b, c, d = [BaseParticle(i) for i in ["A", "B", "C", "D"]]
    p = {
        b: np.array([[1.0, 0.2, 0.3, 0.2]]),
        c: np.array([[2.0, 0.1, 0.3, 0.4]]),
        d: np.array([[3.0, 0.2, 0.5, 0.7]]),
    }
    # st = {b: [b], c: [c], d: [d], a: [b, c, d], r: [b, d]}
    decs = DecayGroup(DecayChain.from_particles(a, [b, c, d]))
    print(decs)
    data = cal_angle_from_momentum(p, decs)
    assert isinstance(data, CalAngleData)
    data = add_weight(data)
    print(data_shape(data, all_list=True))
    print(len(list(split_generator(data, 5000))))
    data = data_to_numpy(data)
    assert data_shape(data) == 1
    data.get_weight()
    data.get_mass("(B, C)")
    dec = data.get_decay().get_decay_chain("(B, C)")
    dec2 = data.get_decay()[0]
    ang = data.get_angle(dec, "B")
    ang2 = data.get_angle("(B, C)", "B")
    assert ang is ang2
    assert "alpha" in ang
    assert "beta" in ang
    assert np.allclose(ang["gamma"], 0)
    p = data.get_momentum("(B, D)")
    assert p.shape[-1] == 4

    data.savetxt("cal_angle_file.txt", ["C", "D"])
    data.savetxt("cal_angle_file.txt")
    hist = data.mass_hist("(C, D)")
