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
    dec = data.get_decay()[0]
    data.get_angle(dec, "B")
