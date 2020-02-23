from tf_pwa.amp import *
from tf_pwa.cal_angle import cal_angle_from_momentum


def get_test_decay():
    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    bd = ParticleLass("BD", 1, 1, mass=1.0, width=1.0)
    cd = Particle("CD", 1, 1, mass=1.0, width=1.0)
    bc = Particle("BC", 1, 1, mass=1.0, width=1.0)
    R = get_particle("R", 1, 1, mass=1.0, width=1.0)
    HelicityDecay(a, [bc, d])
    HelicityDecay(bc, [b, c])
    HelicityDecay(a, [cd, b])
    HelicityDecay(cd, [c, d])
    HelicityDecay(a, [bd, c])
    get_decay(bd, [b, d])
    HelicityDecayNP(a, [R, c])
    HelicityDecayP(R, [b, d])
    de = DecayGroup(a.chain_decay())
    print(de)
    return de, [b, c, d]


test_data = [
    [
        np.array([[2.0, 0.1, 0.2, 0.3]]),
        np.array([[3.0, 0.2, 0.3, 0.4]]),
        np.array([[4.0, 0.3, 0.4, 0.5]])
    ],
    [
        np.array([[2.0, 0.1, 0.2, 0.3],[2.0,0.3,0.2,0.1]]),
        np.array([[3.0, 0.2, 0.3, 0.4],[3.0,0.4,0.3,0.2]]),
        np.array([[4.0, 0.3, 0.4, 0.5],[4.0,0.5,0.4,0.3]])
    ]
]



def test_amp():
    decs, particle = get_test_decay()
    amp = AmplitudeModel(decs)
    for p_data in test_data: 
      p = dict(zip(particle, p_data))
      data = cal_angle_from_momentum(p, decs)
      amp(data)
