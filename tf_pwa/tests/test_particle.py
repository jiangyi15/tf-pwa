from tf_pwa.particle import *


def test_particle():
    a = Particle("a", 1, -1)
    b = Particle("b", 1, -1)
    c = Particle("c", 0, -1)
    d = Particle("d", 1, -1)
    tmp = Particle("tmp", 1, -1)
    tmp2 = Particle("tmp2", 1, -1)
    decay = Decay(a, [tmp, c])
    decay2 = Decay(tmp, [b, d])
    decay3 = Decay(a, [tmp2, d])
    decay4 = Decay(tmp2, [b, c])
    decaychain = DecayChain([decay, decay2])
    decaychain2 = DecayChain([decay3, decay4])
    decaygroup = DecayGroup([decaychain, decaychain2])
    print(decay.get_cg_matrix().T)
    print(np.array(decay.get_ls_list()))
    print(np.array(decay.get_ls_list())[:, 0])
    print(decaychain)
    print(decaychain.sorted_table())
    print(decaygroup)
    print(a.get_resonances())

    
def test_topology():
    a = Particle("a")
    b = Particle("B")
    c = Particle("C")
    d = Particle("D")
    r = Particle("R")
    d1 = Decay(a, [r, c])
    d2 = Decay(r, [b, d])
    dec = DecayChain([d1, d2])
    print(dec.standard_topology())

