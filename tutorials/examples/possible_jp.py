#!/usr/bin/env python3
# from tf_pwa.phasespace import  PhaseSpaceGenerator
import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")


from tf_pwa.amp import get_particle, get_decay, DecayChain
import itertools


def jp_seq(max_J=2):
    """all JP combination"""
    for j in range(max_J + 1):
        for p in [-1, 1]:
            yield j, p


def possible_jp(decay_chain, max_J=2):
    """get possible resonances jp of J <= max_J"""
    ret = []
    A = decay_chain.top
    outs = decay_chain.outs
    jp_list = [jp_seq(max_J) for _ in decay_chain.inner]

    ret = {}
    for jps in itertools.product(*jp_list):
        res_map = {}
        for jp, res in zip(jps, decay_chain.inner):
            j, p = jp
            res_n = get_particle("{}_{}{}".format(res, j, p), J=j, P=p)
            res_map[res] = res_n
        possible_ls = []
        for i in decay_chain:
            core = res_map.get(i.core, i.core)
            outs = [res_map.get(j, j) for j in i.outs]
            dec = get_decay(core, outs, p_break=i.p_break, disable=True)
            # print(core.J, core.P, outs[0].J, outs[0].P, outs[1].J, outs[1].P)
            # print(dec.get_ls_list())
            possible_ls.append(dec.get_ls_list())
        if min([len(i) for i in possible_ls]) > 0:
            ret[jps] = dict(zip(decay_chain, possible_ls))
    return ret


def main():
    a = get_particle("A", J=0, P=-1)
    b = get_particle("B", J=1, P=-1)
    c = get_particle("C", J=0, P=-1)
    d = get_particle("D", J=0, P=-1)
    r = get_particle("R")

    dec1 = get_decay(a, [r, d], p_break=True)
    dec2 = get_decay(r, [b, c])

    decs = DecayChain([dec1, dec2])

    # chains = DecayChain.from_particles(a, [b, c, d])
    # for i in chains:
    jp_list = possible_jp(decs)
    for jps in jp_list:
        sign = lambda x: "+" if x == 1 else "-"
        jp_display = ["{}{}".format(j, sign(p)) for j, p in jps]
        print("Resonances JP: ", dict(zip(decs.inner, jp_display)))
        print("  Possible (l, s): ", jp_list[jps])


if __name__ == "__main__":
    main()
