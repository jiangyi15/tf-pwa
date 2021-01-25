import matplotlib.pyplot as plt

from tf_pwa.particle import BaseDecay, BaseParticle, DecayChain
from tf_pwa.vis import *

ex_result = """
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none, labelfloat=true];
    "A" [shape=none];
    "B" [shape=none];
    "C" [shape=none];
    "D" [shape=none];
    "E" [shape=none];
    "F" [shape=none];
    { rank=same "A" };
    { rank=same "B","D","E","F","C" };
    "A" -> "A->chain0_node_0+D";
    "A->chain0_node_0+D" -> "chain0_node_0->chain0_node_2+chain0_node_3";
    "A->chain0_node_0+D" -> "D";
    "chain0_node_0->chain0_node_2+chain0_node_3" -> "chain0_node_2->B+F";
    "chain0_node_0->chain0_node_2+chain0_node_3" -> "chain0_node_3->C+E";
    "chain0_node_2->B+F" -> "B";
    "chain0_node_2->B+F" -> "F";
    "chain0_node_3->C+E" -> "C";
    "chain0_node_3->C+E" -> "E";
}
"""


def test_dot():

    a = BaseParticle("A")
    c = BaseParticle("C")
    b = BaseParticle("B")
    d = BaseParticle("D")
    f = BaseParticle("E")
    e = BaseParticle("F")
    r = BaseParticle("R")
    BaseDecay(r, [b, d])
    BaseDecay(a, [r, c])
    g = DotGenerator(a)
    chains_0 = a.chain_decay()[0]
    source_0 = g.get_dot_source()[0]
    assert source_0 == DotGenerator.dot_chain(chains_0, True)
    assert source_0 != DotGenerator.dot_chain(chains_0, False)
    chains = DecayChain.from_particles(a, [b, c, d, e, f])
    assert len(chains) == 105  # (2*5-3)!!


def remove_same(decs):
    ret = [decs[0]]
    for i in decs[1:]:
        for j in ret:
            if i.topology_same(j):
                break
        else:
            ret.append(i)
    return ret


def test_plot():
    final = [BaseParticle(i) for i in ["C", "D", "E", "B"]]
    decs = DecayChain.from_particles("A", final)
    decs = remove_same(decs)
    plt.figure(figsize=(15, 9))
    for i in range(len(decs)):
        ax = plt.subplot(3, 5, i + 1)
        plot_decay_struct(decs[i], ax)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig("topo_4.png", dpi=300, pad_inches=0)
