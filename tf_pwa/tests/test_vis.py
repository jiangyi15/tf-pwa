from tf_pwa.vis import *
from tf_pwa.particle import BaseParticle, BaseDecay, DecayChain


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
