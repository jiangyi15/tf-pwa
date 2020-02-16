from .particle import split_particle_type

class DotGenerator():
    dot_head = """
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none, labelfloat=true];
"""
    dot_tail = "}\n"
    dot_ranksame = '    {{ rank=same {} }};\n'
    dot_default_node = '    "{}" [shape=none];\n'
    dot_default_edge = '    "{}" -> "{}";\n'
    dot_label_edge = '    "{}" -> "{}" [label="{}"];\n'

    def __init__(self, top):
        self.top = top

    def get_dot_source(self):
        chains = self.top.chain_decay()
        ret = []
        for i in chains:
            dot_source = self.dot_chain(i)
            ret.append(dot_source)
        return ret

    @staticmethod
    def dot_chain(chains, has_label=True):
        ret = DotGenerator.dot_head
        top, _, outs = split_particle_type(chains)
        def format_particle(ps):
            s = ['"{}"'.format(i) for i in ps]
            return ",".join(s)

        for i in top:
            ret += DotGenerator.dot_default_node.format(i)
        for i in outs:
            ret += DotGenerator.dot_default_node.format(i)
        ret += DotGenerator.dot_ranksame.format(format_particle(top))
        ret += DotGenerator.dot_ranksame.format(format_particle(outs))

        decay_dict = {}
        edges = []
        for i in chains:
            if i.core in top:
                edges.append((i.core, i))
            else:
                decay_dict[i.core] = i
            for j in i.outs:
                edges.append((i, j))

        for i, j in edges:
            if j in decay_dict:
                if has_label:
                    ret += DotGenerator.dot_label_edge.format(i, decay_dict[j], j)
                else:
                    ret += DotGenerator.dot_default_edge.format(i, decay_dict[j])
            else:
                ret += DotGenerator.dot_default_edge.format(i, j)
        ret += DotGenerator.dot_tail
        return ret

def test_dot():
    from .particle import BaseParticle, BaseDecay, DecayChain
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
    chains = DecayChain.from_particles(a, [b, c, d, e, f])
    print(DotGenerator.dot_chain(chains[0], False))
    return g.get_dot_source()

'''
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none, labelfloat=true];
    "A" [shape=none];
    "B" [shape=none];
    "D" [shape=none];
    "E" [shape=none];
    "F" [shape=none];
    "C" [shape=none];
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

['\ndigraph {\n    rankdir=LR;\n    node [shape=point];\n    edge [arrowhead=none, labelfloat=true];\n    "A" [shape=non
e];\n    "B" [shape=none];\n    "D" [shape=none];\n    "C" [shape=none];\n    { rank=same "A" };\n    { rank=same "B","D
","C" };\n    "A" -> "A->R+C";\n    "A->R+C" -> "R->B+D" [label="R"];\n    "A->R+C" -> "C";\n    "R->B+D" -> "B";\n    "
R->B+D" -> "D";\n}\n']
'''