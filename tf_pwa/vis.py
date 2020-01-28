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
        decay_dict = {}
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
