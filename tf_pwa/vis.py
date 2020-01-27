from .particle import BaseParticle, BaseDecay, split_particle_type

class DotGenerator():
    dot_head = """
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none];
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
    def dot_chain(chains):
        ret = DotGenerator.dot_head
        top, inner, outs = split_particle_type(chains)
        def format_particle(ps):
            s = ['"{}"'.format(i) for i in ps]
            return ",".join(s)
        decay_dict = {}
        for i in top:
            ret += DotGenerator.dot_default_node.format(i)
        for i in outs:
            ret += DotGenerator.dot_default_node.format(i)
        for i in inner:
            assert len(i.creators) == 1, ""
            decay_dict[i] = i.creators[0]
        ret += DotGenerator.dot_ranksame.format(format_particle(top))
        ret += DotGenerator.dot_ranksame.format(format_particle(outs))

        for i in chains:
            if i.core in decay_dict:
                ret += DotGenerator.dot_label_edge.format(decay_dict[i.core], i, i.core)
            else:
                ret += DotGenerator.dot_default_edge.format(i.core, i)
            for j in i.outs:
                if j not in decay_dict:
                    ret += DotGenerator.dot_default_edge.format(i, j)
        ret += DotGenerator.dot_tail
        return ret

def test_dot():
    a = BaseParticle("A")
    c = BaseParticle("C")
    b = BaseParticle("B")
    d = BaseParticle("D")
    r = BaseParticle("R")
    BaseDecay(r, [b, d])
    BaseDecay(a, [r, c])
    g = DotGenerator(a)
    return g.get_dot_source()
