from .particle import split_particle_type


class DotGenerator:
    dot_head = """
digraph {
    rankdir=LR;
    node [shape=point];
    edge [arrowhead=none, labelfloat=true];
"""
    dot_tail = "}\n"
    dot_ranksame = "    {{ rank=same {} }};\n"
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
                    ret += DotGenerator.dot_label_edge.format(
                        i, decay_dict[j], j
                    )
                else:
                    ret += DotGenerator.dot_default_edge.format(
                        i, decay_dict[j]
                    )
            else:
                ret += DotGenerator.dot_default_edge.format(i, j)
        ret += DotGenerator.dot_tail
        return ret


def draw_decay_struct(decay_chain, show=False, **kwargs):
    from graphviz import Source

    a = DotGenerator.dot_chain(decay_chain)
    g = Source(a, **kwargs)
    if show:
        g.view()
    else:
        g.render()
