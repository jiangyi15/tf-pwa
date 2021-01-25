import math

import matplotlib.pyplot as plt

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


def get_node_layout(decay_chain):
    stl = decay_chain.sorted_table_layers()
    decay_map = {}
    max_branchs = max([len(i.outs) for i in decay_chain])
    ys = {}
    xs = {}
    for l, p in enumerate(stl[:0:-1]):
        if p is None:
            continue
        for k, v in p:
            ys[k] = ys.get(k, 0)
            xs[k] = l + 1
            for i in decay_chain:
                if i.core == k:
                    decay_map[k] = i
                    n = len(i.outs)
                    for j, m in enumerate(i.outs):
                        ys[m] = ys[k] + (-j - 0.5 + n / 2) / (
                            max_branchs ** (l + 1) + 2
                        )
                    break

    return xs, ys


def reorder_final_particle(decay_chain, ys):
    stl = decay_chain.sorted_table_layers()
    outs = sorted(decay_chain.outs, key=ys.get)

    ys_new = {k: i for i, k in enumerate(outs)}
    for p in stl[2:]:
        if p is None:
            continue
        for l, (k, v) in enumerate(p):
            ys_new[k] = sum(ys_new.get(i) for i in v) / len(v)
    return ys_new


def get_layout(decay_chain, xs, ys):
    stl = decay_chain.sorted_table_layers()
    points = {"__top": (0, ys[decay_chain.top])}
    for i in xs:
        points[i] = xs[i], ys[i]
    for i in decay_chain.outs:
        points[i] = len(stl) - 1, ys[i]
    lines = [("__top", decay_chain.top)]
    for i in decay_chain:
        for j in i.outs:
            lines.append((i.core, j))
    return lines, points


def get_decay_layout(decay_chain):
    xs, ys = get_node_layout(decay_chain)
    ys = reorder_final_particle(decay_chain, ys)
    lines, points = get_layout(decay_chain, xs, ys)
    return lines, points


def plot_decay_struct(decay_chain, ax=plt):
    lines, points = get_decay_layout(decay_chain)
    for a, b in lines:
        x, y = points[a]
        x2, y2 = points[b]
        ax.arrow(
            x,
            y,
            x2 - x,
            y2 - y,
            width=0.01,
            length_includes_head=True,
            label=str(b),
        )
        rotation = math.atan2(y2 - y, x2 - x) / math.pi * 180
        name = str(b)
        if (x2 - x) * (y2 - y) >= 0:
            ax.text(
                (x + x2) / 2,
                (y + y2) / 2 + 0.01,
                name,
                rotation=rotation,
                ha="center",
                va="bottom",
            )
        else:
            ax.text(
                (x + x2) / 2,
                (y + y2) / 2 + 0.01,
                name,
                rotation=rotation,
                ha="center",
                va="bottom",
            )
    ax.axis("off")
