from opt_einsum import get_symbol, contract_path, contract
from .tensorflow_wrapper import tf
import warnings
# from pysnooper import snoop


class Einsum(object):
    def __init__(self, expr, shapes):
        self.expr = expr
        self.shapes = shapes

    def __call__(self, *args):
        return contract(self.expr, *args, backend="tensorflow")


def symbol_generate(base_map):
    if isinstance(base_map, dict):
        base_map = base_map.values()
    for i in range(100):
        symbol = get_symbol(i)
        if symbol not in base_map:
            yield symbol


def replace_ellipsis(expr, shapes):
    ret = expr
    idx = expr.split("->")[0].split(",")[0]
    extra = []
    if "..." in expr:
        extra_size = len(shapes[0]) - len(idx) + 3
        base_map = set(expr) - {".", "-", ">", ","}
        base_map = dict(enumerate(base_map))
        ig = symbol_generate(base_map)
        for i in range(extra_size):
            extra.append(next(ig))
        ret = expr.replace("...", "".join(extra))
    return ret, extra


def _get_order_bound_list(bd_dict, ord_dict, idx, left=0):
    if idx in ord_dict:
        return [ord_dict[idx]]
    assert idx in bd_dict, "not found"
    bd = bd_dict[idx]
    od = []
    for i in bd[left]:
        if i in ord_dict:
            od.append(ord_dict[i])
        else:
            od += _get_order_bound_list(bd_dict, ord_dict, i, left)
    return od


def ordered_indices(expr, shapes):
    """
    find a better order to reduce transpose.

    """
    ein_s = expr.split("->")
    final_index = ein_s[1]
    idx_input = ein_s[0].split(",")
    base_order = dict(zip(final_index, range(len(final_index))))
    max_i = len(expr)
    base_order["_min"] = -1
    base_order["_max"] = max_i

    combined_index = set("".join(idx_input)) - set(final_index)
    bound_dict = {}
    for i in combined_index:
        bound_dict[i] = ([], [])

    for i in combined_index:
        for j in idx_input:
            if i in j:
                pos = j.index(i)
                if pos-1 >= 0:
                    bound_dict[i][0].append(j[pos-1])
                else:
                    bound_dict[i][0].append("_min")
                if pos+1 < len(j):
                    bound_dict[i][1].append(j[pos+1])
                else:
                    bound_dict[i][1].append("_max")

    for i in bound_dict:
        left = max(_get_order_bound_list(bound_dict, base_order, i, 0))
        right = min(_get_order_bound_list(bound_dict, base_order, i, 1))
        if right > left:
            base_order[i] = left * 0.4 + right * 0.6
        else:
            base_order[i] = left + 0.01

    base_order = dict(sorted(base_order.items(), key=lambda x: x[0]))
    return base_order


def remove_size1(expr, *args,extra=None):
    if extra is None:
        extra = []
    sub = expr.split("->")[0].split(",")

    size_map = {}
    for idx, shape in zip(sub, args):
        for i, j in zip(idx, shape.shape):
            l = size_map.get(i, 1)
            if j >= l:
                size_map[i] = j

    remove_idx = []
    for i in size_map:
        if size_map[i] == 1 and i not in extra:
            remove_idx.append(i)

    idxs = expr.split("->")[0].split(",")
    idxs2 = []
    ret = []
    for idx, arg in zip(idxs, args):
        shape = []
        idx2 = []
        for i, j  in zip(idx, arg.shape):
            if i not in remove_idx:
                shape.append(j)
                idx2.append(i)
        ret.append(tf.reshape(arg, shape))
        idxs2.append("".join(idx2))
    
    final_idx = expr.split("->")[1]
    for i in remove_idx:
        final_idx = final_idx.replace(i, "")
    expr2 = ",".join(idxs2)+"->"+final_idx
    
    return expr2, ret, size_map


def einsum(expr, *args, **kwargs):
    path, path_info = contract_path(expr, *args, optimize="auto")
    shapes = [i.shape for i in args]
    expr, extra = replace_ellipsis(expr, shapes)
    final_idx = expr.split("->")[1]
    expr2, args, size_map = remove_size1(expr, *args, extra=extra)
    final_shape = [size_map[i] for i in final_idx]
    base_order = ordered_indices(expr2, shapes)
    ein_s = expr2.split("->")
    final_index = ein_s[1]
    idxs = ein_s[0].split(",")

    data = list(args)
    in_idx = list(idxs)
    for idx in path:
        part_data = [data[i] for i in idx]
        part_in_idx = [in_idx[i] for i in idx]
        for i in sorted(idx)[::-1]:
            del data[i]
            del in_idx[i]
        out_idx = set("".join(part_in_idx)) & set(final_index + "".join(in_idx))
        out_idx = "".join(sorted(out_idx, key=lambda x: base_order[x]))
        in_idx.append(out_idx)
        expr_i = "{}->{}".format(",".join(part_in_idx), out_idx)
        result = tensor_einsum_reduce_sum(expr_i, *part_data, order=base_order)
        data.append(result)
    return tf.reshape(data[0], final_shape)


def tensor_einsum_reduce_sum(expr, *args, order):
    """
    "abe,bcf->acef"  =reshape=> "ab1e1,1bc1f->acef" =product=> "abcef->acef" =reduce_sum=> "acef"
    """
    ein_s = expr.split("->")
    final_index = ein_s[1]
    idxs = ein_s[0].split(",")
    for i in idxs:
        if len(set(i)) != len(i):  # inner product
            warnings.warn("inner product")
            return tf.einsum(expr, *args)

    require_order = sorted(set(ein_s[0]) - {","}, key=lambda x: order[x])

    # transpose
    t_args = []
    n_args = len(idxs)
    def args_it(it):
        i, j = idxs[it], args[it]
        sorted_idx = sorted(i, key=lambda x: order[x])
        if list(i) == sorted_idx:
            return j
        else:
            trans = [i.index(k) for k in sorted_idx]
            return  tf.transpose(j, trans)
    t_args = [args_it(it) for it in range(n_args)]
    # reshape
    sum_idx = set(require_order) - set(final_index)
    sum_idx_idx =[i for i, j in enumerate(require_order) if j in sum_idx]
    shapes = [i.shape for i in args]

    def expand_shape_it(idx, shape):
        shape_dict = dict(zip(idx, shape))
        ex_shape = [shape_dict.get(i, 1) for i in require_order]
        return ex_shape

    expand_shapes = [expand_shape_it(idx, shape) for idx, shape in zip(idxs, shapes)]

    s_args = [tf.reshape(j, i) for i, j in zip(expand_shapes, t_args)]

    # product
    ret_1 = s_args.pop()
    while len(s_args) > 0:
        ret_1 = ret_1 * s_args.pop()

    # reduce_sum
    ret = tf.reduce_sum(ret_1, axis=sum_idx_idx)
    return ret
