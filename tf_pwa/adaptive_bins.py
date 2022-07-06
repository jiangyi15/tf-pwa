"""
adaptive split data into bins.
"""

import functools

import numpy as np

from .data import data_index, data_mask

try:
    import matplotlib.patches as mpathes
except ImportError:
    pass


class AdaptiveBound(object):
    """adaptive bound cut for data value"""

    def __init__(self, base_data, bins):
        if isinstance(bins, int):
            self._base_data = np.array([base_data])
            self.bins = [[bins]]
        elif isinstance(bins, list):
            self._base_data = np.array(base_data)
            self.bins = bins
        else:
            raise TypeError(
                "bins should be int or list of int, "
                + "insteading of {}".format(type(bins))
            )

    @functools.lru_cache()
    def get_bounds_data(self):
        """get split data bounds, and the data after splitting"""
        base_bound = AdaptiveBound.base_bound(self._base_data)
        bounds, datas = AdaptiveBound.loop_split_bound(
            self._base_data, self.bins
        )
        return bounds, datas

    def get_bounds(self):
        """get split data bounds"""
        bounds, _ = self.get_bounds_data()
        return bounds

    def get_bool_mask(self, data):
        """bool mask for splitting data"""
        bounds = self.get_bounds()
        idx_data = data[: bounds[0][0].shape[0]]
        ret = []
        for lb, rb in bounds:
            mask = np.logical_and(
                idx_data >= lb[..., np.newaxis], idx_data < rb[..., np.newaxis]
            )
            mask = np.all(mask, axis=0)
            ret.append(mask)
        return ret

    def split_full_data(self, data, base_index=None):
        """split structure data, (TODO because large IO,  the method is slow.)"""
        base_data = [[data_index(data, i) for i in base_index]]
        mask = self.get_bool_mask(base_data)
        ret = []
        for i in mask:
            ret.append(data_mask(data, i))
        return ret

    def split_data(self, data):
        """split data, the shape is same as base_data"""
        mask = self.get_bool_mask(data)
        ret = []
        for i in mask:
            ret.append(data[..., i])
        return ret

    @staticmethod
    def single_split_bound(data, n=2, base_bound=None):
        """split data in the order of data value

        >>> data = np.array([1.0, 2.0, 1.4, 3.1])
        >>> AdaptiveBound.single_split_bound(data)
        [(1.0, 1.7...), (1.7..., 3.1...)]

        """
        if base_bound is None:
            base_bound = np.min(data), np.max(data) + 1e-6
        num_lb = base_bound[0]
        bounds = []
        for j in range(1, n):
            num_rb = np.percentile(data, j / n * 100, axis=0) + 1e-6
            bounds.append((num_lb, num_rb))
            num_lb = num_rb
        bounds.append((num_lb, base_bound[1]))
        return bounds

    @staticmethod
    def multi_split_bound(datas, n, base_bound=None):
        """multi data for single_split_bound, so `n` is list of int

        >>> data = np.array([[1.0, 2.0, 1.4, 3.1], [2.0, 1.0, 3.0, 1.0]])
        >>> bound, _ = AdaptiveBound.multi_split_bound(data, [2, 1])
        >>> [(i[0][0]+1e-6, i[1][0]+1e-6) for i in bound]
        [(1.0..., 1.7...), (1.7..., 3.1...)]

        """
        datas = np.array(datas)
        if base_bound is None:
            base_bound = AdaptiveBound.base_bound(datas)
        bound_chain = [base_bound]
        data_chain = [datas]
        for idx, size in enumerate(n):
            new_bound_chain = []
            new_data_chain = []
            for bnd, data in zip(bound_chain, data_chain):
                idx_data = data[idx]
                bounds = AdaptiveBound.single_split_bound(
                    idx_data, size, base_bound=(bnd[0][idx], bnd[1][idx])
                )
                for i in bounds:
                    lb, rb = i
                    l_bnd, r_bnd = bnd
                    l_bnd = np.copy(l_bnd)
                    r_bnd = np.copy(r_bnd)
                    l_bnd[idx] = lb
                    r_bnd[idx] = rb
                    new_bound_chain.append((l_bnd, r_bnd))
                    mask = np.logical_and(idx_data >= lb, idx_data < rb)
                    new_data_chain.append(data[:, mask])
            bound_chain = new_bound_chain
            data_chain = new_data_chain
        return bound_chain, data_chain

    @staticmethod
    def loop_split_bound(datas, n, base_bound=None):
        """loop for multi_split_bound, so `n` is list of list of int"""
        datas = np.array(datas)
        if base_bound is None:
            base_bound = AdaptiveBound.base_bound(datas)
        bound_chain = [base_bound]
        data_chain = [datas]
        for idx, size in enumerate(n):
            new_bound_chain = []
            new_data_chain = []
            for bnd, data in zip(bound_chain, data_chain):
                bound, data_i = AdaptiveBound.multi_split_bound(
                    data, size, base_bound=bnd
                )
                new_bound_chain += bound
                new_data_chain += data_i
            bound_chain = new_bound_chain
            data_chain = new_data_chain
        return bound_chain, data_chain

    @staticmethod
    def base_bound(data):
        """base bound for the data"""
        lb = np.min(data, axis=-1) - 1e-6
        rb = np.max(data, axis=-1) + 1e-6
        return (lb, rb)

    def get_bound_patch(self, **kwargs):
        ret = []
        for i, bnd in enumerate(self.get_bounds()):
            min_x, min_y = bnd[0]
            max_x, max_y = bnd[1]
            rect = mpathes.Rectangle(
                (min_x, min_y), max_x - min_x, max_y - min_y, **kwargs
            )  # cmap(weights[i]/max_weight))
            ret.append(rect)
        return ret

    def plot_bound(self, ax, **kwargs):
        for i in self.get_bound_patch(**kwargs):
            ax.add_patch(i)


def binning_shape_function(m, bins):
    adp = AdaptiveBound(m, bins)
    bnds, n = adp.get_bounds_data()
    x1 = [i[0][0] for i in bnds]
    x = np.array(x1 + [bnds[-1][1][0]])
    y1 = [ni.shape[0] / (x[i + 1] - x[i]) for i, ni in enumerate(n)]
    y2 = [(y1[i] + y1[i + 1]) / 2 for i in range(len(y1) - 1)]
    y = np.array([y1[0]] + y2 + [y1[-1]])
    return x, y


def adaptive_shape(m, bins, xmin, xmax):
    x, y = binning_shape_function(m, bins)
    cut = (x < xmax) & (x >= xmin)
    x = x[cut]
    y = y[cut]
    y[0] = y[0] * (x[1] - xmin) / (x[1] - x[0])
    y[-1] = y[-1] * (x[-2] - xmax) / (x[-2] - x[-1])
    x[0] = xmin
    x[-1] = xmax
    from tf_pwa.generator.linear_interpolation import LinearInterp

    return LinearInterp(x, y)


def cal_chi2(numbers, n_fp):
    weights = []
    # print(numbers)
    # chi21 = []
    for ndata, nmc in numbers:
        weight = (ndata - nmc) / np.sqrt(np.abs(ndata))
        weights.append(weight**2)
        # chi21.append(ndata * np.log(nmc))
    max_weight = np.max(weights)
    chi2 = np.sum(weights)
    print("bins: ", len(weights))
    print("number of free parameters: ", n_fp)
    ndf = len(weights) - 1 - n_fp
    print(
        "chi2/ndf: ", np.sum(weights), "/", ndf
    )  # ,"another", np.sum(chi21))
    return chi2, ndf
