import numpy as np

from tf_pwa.adaptive_bins import AdaptiveBound
from tf_pwa.adaptive_bins import cal_chi2 as cal_chi2_o
from tf_pwa.data import (
    data_index,
    data_merge,
    data_shape,
    data_split,
    data_to_numpy,
    load_data,
    save_data,
)

from .config_loader import ConfigLoader


@ConfigLoader.register_function()
def cal_bins_numbers(
    self, adapter, data, phsp, read_data, bg=None, bg_weight=None
):
    data_cut = read_data(data)
    # print(data_cut)
    amp_weight = self.get_amplitude()(phsp).numpy()
    phsp_cut = read_data(
        phsp
    )  # np.array([data_index(phsp, idx) for idx in data_idx])
    phsp_slice = np.concatenate([phsp_cut, [amp_weight]], axis=0)
    phsps = adapter.split_data(phsp_slice)
    datas = adapter.split_data(data_cut)
    bound = adapter.get_bounds()
    weight_scale = self.config["data"].get("weight_scale", False)
    if bg is not None:
        bg_cut = read_data(
            bg
        )  # np.array([data_index(bg, idx) for idx in data_idx])
        bgs = adapter.split_data(bg_cut)
        if weight_scale:
            int_norm = (
                data_cut.shape[-1] * (1 - bg_weight) / np.sum(amp_weight)
            )
        else:
            int_norm = (
                data_cut.shape[-1] - bg_cut.shape[-1] * bg_weight
            ) / np.sum(amp_weight)
    else:
        int_norm = data_cut.shape[-1] / np.sum(amp_weight)
    # print("int norm:", int_norm)
    ret = []
    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        ndata = datas[i].shape[-1]
        print(ndata)
        nmc = np.sum(phsps[i][2]) * int_norm
        if bg is not None:
            nmc += bgs[i].shape[-1] * bg_weight
        n_exp = nmc
        ret.append((ndata, nmc))
    return ret


@ConfigLoader.register_function()
def cal_chi2(self, read_data=None, bins=[[2, 2]] * 3, mass=["R_BD", "R_CD"]):
    if read_data is None:
        data_idx = [self.get_data_index("mass", i) for i in mass]
        read_data = lambda a: np.array(
            [data_index(a, idx) ** 2 for idx in data_idx]
        )
    data, bg, phsp, _ = self.get_all_data()
    bg_weight = self._get_bg_weight()[0]
    all_data = data_merge(*data)
    all_data_cut = read_data(
        all_data
    )  # np.array([data_index(a, idx) for idx in data_idx])
    adapter = AdaptiveBound(all_data_cut, bins)
    numbers = [
        self.cal_bins_numbers(adapter, i, j, read_data, k, l)
        for i, j, k, l in zip(data, phsp, bg, bg_weight)
    ]
    numbers = np.array(numbers)
    numbers = np.sum(numbers, axis=0)
    return cal_chi2_o(numbers, self.get_ndf())
