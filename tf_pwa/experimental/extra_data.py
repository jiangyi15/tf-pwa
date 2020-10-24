from tf_pwa.config_loader.data import register_data_mode, SimpleData
import numpy as np
from tf_pwa.amp import get_particle
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.data import data_shape
import warnings


@register_data_mode("simple_npz")
class NpzData(SimpleData):
    def get_particle_p(self):
        order = self.dic.get("particle_p", None)
        if order is None:
            return self.get_dat_order()
        return order

    def get_data(self, idx) -> dict:
        if self.cached_data is not None:
            data = self.cached_data.get(idx, None)
            if data is not None:
                return data
        files = self.get_data_file(idx)
        weight_sign = self.get_weight_sign(idx)
        return self.load_data(files, weight_sign)

    def load_data(
        self, files, weights=None, weights_sign=1, charge=None
    ) -> dict:
        # print(files, weights)
        if files is None:
            return None
        order = self.get_dat_order()
        p_list = self.get_particle_p()
        center_mass = self.dic.get("center_mass", True)
        r_boost = self.dic.get("r_boost", False)
        random_z = self.dic.get("random_z", False)
        npz_data = np.load(files)
        p = {
            get_particle(str(v)): npz_data[str(k)]
            for k, v in zip(p_list, order)
        }
        data = cal_angle_from_momentum(
            p,
            self.decay_struct,
            center_mass=center_mass,
            r_boost=r_boost,
            random_z=random_z,
        )
        if "weight" in npz_data:
            data["weight"] = npz_data["weight"]
        if "charge_conjugation" in npz_data:
            data["charge_conjugation"] = npz_data["charge_conjugation"]
        else:
            data["charge_conjugation"] = np.ones((data_shape(data),))
        return data


@register_data_mode("multi_npz")
class MultiNpzData(NpzData):
    def __init__(self, *args, **kwargs):
        super(MultiNpzData, self).__init__(*args, **kwargs)
        self._Ngroup = 0

    def get_data(self, idx) -> list:
        if self.cached_data is not None:
            data = self.cached_data.get(idx, None)
            if data is not None:
                return data
        files = self.get_data_file(idx)
        if files is None:
            return None
        if not isinstance(files[0], list):
            files = [files]
        weight_sign = self.get_weight_sign(idx)
        ret = [self.load_data(i, weight_sign) for i in files]
        if self._Ngroup == 0:
            self._Ngroup = len(ret)
        elif idx != "phsp_noeff" and self._Ngroup != len(ret):
            warnings.warn("not the same data group")
        return ret

    def get_phsp_noeff(self):
        if "phsp_noeff" in self.dic:
            phsp_noeff = self.get_data("phsp_noeff")
            assert len(phsp_noeff) == 1
            return phsp_noeff[0]
        warnings.warn(
            "No data file as 'phsp_noeff', using the first 'phsp' file instead."
        )
        return self.get_data("phsp")[0]
