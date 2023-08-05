from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.data import HeavyCall


class BasePreProcessor(HeavyCall):
    def __init__(self, decay_config, **kwargs):
        self.decay_config = decay_config
        self.decay_struct = decay_config.decay_struct
        self.kwargs = kwargs

    def __call__(self, p4):
        return cal_angle_from_momentum(p4, self.decay_struct, **self.kwargs)
