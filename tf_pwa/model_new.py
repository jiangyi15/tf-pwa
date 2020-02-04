from .data import data_shape, split_generator, data_merge, data_split
from .tensorflow_wrapper import tf


def loop_generator(var):
    while True:
        yield var


def sum_gradient(f, data, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    ys = []
    gs = []
    for data_i, weight_i in zip(data, weight):
        with tf.GradientTape() as tape:
            part_y = trans(f(data_i, *args, **kwargs))
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
        g_i = tape.gradient(y_i, var)
        ys.append(y_i)
        gs.append(g_i)
    return sum(ys), list(map(sum, zip(*gs)))


class Model(object):
    def __init__(self, amp, w_bkg=1.0):
        self.Amp = amp
        self.w_bkg = w_bkg

    def nll(self, data, mcdata, weight: tf.Tensor = 1.0, batch=None):
        r"""
        calculate negative log-likelihood

        .. math::
          -\ln L = -\sum_{x_i \in data } w_i \ln f(x_i;\theta_i) +  (\sum w_i ) \ln \sum_{x_i \in mc } f(x_i;\theta_i)

        """
        if isinstance(weight, float):
            sw = data_shape(data) * weight
        else:
            sw = tf.reduce_sum(weight)
        ln_data = tf.math.log(self.Amp(data))
        int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
        nll_0 = - tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
        return nll_0 + sw * int_mc

    def nll_grad(self, data, mcdata, weight=1.0, batch=65000):
        n_data = data_shape(data)
        n_mc = data_shape(mcdata)
        ln_data, g_ln_data = sum_gradient(self.Amp, split_generator(data, batch),
                                          self.Amp.trainable_variables, weight=weight, trans=tf.math.log)
        int_mc, g_int_mc = sum_gradient(self.Amp, split_generator(mcdata, batch),
                                        self.Amp.trainable_variables)
        sw = n_data
        g = list(map(lambda x: x[0] + sw * x[1] / int_mc, zip(g_ln_data, g_int_mc)))
        return - ln_data + sw * tf.math.log(int_mc / n_mc), g


class CachedModel(Model):
    def __init__(self, amp, data, mcdata, w_bkg, bg=None, batch=65000):
        super(CachedModel, self).__init__(amp, w_bkg)
        n_data = data_shape(data)
        n_mcdata = data_shape(mcdata)
        n_bg = 0
        if bg is not None:
            n_bg = data_shape(bg)
            data = data_merge(data, bg)
        self.weight = tf.convert_to_tensor([1.0]*n_data + [-w_bkg] * n_bg)
        self.sw = tf.reduce_sum(self.weight)
        self.data = data
        self.mcdata = mcdata
        self.batch = batch

    def cal_nll(self, params):
        self.Amp.set_params(params)
        nll = self.nll(self.data, self.mcdata, weight=self.weight)
        return nll

    def cal_nll_grad(self, params):
        self.Amp.set_params(params)
        nll, g = self.nll_grad(self.data, self.mcdata,
                               weight=data_split(self.weight, self.batch), batch=self.batch)
        return nll, g


class FCN(object):
    def __init__(self, cache_model):
        self.model = cache_model
        self.n_call = 0
        self.n_grad = 0
        self.cached_nll = None

    # @time_print
    def __call__(self, x):
        nll = self.model.cal_nll(x)
        self.cached_nll = nll
        self.n_call += 1
        return nll

    def grad(self, x):
        nll, g = self.model.cal_nll_grad(x)
        self.cached_nll = nll
        self.n_call += 1
        return g



