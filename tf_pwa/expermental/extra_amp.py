from tf_pwa.amp import regist_particle, Particle
from tf_pwa.tensorflow_wrapper import tf


@regist_particle("interp")
class Interp(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def __init__(self, *args, **kwargs):
        super(Interp, self).__init__(*args, **kwargs)
        dx = (self.max_m - self.min_m)/self.interp_N
        self.points = [(self.min_m + dx *i, self.min_m + dx * i +dx) for i in range(self.interp_N)]
        print(self.points)
    def init_params(self):
        # self.a = self.add_var("a")
        self.point = self.add_var("point",shape=(self.interp_N+1,))
        #self.point.set_fix_idx(fix_idx=self.interp_N//2, fix_vals= 1.0)

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        zeros = tf.zeros_like(m)
        p = tf.abs(self.point())
        def add_f(x, bl, br, pl, pr):
            return tf.where((x > bl)&(x<=br), (x-bl)/(br-bl)*(pr-pl)+pl, zeros)
        ret = [add_f(m, *self.points[i], p[i], p[i+1]) for i in range(self.interp_N)]
        return tf.complex(tf.reduce_sum(ret, axis=0), zeros)


@regist_particle("interp1dc")
class Interp1D(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def __init__(self, *args, **kwargs):
        super(Interp1D, self).__init__(*args, **kwargs)
        dx = (self.max_m - self.min_m)/self.interp_N
        self.points = [(self.min_m + dx * i - dx, self.min_m + dx *i, self.min_m + dx * i +dx) for i in range(self.interp_N)]
        print(self.points)
    def init_params(self):
        # self.a = self.add_var("a")
        self.point = self.add_var("point", is_complex=True, shape=(self.interp_N,))
        #self.point.set_fix_idx(fix_idx=self.interp_N//2, fix_vals= 1.0)

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        zeros = tf.zeros_like(m)
        p = self.point()
        def add_f(x, bl, bm, br):
            return tf.where((x > bl)&(x<=bm), (x-bl)/(bm-bl), zeros) + tf.where((x > bm)&(x<=br), (x-br)/(bm-br), zeros)
        ret = [add_f(m, *self.points[i]) for i in range(self.interp_N)]
        ret = tf.complex(ret, [zeros for i in range(self.interp_N)])
        return tf.reduce_sum(ret * tf.reshape(p, (-1,1)), axis=0)

