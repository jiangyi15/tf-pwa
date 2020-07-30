from tf_pwa.amp import regist_particle, Particle
from tf_pwa.tensorflow_wrapper import tf
import numpy as np

# pylint: disable=no-member

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


@regist_particle("spline_c")
class Interp1DSpline(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def __init__(self, *args, **kwargs):
        super(Interp1DSpline, self).__init__(*args, **kwargs)
        if not hasattr(self,"points"):
            dx = (self.max_m - self.min_m)/(self.interp_N - 1)
            self.points = [self.min_m + dx * i for i in range(self.interp_N)]
        else:
            self.interp_N = len(self.points)
        assert self.interp_N >=2, "points need large than 2"
        self.hi = [self.points[i+1] - self.points[i] for i in range(self.interp_N - 1)]
        self.h_matrix = np.zeros((self.interp_N, self.interp_N))
        self.h_matrix[0,0] = 2 * self.hi[0]
        self.h_matrix[0,1] = self.hi[0]
        for i in range(1, self.interp_N-1):
            self.h_matrix[i, i-1] = self.hi[i-1]
            self.h_matrix[i, i-1] = 2*(self.hi[i-1] + self.hi[i])
            self.h_matrix[i, i+1] = self.hi[i]
        self.h_matrix[-1, -2] = self.hi[-1]
        self.h_matrix[-1, -1] = 2*self.hi[-1]
        self.h_matrix_inv = np.linalg.inv(self.h_matrix)
        self.y_matrix = np.zeros((self.interp_N, self.interp_N-1))
        self.y_matrix[0,0] = 6 / self.hi[0]
        for i in range(1, self.interp_N-1):
            if i-2>0:
                self.y_matrix[i,i-2] = 6/self.hi[i-1]
            self.y_matrix[i,i-1] = 6*(1/self.hi[i] - 1/self.hi[i-1])
            if i < self.interp_N -2:
                self.y_matrix[i,i] = 6 / self.hi[i]
        self.y_matrix[-1,-1] = 6 / self.hi[-1]
        self.hy_matrix = np.dot(self.h_matrix_inv, self.y_matrix)

    def init_params(self):
        # self.a = self.add_var("a")
        self.point = self.add_var("point", is_complex=True, shape=(self.interp_N-2,))
        self.point.set_fix_idx(fix_idx=0, fix_vals= 1.0)

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        zeros = tf.zeros_like(m)
        p = self.point()
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        m0 = tf.ones_like(m)
        m1 = m
        m2 = m * m
        m3 = m2 * m
        ret =[]
        ai, bi, ci, di = [],[],[],[]
        for i in range(self.interp_N-1):
            ret.append(tf.where(m >= self.points[i] ,
                     tf.where(m < self.points[i+1],
                              ai[i] + bi[i]*m +ci[i]*m2+di[i]*m3,
                              zeros), zeros))
        return ret


@regist_particle("interp1d3")
class Interp1D3(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def __init__(self, *args, **kwargs):
        super(Interp1D3, self).__init__(*args, **kwargs)
        if not hasattr(self,"points"):
            dx = (self.max_m - self.min_m)/(self.interp_N+1)
            self.points = [self.min_m + dx * i + dx for i in range(self.interp_N)]
        else:
            self.interp_N = len(self.points)
        assert self.interp_N >=2, "points need large than 2"
        self.bound = [(self.min_m, self.points[0])]
        for i in range(1,self.interp_N):
            self.bound.append((self.points[i-1], self.points[i]))
        self.bound.append((self.points[-1], self.max_m))

    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var("point", is_complex=True, shape=(self.interp_N-2,))
        #self.point_value.set_fix_idx(fix_idx=0, fix_vals=(0.0,0.))
        self.point_value.set_fix_idx(fix_idx=1, fix_vals=(1.0,0.))
        #self.point_value.set_fix_idx(fix_idx=-1, fix_vals=(0.0,0.))

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        xi = [self.min_m] + self.points + [self.max_m]
        p = self.point_value()
        ret = interp1d3(m, self.points, tf.stack(p))
        return ret


def interp1d3(x, xi, yi):
    h, b = get_matrix_interp1d3(x, xi) # (..., N), (...,)
    ret = tf.reshape(tf.matmul(tf.cast(h, yi.dtype), tf.reshape(yi,(-1,1))),b.shape) + tf.cast(b, yi.dtype)
    return ret


def get_matrix_interp1d3(x, xi):
    N = len(xi) - 1
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    # @pysnooper.snoop()
    def poly_i(i):
        tmp = zeros
        for j in range(i-1, i+3):
            if j < 0 or j > N-1:
                continue
            r = ones
            for k in range(j-1, j+3):
                if k==i or k < 0 or k>N:
                    continue
                r = r * (x - xi[k])/(xi[i]-xi[k])
            r = tf.where((x >= xi[j]) & (x<xi[j+1]), r, zeros)
            tmp = tmp + r
        return tmp
    h = tf.stack([poly_i(i) for i in range(1, N)], axis=-1)
    b = tf.zeros_like(x)
    return h, b


@regist_particle("interp_lagrange")
class Interp1DLang(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def __init__(self, *args, **kwargs):
        super(Interp1DLang, self).__init__(*args, **kwargs)
        if not hasattr(self,"points"):
            dx = (self.max_m - self.min_m)/self.interp_N
            self.points = [self.min_m + dx * i + dx*0.5 for i in range(self.interp_N)]
        else:
            self.interp_N = len(self.points)
        assert self.interp_N >=2, "points need large than 2"

    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var("point", is_complex=True, shape=(self.interp_N,))
        self.point_value.set_fix_idx(fix_idx=0, fix_vals=(1.0,0.))

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        zeros = tf.zeros_like(m)
        p = self.point_value()
        xs = []
        def poly_i(i):
            x = 1.0
            for j in range(self.interp_N):
                if i==j:
                    continue
                x = x * (m - self.points[j])/(self.points[i] - self.points[j])
            return x
        xs = tf.stack([poly_i(i) for i in range(self.interp_N)], axis=-1)
        zeros = tf.zeros_like(xs)
        xs = tf.complex(xs, zeros)
        ret = tf.reduce_sum(xs * p, axis=-1)
        return ret
