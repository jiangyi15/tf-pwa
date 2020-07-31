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
    """Spline interpolation function for model independent resonance"""
    def __init__(self, *args, **kwargs):
        self.points = None
        self.max_m = None
        self.min_m = None
        self.interp_N = None
        super(Interp1DSpline, self).__init__(*args, **kwargs)
        if self.points is None:
            dx = (self.max_m - self.min_m)/(self.interp_N - 1)
            self.points = [self.min_m + dx * i for i in range(self.interp_N)]
        else:
            self.interp_N = len(self.points)
        assert self.interp_N > 2, "points need large than 2"
        self.h_matrix = None

    def init_params(self):
        # self.a = self.add_var("a")
        self.point = self.add_var("point", is_complex=True, shape=(self.interp_N-2,))
        self.point.set_fix_idx(fix_idx=0, fix_vals= 1.0)
        h_matrix = spline_xi_matrix(self.points)
        self.h_matrix = tf.convert_to_tensor(h_matrix[...,1:-1])

    def get_amp(self, data, data_extra, *args, **kwargs):
        m = data["m"]
        zeros = tf.zeros_like(m)
        p = self.point()
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        xi_m = self.h_matrix
        x_m = spline_x_matrix(m, self.points)
        x_m = tf.expand_dims(x_m, axis=-1)
        m_xi = tf.reduce_sum(xi_m*x_m, axis=[-3, -2])
        ret_r = tf.reduce_sum(tf.cast(m_xi, p_r.dtype) * p_r, axis=-1)
        ret_i = tf.reduce_sum(tf.cast(m_xi, p_i.dtype) * p_i, axis=-1)
        return tf.complex(ret_r, ret_i)


def spline_x_matrix(x, xi):
    """build matrix of x for spline interpolation"""
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    x2 = x * x
    x3 = x2 * x
    def poly_i(i):
        cut = (x >= xi[i]) & (x<xi[i+1])
        x0_c = tf.where(cut, ones, zeros)
        x1_c = tf.where(cut, x, zeros)
        x2_c = tf.where(cut, x2, zeros)        
        x3_c = tf.where(cut, x3, zeros)
        return tf.stack([x0_c, x1_c, x2_c, x3_c], axis=-1)
    xs = [poly_i(i) for i in range(len(xi)-1)]
    return tf.stack(xs, axis=-2)


def spline_matrix(x, xi, yi):
    """calculate spline interpolation"""
    xi_m = spline_xi_matrix(xi) # (N_range, 4, N_yi)
    x_m = spline_x_matrix(x, xi) # (..., N_range, 4) 
    x_m = tf.expand_dims(x_m, axis=-1)
    m = tf.reduce_sum(xi_m*x_m, axis=[-3, -2])
    return tf.reduce_sum(tf.cast(m, yi.dtype) * yi, axis=-1)


def spline_xi_matrix(xi):
    """build matrix of xi for spline interpolation"""
    N = len(xi)
    hi = [xi[i+1] - xi[i] for i in range(N-1)]

    h_matrix = np.zeros((N, N))
    h_matrix[0,0] = 2 * hi[0]
    h_matrix[0,1] = hi[0]
    for i in range(1, N-1):
        h_matrix[i, i-1] = hi[i-1]
        h_matrix[i, i] = 2*(hi[i-1] + hi[i])
        h_matrix[i, i+1] = hi[i]
    h_matrix[-1, -2] = hi[-1]
    h_matrix[-1, -1] = 2*hi[-1]

    h_matrix_inv = np.linalg.inv(h_matrix)
    y_matrix = np.zeros((N, N))
    y_matrix[0,0] = 6 / hi[0]
    for i in range(1, N-1):
        y_matrix[i,i-1] = 6/hi[i-1]
        y_matrix[i,i] = -6*(1/hi[i] + 1/hi[i-1])
        y_matrix[i,i+1] = 6 / hi[i]
    y_matrix[-1,-1] = -6 / hi[-1]

    hy_matrix = np.dot(h_matrix_inv, y_matrix)

    hi = np.array(hi)[:,np.newaxis]
    I = np.eye(N)
    ci = hy_matrix[:-1]/2
    di = (hy_matrix[1:]-hy_matrix[:-1])/6/hi
    bi = (I[1:]- I[:-1])/hi - ci * hi - di * hi * hi
    ai = I[:-1]

    x1 = np.array(xi[:-1])[:,np.newaxis]
    x2 = x1 * x1
    x3 = x2 * x1
    ai_2 = ai - bi*x1 + ci * x2  - di * x3
    bi_2 = bi - 2*ci*x1 + 3 *di* x2
    ci_2 = ci - 3*di*x1
    di_2 = di
    ret= np.stack([ai_2, bi_2, ci_2, di_2], axis=-2)
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
