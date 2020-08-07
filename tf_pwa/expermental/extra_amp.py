from tf_pwa.amp import regist_particle, Particle
from tf_pwa.tensorflow_wrapper import tf
import numpy as np

# pylint: disable=no-member


class InterpolationPartilce(Particle):
    def __init__(self, *args, **kwargs):
        self.points = None
        self.max_m = None
        self.min_m = None
        self.interp_N = None
        self.polar = True
        super(InterpolationPartilce, self).__init__(*args, **kwargs)
        if self.points is None:
            dx = (self.max_m - self.min_m)/(self.interp_N - 1)
            self.points = [self.min_m + dx * i for i in range(self.interp_N)]
        else:
            self.interp_N = len(self.points)
        self.bound = [(self.points[i], self.points[i+1]) for i in range(0,self.interp_N-1)]

    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var("point", is_complex=True, shape=(self.interp_N-2,), polar=self.polar)
        self.point_value.set_fix_idx(fix_idx=0, fix_vals= 1.0)

    def get_amp(self, data, *args, **kwargs):
        m = data["m"]
        fm = self.interp(m)
        return fm

    def __call__(self, mass):
        return self.interp(mass)

    def interp(self, mass):
        raise NotImplementedError

    def get_point_values(self):
        p = self.point_value()
        v_r = [0.0] + [tf.math.real(i) for i in p] + [0.0]
        v_i = [0.0] + [tf.math.real(i) for i in p] + [0.0]
        return self.points, v_r, v_i


@regist_particle("interp")
class Interp(InterpolationPartilce):
    """"""
    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var("point",shape=(self.interp_N+1,))
        self.point_value.set_fix_idx(fix_idx=0, fix_vals= 1.0)

    def interp(self, m):
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        zeros = tf.zeros_like(m)
        p = tf.abs(self.point())
        def add_f(x, bl, br, pl, pr):
            return tf.where((x > bl)&(x<=br), (x-bl)/(br-bl)*(pr-pl)+pl, zeros)
        ret = [add_f(m, *self.points[i], p[i], p[i+1]) for i in range(self.interp_N)]
        return tf.complex(tf.reduce_sum(ret, axis=0), zeros)


@regist_particle("spline_c")
class Interp1DSpline(InterpolationPartilce):
    """Spline interpolation function for model independent resonance"""
    def __init__(self, *args, **kwargs):
        super(Interp1DSpline, self).__init__(*args, **kwargs)
        assert self.interp_N > 2, "points need large than 2"
        self.h_matrix = None

    def init_params(self):
        super(Interp1DSpline, self).init_params()
        h_matrix = spline_xi_matrix(self.points)
        self.h_matrix = tf.convert_to_tensor(h_matrix[...,1:-1])

    def interp(self, m):
        zeros = tf.zeros_like(m)
        p = self.point_value()
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
    x2 = x * x
    x3 = x2 * x
    x_p = tf.stack([ones, x, x2, x3], axis=-1)
    x = tf.expand_dims(x, axis=-1)
    zeros = tf.zeros_like(x)
    def poly_i(i):
        cut = (x >= xi[i]) & (x<xi[i+1])
        return tf.where(cut, x_p, zeros)
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
    """build matrix of xi for spline interpolation
    solve equation

    .. math::
        S_i'(x_i) = S_{i-1}'(x_i)

    and two bound condition. :math:`S_0'(x_0) = S_{n-1}'(x_n) = 0`
    """
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

    # Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    hi = np.array(hi)[:,np.newaxis]
    I = np.eye(N)
    ci = hy_matrix[:-1]/2
    di = (hy_matrix[1:]-hy_matrix[:-1])/6/hi
    bi = (I[1:]- I[:-1])/hi - ci * hi - di * hi * hi
    ai = I[:-1]

    # Si(x) = ai + bi x + ci x^2 + di x^3
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
class Interp1D3(InterpolationPartilce):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def interp(self, m):
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
class Interp1DLang(InterpolationPartilce):
    """example Particle model define, can be used in config.yml as `model: interp_lagrange`"""

    def interp(self, m):
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
