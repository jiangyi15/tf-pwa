import numpy as np

from tf_pwa.amp.core import Particle, register_particle
from tf_pwa.tensorflow_wrapper import tf

# pylint: disable=no-member


class InterpolationParticle(Particle):
    def __init__(self, *args, **kwargs):
        self.points = None
        self.max_m = None
        self.min_m = None
        self.interp_N = None
        self.polar = True
        self.fix_idx = -1
        self.with_bound = False
        super(InterpolationParticle, self).__init__(*args, **kwargs)
        self.fix_width = True
        if self.points is None:
            dx = (self.max_m - self.min_m) / (self.interp_N - 1)
            self.points = [self.min_m + dx * i for i in range(self.interp_N)]
        else:
            self.fix_width = False
            self.interp_N = len(self.points)
        self.bound = [
            (self.points[i], self.points[i + 1])
            for i in range(0, self.interp_N - 1)
        ]
        self.n_bins = len(self.bound)
        if self.fix_idx is not None and self.fix_idx < 0:
            self.fix_idx = self.interp_N // 2 - 1

    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var(
            "point",
            is_complex=True,
            shape=(self.n_points(),),
            polar=self.polar,
        )
        if self.fix_idx is not None:
            self.point_value.set_fix_idx(fix_idx=self.fix_idx, fix_vals=1.0)

    def get_amp(self, data, *args, **kwargs):
        m = data["m"]
        fm = self.interp(m)
        return fm

    def n_points(self):
        if self.with_bound:
            return self.interp_N
        return self.interp_N - 2

    def __call__(self, mass):
        return self.interp(mass)

    def interp(self, mass):
        raise NotImplementedError

    def get_point_values(self):
        p = self.point_value()
        if self.with_bound:
            v_r = [tf.math.real(i) for i in p]
            v_i = [tf.math.imag(i) for i in p]
        else:
            v_r = [0.0] + [tf.math.real(i) for i in p] + [0.0]
            v_i = [0.0] + [tf.math.imag(i) for i in p] + [0.0]
        return self.points, v_r, v_i

    def get_bin_index(self, m):
        if self.fix_width:
            m_min = tf.convert_to_tensor(self.points[0], m.dtype)
            m_max = tf.convert_to_tensor(self.points[-1], m.dtype)
            delta_width = (m_max - m_min) / (self.interp_N - 1)
            bin_idx = tf.histogram_fixed_width_bins(
                m,
                [m_min - delta_width, m_max + delta_width],
                nbins=self.interp_N + 1,
                dtype=tf.dtypes.int64,
            )
        else:
            # dig = lambda x, y: tf.numpy_function(np.digitize, [x, y], tf.int64)
            # bin_idx = dig(m, self.points)
            bin_idx = tf.raw_ops.Bucketize(input=m, boundaries=self.points)
        bin_idx = bin_idx - 1
        # print(tf.reduce_max(bin_idx), tf.reduce_min(bin_idx))
        bin_idx = tf.stop_gradient(bin_idx)
        return bin_idx


@register_particle("linear_npy")
class InterpLinearNpy(InterpolationParticle):
    def __init__(self, *args, **kwargs):
        self.input_file = kwargs.get("file")
        self.data = np.load(self.input_file)
        points = self.data[:, 0]
        kwargs["points"] = points
        super().__init__(*args, **kwargs)

    def init_params(self):
        pass

    def get_point_values(self):
        v_r = np.concatenate([[0.0], self.data[:, 1], [0.0]])
        v_i = np.concatenate([[0.0], self.data[:, 1], [0.0]])
        return self.data[:, 0], v_r, v_i

    def interp(self, m):
        x, p_r, p_i = self.get_point_values()
        bin_idx = tf.raw_ops.Bucketize(input=m, boundaries=x)
        bin_idx = (bin_idx + len(self.bound)) % len(self.bound)
        ret_r_l = tf.gather(p_r[1:], bin_idx)
        ret_i_l = tf.gather(p_r[1:], bin_idx)
        ret_r_r = tf.gather(p_r[:-1], bin_idx)
        ret_i_r = tf.gather(p_r[:-1], bin_idx)
        delta = np.concatenate(
            [[1.0], self.data[1:, 1] - self.data[:-1, 1], [1.0]]
        )
        x_left = np.concatenate([[x[0] - 1], x])
        delta = tf.gather(delta, bin_idx)
        x_left = tf.gather(x_left, bin_idx)
        step = (m - x_left) / delta
        a = step * (ret_r_l - ret_r_r)
        b = step * (ret_i_l - ret_i_r)
        return tf.complex(a, b)


@register_particle("interp")
class Interp(InterpolationParticle):
    """linear interpolation for real number"""

    def init_params(self):
        # self.a = self.add_var("a")
        self.point_value = self.add_var("point", shape=(self.interp_N + 1,))
        self.point_value.set_fix_idx(fix_idx=self.fix_idx, fix_vals=1.0)

    def interp(self, m):
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        zeros = tf.zeros_like(m)
        p = tf.abs(self.point_value())

        def add_f(x, bl, br, pl, pr):
            return tf.where(
                (x > bl) & (x <= br),
                (x - bl) / (br - bl) * (pr - pl) + pl,
                zeros,
            )

        ret = [
            add_f(m, self.points[i], self.points[i + 1], p[i], p[i + 1])
            for i in range(self.interp_N - 1)
        ]
        return tf.complex(tf.reduce_sum(ret, axis=0), zeros)


@register_particle("interp_c")
class Interp(InterpolationParticle):
    """linear interpolation for complex number"""

    def interp(self, m):
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        p = self.point_value()
        zeros = tf.zeros_like(m)
        ones = tf.ones_like(m)

        def poly_i(i, xi):
            tmp = zeros
            for j in range(i - 1, i + 1):
                if j < 0 or j > self.interp_N - 1:
                    continue
                r = ones
                for k in range(j, j + 2):
                    if k == i:
                        continue
                    r = r * (m - xi[k]) / (xi[i] - xi[k])
                r = tf.where((m >= xi[j]) & (m < xi[j + 1]), r, zeros)
                tmp = tmp + r
            return tmp

        h = tf.stack(
            [poly_i(i, self.points) for i in range(1, self.interp_N - 1)],
            axis=-1,
        )
        h = tf.stop_gradient(h)
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        ret_r = tf.reduce_sum(h * p_r, axis=-1)
        ret_i = tf.reduce_sum(h * p_i, axis=-1)
        return tf.complex(ret_r, ret_i)


@register_particle("spline_c")
class Interp1DSpline(InterpolationParticle):
    """Spline interpolation function for model independent resonance"""

    def __init__(self, *args, **kwargs):
        self.bc_type = "not-a-knot"
        super(Interp1DSpline, self).__init__(*args, **kwargs)
        assert self.interp_N > 2, "points need large than 2"
        self.h_matrix = None

    def init_params(self):
        super(Interp1DSpline, self).init_params()
        h_matrix = spline_xi_matrix(self.points, self.bc_type)
        if self.with_bound:
            self.h_matrix = tf.convert_to_tensor(h_matrix)
        else:
            self.h_matrix = tf.convert_to_tensor(h_matrix[..., 1:-1])

    def interp(self, m):
        zeros = tf.zeros_like(m)
        p = self.point_value()
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        xi_m = self.h_matrix
        x_m = spline_x_matrix(m, self.points)
        x_m = tf.expand_dims(x_m, axis=-1)
        m_xi = tf.reduce_sum(xi_m * x_m, axis=[-3, -2])
        m_xi = tf.stop_gradient(m_xi)
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
        cut = (x >= xi[i]) & (x < xi[i + 1])
        return tf.where(cut, x_p, zeros)

    xs = [poly_i(i) for i in range(len(xi) - 1)]
    return tf.stack(xs, axis=-2)


def spline_matrix(x, xi, yi, bc_type="not-a-knot"):
    """calculate spline interpolation"""
    xi_m = spline_xi_matrix(xi)  # (N_range, 4, N_yi)
    x_m = spline_x_matrix(x, xi)  # (..., N_range, 4)
    x_m = tf.expand_dims(x_m, axis=-1)
    m = tf.reduce_sum(xi_m * x_m, axis=[-3, -2])
    return tf.reduce_sum(tf.cast(m, yi.dtype) * yi, axis=-1)


def spline_xi_matrix(xi, bc_type="not-a-knot"):
    """build matrix of xi for spline interpolation
    solve equation

    .. math::
        S_i'(x_i) = S_{i-1}'(x_i)

    and two bound condition. :math:`S_0'(x_0) = S_{n-1}'(x_n) = 0`
    """
    N = len(xi)
    hi = [xi[i + 1] - xi[i] for i in range(N - 1)]

    h_matrix = np.zeros((N, N))
    if bc_type == "not-a-knot":
        h_matrix[0, 0] = -hi[1]
        h_matrix[0, 1] = hi[0] + hi[1]
        h_matrix[0, 2] = -hi[0]
    elif bc_type == "clamped":
        h_matrix[0, 0] = 2 * hi[0]
        h_matrix[0, 1] = hi[0]
    elif bc_type == "natural":
        h_matrix[0, 0] = 1
    else:
        raise ValueError("bc_type={} not in {not-a-knot,clamped,natural}")
    for i in range(1, N - 1):
        h_matrix[i, i - 1] = hi[i - 1]
        h_matrix[i, i] = 2 * (hi[i - 1] + hi[i])
        h_matrix[i, i + 1] = hi[i]
    if bc_type == "not-a-knot":
        h_matrix[-1, -3] = -hi[-1]
        h_matrix[-1, -2] = hi[-1] + hi[-2]
        h_matrix[-1, -1] = -hi[-2]
    elif bc_type == "clamped":
        h_matrix[-1, -2] = hi[-1]
        h_matrix[-1, -1] = 2 * hi[-1]
    elif bc_type == "natural":
        h_matrix[-1, -1] = 1
    h_matrix_inv = np.linalg.inv(h_matrix)
    y_matrix = np.zeros((N, N))
    if bc_type == "not-a-knot":
        y_matrix[0, 0] = 0  # 6 / hi[0]
    elif bc_type == "clamped":
        y_matrix[0, 0] = 6 / hi[0]
    elif bc_type == "natural":
        y_matrix[0, 0] = 0  # 6 / hi[0]
    for i in range(1, N - 1):
        y_matrix[i, i - 1] = 6 / hi[i - 1]
        y_matrix[i, i] = -6 * (1 / hi[i] + 1 / hi[i - 1])
        y_matrix[i, i + 1] = 6 / hi[i]
    if bc_type == "not-a-knot":
        y_matrix[-1, -1] = 0  # -6 / hi[-1]
    elif bc_type == "clamped":
        y_matrix[-1, -1] = -6 / hi[-1]
    elif bc_type == "natural":
        y_matrix[-1, -1] = 0  # -6 / hi[-1]

    hy_matrix = np.dot(h_matrix_inv, y_matrix)

    # Si(x) = ai + bi(x-xi) + ci(x-xi)^2 + di(x-xi)^3
    hi = np.array(hi)[:, np.newaxis]
    I = np.eye(N)
    ci = hy_matrix[:-1] / 2
    di = (hy_matrix[1:] - hy_matrix[:-1]) / 6 / hi
    bi = (I[1:] - I[:-1]) / hi - ci * hi - di * hi * hi
    ai = I[:-1]

    # Si(x) = ai + bi x + ci x^2 + di x^3
    x1 = np.array(xi[:-1])[:, np.newaxis]
    x2 = x1 * x1
    x3 = x2 * x1
    ai_2 = ai - bi * x1 + ci * x2 - di * x3
    bi_2 = bi - 2 * ci * x1 + 3 * di * x2
    ci_2 = ci - 3 * di * x1
    di_2 = di
    ret = np.stack([ai_2, bi_2, ci_2, di_2], axis=-2)
    return ret


@register_particle("interp1d3")
class Interp1D3(InterpolationParticle):
    """Piecewise third order interpolation"""

    def interp(self, m):
        p = self.point_value()
        ret = interp1d3(m, self.points, tf.stack(p))
        return ret


def interp1d3(x, xi, yi):
    h, b = get_matrix_interp1d3(x, xi)  # (..., N), (...,)
    ret = tf.reshape(
        tf.matmul(tf.cast(h, yi.dtype), tf.reshape(yi, (-1, 1))), b.shape
    ) + tf.cast(b, yi.dtype)
    return ret


def get_matrix_interp1d3(x, xi):
    N = len(xi) - 1
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    # @pysnooper.snoop()
    def poly_i(i):
        tmp = zeros
        for j in range(i - 1, i + 3):
            if j < 0 or j > N - 1:
                continue
            r = ones
            for k in range(j - 1, j + 3):
                if k == i or k < 0 or k > N:
                    continue
                r = r * (x - xi[k]) / (xi[i] - xi[k])
            r = tf.where((x >= xi[j]) & (x < xi[j + 1]), r, zeros)
            tmp = tmp + r
        return tmp

    h = tf.stack([poly_i(i) for i in range(1, N)], axis=-1)
    b = tf.zeros_like(x)
    return h, b


@register_particle("interp_lagrange")
class Interp1DLang(InterpolationParticle):
    """Lagrange interpolation"""

    def interp(self, m):
        zeros = tf.zeros_like(m)
        p = self.point_value()
        xs = []

        def poly_i(i):
            x = 1.0
            for j in range(self.interp_N):
                if i == j:
                    continue
                x = (
                    x
                    * (m - self.points[j])
                    / (self.points[i] - self.points[j])
                )
            return x

        xs = tf.stack([poly_i(i) for i in range(self.interp_N)], axis=-1)
        zeros = tf.zeros_like(xs)
        xs = tf.complex(xs, zeros)
        ret = tf.reduce_sum(xs[:, 1:-1] * p, axis=-1)
        return ret


@register_particle("interp_hist")
class InterpHist(InterpolationParticle):
    """Interpolation for each bins as constant"""

    def interp(self, m):
        p = self.point_value()
        ones = tf.ones_like(m)
        zeros = tf.zeros_like(m)

        def add_f(x, bl, br):
            return tf.where((x > bl) & (x <= br), ones, zeros)

        x_bin = tf.stack(
            [
                add_f(
                    m,
                    (self.points[i] + self.points[i + 1]) / 2,
                    (self.points[i + 1] + self.points[i + 2]) / 2,
                )
                for i in range(self.interp_N - 2)
            ],
            axis=-1,
        )
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        x_bin = tf.stop_gradient(x_bin)
        ret_r = tf.reduce_sum(x_bin * p_r, axis=-1)
        ret_i = tf.reduce_sum(x_bin * p_i, axis=-1)
        return tf.complex(ret_r, ret_i)


class HistParticle(InterpolationParticle):
    def n_points(self):
        return self.interp_N - 1


@register_particle("hist_idx")
class InterpHistIdx(HistParticle):
    """Interpolation for each bins as constant

    use

    .. code::

        min_m: 0.19
        max_m: 0.91
        interp_N: 8
        with_bound: True

    for mass range [0.19, 0.91] and 7 bins

    The first and last are fixed to zero unless set :code:`with_bound: True`.

    This is an example of :math:`k\exp (i k)` for point k.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> params = {}
        >>> for i in range(7):
        ...     params[f"R_BC_point_{i}r"] = i
        ...     params[f"R_BC_point_{i}i"] = i
        ...
        >>> axis = plot_particle_model("hist_idx", {"max_m": 0.91, "min_m": 0.19, "interp_N": 8}, params, special_points=np.linspace(0.19, 0.91, 8)[1:-1])

    """

    def interp(self, m):
        _, p_r, p_i = self.get_point_values()
        bin_idx = self.get_bin_index(m)
        bin_idx = (bin_idx + len(self.bound)) % len(self.bound)
        ret_r = tf.gather(p_r[1:], bin_idx)
        ret_i = tf.gather(p_i[1:], bin_idx)
        return tf.complex(ret_r, ret_i)


@register_particle("spline_c_idx")
class Interp1DSplineIdx(InterpolationParticle):
    """Spline function in index way.

    use

    .. code::

        min_m: 0.19
        max_m: 0.91
        interp_N: 8
        with_bound: True

    for mass range [0.19, 0.91] and 8 interpolation points

    The first and last are fixed to zero unless set :code:`with_bound: True`.

    This is an example of :math:`k\exp (i k)` for point k.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> params = {}
        >>> for i in range(8):
        ...     params[f"R_BC_point_{i}r"] = i
        ...     params[f"R_BC_point_{i}i"] = i
        ...
        >>> axis = plot_particle_model("spline_c_idx", {"max_m": 0.91, "min_m": 0.19, "interp_N": 8, "with_bound": True}, params, special_points=np.linspace(0.19, 0.91, 8)[1:-1])

    """

    def __init__(self, *args, **kwargs):
        self.bc_type = "not-a-knot"
        super().__init__(*args, **kwargs)
        assert self.interp_N > 2, "points need large than 2"
        self.h_matrix = None

    def init_params(self):
        super(Interp1DSplineIdx, self).init_params()
        h_matrix = spline_xi_matrix(self.points, self.bc_type)
        if self.with_bound:
            self.h_matrix = tf.convert_to_tensor(h_matrix.transpose((1, 0, 2)))
        else:
            self.h_matrix = tf.convert_to_tensor(
                h_matrix.transpose((1, 0, 2))[..., 1:-1]
            )

    def interp(self, m):
        p = self.point_value()
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        idx = self.get_bin_index(m)
        idx = tf.clip_by_value(idx, 0, self.h_matrix.shape[1] - 1)
        ret_r = do_spline_hmatrix(self.h_matrix, p_r, m, idx)
        ret_i = do_spline_hmatrix(self.h_matrix, p_i, m, idx)
        return tf.complex(ret_r, ret_i)


def do_spline_hmatrix(h_matrix, y, m, idx):
    ai, bi, ci, di = tf.unstack(tf.reduce_sum(h_matrix * y, axis=-1), axis=0)
    a, b, c, d = (
        tf.gather(ai, idx),
        tf.gather(bi, idx),
        tf.gather(ci, idx),
        tf.gather(di, idx),
    )
    ret = a + m * (b + m * (c + d * m))
    return ret


@register_particle("interp_l3")
class InterpL3(InterpolationParticle):
    def interp(self, m):
        p = self.point_value()
        ones = tf.ones_like(m)
        zeros = tf.zeros_like(m)
        p_r = tf.math.real(p)
        p_i = tf.math.imag(p)
        h, b = get_matrix_interp1d3_v2(m, self.points)
        h = tf.stop_gradient(h)
        f = lambda x: tf.reshape(
            tf.matmul(tf.cast(h, x.dtype), tf.reshape(x, (-1, 1))), b.shape
        ) + tf.cast(b, x.dtype)
        ret_r = f(p_r)
        ret_i = f(p_i)
        return tf.complex(ret_r, ret_i)


def get_matrix_interp1d3_v2(x, xi):
    N = len(xi) - 1
    zeros = tf.zeros_like(x)
    ones = tf.ones_like(x)
    # @pysnooper.snoop()
    def poly_i(i):
        tmp = zeros
        x_i = (xi[i] + xi[i - 1]) / 2
        for j in range(i - 1, i + 3):
            if j < 0 or j > N - 1:
                continue
            r = ones
            for k in range(j - 1, j + 3):
                if k == i or k < 1 or k > N:
                    continue
                x_k = (xi[k] + xi[k - 1]) / 2
                r = r * (x - x_k) / (x_i - x_k)
            r = tf.where(
                (x >= (xi[j] + xi[j - 1]) / 2) & (x < (xi[j] + xi[j + 1]) / 2),
                r,
                zeros,
            )
            tmp = tmp + r
        return tmp

    h = tf.stack([poly_i(i) for i in range(1, N)], axis=-1)
    b = tf.zeros_like(x)
    return h, b


@register_particle("sppchip")
class InterpSPPCHIP(InterpolationParticle):
    """
    Shape-Preserving Piecewise Cubic Hermite Interpolation Polynomial.
    It is monotonic in each interval.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> params = {}
        >>> for i in range(8):
        ...     params[f"R_BC_point_{i}r"] = i
        ...     params[f"R_BC_point_{i}i"] = i
        ...
        >>> axis = plot_particle_model("sppchip", {"max_m": 0.91, "min_m": 0.19, "interp_N": 8, "with_bound": True}, params, special_points=np.linspace(0.19, 0.91, 8)[1:-1])

    """

    def init_params(self):
        super().init_params()
        self.x_matrix = create_sppchip_matrix(self.points)

    def interp(self, m):
        p, p_r, p_i = self.get_point_values()
        idx = self.get_bin_index(m)
        ret_r = sppchip(m, p, p_r, idx, self.x_matrix)
        ret_i = sppchip(m, p, p_i, idx, self.x_matrix)
        return tf.complex(ret_r, ret_i)


def create_sppchip_matrix(points):
    """
    matrix to solve f(xi),f(xi+1),f'(xi),f'(xi+1)

    """
    x = np.array(points)
    a = x[:-1]
    b = x[1:]
    one = np.ones_like(a)
    zero = np.zeros_like(a)
    matrix = [
        [one, a, a**2, a**3],
        [one, b, b**2, b**3],
        [zero, one, 2 * a, 3 * a**2],
        [zero, one, 2 * b, 3 * b**2],
    ]
    matrix = np.stack(matrix)
    return np.linalg.inv(np.transpose(matrix, (2, 0, 1)))


def sppchip(m, xi, y, idx=None, matrix=None):
    """
    Shape-Preserving Piecewise Cubic Hermite Interpolation Polynomial.
    It is monotonic in each interval.

    >>> from scipy.interpolate import pchip_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed)-1e-12, num=100)
    >>> y = pchip_interpolate(x_observed, y_observed, x)
    >>> assert np.allclose(y, sppchip(x, x_observed, y_observed).numpy())

    """
    y = tf.stack(y)
    if idx is None:
        idx = tf.raw_ops.Bucketize(input=m, boundaries=list(xi)) - 1
    xi = tf.cast(tf.stack(xi), y.dtype)
    coeffs = sppchip_coeffs(xi, y, matrix)
    ai, bi, ci, di = tf.unstack(coeffs, axis=-1)
    a, b, c, d = (
        tf.gather(ai, idx),
        tf.gather(bi, idx),
        tf.gather(ci, idx),
        tf.gather(di, idx),
    )
    ret = a + m * (b + m * (c + d * m))
    return ret


def sppchip_coeffs(xi, y, matrix=None, eps=1e-12):
    if matrix is None:
        matrix = create_sppchip_matrix(xi)
    h = xi[1:] - xi[:-1]
    delta = (y[1:] - y[:-1]) / h
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    cond1 = tf.abs(delta) < eps
    cond2 = delta[:-1] * delta[1:] <= 0
    delta_wrap = tf.where(cond1, tf.ones_like(delta), delta)
    d = (w1 + w2) / (w1 / delta_wrap[:-1] + w2 / delta_wrap[1:])
    d = tf.where(cond2 | cond1[1:] | cond1[:-1], tf.zeros_like(d), d)

    def cond(p, q):
        d_tmp = ((2 * h[p] + h[q]) * delta[p] - h[p] * delta[q]) / (
            h[p] + h[q]
        )
        d_tmp = tf.where(
            d_tmp * delta[p] < 0,
            tf.zeros_like(d_tmp),
            tf.where(
                (delta[p] * delta[q] < 0)
                & (tf.abs(d_tmp) > 3 * tf.abs(delta[p])),
                2 * delta[p],
                d_tmp,
            ),
        )
        return d_tmp

    d1 = cond(0, 1)
    dn = cond(-1, -2)
    d = tf.concat([[d1], d, [dn]], axis=0)

    v = tf.stack([y[:-1], y[1:], d[:-1], d[1:]], axis=-1)

    coeffs = tf.linalg.matvec(matrix, v)
    return coeffs
