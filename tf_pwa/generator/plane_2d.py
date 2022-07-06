import numpy as np
import sympy


def _get_solve_3():
    x = sympy.var("x")
    a1, a2, a3 = sympy.var("a1,a2,a3")
    x0 = sympy.var("x0")
    y = sympy.var("y")

    def f(x):
        return a1 * x + a2 * x**2 + a3 * x**3

    eq = f(x) - f(x0) - y
    solution = sympy.solve(eq, x)
    # print(solution[1])
    # print(sympy.cancel(sympy.simplify(solution[1])))

    return sympy.lambdify([a1, a2, a3, x0, y], solution, "numpy")


_solve_3 = _get_solve_3()


def _solve_2(a1, a2, x0, y):
    from numpy import sqrt

    return (
        -a1
        + sqrt(a1**2 + 4 * a1 * a2 * x0 + 4 * a2**2 * x0**2 + 4 * a2 * y)
    ) / (2 * a2)


def solve_2(a2, a1, x0, y):
    """
    solve (a2 x**2 + a1 x)|_{x0}^{x} = y
    """
    a2_0 = np.where(np.abs(a2) > 1e-8, a2, np.ones_like(a2))
    a1_0 = np.where(a1 != 0, a1, np.ones_like(a1))
    ret = np.where(
        np.abs(a2) > 1e-8,
        np.real(_solve_2(a1, a2_0, x0, y + 0.0j)),
        y / a1_0 + x0,
    )
    # print("ret", ret)
    return ret


def solve_3(a3, a2, a1, x0, x_max, y):
    """
    solve (a3 x**3 + a2 x**2 + a1 x**1)|_{x0}^{x} = y
    """
    with np.errstate(all="ignore"):
        s3 = _solve_3(a1, a2, a3, x0, y + 0.0j)
    s3 = [np.real(i) for i in s3]
    s3 = [np.where(np.isnan(i), np.inf * np.ones_like(i), i) for i in s3]
    s31 = np.where(
        (s3[0] >= x0) & (s3[0] < x_max),
        s3[0],
        np.where(
            (s3[1] >= x0) & (s3[1] < x_max),
            s3[1],
            np.where((s3[2] >= x0) & (s3[2] < x_max), s3[2], x0),
        ),
    )
    # print("so", s31, s3, x0, x_max)
    ret = np.where(np.abs(a3) > 1e-8, s31, solve_2(a2, a1, x0, y + 0.0j))
    # assert np.all(np.abs(np.imag(ret)) < 1e-7), ret
    # print("ret2", ret, a3, a2, a1, x0, y)
    ret = np.real(ret)
    np.where(np.isnan(ret), np.zeros_like(ret), ret)
    return ret


class TriangleGenerator:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.cal_coeffs()
        self.cal_1d_shape()

    def cal_coeffs(self):
        """
        z = a x  + b y + c

        [ x_1 , y_1 , 1 ] [a]   [z_1]
        [ x_2 , y_2 , 1 ] [b] = [z_2]
        [ x_3 , y_3 , 1 ] [c] = [z_3]

        z_c = a x + b y + c

        s^2 = (x-x_c)**2 + (y-y_c)**2
        s = 1/b sqrt(a**2+b**2)(x-x_c)
        x = b s / sqrt(a**2+b**2) + x_c

        """
        m = np.stack([self.x, self.y, np.ones_like(self.y)], axis=-1)
        self.coeff = np.sum(np.linalg.inv(m) * self.z[..., None, :], axis=-1)
        self.theta = np.arctan2(-self.coeff[..., 0], self.coeff[..., 1])
        self.rotation_matrix = np.stack(
            [
                np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1),
                np.stack([-np.sin(self.theta), np.cos(self.theta)], axis=-1),
            ],
            axis=-2,
        )
        self.inv_rm = np.linalg.inv(self.rotation_matrix)

        self.center_x = np.mean(self.x, axis=-1)
        self.center_y = np.mean(self.y, axis=-1)
        self.center_z = (
            self.center_x * self.coeff[..., 0]
            + self.center_y * self.coeff[..., 1]
            + self.coeff[..., 2]
        )
        st = np.stack(
            [self.cal_st_xy(self.x[..., i], self.y[..., i]) for i in range(3)],
            axis=-2,
        )
        # print(st, self.z)
        st_z = np.concatenate([st, self.z[..., None]], axis=-1)
        sort_index = np.argsort(st_z[..., 0], axis=-1)
        # print(sort_index)
        # print("st_z", st_z)
        # print(st_z[sort_index])
        st_z_shape = st_z.shape
        st_z = st_z.reshape((-1, 3, 3))
        st_z = np.stack([i[j] for i, j in zip(st_z, sort_index)]).reshape(
            st_z_shape
        )  #  np.sort(st_z, axis=-2)
        # print("st_z", st_z.reshape)
        self.st = st_z[..., :2]  # st[st_index]
        self.st_z = st_z[..., -1]  # self.z[st_index]
        self.s_min = self.st[..., 0, 0]
        self.s_max = self.st[..., -1, 0]
        self.t_min = np.min(st[..., :, 1])
        self.t_max = np.max(st[..., :, 1])

        # print(self.s_min, self.s_max)
        # print(self.t_min, self.t_max)
        # st = self.st.reshape((-1, 3, 2)).transpose((0,2,1))
        # print("st1", st)
        #  st = np.sort(st, axis=-1)
        # self.st = np.transpose(st, (0,2,1)).reshape(st_shape)

    def __call__(self, x, y, bin_index):
        coeff = self.coeff[bin_index]
        a = coeff[..., 0]
        b = coeff[..., 1]
        c = coeff[..., 2]
        return a * x + b * y + c

    def cal_st_xy(self, x, y, bin_index=slice(None)):
        a = np.stack([x, y], axis=-1)
        # print(self.rotation_matrix[bin_index].shape, a.shape)
        return np.einsum(
            "...ij,...j->...i", self.rotation_matrix[bin_index], a
        )

    def cal_xy_st(self, s, t, bin_index=slice(None)):
        a = np.stack([s, t], axis=-1)
        return np.einsum("...ij,...j->...i", self.inv_rm[bin_index], a)

    def cal_1d_shape(self):
        st = self.st
        self.s_2 = self.st[..., 1, 0]
        # y = (y_a - y_b)/(x_a -x_b) * (x-x_b) + y_b
        dom = st[..., 1, 0] - st[..., 0, 0]
        dom2 = np.where(np.abs(dom) < 1e-8, np.ones_like(dom), dom)
        self.k12 = np.where(
            np.abs(dom) < 1e-8,
            np.ones_like(dom),
            (st[..., 1, 1] - st[..., 0, 1]) / dom2,
        )
        dom = st[..., 2, 0] - st[..., 1, 0]
        dom2 = np.where(np.abs(dom) < 1e-8, np.ones_like(dom), dom)
        self.k23 = np.where(
            np.abs(dom) < 1e-8,
            np.ones_like(dom),
            (st[..., 2, 1] - st[..., 1, 1]) / dom2,
        )
        dom = st[..., 2, 0] - st[..., 0, 0]
        dom2 = np.where(np.abs(dom) < 1e-8, np.ones_like(dom), dom)
        self.k13 = np.where(
            np.abs(dom) < 1e-8,
            np.ones_like(dom),
            (st[..., 2, 1] - st[..., 0, 1]) / dom2,
        )
        self.b12 = -self.k12 * st[..., 1, 0] + st[..., 1, 1]
        self.b23 = -self.k23 * st[..., 1, 0] + st[..., 1, 1]
        self.b13 = -self.k13 * st[..., 0, 0] + st[..., 0, 1]
        self.k_plane = (
            self.inv_rm[..., 0, 1] * self.coeff[..., 0]
            + self.inv_rm[..., 1, 1] * self.coeff[..., 1]
        )
        self.b_plane = -self.k_plane * st[..., 0, 1] + self.st_z[..., 0]

        self.center_st = self.cal_st_xy(self.center_x, self.center_y)
        self.center_cond = self.center_st[..., 1] < self.st[..., 1, 1]

        self.cal_coeff_left()
        self.cal_coeff_right()
        self.int_all = self.int_left + self.int_right
        # print("int all", self.int_all, self.int_left , self.int_right)
        self.int_step = np.cumsum(self.int_all)
        # print(self.int_step)

    def cal_coeff_left(self):
        # \int_{s_1}^{s_2} | \int_{k13 s + b13}^{k12 s + b12} (k_{plane} t + b_{plane}) d t | d s = \int_{0}^{s_g} d s_g

        a_s3 = 1 / 6 * self.k_plane * (self.k12**2 - self.k13**2)
        a_s2 = (
            1
            / 2
            * (
                self.k_plane * (self.k12 * self.b12 - self.k13 * self.b13)
                + self.b_plane * (self.k12 - self.k13)
            )
        )
        a_s1 = 1 / 2 * self.k_plane * (
            self.b12**2 - self.b13**2
        ) + self.b_plane * (self.b12 - self.b13)

        s_2 = self.s_2
        int_left = (
            a_s3 * (s_2**3 - self.s_min**3)
            + a_s2 * (s_2**2 - self.s_min**2)
            + a_s1 * (self.st[..., 1, 0] - self.s_min)
        )
        # print(self.int_left)
        self.left_cond = int_left > 0
        self.int_left = np.abs(int_left)
        a_s3 = np.where(self.left_cond, a_s3, -a_s3)
        a_s2 = np.where(self.left_cond, a_s2, -a_s2)
        a_s1 = np.where(self.left_cond, a_s1, -a_s1)
        self.left_coeff = np.stack([a_s3, a_s2, a_s1], axis=-1)
        # print("int_left", self.int_left)
        # self.solve_left = lambda y: solve_3(a_s3, a_s2, a_s1, self.s_min, y)

    def solve_left(self, y, bin_index=slice(None)):
        a_si = self.left_coeff[bin_index]
        a_s3 = a_si[..., 0]
        a_s2 = a_si[..., 1]
        a_s1 = a_si[..., 2]
        # print("solve left", a_s3, a_s2, a_s1, self.s_min[bin_index], y)
        return solve_3(
            a_s3, a_s2, a_s1, self.s_min[bin_index], self.s_2[bin_index], y
        )

    def cal_coeff_right(self):
        # \int_{s_2}^{s_3} | \int_{k13 s + b13}^{k23 s + b23} (k_{plane} t + b_{plane}) d t | d s = \int_{s_{g,2}}^{s_g} d s_g
        # self.center_st = self.cal_st_xy(self.center_x[...,None], self.center_y[...,None])[...,0,:]

        a_s3 = 1 / 6 * self.k_plane * (self.k23**2 - self.k13**2)
        a_s2 = (
            1
            / 2
            * (
                self.k_plane * (self.k23 * self.b23 - self.k13 * self.b13)
                + self.b_plane * (self.k23 - self.k13)
            )
        )
        a_s1 = 1 / 2 * self.k_plane * (
            self.b23**2 - self.b13**2
        ) + self.b_plane * (self.b23 - self.b13)
        s_2 = self.s_2
        int_right = (
            a_s3 * (self.s_max**3 - s_2**3)
            + a_s2 * (self.s_max**2 - s_2**2)
            + a_s1 * (self.s_max - s_2)
        )
        self.right_cond = int_right > 0
        self.int_right = np.abs(int_right)
        a_s3 = np.where(self.right_cond, a_s3, -a_s3)
        a_s2 = np.where(self.right_cond, a_s2, -a_s2)
        a_s1 = np.where(self.right_cond, a_s1, -a_s1)
        self.right_coeff = np.stack([a_s3, a_s2, a_s1], axis=-1)

        # print("int_right", self.int_right)

    def solve_right(self, y, bin_index=slice(None)):
        a_si = self.right_coeff[bin_index]
        a_s3 = a_si[..., 0]
        a_s2 = a_si[..., 1]
        a_s1 = a_si[..., 2]
        s_2 = self.s_2[bin_index]
        # print("solve", a_s3, a_s2, a_s1, s_2, y - self.int_left[bin_index])
        return solve_3(
            a_s3,
            a_s2,
            a_s1,
            s_2,
            self.s_max[bin_index],
            y - self.int_left[bin_index],
        )

    def solve_s(self, s_r, bin_index=slice(None)):
        """
        int (k_1 s + b_1 )^2 - (k_2 s + b_2)^2 ds = int d s_r
        """
        s_r = s_r * (self.int_left[bin_index] + self.int_right[bin_index])
        # print(s_r, (self.int_left[bin_index], self.int_right[bin_index]))

        ret = np.where(
            s_r < self.int_left[bin_index],
            self.solve_left(s_r, bin_index),
            self.solve_right(s_r, bin_index),
        )
        # print("solve_ret", ret)
        return ret

    def t_min_max(self, s, bin_index=slice(None)):
        s_2 = self.s_2[bin_index]
        k12 = self.k12[bin_index]
        k23 = self.k23[bin_index]
        k13 = self.k13[bin_index]
        b12 = self.b12[bin_index]
        b23 = self.b23[bin_index]
        b13 = self.b13[bin_index]
        center_cond = self.center_cond[bin_index]

        k_2 = np.where(s < s_2, k12, k23)
        b_2 = np.where(s < s_2, b12, b23)
        t1_max = k_2 * s + b_2
        t1_min = k13 * s + b13
        t_max = np.where(t1_max > t1_min, t1_max, t1_min)
        t_min = np.where(t1_max > t1_min, t1_min, t1_max)
        return t_min, t_max

    def generate_st(self, N):
        int_range = np.random.random(N) * self.int_step[-1]
        bin_index = np.digitize(int_range, self.int_step[:-1])
        s_r = np.random.random(N)
        s = self.solve_s(s_r, bin_index)  #
        # s = np.zeros(N)
        # print("s", s)
        t_min, t_max = self.t_min_max(s, bin_index)
        y_max = 0.5 * self.k_plane[bin_index] * (
            t_max**2 - t_min**2
        ) + self.b_plane[bin_index] * (t_max - t_min)
        # print(self.k_plane[bin_index], self.b_plane[bin_index])
        # print("s", t_min, t_max, y_max)
        y = np.random.random(N) * y_max
        t = solve_2(
            0.5 * self.k_plane[bin_index], self.b_plane[bin_index], t_min, y
        )
        # print(s,t)
        return s, t, bin_index

    def generate(self, N):
        s, t, bin_index = self.generate_st(N)
        return self.cal_xy_st(s, t, bin_index)


class Interp2D(TriangleGenerator):
    def __init__(self, x, y, z):
        self.all_x = x
        self.all_y = y
        self.all_z = z
        left_x = x[:-1]
        right_x = x[1:]
        up_y = y[1:]
        down_y = y[:-1]
        center_x = (left_x + right_x) / 2
        center_y = (up_y + down_y) / 2
        triange_x = []
        triange_y = []
        triange_z = []
        for i in range(left_x.shape[0]):
            for j in range(up_y.shape[0]):
                triange_x.append(left_x[i])
                triange_y.append(down_y[j])
                triange_z.append(z[i][j])
                triange_x.append(right_x[i])
                triange_y.append(down_y[j])
                triange_z.append(z[i + 1][j])
                triange_x.append(center_x[i])
                triange_y.append(center_y[j])
                triange_z.append(
                    (z[i][j] + z[i + 1][j] + z[i][j + 1] + z[i + 1][j + 1]) / 4
                )
                triange_x.append(right_x[i])
                triange_y.append(down_y[j])
                triange_z.append(z[i + 1][j])
                triange_x.append(right_x[i])
                triange_y.append(up_y[j])
                triange_z.append(z[i + 1][j + 1])
                triange_x.append(center_x[i])
                triange_y.append(center_y[j])
                triange_z.append(
                    (z[i][j] + z[i + 1][j] + z[i][j + 1] + z[i + 1][j + 1]) / 4
                )
                triange_x.append(right_x[i])
                triange_y.append(up_y[j])
                triange_z.append(z[i + 1][j + 1])
                triange_x.append(left_x[i])
                triange_y.append(up_y[j])
                triange_z.append(z[i][j + 1])
                triange_x.append(center_x[i])
                triange_y.append(center_y[j])
                triange_z.append(
                    (z[i][j] + z[i + 1][j] + z[i][j + 1] + z[i + 1][j + 1]) / 4
                )
                triange_x.append(left_x[i])
                triange_y.append(up_y[j])
                triange_z.append(z[i][j + 1])
                triange_x.append(left_x[i])
                triange_y.append(down_y[j])
                triange_z.append(z[i][j])
                triange_x.append(center_x[i])
                triange_y.append(center_y[j])
                triange_z.append(
                    (z[i][j] + z[i + 1][j] + z[i][j + 1] + z[i + 1][j + 1]) / 4
                )
        triange_x = np.stack(triange_x).reshape((-1, 3))
        triange_y = np.stack(triange_y).reshape((-1, 3))
        triange_z = np.stack(triange_z).reshape((-1, 3))
        super().__init__(triange_x, triange_y, triange_z)

    def __call__(self, x, y):
        bin_x = np.digitize(x, self.all_x[1:-1])
        bin_y = np.digitize(y, self.all_y[1:-1])
        a = (x - self.all_x[bin_x]) / (
            self.all_x[bin_x + 1] - self.all_x[bin_x]
        )
        b = (y - self.all_y[bin_y]) / (
            self.all_y[bin_y + 1] - self.all_y[bin_y]
        )
        c1 = b > a
        c2 = ((1 - a) > b) ^ c1
        c = c1 * 2 + c2
        new_bin = bin_x * 4 * (self.all_y.shape[0] - 1) + bin_y * 4 + c
        return super().__call__(x, y, new_bin)
