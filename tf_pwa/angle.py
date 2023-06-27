"""
This module implements three classes **Vector3**, **LorentzVector**, **EulerAngle** .
"""
from .tensorflow_wrapper import numpy_cross, tf

_epsilon = 1.0e-14


# import functools
# from pysnooper import snoop


class Vector3(tf.Tensor):
    """
    This class provides methods for 3-d vectors (X,Y,Z)
    """

    def get_X(self):
        return self[..., 0]

    def get_Y(self):
        return self[..., 1]

    def get_Z(self):
        return self[..., 2]

    def norm2(self):
        """
        The norm square
        """
        return tf.reduce_sum(self * self, axis=-1)

    def norm(self):
        return tf.norm(self, axis=-1)

    def dot(self, other):
        """
        Dot product with another Vector3 object
        """
        ret = tf.reduce_sum(self * other, axis=-1)
        return ret

    def cross(self, other):
        """
        Cross product with another Vector3 instance
        """
        p = numpy_cross(self, other)
        return p

    def unit(self):
        """
        The unit vector of itself. It has interface to *tf.linalg.normalize()*.
        """
        p, _n = tf.linalg.normalize(self, axis=-1)
        return p

    def cross_unit(self, other):
        """
        The unit vector of the cross product with another Vector3 object. It has interface to *tf.linalg.normalize()*.
        """
        cro = numpy_cross(self, other)
        norm_cro = tf.expand_dims(tf.norm(cro, axis=-1), -1)
        mask = norm_cro < _epsilon
        bias_other = tf.ones_like(norm_cro) + other
        cro = tf.where(mask, numpy_cross(self, bias_other), cro)
        p, _n = tf.linalg.normalize(cro, axis=-1)
        return p

    def angle_from(self, x, y):
        """
        The angle from x-axis providing the x,y axis to define a 3-d coordinate.

        :param x: A Vector3 instance as x-axis
        :param y: A Vector3 instance as y-axis. It should be perpendicular to the x-axis.
        """
        return tf.math.atan2(Vector3.dot(self, y), Vector3.dot(self, x))

    def cos_theta(self, other):
        """
        cos theta of included angle
        """
        d = Vector3.dot(self, other)
        return d / Vector3.norm(self) / Vector3.norm(other)


class LorentzVector(tf.Tensor):
    """
    This class provides methods for Lorentz vectors (T,X,Y,Z). or -T???
    """

    @staticmethod
    def from_p4(p_0, p_1, p_2, p_3):
        """
        Given **p_0** is a real number, it will make it transform into the same shape with **p_1**.
        """
        zeros = tf.zeros_like(p_1)
        return tf.stack([p_0 + zeros, p_1, p_2, p_3], axis=-1)

    def get_X(self):
        return self[..., 1]

    def get_Y(self):
        return self[..., 2]

    def get_Z(self):
        return self[..., 3]

    def get_T(self):
        return self[..., 0]

    def get_e(self):
        """rm???"""
        return self[..., 0]

    def boost_vector(self):
        """
        :math:`\\beta=(X,Y,Z)/T`
        :return: 3-d vector :math:`\\beta`
        """
        return self[..., 1:4] / self[..., 0:1]

    def vect(self):
        """
        It returns the 3-d vector (X,Y,Z).
        """
        return self[..., 1:4]

    def rest_vector(self, other):
        """
        Boost another Lorentz vector into the rest frame of :math:`\\beta`.
        """
        p = -LorentzVector.boost_vector(self)
        ret = LorentzVector.boost(other, p)
        return ret

    def boost(self, p):
        """
        Boost this Lorentz vector into the frame indicated by the 3-d vector p.
        """
        # pb = Vector3(p)
        pb = p
        beta2 = Vector3.norm2(pb)
        gamma = 1.0 / tf.sqrt(1 - beta2)
        bp = Vector3.dot(pb, LorentzVector.vect(self))
        gamma2 = tf.where(beta2 > _epsilon, (gamma - 1.0) / beta2, 0.0)
        p_r = LorentzVector.vect(self)
        p_r += tf.expand_dims(gamma2 * bp, axis=-1) * pb
        p_r += tf.expand_dims(gamma * LorentzVector.get_T(self), axis=-1) * pb
        T_r = tf.expand_dims(gamma * (LorentzVector.get_T(self) + bp), axis=-1)
        ret = tf.concat([T_r, p_r], -1)
        return ret

    def boost_matrix(self):
        pb = LorentzVector.boost_vector(self)
        beta2 = Vector3.norm2(pb)
        gamma = 1.0 / tf.sqrt(1 - beta2)
        gamma2 = tf.where(beta2 > _epsilon, (gamma - 1.0) / beta2, 0.0)

        # bp = pb_i v_i
        # p_r_i = v_i
        # p_r_i += gamma2 * bp * pb_i
        # p_r_i += gamma * E * pb_i
        # T_r = gamma * (E + bp)

        # T_r = gamma * (E + pb_i vi)
        # p_r_i = v_i + gamma2 pb_j v_j pb_i + gamma E pb_i
        # [ T ] = [ gamma       |      gamma pb_x        |    gamma pb_y         |    gamma pb_z       ]
        # [ x ] = [ gamma pb_x  |  1 + gamma2 pb_x pb_x  |    gamma2 pb_x pb_y   |    gamma2 pb_x pb_z ]
        # [ y ] = [ gamma pb_y  |      gamma2 pb_y pb_x  |  1+gamma2 pb_y pb_y   |    gamma2 pb_y pb_z ]
        # [ z ] = [ gamma pb_z  |      gamma2 pb_z pb_x  |    gamma2 pb_z pb_y   |  1+gamma2 pb_z pb_z ]
        ret00 = gamma
        ret0x = tf.expand_dims(gamma, axis=-1) * pb
        retx0 = tf.expand_dims(gamma, axis=-1) * pb
        retxx = tf.eye(3, dtype=pb.dtype) + tf.expand_dims(
            tf.expand_dims(gamma2, axis=-1) * pb, axis=-1
        ) * tf.expand_dims(pb, axis=-2)

        ret0 = tf.concat([tf.expand_dims(gamma, axis=-1), ret0x], axis=-1)
        # print(ret0)
        retx = tf.concat([tf.expand_dims(retx0, axis=-2), retxx], axis=-2)
        # print(retx)
        ret = tf.concat([tf.expand_dims(ret0, axis=-1), retx], axis=-1)
        # print(ret)
        return ret

    def gamma(self):
        pb = LorentzVector.boost_vector(self)
        beta2 = Vector3.norm2(pb)
        beta2 = tf.where(beta2 < 1, beta2, tf.zeros_like(beta2))
        gamma = 1.0 / tf.sqrt(1 - beta2)
        return gamma

    def beta(self):
        pb = LorentzVector.boost_vector(self)
        beta = Vector3.norm(pb)
        return beta

    def omega(self):
        gamma = LorentzVector.gamma(self)
        omega = tf.acosh(gamma)
        return omega

    def get_metric(self):
        """
        The metric is (1,-1,-1,-1) by default
        """
        return tf.cast(tf.constant([1.0, -1.0, -1.0, -1.0]), self.dtype)

    def M2(self):
        """
        The invariant mass squared
        """
        s = self * self * LorentzVector.get_metric(self)
        return tf.reduce_sum(s, axis=-1)

    def M(self):
        """
        The invariant mass
        """
        m2 = LorentzVector.M2(self)
        return tf.sqrt(tf.abs(m2))

    def neg(self):
        """
        The negative vector
        """
        return tf.concat([self[..., 0:1], -self[..., 1:4]], axis=-1)


class EulerAngle(dict):
    """
    This class provides methods for Eular angle :math:`(\\alpha,\\beta,\\gamma)`
    """

    def __init__(self, alpha=0.0, beta=0.0, gamma=0.0):
        super(EulerAngle, self).__init__()
        self["alpha"] = alpha
        self["beta"] = beta
        self["gamma"] = gamma

    @staticmethod
    def angle_zx_zx(z1, x1, z2, x2):
        """
        The Euler angle from coordinate 1 to coordinate 2 (right-hand coordinates).

        :param z1: Vector3 z-axis of the initial coordinate
        :param x1: Vector3 x-axis of the initial coordinate
        :param z2: Vector3 z-axis of the final coordinate
        :param x2: Vector3 x-axis of the final coordinate
        :return: Euler Angle object
        """
        u_z1 = Vector3.unit(z1)
        u_z2 = Vector3.unit(z2)
        u_y1 = Vector3.cross_unit(z1, x1)
        u_x1 = Vector3.cross_unit(u_y1, z1)
        u_yr = Vector3.cross_unit(z1, z2)
        u_xr = Vector3.cross_unit(u_yr, z1)
        u_y2 = Vector3.cross_unit(z2, x2)
        u_x2 = Vector3.cross_unit(u_y2, z2)
        # np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
        alpha = Vector3.angle_from(u_xr, u_x1, u_y1)
        # np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
        beta = Vector3.angle_from(u_z2, u_z1, u_xr)
        # np.arctan2(u_xr.Dot(u_y2),u_xr.Dot(u_x2))
        gamma = -Vector3.angle_from(u_yr, u_y2, -u_x2)
        return EulerAngle(alpha, beta, gamma)

    @staticmethod
    # @pysnooper.snoop()
    def angle_zx_z_getx(z1, x1, z2):
        """
        The Eular angle from coordinate 1 to coordinate 2. Only the z-axis is provided for coordinate 2, so
        :math:`\\gamma` is set to be 0.

        :param z1: Vector3 z-axis of the initial coordinate
        :param x1: Vector3 x-axis of the initial coordinate
        :param z2: Vector3 z-axis of the final coordinate
        :return eular_angle: EularAngle object with :math:`\\gamma=0`.
        :return x2: Vector3 object, which is the x-axis of the final coordinate when :math:`\\gamma=0`.
        """
        u_z1 = Vector3.unit(z1)
        u_z2 = Vector3.unit(z2)
        u_y1 = Vector3.cross_unit(z1, x1)
        u_x1 = Vector3.cross_unit(u_y1, z1)
        u_yr = Vector3.cross_unit(z1, z2)
        u_xr = Vector3.cross_unit(u_yr, z1)
        # np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
        alpha = Vector3.angle_from(u_xr, u_x1, u_y1)
        # np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
        beta = Vector3.angle_from(u_z2, u_z1, u_xr)
        gamma = tf.zeros_like(beta)
        u_x2 = Vector3.cross_unit(u_yr, u_z2)
        return (EulerAngle(alpha, beta, gamma), u_x2)

    @staticmethod
    def angle_zx_zzz_getx(z, x, zi):
        """
        The Eular angle from coordinate 1 to coordinate 2.
        Z-axis of coordinate 2 is the normal vector of a plane.

        :param z1: Vector3 z-axis of the initial coordinate
        :param x1: Vector3 x-axis of the initial coordinate
        :param z: list of Vector3 of the plane point.
        :return eular_angle: EularAngle object.
        :return x2: list of Vector3 object, which is the x-axis of the final coordinate in zi.
        """
        z1, z2, z3 = zi
        zz = Vector3.cross_unit(z1 - z2, z2 - z3)
        xi = [Vector3.cross_unit(i, zz) for i in zi]
        ang = EulerAngle.angle_zx_zx(z, x, zz, xi[2])
        return ang, xi


class SU2M(dict):
    def __init__(self, x):
        self["x"] = x

    @staticmethod
    def Boost_z(omega):
        zeros = tf.zeros_like(omega)
        a = tf.complex(tf.exp(omega / 2), zeros)
        zeros_c = tf.complex(zeros, zeros)
        ret = SU2M([[1 / a, zeros_c], [zeros_c, a]])
        return ret

    @staticmethod
    def Boost_z_from_p(p):
        omega = LorentzVector.omega(p)
        return SU2M.Boost_z(omega)

    @staticmethod
    def Rotation_z(alpha):
        zeros = tf.zeros_like(alpha)
        zeros_c = tf.complex(zeros, zeros)
        a = tf.exp(tf.complex(zeros, alpha / 2))
        return SU2M([[1 / a, zeros_c], [zeros_c, a]])

    @staticmethod
    def Rotation_y(beta):
        zeros = tf.zeros_like(beta)
        s = tf.complex(tf.sin(beta / 2), zeros)
        c = tf.complex(tf.cos(beta / 2), zeros)
        return SU2M([[c, -s], [s, c]])

    def __mul__(self, other):
        x = self["x"]
        y = other["x"]
        aa = x[0][0] * y[0][0] + x[0][1] * y[1][0]
        ab = x[0][0] * y[0][1] + x[0][1] * y[1][1]
        ba = x[1][0] * y[0][0] + x[1][1] * y[1][0]
        bb = x[1][0] * y[0][1] + x[1][1] * y[1][1]
        return SU2M([[aa, ab], [ba, bb]])

    def inv(self):
        aa, ab = self["x"][0]
        ba, bb = self["x"][1]
        return SU2M([[bb, -ab], [-ba, aa]])

    def get_euler_angle(self):
        x = self["x"]
        cosbeta = tf.math.real(x[0][0] * x[1][1] + x[0][1] * x[1][0])
        cosbeta = tf.clip_by_value(cosbeta, -1, 1)
        zeros = tf.zeros_like(cosbeta)
        beta = tf.math.acos(cosbeta)
        m_1 = tf.abs(x[0][0])
        m_2 = tf.abs(x[1][0])
        # alpha_p_gamma = tf.math.imag(
        #     tf.math.log(x[1][1] / tf.complex(m_1, zeros))
        # )
        alpha_p_gamma = tf.math.angle(x[1][1])
        alpha_m_gamma = -tf.math.angle(x[1][0])
        # -tf.math.imag(
        #    tf.math.log(x[1][0] / tf.complex(m_2, zeros))
        # )
        alpha = alpha_p_gamma + alpha_m_gamma
        gamma = alpha_p_gamma - alpha_m_gamma
        return EulerAngle(alpha, beta, gamma)

    def __repr__(self):
        return str(tf.stack(self["x"]))


class AlignmentAngle(EulerAngle):
    @staticmethod
    def angle_px_px(p1, x1, p2, x2):
        z1 = LorentzVector.vect(p1)
        zr = LorentzVector.vect(p2 - p1)
        z2 = LorentzVector.vect(p2)
        r1, xr = EulerAngle.angle_zx_z_getx(z1, x1, zr)
        m_p = LorentzVector.M(p1)
        p3_1 = Vector3.norm(LorentzVector.vect(p1))
        p3_2 = Vector3.norm(LorentzVector.vect(p2))
        zeros = tf.zeros_like(p3_1)
        delta_p = p3_2 - p3_1
        p4_2 = LorentzVector.from_p4(
            tf.sqrt(m_p * m_p + delta_p * delta_p), zeros, zeros, delta_p
        )
        b1 = SU2M.Boost_z_from_p(p4_2)
        r2 = EulerAngle.angle_zx_zx(zr, xr, z2, x2)
        r_1 = b1 * SU2M.Rotation_y(r1["beta"]) * SU2M.Rotation_z(r1["alpha"])
        r_2 = (
            SU2M.Rotation_z(r2["gamma"])
            * SU2M.Rotation_y(r2["beta"])
            * SU2M.Rotation_z(r2["alpha"])
        )
        r = r_2 * r_1
        return r.get_eular_angle()


def kine_min_max(s12, m0, m1, m2, m3):
    """min max s23 for s12 in p0 -> p1 p2 p3"""
    m12 = tf.sqrt(s12)
    m12 = tf.where(m12 > (m1 + m2), m12, m1 + m2)
    m12 = tf.where(m12 < (m0 - m3), m12, m0 - m3)
    # if(mz < (m_d+m_pi)) return 0;
    # if(mz > (m_b-m_pi)) return 0;
    E2st = 0.5 * (m12 * m12 - m1 * m1 + m2 * m2) / m12
    E3st = 0.5 * (m0 * m0 - m12 * m12 - m3 * m3) / m12
    p2st = tf.sqrt(tf.abs(E2st * E2st - m2 * m2))
    p3st = tf.sqrt(tf.abs(E3st * E3st - m3 * m3))
    s_min = (E2st + E3st) ** 2 - (p2st + p3st) ** 2
    s_max = (E2st + E3st) ** 2 - (p2st - p3st) ** 2
    return s_min, s_max


def kine_min(s12, m0, m1, m2, m3):
    """min s23 for s12 in p0 -> p1 p2 p3"""
    s_min, s_max = kine_min_max(s12, m0, m1, m2, m3)
    return s_min


def kine_max(s12, m0, m1, m2, m3):
    """max s23 for s12 in p0 -> p1 p2 p3"""
    s_min, s_max = kine_min_max(s12, m0, m1, m2, m3)
    return s_max
