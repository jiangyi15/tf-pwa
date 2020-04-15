"""
This module implements three classes **Vector3**, **LorentzVector**, **EulerAngle** .
"""
from .tensorflow_wrapper import tf, numpy_cross

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
        p =  numpy_cross(self, other)
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
        cro =  numpy_cross(self, other)
        norm_cro = tf.expand_dims(tf.norm(cro, axis=-1), -1)
        mask = norm_cro < _epsilon
        bias_other = tf.ones_like(norm_cro) + other
        cro = tf.where(mask,  numpy_cross(self, bias_other), cro)
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
        p_r += tf.reshape(gamma2 * bp, (-1, 1)) * pb
        p_r += tf.reshape(gamma * LorentzVector.get_T(self), (-1, 1)) * pb
        T_r = tf.reshape(gamma * (LorentzVector.get_T(self) + bp), (-1, 1))
        ret = tf.concat([T_r, p_r], -1)
        return ret

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
        return tf.sqrt(LorentzVector.M2(self))

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
