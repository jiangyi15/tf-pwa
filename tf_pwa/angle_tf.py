from .tensorflow_wrapper import tf
#import functools
#from pysnooper import snoop


class Vector3(tf.Tensor):
    """
    3-dim Vector functions
    """
    def get_X(self):
        return self[..., 0]

    def get_Y(self):
        return self[..., 1]

    def get_Z(self):
        return self[..., 2]

    def norm2(self):
        return tf.reduce_sum(self * self, axis=-1)

    def norm(self):
        return tf.norm(self, axis=-1)

    def dot(self, other):
        ret = tf.reduce_sum(self * other, axis=-1)
        return ret

    def cross(self, other):
        p = tf.linalg.cross(self, other)
        return p

    def unit(self):
        p, _n = tf.linalg.normalize(self, axis=-1)
        return p

    def cross_unit(self, other):
        shape = tf.broadcast_dynamic_shape(self.shape, other.shape)
        a = tf.broadcast_to(self, shape)
        b = tf.broadcast_to(other, shape)
        p, _n = tf.linalg.normalize(tf.linalg.cross(a, b), axis=-1)
        return p

    def angle_from(self, x, y):
        return tf.math.atan2(Vector3.dot(self, y), Vector3.dot(self, x))

_epsilon = 1.0e-14

class LorentzVector(tf.Tensor):
    """
    LorentzVector functions
    """
    def get_X(self):
        return self[..., 1]

    def get_Y(self):
        return self[..., 2]

    def get_Z(self):
        return self[..., 3]

    def get_T(self):
        return self[..., 0]

    def get_e(self):
        return self[..., 0]

    def boost_vector(self):
        return self[..., 1:4]/self[..., 0:1]

    def vect(self):
        return self[..., 1:4]

    def rest_vector(self, other):
        p = -LorentzVector.boost_vector(self)
        ret = LorentzVector.boost(other, p)
        return ret

    def boost(self, p):
        #pb = Vector3(p)
        pb = p
        beta2 = Vector3.norm2(pb)
        gamma = 1.0/tf.sqrt(1-beta2)
        bp = Vector3.dot(pb, LorentzVector.vect(self))
        gamma2 = tf.where(beta2 > _epsilon, (gamma-1.0)/beta2, 0.0)
        p_r = LorentzVector.vect(self)
        p_r += tf.reshape(gamma2*bp, (-1, 1))*pb
        p_r += tf.reshape(gamma*LorentzVector.get_T(self), (-1, 1))*pb
        T_r = tf.reshape(gamma*(LorentzVector.get_T(self) + bp), (-1, 1))
        ret = tf.concat([T_r, p_r], -1)
        return ret

    def get_metric(self):
        return tf.cast(tf.constant([1.0, -1.0, -1.0, -1.0]), self.dtype)

    def M2(self):
        s = self*self * LorentzVector.get_metric(self)
        return tf.reduce_sum(s, axis=-1)

    def M(self):
        return tf.sqrt(LorentzVector.M2(self))


class EularAngle(dict):
    """
    EularAngle functions
    """
    def __init__(self, alpha=0.0, beta=0.0, gamma=0.0):
        super(EularAngle, self).__init__()
        self["alpha"] = alpha
        self["beta"] = beta
        self["gamma"] = gamma

    @staticmethod
    def angle_zx_zx(z1, x1, z2, x2):
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
        return EularAngle(alpha, beta, gamma)

    @staticmethod
    # @pysnooper.snoop()
    def angle_zx_z_getx(z1, x1, z2):
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
        return (EularAngle(alpha, beta, gamma), u_x2)
