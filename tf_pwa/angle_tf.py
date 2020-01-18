import tensorflow as tf
import functools
#from pysnooper import snoop

Tensor_Type = type(tf.Variable(0.0))

class Vector3(Tensor_Type):
  def __init__(self,value):
    super(Vector3,self).__init__(value)
  @property
  def X(self):
    return self[:,0]
  @property
  def Y(self):
    return self[:,1]
  @property
  def Z(self):
    return self[:,2]
  
  def norm2(self):
    return tf.reduce_sum(self * self,-1)
  
  def norm(self):
    return tf.sqrt(self.norm2())
  
  def dot(self,other):
    ret = tf.reduce_sum(self * other,-1)
    return ret
  
  def cross(self,other):
    p = tf.linalg.cross(self,other)
    return Vector3(p)
  
  def unit(self):
    p, n = tf.linalg.normalize(self,axis=-1)
    return Vector3(p)
  def angle_from(self,x,y):
    return tf.math.atan2(self.dot(y),self.dot(x))
  
  def __add__(self,other):
    return Vector3(super(Vector3,self).__add__(other))
  def __sub__(self,other):
    return Vector3(super(Vector3,self).__sub__(other))
  def __neg__(self):
    return Vector3(super(Vector3,self).__neg__())
    
class LorentzVector(Tensor_Type):
  def __init__(self,value):
    super(LorentzVector,self).__init__(value)
  @property
  def X(self):
    return self[:,1]
  @property
  def Y(self):
    return self[:,2]
  @property
  def Z(self):
    return self[:,3]
  @property
  def T(self):
    return self[:,0]
  @property
  def e(self):
    return self[:,0]
  
  def boost_vector(self):
    return Vector3(self[:,1:4]/self[:,0:1])
  
  def vect(self):
    return Vector3(self[:,1:4])
  
  def rest_vector(self,other):
    rest = LorentzVector(other)
    p = -self.boost_vector()
    ret = rest.boost(p)
    return ret

  def boost(self,p,epslion=1e-14):
    pb = Vector3(p)
    beta2 = pb.norm2()
    gamma = 1.0/tf.sqrt(1-beta2)
    bp = pb.dot(self.vect())
    gamma2 = tf.where(beta2 > epslion,(gamma-1.0)/beta2,0.0)
    p = self.vect() + tf.reshape(gamma2*bp,(-1,1))*pb + tf.reshape(gamma*self.T,(-1,1))*pb
    T = tf.reshape(gamma*(self.T + bp),(-1,1))
    ret = tf.concat([T,p],-1)
    return LorentzVector(ret)
  
  def get_metric(self):
    return tf.cast(tf.constant([1.0,-1.0,-1.0,-1.0]),self.dtype)
  
  def M2(self):
    s = self*self* self.get_metric()
    return tf.reduce_sum(s,axis=-1)
  
  def M(self):
    return tf.sqrt(self.M2())
  
  def __add__(self,other):
    return LorentzVector(super(LorentzVector,self).__add__(other))
  
  def __sub__(self,other):
    return LorentzVector(super(LorentzVector,self).__sub__(other))
  
  def __neg__(self):
    return LorentzVector(super(LorentzVector,self).__neg__())
  
class EularAngle(dict):
  def __init__(self,alpha=0.0,beta=0.0,gamma=0.0):
    self["alpha"] = alpha
    self["beta"]  = beta
    self["gamma"] = gamma

  @staticmethod
  def angle_zx_zx(z1,x1,z2,x2):
    u_z1 = z1.unit()
    u_z2 = z2.unit()
    u_y1 = z1.cross(x1).unit()
    u_x1 = u_y1.cross(z1).unit()
    u_yr = z1.cross(z2).unit()
    u_xr = u_yr.cross(z1).unit()
    u_y2 = z2.cross(x2).unit()
    u_x2 = u_y2.cross(z2).unit()
    alpha = u_xr.angle_from(u_x1,u_y1)#np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = u_z2.angle_from(u_z1,u_xr)#np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
    gamma = -u_yr.angle_from(u_y2,-u_x2)#np.arctan2(u_xr.Dot(u_y2),u_xr.Dot(u_x2))
    return EularAngle(alpha,beta,gamma)

  @staticmethod
  #@pysnooper.snoop()
  def angle_zx_z_gety(z1,x1,z2):
    u_z1 = z1.unit()
    u_z2 = z2.unit()
    u_y1 = z1.cross(x1).unit()
    u_x1 = u_y1.cross(z1).unit()
    u_yr = z1.cross(z2).unit()
    u_xr = u_yr.cross(z1).unit()
    alpha = u_xr.angle_from(u_x1,u_y1)#np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = u_z2.angle_from(u_z1,u_xr)#np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
    gamma = tf.zeros_like(beta)
    u_x2 = u_yr.cross(u_z2).unit()
    return (EularAngle(alpha,beta,gamma),u_x2)
