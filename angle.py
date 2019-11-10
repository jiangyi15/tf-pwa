import numpy as np
import pysnooper

class Vector3(object):
  def __init__(self,x=0.0,y=0.0,z=0.0):
    
    self.data = np.array([x,y,z])
  @property
  def X(self):
    return self.data[0]
  @property
  def Y(self):
    return self.data[1]
  @property
  def Z(self):
    return self.data[2]
  def __neg__(self):
    return Vector3(-self.X,-self.Y,-self.Z)
  def Norm2(self):
    return np.sum(self.data * self.data,0)
  def Norm(self):
    return np.sqrt(np.sum(self.data * self.data,0))
  def Dot(self,other):
    ret = np.sum(self.data * other.data,0)
    return ret
  def __add__(self,o):
    return Vector3(self.X+o.X,self.Y+o.Y,self.Z+o.Z)
  def __sub__(self,o):
    return Vector3(self.X-o.X,self.Y-o.Y,self.Z-o.Z)
  def Cross(self,o):
    p = np.cross(self.data,o.data,axis=0)
    return Vector3(p[0],p[1],p[2])
  def Unit(self):
    p = np.where(self.Norm() == 0.,self.data,self.data/self.Norm())
    return Vector3(p[0],p[1],p[2])
  def __repr__(self):
    return str(self.data)+"\n"

class LorentzVector(object):
  def __init__(self,x=0.0,y=0.0,z=0.0,e=0.0):
    self.p ,self.e = np.array([x, y, z]), np.array(e)
  @property
  def X(self):
    return self.p[0]
  @property
  def Y(self):
    return self.p[1]
  @property
  def Z(self):
    return self.p[2]
  @property
  def T(self):
    return self.e
  def BoostVector(self):
    return Vector3(self.p[0]/self.e,self.p[1]/self.e,self.p[2]/self.e)
  
  def Vect(self):
    return Vector3(self.p[0],self.p[1],self.p[2])
  
  def Rest_Vector(self,other):
    ret = LorentzVector(other.p[0],other.p[1],other.p[2],other.e)
    p = -self.BoostVector()
    ret.Boost(p.X,p.Y,p.Z)
    return ret
    
  def __neg__(self):
    return LorentzVector(-self.p[0],-self.p[1],-self.p[2],self.e)
  
  def Boost(self,px,py,pz):
    pb = Vector3(px,py,pz)
    beta2 = pb.Norm2()
    gamma = 1.0/np.sqrt(1-beta2)
    bp = pb.Dot(self.p)
    gamma2 = np.where(beta2 >0,(gamma-1.0)/beta2,0.0)
    self.p = self.p + gamma2*bp*pb.data + gamma*pb.data*self.e
    self.e = gamma*(self.e + bp)
  
  def M2(self):
    return self.e * self.e - np.sum(self.p *self.p,0)
  
  def __add__(self,o):
    return LorentzVector(self.X+o.X,self.Y+o.Y,self.Z+o.Z,self.T+o.T)
  def __sub__(self,o):
    return LorentzVector(self.X-o.X,self.Y-o.Y,self.Z-o.Z,self.T-o.T)
  def __repr__(self):
    return str(self.p)+"\n"+str(self.e)+"\n"
  
class EularAngle(object):
  def __init__(self,alpha=0.0,beta=0.0,gamma=0.0):
    self.alpha = alpha
    self.beta  = beta
    self.gamma = gamma

  @staticmethod
  def angle_zy_zy(z1,y1,z2,y2):
    u_z1 = z1.Unit()
    u_z2 = z2.Unit()
    u_x1 = y1.Cross(z1).Unit()
    u_xr = z1.Cross(z2).Unit()
    u_y1 = z1.Cross(u_x1).Unit()
    u_yr = z1.Cross(u_xr).Unit()
    u_x2 = y2.Cross(z2).Unit()
    u_y2 = z2.Cross(u_x2).Unit()
    alpha = np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = np.arctan2(u_z2.Dot(-u_yr),u_z2.Dot(u_z1))
    gamma = -np.arctan2(u_xr.Dot(u_y2),u_xr.Dot(u_x2))
    return EularAngle(alpha,beta,gamma)

  
  @staticmethod
  def angle_zy_z_gety(z1,y1,z2):
    u_z1 = z1.Unit()
    u_x1 = y1.Cross(z1).Unit()
    u_y1 = z1.Cross(u_x1).Unit()
    u_xr = z1.Cross(z2).Unit()
    u_yr = z1.Cross(u_xr).Unit()
    u_z2 = z2.Unit()
    alpha = np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = np.arctan2(u_z2.Dot(-u_yr),u_z2.Dot(u_z1))
    gamma = np.zeros_like(beta)
    u_y2 = z2.Cross(u_xr).Unit()
    return (EularAngle(alpha,beta,gamma),u_y2)
  def __repr__(self):
    return """{{
"alpha":{},
"beta" :{},
"gamma":{}
}}""".format(self.alpha,self.beta,self.gamma)

#@pysnooper.snoop()
def cal_angle(p_B,p_C,p_D):
  p_A = p_B + p_C + p_D
  p_B_A = p_A.Rest_Vector(p_B)
  p_C_A = p_A.Rest_Vector(p_C)
  p_D_A = p_A.Rest_Vector(p_D)
  return cal_angle_rest(p_B_A,p_C_A,p_D_A)

#@pysnooper.snoop()
def cal_angle_rest(p4_B,p4_C,p4_D):
  p4_BD = p4_B + p4_D
  p4_BC = p4_B + p4_C
  p4_CD = p4_C + p4_D
  p4_B_BD = p4_BD.Rest_Vector(p4_B)
  p4_B_BC = p4_BC.Rest_Vector(p4_B)
  p4_D_CD = p4_CD.Rest_Vector(p4_D)
  
  zeros = np.zeros_like(p4_B.e)
  ones = np.ones_like(p4_B.e)
  u_z = Vector3(zeros,zeros,ones)
  u_y = Vector3(-ones,zeros,zeros)
  ang_BC,y_BC = EularAngle.angle_zy_z_gety(u_z,u_y,p4_BC.Vect())
  ang_B_BC,y_B_BC = EularAngle.angle_zy_z_gety(p4_BC.Vect(),y_BC,p4_B_BC.Vect())
  ang_BD,y_BD = EularAngle.angle_zy_z_gety(u_z,u_y,p4_BD.Vect())
  ang_B_BD,y_B_BD = EularAngle.angle_zy_z_gety(p4_BD.Vect(),y_BD,p4_B_BD.Vect())
  ang_CD,y_CD = EularAngle.angle_zy_z_gety(u_z,u_y,p4_CD.Vect())
  ang_D_CD,y_D_CD = EularAngle.angle_zy_z_gety(p4_CD.Vect(),y_CD,p4_D_CD.Vect())
  
  ang_BD_B = EularAngle.angle_zy_zy(p4_B_BD.Vect(),y_B_BD,p4_B.Vect(),-y_CD)
  ang_BC_B = EularAngle.angle_zy_zy(p4_B_BC.Vect(),y_B_BC,p4_B.Vect(),-y_CD)
  ang_BD_D = EularAngle.angle_zy_zy(-p4_B_BD.Vect(),-y_B_BD,p4_D.Vect(),-y_BC)
  ang_CD_D = EularAngle.angle_zy_zy(p4_D_CD.Vect(),y_D_CD,p4_D.Vect(),-y_BC)
  
  return {
    "ang_BC":ang_BC,  
    "ang_BD":ang_BD,
    "ang_CD":ang_CD,
    "ang_B_BC":ang_B_BC,
    "ang_B_BD":ang_B_BD,
    "ang_D_CD":ang_D_CD,
    "ang_BD_B":ang_BD_B,
    "ang_BD_D":ang_BD_D,
    "ang_BC_B":ang_BC_B,
    "ang_CD_D":ang_CD_D
  }
  
def cal_ang_file(fname,dtype="float64"):
  data = np.loadtxt(fname,dtype=dtype)
  pb = data[0::3]
  lpb = LorentzVector(pb[:,1],pb[:,2],pb[:,3],pb[:,0])
  pd = data[1::3]
  lpd = LorentzVector(pd[:,1],pd[:,2],pd[:,3],pd[:,0])
  pc = data[2::3]
  lpc = LorentzVector(pc[:,1],pc[:,2],pc[:,3],pc[:,0])
  ret = cal_angle(lpd,lpc,lpb)
  return ret

if __name__ == "__main__":
  f = ["data/data4600_new.dat","data/bg4600_new.dat","data/PHSP4600_new.dat"]
  t = ["data","bg","PHSP"]
  data = {}
  for i in range(3):
    data[t[i]] = cal_ang_file(f[i])
  import json
  class MyEncoder(json.JSONEncoder):
    def default(self, obj):
      if isinstance(obj, EularAngle):
        return {
          "alpha":obj.alpha.tolist(),
          "beta":obj.beta.tolist(),
          "gamma":obj.gamma.tolist()
        }
      return json.JSONEncoder.default(self, obj)
  def save_data(dat,name):
    with open("data/"+name+"_ang_struct.json","w") as f:
      json.dump(dat,f,cls=MyEncoder,indent=2)
  for i in t:
    save_data(data[i],i)
