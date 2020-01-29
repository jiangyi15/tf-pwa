import numpy as np
from .utils import AttrDict


class Vector3(object): # 3-d vector
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
  
  def __neg__(self): # opposite
    return Vector3(-self.X,-self.Y,-self.Z)
  def Norm2(self): # norm square
    return np.sum(self.data * self.data,0)
  def Norm(self):
    return np.sqrt(np.sum(self.data * self.data,0))
  def __add__(self,o):
    return Vector3(self.X+o.X,self.Y+o.Y,self.Z+o.Z)
  def __sub__(self,o):
    return Vector3(self.X-o.X,self.Y-o.Y,self.Z-o.Z)
  def Dot(self,other):
    ret = np.sum(self.data * other.data,0)
    return ret
  def Cross(self,o):
    p = np.cross(self.data,o.data,axis=0)
    return Vector3(p[0],p[1],p[2])
  def Unit(self): # if zero, the unit is itself
    p = np.where(self.Norm() == 0.,self.data,self.data/self.Norm())
    return Vector3(p[0],p[1],p[2])
  def ang_from(self,x,y): # angle from x-axis (x⊥y and both are unit vectors)
    return np.arctan2(self.Dot(y),self.Dot(x))

  def __repr__(self):
    return str(self.data)+"\n"


class LorentzVector(object): # 4-d vector (x,y,z,-t)
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
  
  def BoostVector(self): # 3-d β=p/e 
    return Vector3(self.p[0]/self.e,self.p[1]/self.e,self.p[2]/self.e)
  
  def Vect(self): # 3-d p
    return Vector3(self.p[0],self.p[1],self.p[2])
  
  def Rest_Vector(self,other): # boost other to the RF of self
    ret = LorentzVector(other.p[0],other.p[1],other.p[2],other.e)
    p = -self.BoostVector()
    ret.Boost(p.X,p.Y,p.Z)
    return ret
    
  def __neg__(self):
    return LorentzVector(-self.p[0],-self.p[1],-self.p[2],self.e)
  
  def Boost(self,px,py,pz,epslion=1e-14): # switch to frame β #洛伦兹变换公式
    pb = Vector3(px,py,pz) #β
    beta2 = pb.Norm2()
    gamma = 1.0/np.sqrt(1-beta2) #γ
    bp = pb.Dot(self.p) #β·p
    gamma2 = np.where(beta2 > epslion,(gamma-1.0)/beta2,0.0) #Γ # epsilon防止beta2为0？
    self.p = self.p + gamma2*bp*pb.data + gamma*pb.data*self.e
    self.e = gamma*(self.e + bp)
  
  def M2(self):
    return self.e * self.e - np.sum(self.p *self.p,0)
  
  def M(self): # invariant mass
    return np.sqrt(self.e * self.e - np.sum(self.p *self.p,0))
  
  def __add__(self,o):
    return LorentzVector(self.X+o.X,self.Y+o.Y,self.Z+o.Z,self.T+o.T)
  def __sub__(self,o):
    return LorentzVector(self.X-o.X,self.Y-o.Y,self.Z-o.Z,self.T-o.T)
  def __repr__(self):
    return str(self.p)+"\n"+str(self.e)+"\n"


class EularAngle(AttrDict):
  def __init__(self,alpha=0.0,beta=0.0,gamma=0.0):
    self.alpha = alpha
    self.beta  = beta
    self.gamma = gamma

  @staticmethod
  def angle_zx_zx(z1,x1,z2,x2): #计算从Frame 1经过Frame r转到Frame 2的欧拉角 (输入都是Vector3类实例)
    u_z1 = z1.Unit()
    u_z2 = z2.Unit()
    u_y1 = z1.Cross(x1).Unit()
    u_x1 = u_y1.Cross(z1).Unit()
    u_yr = z1.Cross(z2).Unit() #共同y轴(三步转动过程中间那个参考系)
    u_xr = u_yr.Cross(z1).Unit() #共同y轴的x1轴
    u_y2 = z2.Cross(x2).Unit()
    u_x2 = u_y2.Cross(z2).Unit()
    alpha = u_xr.ang_from(u_x1,u_y1)#np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = u_z2.ang_from(u_z1,u_xr)#np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
    gamma = -u_yr.ang_from(u_y2,-u_x2)#np.arctan2(u_xr.Dot(u_y2),u_xr.Dot(u_x2))
    return EularAngle(alpha,beta,gamma)
  
  @staticmethod
  def angle_zx_z_gety(z1,x1,z2):
    u_z1 = z1.Unit()
    u_z2 = z2.Unit()
    u_y1 = z1.Cross(x1).Unit()
    u_x1 = u_y1.Cross(z1).Unit()
    u_yr = z1.Cross(z2).Unit()
    u_xr = u_yr.Cross(z1).Unit()
    alpha = u_xr.ang_from(u_x1,u_y1)#np.arctan2(u_xr.Dot(u_y1),u_xr.Dot(u_x1))
    beta  = u_z2.ang_from(u_z1,u_xr)#np.arctan2(u_z2.Dot(u_xr),u_z2.Dot(u_z1))
    gamma = np.zeros_like(beta) #不用转第三步了，γ设为0
    u_x2 = u_yr.Cross(u_z2).Unit() #γ为0对应的x2轴
    return (EularAngle(alpha,beta,gamma),u_x2)


def cal_angle(p_B,p_C,p_D): #从末态粒子四动量计算amplitude PDF中用到的角度变量
  p_A = p_B + p_C + p_D # conservation of 4-vector
  p_B_A = p_A.Rest_Vector(p_B) # p_B in the RF of A
  p_C_A = p_A.Rest_Vector(p_C)
  p_D_A = p_A.Rest_Vector(p_D)
  return cal_angle_rest(p_B_A,p_C_A,p_D_A)

def cal_angle_rest(p4_B,p4_C,p4_D):
  p4_BD = p4_B + p4_D
  p4_BC = p4_B + p4_C
  p4_CD = p4_C + p4_D
  p4_B_BD = p4_BD.Rest_Vector(p4_B)
  p4_B_BC = p4_BC.Rest_Vector(p4_B)
  p4_D_CD = p4_CD.Rest_Vector(p4_D)
  
  zeros = np.zeros_like(p4_B.e) # length of data
  ones = np.ones_like(p4_B.e)
  u_z = Vector3(zeros,zeros,ones) # the current frame
  u_x = Vector3(ones,zeros,zeros)
  ang_BC,x_BC = EularAngle.angle_zx_z_gety(u_z,u_x,p4_BC.Vect()) #z轴转到BC方向
  ang_B_BC,x_B_BC = EularAngle.angle_zx_z_gety(p4_BC.Vect(),x_BC,p4_B_BC.Vect())
  ang_BD,x_BD = EularAngle.angle_zx_z_gety(u_z,u_x,p4_BD.Vect())
  ang_B_BD,x_B_BD = EularAngle.angle_zx_z_gety(p4_BD.Vect(),x_BD,p4_B_BD.Vect())
  ang_CD,x_CD = EularAngle.angle_zx_z_gety(u_z,u_x,p4_CD.Vect())
  ang_D_CD,x_D_CD = EularAngle.angle_zx_z_gety(p4_CD.Vect(),x_CD,p4_D_CD.Vect())
  
  ang_BD_B = EularAngle.angle_zx_zx( p4_B_BD.Vect() , x_B_BD, p4_B.Vect(), x_CD)
  ang_BC_B = EularAngle.angle_zx_zx( p4_B_BC.Vect() , x_B_BC, p4_B.Vect(), x_CD)
  ang_BD_D = EularAngle.angle_zx_zx(-p4_B_BD.Vect() , x_B_BD, p4_D.Vect(), x_BC)
  ang_CD_D = EularAngle.angle_zx_zx( p4_D_CD.Vect() , x_D_CD, p4_D.Vect(), x_BC)

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
    "ang_CD_D":ang_CD_D,
  }


def cal_angle4(p_B,p_C,p_E,p_F): #四体A->BC(EF)
  p_D = p_E + p_F
  p_A = p_B + p_C + p_D
  p_B_A = p_A.Rest_Vector(p_B)
  p_C_A = p_A.Rest_Vector(p_C)
  p_E_A = p_A.Rest_Vector(p_E)
  p_F_A = p_A.Rest_Vector(p_F)
  return  cal_angle_rest4(p_B_A,p_C_A,p_E_A,p_F_A)

def cal_angle_rest4(p4_B,p4_C,p4_E,p4_F):
  p4_D = p4_E + p4_F
  p4_BD = p4_B + p4_D
  p4_BC = p4_B + p4_C
  p4_CD = p4_C + p4_D
  p4_B_BD = p4_BD.Rest_Vector(p4_B)
  p4_D_BD = p4_BD.Rest_Vector(p4_D)
  p4_B_BC = p4_BC.Rest_Vector(p4_B)
  p4_D_CD = p4_CD.Rest_Vector(p4_D)
  
  p4_E_CD = p4_CD.Rest_Vector(p4_E)
  p4_E_CD = p4_D_CD.Rest_Vector(p4_E_CD)
  p4_E_BD = p4_BD.Rest_Vector(p4_E)
  p4_E_BD = p4_D_BD.Rest_Vector(p4_E_BD)
  
  p4_E_BC = p4_D.Rest_Vector(p4_E)
  
  zeros = np.zeros_like(p4_B.e)
  ones = np.ones_like(p4_B.e)
  u_z = Vector3(zeros,zeros,ones)
  u_x = Vector3(ones,zeros,zeros)
  ang_BC,x_BC = EularAngle.angle_zx_z_gety(u_z,u_x,p4_BC.Vect())
  ang_B_BC,x_B_BC = EularAngle.angle_zx_z_gety(p4_BC.Vect(),x_BC,p4_B_BC.Vect())
  ang_BD,x_BD = EularAngle.angle_zx_z_gety(u_z,u_x,p4_BD.Vect())
  ang_B_BD,x_B_BD = EularAngle.angle_zx_z_gety(p4_BD.Vect(),x_BD,p4_B_BD.Vect())
  ang_CD,x_CD = EularAngle.angle_zx_z_gety(u_z,u_x,p4_CD.Vect())
  ang_D_CD,x_D_CD = EularAngle.angle_zx_z_gety(p4_CD.Vect(),x_CD,p4_D_CD.Vect())
  
  ang_BD_B = EularAngle.angle_zx_zx(p4_B_BD.Vect(),x_B_BD,p4_B.Vect(),x_CD)
  ang_BC_B = EularAngle.angle_zx_zx(p4_B_BC.Vect(),x_B_BC,p4_B.Vect(),x_CD)
  ang_BD_D = EularAngle.angle_zx_zx(-p4_B_BD.Vect(),x_B_BD,p4_D.Vect(),x_BC)
  ang_CD_D = EularAngle.angle_zx_zx(p4_D_CD.Vect(),x_D_CD,p4_D.Vect(),x_BC)

  ang_E_BC,x_E_BC = EularAngle.angle_zx_z_gety(p4_D.Vect(),x_BC,p4_E_BC.Vect())
  ang_E_BD = EularAngle.angle_zx_zx(p4_D_BD.Vect(),x_BD,p4_E_BD.Vect(),x_E_BC)
  ang_E_CD = EularAngle.angle_zx_zx(p4_D_CD.Vect(),x_CD,p4_E_CD.Vect(),x_E_BC)
  
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
    "ang_CD_D":ang_CD_D,
    "ang_E_BC":ang_E_BC,
    "ang_E_BD":ang_E_BD,
    "ang_E_CD":ang_E_CD
  }
 

def cal_ang_file(fname,dtype="float64"): #从末态粒子四动量信息文件计算拟合变量
  data = np.loadtxt(fname,dtype=dtype) #三个末态粒子的四动量信息 [[dt,dx,dy,dz],[b...],[c...],[d...]]
  pd = data[0::3] #从第0个开始每隔三个
  lpd = LorentzVector(pd[:,1],pd[:,2],pd[:,3],pd[:,0])
  pb = data[1::3]
  lpb = LorentzVector(pb[:,1],pb[:,2],pb[:,3],pb[:,0])
  pc = data[2::3]
  lpc = LorentzVector(pc[:,1],pc[:,2],pc[:,3],pc[:,0])
  
  ret = cal_angle(lpb,lpc,lpd) #从末态粒子四动量计算amplitude PDF中用到的角度变量
  
  ret["m_BC"] = (lpb + lpc).M() # mass parameters
  ret["m_CD"] = (lpd + lpc).M()
  ret["m_BD"] = (lpb + lpd).M()
  ret["m_A"] = (lpb + lpc + lpd).M()
  ret["m_B"] = lpb.M()
  ret["m_C"] = lpc.M()
  ret["m_D"] = lpd.M()
  return ret


def cal_ang_file4(fname,dst_fname,dtype="float64"):
  data = np.loadtxt(fname,dtype=dtype)
  data2 = np.loadtxt(dst_fname,dtype=dtype)
  pb = data[1::3]
  lpb = LorentzVector(pb[:,1],pb[:,2],pb[:,3],pb[:,0])
  pc = data[2::3]
  lpc = LorentzVector(pc[:,1],pc[:,2],pc[:,3],pc[:,0])
  pd = data[0::3]
  lpd = LorentzVector(pd[:,1],pd[:,2],pd[:,3],pd[:,0])
  pe = data2[0::2]
  lpe = LorentzVector(pe[:,1],pe[:,2],pe[:,3],pe[:,0])
  pf = data2[1::2]
  lpf = LorentzVector(pf[:,1],pf[:,2],pf[:,3],pf[:,0])
  ret = cal_angle4(lpb,lpc,lpe,lpf)
  
  ret["m_BC"] = (lpb + lpc).M()
  ret["m_CD"] = (lpd + lpc).M()
  ret["m_BD"] = (lpb + lpd).M()
  
  ret["m_A"] = (lpb + lpc + lpd).M()
  ret["m_B"] = lpb.M()
  ret["m_C"] = lpc.M()
  ret["m_D"] = lpd.M()
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
      if isinstance(obj, EularAngle): #把三个欧拉角展开
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
    save_data(data[i],i) #产生"data","bg","PHSP"三个json文件
