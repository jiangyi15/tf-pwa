from .tensorflow_wrapper import tf
import numpy as np

from contextlib import contextmanager
from .particle import Particle,Decay
from .variable import Variable
from .dfun_tf import D_Cache as D_fun_Cache
from .breit_wigner import barrier_factor,breit_wigner_dict as bw_dict
from .config import regist_config

import os
import copy
import functools

from .utils import print_dic


param_list = [
  "m_A","m_B","m_C","m_D","m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D"
]


def Getp(M_0, M_1, M_2) :
  M12S = M_1 + M_2
  M12D = M_1 - M_2
  p = (M_0 - M12S) * (M_0 + M12S) * (M_0 - M12D) * (M_0 + M12D)
  q = (p + tf.abs(p))/2 # if p is negative, which results from bad data, the return value is 0.0
  return tf.sqrt(q) / (2 * M_0)

@regist_config("amp")
class AllAmplitude(tf.keras.Model):
  def __init__(self,res,polar=True):
    super(AllAmplitude,self).__init__()
    self.JA = 1
    self.JB = 1
    self.JC = 0
    self.JD = 1
    self.ParA = -1
    self.ParB = -1
    self.ParC = -1
    self.ParD = -1
    self.A = Particle("A",self.JA,self.ParA,spins=[-1,1])
    self.B = Particle("B",self.JB,self.ParB)
    self.C = Particle("C",self.JC,self.ParC)
    self.D = Particle("D",self.JD,self.ParD)
    
    self.res = copy.deepcopy(res) # RESON Params #直接用等号会修改res
    self.res_decay = self.init_res_decay() # DECAY for each type of process
    self.polar = polar
    fix_dic = {}#"D1_2420r":3.12}
    bnd_dic = {}#"D1_2420r":(0.3334301945,0.3334301945)}
    self.fit_params = Variable(self.res,self.res_decay,self.polar,fix_dic=fix_dic,bnd_dic=bnd_dic,dtype=tf.float64)
    self.fit_params.init_fit_params() # initialize FPs
    
    #print_dic(self.fit_params.variables)
    #print_dic(self.fit_params.trainable_vars)
    
    self.coef = self.fit_params.coef # FP gls inside H
    self.coef_norm = self.fit_params.coef_norm # FP norm factor for each resonance
    self.init_used_res() # used RESON'NAMES in config
 

  def init_used_res(self):
    self.used_res = [i for i in self.res]
    

  def init_res_decay(self):
    ret = {}
    for i in self.res:
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      chain = self.res[i]["Chain"]
      if "bw" in self.res[i]: # BW的形式
        self.res[i]["bwf"] = bw_dict[self.res[i]["bw"]]
      else:
        self.res[i]["bwf"] = bw_dict["default"]
      tmp = Particle(i,J_reson,P_reson) # resonance(the intermediate particle)
      if (chain < 0) : # A->(DB)C
        d_tmp_0 = Decay(self.A,[tmp,self.C],i+"_0")
        d_tmp_1 = Decay(tmp,[self.B,self.D],i+"_1")
        ret[i] = [d_tmp_0,d_tmp_1]
      elif (chain > 0 and chain < 100) : # A->(BC)D 
        d_tmp_0 = Decay(self.A,[tmp,self.D],i+"_0")
        d_tmp_1 = Decay(tmp,[self.B,self.C],i+"_1")
        ret[i] = [d_tmp_0,d_tmp_1]
      elif (chain > 100 and chain < 200) : # A->B(CD) 
        d_tmp_0 = Decay(self.A,[tmp,self.B],i+"_0")
        d_tmp_1 = Decay(tmp,[self.D,self.C],i+"_1")
        ret[i] = [d_tmp_0,d_tmp_1]
      else :
        raise Exception("unknown chain")
    return ret


  def Get_BWReson(self,m_A,m_B,m_C,m_D,m_BC,m_BD,m_CD):
    ret = {}
    for i in self.used_res:
      m = self.res[i]["m0"] # either var or const
      g = self.res[i]["g0"]
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      chain = self.res[i]["Chain"]
      if (chain < 0) : # A->(BD)C
        p = Getp(m_A, m_BD, m_C)
        p0 = Getp(m_A, m, m_C)
        q = Getp(m_BD, m_B, m_D)
        q0 = Getp(m, m_B, m_D)
        l = self.res_decay[i][1].get_min_l()
        bw = self.res[i]["bwf"](m_BD, m, g, q, q0, l, 3.0) # d=3.0
        ret[i] = [p,p0,q,q0,bw]
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        p = Getp(m_A, m_BC, m_D)
        p0 = Getp(m_A, m, m_D)
        q = Getp(m_BC, m_B, m_C)
        q0 = Getp(m, m_B, m_C)
        l = self.res_decay[i][1].get_min_l()
        bw = self.res[i]["bwf"](m_BC, m, g, q, q0, l, 3.0)
        ret[i] = [p,p0,q,q0,bw]
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        p = Getp(m_A, m_CD, m_B)
        p0 = Getp(m_A, m, m_B)
        q = Getp(m_CD, m_C, m_D)
        q0 = Getp(m, m_C, m_D)
        l = self.res_decay[i][1].get_min_l()
        bw = self.res[i]["bwf"](m_CD, m, g, q, q0, l, 3.0)
        ret[i] = [p,p0,q,q0,bw]
      else :
        raise Exception("unknown chain")
    return ret
  

  def GetA2BC_LS(self,idx,layer,q,q0,d): #某个H里各项构成的三维数组
    decay = self.res_decay[idx][layer]
    ja = decay.core.J
    jb = decay.outs[0].J
    jc = decay.outs[1].J
    M_r = []
    M_i = []
    for r,i in self.coef[idx][layer]:
      M_r.append(self.fit_params.get(r))
      M_i.append(self.fit_params.get(i))
    M_r = tf.stack(M_r)
    M_i = tf.stack(M_i)
    bf = barrier_factor(decay.get_l_list(),q,q0,d)
    # switch temporately into xy coordinates
    if self.polar:
      norm_r = tf.linalg.diag(M_r*tf.cos(M_i))
      norm_i = tf.linalg.diag(M_r*tf.sin(M_i))
    else:
      norm_r = tf.linalg.diag(M_r)
      norm_i = tf.linalg.diag(M_i)
    mdep_r = tf.matmul(norm_r,bf)
    mdep_i = tf.matmul(norm_i,bf)
    cg_trans = tf.cast(decay.get_cg_matrix(),M_r.dtype)
    H_r = tf.matmul(cg_trans,mdep_r)
    H_i = tf.matmul(cg_trans,mdep_i)
    ret = tf.reshape(tf.complex(H_r,H_i),(2*jb+1,2*jc+1,-1))
    return ret


  def get_res_total(self,idx): # get??? change a name
    r,i =  self.coef_norm[idx]
    M_r = r*tf.cos(i)
    M_i = r*tf.sin(i)
    return tf.complex(M_r,M_i) #switch norm factor into xy coordinates
 

  def get_amp2s(self,*x):
    data = self.cache_data(*x)
    sum_A = self.get_amp2s_matrix(*data)
    return sum_A
  
  def cache_data(self,m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD, 
      Theta_BC,Theta_B_BC, phi_BC, phi_B_BC,
      Theta_BD,Theta_B_BD,phi_BD, phi_B_BD, 
      Theta_CD,Theta_D_CD, phi_CD,phi_D_CD,
      Theta_BD_B,Theta_BC_B,Theta_BD_D,Theta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2,split=None,batch=None):
    
    if split is None and batch is None:
      ang_BD_B = D_fun_Cache(phi_BD_B,Theta_BD_B, phi_BD_B2)
      ang_BD_D = D_fun_Cache(phi_BD_D,Theta_BD_D, phi_BD_D2)
      ang_BD = D_fun_Cache(phi_BD,Theta_BD, 0.0)
      ang_B_BD = D_fun_Cache(phi_B_BD,Theta_B_BD, 0.0)
      ang_BC_B = D_fun_Cache(phi_BC_B, Theta_BC_B,phi_BC_B2)
      ang_BC = D_fun_Cache(phi_BC, Theta_BC,0.0)
      ang_B_BC = D_fun_Cache(phi_B_BC, Theta_B_BC,0.0)
      ang_CD_D = D_fun_Cache(phi_CD_D, Theta_CD_D,phi_CD_D2)
      ang_CD = D_fun_Cache(phi_CD, Theta_CD,0.0)
      ang_D_CD = D_fun_Cache(phi_D_CD, Theta_D_CD,0.0)
      return [m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D]
    
    else :
      data = [m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD, 
      Theta_BC,Theta_B_BC, phi_BC, phi_B_BC,
      Theta_BD,Theta_B_BD,phi_BD, phi_B_BD, 
      Theta_CD,Theta_D_CD, phi_CD,phi_D_CD,
      Theta_BD_B,Theta_BC_B,Theta_BD_D,Theta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2]
      n = m_BC.shape[0]
      if batch is None: # split个一组，共batch组
        l = (n+split-1)//split
      else:
        split = (n+batch-1)//batch
        l = batch
      ret = []
      for i in range(split):
        data_part = [ arg[i*l:min(i*l+l,n)] for arg in data ]
        ret.append(self.cache_data(*data_part)) #递归
      return ret
  
  def get_amp2s_matrix(self,m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D):
    d = 3.0
    res_cache = self.Get_BWReson(m_A,m_B,m_C,m_D,m_BC,m_BD,m_CD)
    sum_A = 0.1
    ret = []
    ns_a = 2 #ns个自旋求和项
    ns_b = 3
    ns_c = 1
    ns_d = 3
    for i in self.used_res:
      chain = self.res[i]["Chain"]
      if chain == 0:
        continue
      JReson = self.res[i]["J"]
      ParReson = self.res[i]["Par"]

      if chain < 0: # A->(DB)C aligned B,D
        lambda_BD = list(range(-JReson,JReson+1))
        ns_bd = len(lambda_BD)
        H_0 = self.GetA2BC_LS(i,0,res_cache[i][0],res_cache[i][1],d) # H factor 3-d array (layer 0)
        H_1 = self.GetA2BC_LS(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BD.get_lambda(self.A.J,self.A.spins,lambda_BD,self.C.spins) # D^A
        df_b = ang_B_BD.get_lambda(JReson,lambda_BD,self.B.spins,self.D.spins) # D^{BD}
        aligned_B = ang_BD_B.get_lambda(self.B.J,self.B.spins,self.B.spins) # alignment B
        aligned_D = ang_BD_D.get_lambda(self.D.J,self.D.spins,self.D.spins) # alignment D
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(ns_a,ns_bd,1,ns_c,1,-1)) * tf.reshape(HD2,(1,ns_bd,ns_b,1,ns_d,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abxcdi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_c,ns_d,-1)) * tf.reshape(aligned_B,(1,ns_b,ns_b,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        abcdyi = tf.reshape(abcdi,(ns_a,ns_b,ns_c,ns_d,1,-1))*tf.reshape(aligned_D,(1,1,1,ns_d,ns_d,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        s = abcdi
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))

      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        lambda_BC = list(range(-JReson,JReson+1))
        ns_bc = len(lambda_BC)
        H_0 = self.GetA2BC_LS(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BC.get_lambda(self.A.J,self.A.spins,lambda_BC,self.D.spins)
        df_b = ang_B_BC.get_lambda(JReson,lambda_BC,self.B.spins,self.C.spins)
        aligned_B = ang_BC_B.get_lambda(self.B.J,self.B.spins,self.B.spins)#(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(ns_a,ns_bc,1,1,ns_d,-1)) * tf.reshape(HD2,(1,ns_bc,ns_b,ns_c,1,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abxcdi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_c,ns_d,-1)) * tf.reshape(aligned_B,(1,ns_b,ns_b,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        s = abcdi
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))

      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        lambda_CD = list(range(-JReson,JReson+1))
        ns_cd  = len(lambda_CD)
        H_0 = self.GetA2BC_LS(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_CD.get_lambda(self.A.J,self.A.spins,lambda_CD,self.B.spins)
        df_b = ang_D_CD.get_lambda(JReson,lambda_CD,self.D.spins,self.C.spins)
        aligned_D = ang_CD_D.get_lambda(self.D.J,self.D.spins,self.D.spins)#(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        if self.C.J != 0: # change order into BCD
          HD2 = tf.transpose(HD2, perm=[0, 2, 1, 3])
        arbcdi = tf.reshape(HD1,(ns_a,ns_cd,ns_b,1,1,-1)) * tf.reshape(HD2,(1,ns_cd,1,ns_c,ns_d,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abcdyi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_d,1,-1))*tf.reshape(aligned_D,(1,1,1,ns_d,ns_d,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        s = abcdi 
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))

      else:
        print("unknown chain")

    ret = tf.stack(ret)
    amp = tf.reduce_sum(ret,axis=[0])
    amp2s = tf.math.real(amp*tf.math.conj(amp))
    sum_A = tf.reduce_sum(amp2s,[0,1,2,3])
    return sum_A
   
  def call(self,x,cached=False):
    """
    
    """
    if cached:
      return self.get_amp2s_matrix(*x)
    return self.get_amp2s(*x)


  def trans_params(self,polar=True,force=False):
    """
    transform parameters for self.polar to polar coordinates

.. math::
    r + ij \leftrightarrow r e^{ij} 
    
    """
    if self.polar is polar and not force:
      return self.get_params()#本身就是想要的坐标系了
    t_p = set() # the order doesn't matter
    for i in self.coef:
      for j in self.coef[i]:
        for k in j:
          t_p.add(k)
    for r,i in t_p:
      o_r = self.fit_params.get(r)
      o_i = self.fit_params.get(i)
      if self.polar: # rp2xy
        o_r,o_i = o_r * tf.cos(o_i), o_r * tf.sin(o_i)
      if polar: # xy2rp
        n_r = tf.sqrt(o_r*o_r + o_i*o_i)
        n_i = tf.math.atan2(o_i, o_r)
      else: # if force, xy remains the same, rp will be standardized
        n_r, n_i = o_r, o_i
      self.fit_params.set(r,n_r)
      self.fit_params.set(i,n_i)
    self.polar = polar
    return self.get_params()
  

  @contextmanager
  def params_form(self,polar=True): # switch temporately between xy and rp
    origin_polar = self.polar
    self.trans_params(polar)
    yield self.get_params()
    self.trans_params(origin_polar)
  

  def _std_polar_total(self): # standardize polar expression for norm factors
    polar_sign = {}
    for idx in self.res:
      r,i = self.coef_norm[idx]
      polar_sign[r.name] = np.sign(r.numpy())
    
    for idx in self.res:
      r,i =  self.coef_norm[idx]
      r.assign(tf.abs(r)) # r is positive
      if polar_sign[r.name] < 0:
        i.assign_add(np.pi) # -r->r, then p+=np.pi
      while i.numpy() >= np.pi:
        i.assign_add(-2*np.pi) # p<=pi
      while i.numpy() < -np.pi:
        i.assign_add(2*np.pi) # p>-pi
    
  def get_params(self):
    ret = {}
    self._std_polar_total()
    for i in self.variables:
      tmp = i.numpy()
      ret[i.name] = float(tmp)
    return ret
  
  def set_params(self,param):
    for j in param:
      for i in self.variables:
        if j == i.name:
          tmp = param[i.name]
          i.assign(tmp)
  
  def set_used_res(self,ires):
    ret = []
    for i in ires:
      if i in self.res:
        ret.append(i)
      else:
        raise Exception("unknown res {}".format(i))
    self.used_res = ret

param_list = [
  "m_A","m_B","m_C","m_D","m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D"
]
