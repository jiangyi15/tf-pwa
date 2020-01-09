import tensorflow as tf
import numpy as np
from .cg import get_cg_coef
from .d_function_new import d_function_cos
from .complex_F import Complex_F
from .res_cache import Particle,Decay
from .variable import Vars
from .dfun_tf import dfunctionJ,D_Cache
import os
from pysnooper import snoop
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import functools
from .breit_wigner import barrier_factor,breit_wigner_dict as bw_dict
from .amplitude import AllAmplitude
#print(bw_dict)

class AllAmplitude4(AllAmplitude):
  def __init__(self,res):
    super(AllAmplitude4,self).__init__(res)
  
  def cache_data(self,m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD, 
      Theta_BC,Theta_B_BC, phi_BC, phi_B_BC,
      Theta_BD,Theta_B_BD,phi_BD, phi_B_BD, 
      Theta_CD,Theta_D_CD, phi_CD,phi_D_CD,
      Theta_BD_B,Theta_BC_B,Theta_BD_D,Theta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2,
      alpha_E_BC,beta_E_BC,alpha_E_BD,beta_E_BD,alpha_E_CD,beta_E_CD,
      split=None,batch=None):
    D_fun_Cache = D_Cache
    if split is None and batch is None:
      ang_BD_B = D_fun_Cache(phi_BD_B, Theta_BD_B, phi_BD_B2)
      ang_BD_D = D_fun_Cache(phi_BD_D, Theta_BD_D, phi_BD_D2)
      ang_BD = D_fun_Cache(phi_BD, Theta_BD, 0.0)
      ang_B_BD = D_fun_Cache(phi_B_BD, Theta_B_BD, 0.0)
      ang_BC_B = D_fun_Cache(phi_BC_B, Theta_BC_B,phi_BC_B2)
      ang_BC = D_fun_Cache(phi_BC, Theta_BC,0.0)
      ang_B_BC = D_fun_Cache(phi_B_BC, Theta_B_BC,0.0)
      ang_CD_D = D_fun_Cache(phi_CD_D, Theta_CD_D,phi_CD_D2)
      ang_CD = D_fun_Cache(phi_CD, Theta_CD,0.0)
      ang_D_CD = D_fun_Cache(phi_D_CD, Theta_D_CD,0.0)
      ang_E_BC = D_fun_Cache(alpha_E_BC, beta_E_BC,0.0)
      ang_E_BD = D_fun_Cache(alpha_E_BD, beta_E_BD,0.0)
      ang_E_CD = D_fun_Cache(alpha_E_CD, beta_E_CD,0.0)
      return [m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D,ang_E_BC,ang_E_BD,ang_E_CD]
    else :
      data = [m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD, 
      Theta_BC,Theta_B_BC, phi_BC, phi_B_BC,
      Theta_BD,Theta_B_BD,phi_BD, phi_B_BD, 
      Theta_CD,Theta_D_CD, phi_CD,phi_D_CD,
      Theta_BD_B,Theta_BC_B,Theta_BD_D,Theta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2,
      alpha_E_BC,beta_E_BC,alpha_E_BD,beta_E_BD,alpha_E_CD,beta_E_CD
      ]
      n = m_BC.shape[0]
      if batch is None:
        l = (n+split-1) // split
      else:
        l = batch
        split = (n +batch-1)//batch
      ret = []
      for i in range(split):
        data_part = [ arg[i*l:min(i*l+l,n)] for arg in data]
        ret.append(self.cache_data(*data_part))
      return ret
  #@snoop()
  def get_amp2s_matrix(self,m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D,ang_E_BC,ang_E_BD,ang_E_CD):
    d = 3.0
    res_cache = self.Get_BWReson(m_A,m_B,m_C,m_D,m_BC,m_BD,m_CD)
    sum_A = 0.1
    ret = []
    for i in self.used_res:
      chain = self.res[i]["Chain"]
      if chain == 0:
        continue
      JReson = self.res[i]["J"]
      ParReson = self.res[i]["Par"]
      if chain < 0: # A->(DB)C
        lambda_BD = list(range(-JReson,JReson+1))
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        #print(i,H_0,H_1)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BD.get_lambda(1,[-1,1],lambda_BD,[0])
        df_b = ang_B_BD.get_lambda(JReson,lambda_BD,[-1,0,1],[-1,0,1])
        aligned_B = ang_BD_B(1)
        aligned_D = ang_BD_D(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(2,JReson*2+1,1,1,1,-1)) * tf.reshape(HD2,(1,JReson*2+1,3,1,3,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abxcdi = tf.reshape(abcdi,(2,3,1,1,3,-1)) * tf.reshape(aligned_B,(1,3,3,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        #abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
        #abcdi = tf.reduce_sum(abcdyi,3)
        e_d = ang_E_BD.get_lambda(1,[-1,0,1],[0],[0])
        s = abcdi * tf.reshape(e_d,(3,-1))
        s = tf.reduce_sum(s,axis=3)
        #s = tf.einsum("arci,rbdi,bxi,dyi->axcyi",HD1,HD2,aligned_B,aligned_D)
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        lambda_BD = list(range(-JReson,JReson+1))
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BC.get_lambda(1,[-1,1],lambda_BD,[-1,0,1])
        df_b = ang_B_BC.get_lambda(JReson,lambda_BD,[-1,0,1],[0])
        aligned_B = ang_BC_B(1)        
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(2,JReson*2+1,1,1,3,-1)) * tf.reshape(HD2,(1,JReson*2+1,3,1,1,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abxcdi = tf.reshape(abcdi,(2,3,1,1,3,-1)) * tf.reshape(aligned_B,(1,3,3,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        e_d = ang_E_BC.get_lambda(1,[-1,0,1],[0],[0])
        
        s = abcdi * tf.reshape(e_d,(3,-1))
        s = tf.reduce_sum(s,axis=3)
        #s = tf.einsum("ardi,rbci,bxi->axcdi",HD1,HD2,aligned_B)
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        lambda_BD = list(range(-JReson,JReson+1))
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_CD.get_lambda(1,[-1,1],lambda_BD,[-1,0,1])
        df_b = ang_D_CD.get_lambda(JReson,lambda_BD,[-1,0,1],[0])
        aligned_D = ang_CD_D(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(2,JReson*2+1,3,1,1,-1)) * tf.reshape(HD2,(1,JReson*2+1,1,1,3,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        #abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
        #abcdi = tf.reduce_sum(abcdyi,3)
        e_d = ang_E_CD.get_lambda(1,[-1,0,1],[0],[0])
        s = abcdi * tf.reshape(e_d,(3,-1))
        s = tf.reduce_sum(s,axis=3)
        #s = tf.einsum("arbi,rdci,dyi->abcyi",HD1,HD2,aligned_D)
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      else:
        pass
        #std::cerr << "unknown chain" << std::endl
    ret = tf.stack(ret)
    amp = tf.reduce_sum(ret,axis=0)
    amp2s = tf.math.real(amp*tf.math.conj(amp))
    sum_A = tf.reduce_sum(amp2s,[0,1,2])
    return sum_A
  

param_list = [
  "m_A", "m_B", "m_C", "m_D", "m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D",
  "alpha_E_BC","beta_E_BC","alpha_E_BD","beta_E_BD","alpha_E_CD","beta_E_CD",
]
