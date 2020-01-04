import tensorflow as tf
import numpy as np
from .cg import get_cg_coef
from .d_function_new import d_function_cos
from .complex_F import Complex_F
from .res_cache import Particle,Decay
from .variable import Vars
from .dfun_tf import dfunctionJ,D_Cache
import os
# from pysnooper import snoop
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import functools
from .breit_wigner import barrier_factor,breit_wigner_dict as bw_dict

#print(bw_dict)

def is_complex(x):
  try:
    y = complex(x)
  except:
    return False
  return True

def dfunction(j,m1,m2,cos_theta):
  return d_function_cos(j,m1,m2)(cos_theta)

def cg_coef(j1,j2,m1,m2,j,m):
  ret = get_cg_coef(j1,j2,m1,m2,j,m)
  #print(j1,j2,m1,m2,j,m,ret)
  return ret

def Getp(M_0, M_1, M_2) :
  M12S = M_1 + M_2
  M12D = M_1 - M_2
  p = (M_0 - M12S) * (M_0 + M12S) * (M_0 - M12D) * (M_0 + M12D)
  q = (p + tf.abs(p))/2
  #print(M_0,M_1,M_2,tf.sqrt(p) / (2 * M_0))
  return tf.sqrt(q) / (2 * M_0)

def GetMinL(J1,J2,J3,P1,P2,P3):
  dl = not (P1*P2*P3==1)
  s_min = abs(J2-J3)
  s_max = J2+J3
  minL = 10000
  for s in range(s_min,s_max+1,1):
    for l in range(abs(J1-s),J1+s+1,1):
      if l%2==dl:
        minL = min(l,minL)
  return minL

def fix_value(x):
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x,dtype=dtype)
    return x
  return f

class AllAmplitude(tf.keras.Model):
  def __init__(self,res):
    super(AllAmplitude,self).__init__()
    self.JA = 1
    self.JB = 1
    self.JC = 0
    self.JD = 1
    self.ParA = -1
    self.ParB = -1
    self.ParC = -1
    self.ParD = -1
    self.m0_A = 4.59925
    self.m0_B = 2.01026
    self.m0_C = 0.13957061
    self.m0_D = 2.00685
    
    self.A = Particle("A",self.JA,self.ParA,[-1,1])
    self.B = Particle("B",self.JB,self.ParB)
    self.C = Particle("C",self.JC,self.ParC)
    self.D = Particle("D",self.JD,self.ParD)
    self.add_var = Vars(self)
    self.res = res.copy()
    self.polar = True
    #if "Zc_4160" in self.res:
      #self.res["Zc_4160"]["m0"] = self.add_var(name="Zc_4160_m0",var = self.res["Zc_4160"]["m0"],trainable=True)
      #self.res["Zc_4160"]["g0"] = self.add_var(name="Zc_4160_g0",var = self.res["Zc_4160"]["g0"],trainable=True)
    self.res_decay = self.init_res_decay()
    self.coef = {}
    self.coef_norm = {}
    self.res_cache = {}
    self.init_res_param()
    self.init_used_res()
  
  def init_used_res(self):
    self.used_res = [i for i in self.res]
    
    
  def init_res_decay(self):
    ret = {}
    for i in self.res:
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      m0 = self.res[i]["m0"]
      g0 = self.res[i]["g0"]
      chain = self.res[i]["Chain"]
      if "bw" in self.res[i]:
        self.res[i]["bwf"] = bw_dict[self.res[i]["bw"]]
      else:
        self.res[i]["bwf"] = bw_dict["default"]
      tmp = Particle(i,J_reson,P_reson)
      if (chain < 0) : # A->(DB)C
        d_tmp_0 = Decay(i+"_0",self.A,[tmp,self.C])
        d_tmp_1 = Decay(i+"_1",tmp,[self.B,self.D])
        ret[i] = [d_tmp_0,d_tmp_1]
      elif (chain > 0 and chain < 100) : # A->(BC)D 
        d_tmp_0 = Decay(i+"_0",self.A,[tmp,self.D])
        d_tmp_1 = Decay(i+"_1",tmp,[self.B,self.C])
        ret[i] = [d_tmp_0,d_tmp_1]
      elif (chain > 100 and chain < 200) : # A->B(CD) 
        d_tmp_0 = Decay(i+"_0",self.A,[tmp,self.B])
        d_tmp_1 = Decay(i+"_1",tmp,[self.D,self.C])
        ret[i] = [d_tmp_0,d_tmp_1]
      else :
        raise "unknown chain"
    return ret
  
  def init_res_param(self):
    const_first = True
    for i in self.res:
      if "total" in self.res[i]:
        const_first = False
    res_tmp = [i for i in self.res]
    res_all = []
    # order for coef_head
    while len(res_tmp) > 0:
      i = res_tmp.pop()
      if "coef_head" in self.res[i]:
        coef_head = self.res[i]["coef_head"]
        if coef_head in res_tmp:
          res_all.append(coef_head)
          res_tmp.remove(coef_head)
      res_all.append(i)
    for i in res_all:
      self.init_res_param_sig(i,self.res[i],const_first=const_first)
      if const_first:
        const_first = False
    
  def init_res_param_sig(self,head,config,const_first=False):
    self.res_cache[head] = {}
    self.res_cache[head]["ls"] = []
    self.coef[head] = []
    chain = config["Chain"]
    coef_head = head
    if "coef_head" in config:
      coef_head = config["coef_head"]
    if chain < 0:
        jc,jd,je = self.JC,self.JB,self.JD
    elif chain>0 and chain< 100:
        jc,jd,je = self.JD,self.JB,self.JC
    elif chain>100 :
        jc,jd,je = self.JB,self.JD,self.JC
    if "total" in config:
      N_tot = config["total"]
      if is_complex(N_tot):
        N_tot = complex(N_tot)
        rho,phi = N_tot.real,N_tot.imag
      else:
        rho,phi = N_tot
      r = self.add_var(name=coef_head+"r",initializer=fix_value(rho),trainable=False)
      i = self.add_var(name=head+"i",initializer=fix_value(phi),trainable=False)
    elif const_first:
      r = self.add_var(name=coef_head+"r",initializer=fix_value(1.0),trainable=False)
      i = self.add_var(name=head+"i",initializer=fix_value(0.0),trainable=False)
    else:
      r = self.add_var(name=coef_head+"r",size=2.0)
      i = self.add_var(name=head+"i",size=6.283185307179586)
    self.coef_norm[head] = [r,i]
    if "const" in config:
      const = list(config["const"])
    else:
      const = [0,0]
    ls,arg = self.gen_coef(head,0,coef_head+"_",const[0])
    self.coef[head].append(arg)
    self.res_cache[head]["ls"].append(ls)
    ls,arg = self.gen_coef(head,1,coef_head+"_d_",const[1])
    self.coef[head].append(arg)
    self.res_cache[head]["ls"].append(ls)
    return const_first
    
  def gen_coef(self,idx,layer,coef_head,const = 0) :
    if const is None:
      const = 0
    if isinstance(const,int):
      const = [const]
    ls = self.res_decay[idx][layer].get_ls_list()
    n_ls = len(ls)
    const_list = []
    for i in const:
      if i<0:
        const_list.append(n_ls + i)
      else:
        const_list.append(i)
    arg_list = []
    for i in range(n_ls):
      l,s = ls[i]
      name = "{head}BLS_{l}_{s}".format(head=coef_head,l=l,s=s)
      if i in const_list:
        tmp_r = self.add_var(name=name+"r",initializer=fix_value(1.0),trainable=False)
        tmp_i = self.add_var(name=name+"i",initializer=fix_value(0.0),trainable=False)
        arg_list.append([tmp_r,tmp_i])
      else :
        if self.polar:
          tmp_r = self.add_var(name=name+"r",size=2.0)
          tmp_i = self.add_var(name=name+"i",size=6.283185307179586)
        else:
          tmp_r = self.add_var(name=name+"r",range=(-1,1))
          tmp_i = self.add_var(name=name+"i",range=(-1,1))
        arg_list.append([tmp_r,tmp_i])
    return ls,arg_list
  
  def init_res_chain(self):
    for i in self.res:
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      self.res_cache[i]["cgls"] = []
  
  def Get_BWReson(self,m_A,m_B,m_C,m_D,m_BC,m_BD,m_CD):
    ret = {}
    for i in self.used_res:
      m = self.res[i]["m0"]
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
        bw = self.res[i]["bwf"](m_BD, m, g, q, q0, l, 3.0)
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
        raise "unknown chain"
    return ret
  
  def GetA2BC_LS(self,idx,ja,jb,jc,pa,pb,pc,lambda_b,lambda_c,layer,q,q0,d):
    dl = 0 if pa * pb * pc == 1 else  1 # pa = pb * pc * (-1)^l
    s_min = abs(jb - jc);
    s_max = jb + jc;
    ns = s_max - s_min + 1
    ret = complex(0.0,0.0)
    ptr = 0
    M_r = []
    M_i = []
    for i in self.coef[idx][layer]:
      M_r.append(i.r)
      M_i.append(i.i)
    M_r = tf.stack(M_r)
    M_i = tf.stack(M_i)
    l_s = self.res_decay[idx][layer].get_l_list()
    
    ls_norm = tf.linalg.diag(M_r * tf.sqrt((2*tf.cast(l_s,M_r.dtype)+1.0)/(2*ja+1.0)))
    mdep = tf.matmul(ls_norm,barrier_factor(l_s,q,q0,d))
    cg_trans = tf.cast(self.res_decay[idx][layer].cg_matrix,M_r.dtype)
    for l,s in self.res_cache[idx]["ls"][layer]:
      M = self.coef[idx][layer][ptr]
      ptr += 1
      ret = ret + M * \
               cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) * \
               cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c) * q**l * Bprime(l,q,q0,d) * tf.sqrt((2*l+1.0)/(2*ja+1.0))
    #print(tf.reshape(tf.matmul(tf.reshape(M_r,(1,-1)),tf.cast(self.res_decay[idx][layer].cg_matrix(),M_r.dtype)),(jb*2+1,jc*2+1,-1)))
    
    return ret
  
  #@snoop()
  def GetA2BC_LS_mat(self,idx,layer,q,q0,d):
    decay = self.res_decay[idx][layer]
    ja = decay.mother.J
    jb = decay.outs[0].J
    jc = decay.outs[1].J
    M_r = []
    M_i = []
    for r,i in self.coef[idx][layer]:
      M_r.append(r)
      M_i.append(i)
    M_r = tf.stack(M_r)
    M_i = tf.stack(M_i)
    bf = decay.barrier_factor(q,q0)
    # switch for rectangular and polar coordinates of params
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
    #print(idx,layer,ret)
    return ret
  
  def get_res_total(self,idx):
    r,i =  self.coef_norm[idx]
    M_r = r*tf.cos(i)
    M_i = r*tf.sin(i)
    return tf.complex(M_r,M_i) 

  
  @staticmethod
  def GetA2BC_LS_list(ja,jb,jc,pa,pb,pc):
    dl = 0 if pa * pb * pc == 1 else  1 # pa = pb * pc * (-1)^l
    s_min = abs(jb - jc);
    s_max = jb + jc;
    ns = s_max - s_min + 1
    ret = []
    for s in range(s_min,s_max+1):
      for l in range(abs(ja - s),ja + s +1 ):
        if l % 2 == dl :
          ret.append((l,s))
    return ret
  
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
    D_fun_Cache = D_Cache
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
  
  def get_amp2s_matrix(self,m_A,m_B,m_C,m_D,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D):
    d = 3.0
    res_cache = self.Get_BWReson(m_A,m_B,m_C,m_D,m_BC,m_BD,m_CD)
    sum_A = 0.1
    ret = []
    ns_a = 2
    ns_b = 3
    ns_c = 1
    ns_d = 3
    for i in self.used_res:
      chain = self.res[i]["Chain"]
      if chain == 0:
        continue
      JReson = self.res[i]["J"]
      ParReson = self.res[i]["Par"]
      if chain < 0: # A->(DB)C
        lambda_BD = list(range(-JReson,JReson+1))
        ns_bd = len(lambda_BD)
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        #print(i,H_0,H_1)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BD.get_lambda(self.A.J,self.A.spins,lambda_BD,self.C.spins)
        df_b = ang_B_BD.get_lambda(JReson,lambda_BD,self.B.spins,self.D.spins)
        aligned_B = ang_BD_B.get_lambda(self.B.J,self.B.spins,self.B.spins)#(1)
        aligned_D = ang_BD_D.get_lambda(self.D.J,self.D.spins,self.D.spins)#(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(ns_a,ns_bd,1,ns_c,1,-1)) * tf.reshape(HD2,(1,ns_bd,ns_b,1,ns_d,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abxcdi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_c,ns_d,-1)) * tf.reshape(aligned_B,(1,ns_b,ns_b,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        abcdyi = tf.reshape(abcdi,(ns_a,ns_b,ns_c,ns_d,1,-1))*tf.reshape(aligned_D,(1,1,1,ns_d,ns_d,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        s = abcdi
        #s = tf.einsum("arci,rbdi,bxi,dyi->axcyi",HD1,HD2,aligned_B,aligned_D)
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        lambda_BC = list(range(-JReson,JReson+1))
        ns_bc = len(lambda_BC)
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_BC.get_lambda(self.A.J,self.A.spins,lambda_BC,self.D.spins)
        df_b = ang_B_BC.get_lambda(JReson,lambda_BC,self.B.spins,self.C.spins)
        aligned_B = ang_BC_B.get_lambda(self.B.J,self.B.spins,self.B.spins)#(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        arbcdi = tf.reshape(HD1,(ns_a,ns_bc,1,1,ns_d,-1)) * tf.reshape(HD2,(1,ns_bc,ns_b,ns_c,1,-1))
        #print(arbcdi)
        abcdi = tf.reduce_sum(arbcdi,1)
        #print(tf.reshape(abcdi,(2,JReson*2+1,3,1,1,3,-1)) * tf.reshape((aligned_B),(1,1,3,3,1,1,-1)))
        abxcdi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_c,ns_d,-1)) * tf.reshape(aligned_B,(1,ns_b,ns_b,1,1,-1))
        abcdi = tf.reduce_sum(abxcdi,1)
        s = abcdi
        #s = tf.einsum("ardi,rbci,bxi->axcdi",HD1,HD2,aligned_B)
        #print(res_cache[i][-1],self.get_res_total(i))
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        lambda_CD = list(range(-JReson,JReson+1))
        ns_cd  = len(lambda_CD)
        H_0 = self.GetA2BC_LS_mat(i,0,res_cache[i][0],res_cache[i][1],d)
        H_1 = self.GetA2BC_LS_mat(i,1,res_cache[i][2],res_cache[i][3],d)
        df_a = ang_CD.get_lambda(self.A.J,self.A.spins,lambda_CD,self.B.spins)
        df_b = ang_D_CD.get_lambda(JReson,lambda_CD,self.D.spins,self.C.spins)
        aligned_D = ang_CD_D.get_lambda(self.D.J,self.D.spins,self.D.spins)#(1)
        HD1 = H_0*df_a
        HD2 = H_1*df_b
        ## TODO c,d order
        # HD2 = tf.transpose(HD2, perm=[0, 2, 1, 3])
        arbcdi = tf.reshape(HD1,(ns_a,ns_cd,ns_b,1,1,-1)) * tf.reshape(HD2,(1,ns_cd,1,ns_c,ns_d,-1))
        abcdi = tf.reduce_sum(arbcdi,1)
        abcdyi = tf.reshape(abcdi,(ns_a,ns_b,1,ns_d,1,-1))*tf.reshape(aligned_D,(1,1,1,ns_d,ns_d,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        s = abcdi 
        #s = tf.einsum("arbi,rdci,dyi->abcyi",HD1,HD2,aligned_D)
        ret.append(s*res_cache[i][-1]*self.get_res_total(i))
      else:
        pass
        #std::cerr << "unknown chain" << std::endl
    ret = tf.stack(ret)
    amp = tf.reduce_sum(ret,axis=[0])
    amp2s = tf.math.real(amp*tf.math.conj(amp))
    sum_A = tf.reduce_sum(amp2s,[0,1,2,3])
    return sum_A
  
  
  def call(self,x,cached=False):
    if cached:
      return self.get_amp2s_matrix(*x)
    return self.get_amp2s(*x)
  
  def get_params(self):
    ret = {}
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
        raise "unknow res {}".format(i)
    self.used_res = ret

param_list = [
  "m_A","m_B","m_C","m_D","m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D"
]
