import tensorflow as tf
import numpy as np
from cg import get_cg_coef
from d_function_new import d_function_cos
from complex_F import Complex_F
from res_cache import Particle,Decay
from variable import Vars
from dfun_tf import dfunctionJ
import os
import pysnooper
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import functools

complex = lambda x,y:Complex_F(tf,x,y)

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
  return tf.sqrt(q) / (2 * M_0)

def BW(m, m0,g0,q,q0,L,d):
  gamma = Gamma(m, g0, q, q0, L, m0, d)
  num = complex(1, 0)
  denom = tf.complex((m0 + m) * (m0 - m), -m0 * gamma)
  return 1.0/denom

def Gamma(m, gamma0, q, q0, L, m0,d):
  gammaM = gamma0 * (q / tf.cast(q0,q.dtype))**(2 * L + 1) * (tf.cast(m0,m.dtype) / m) * Bprime(L, q, q0, d)**2
  return gammaM

def Bprime(L, q, q0, d):
  z = (q * d)**2
  z0 = (tf.cast(q0,q.dtype) * d)**2
  if L == 0:
    return 1.0
  if L == 1:
    return tf.sqrt((1.0 + z0) / (1.0 + z))
  if L == 2:
    return tf.sqrt((9. + (3. + z0) * z0) / (9. + (3. + z) * z))
  if L == 3:
    return tf.sqrt((z0 * (z0 * (z0 + 6.) + 45.) + 225.) /
                (z * (z * (z + 6.) + 45.) + 225.))
  if L == 4:
    return tf.sqrt((z0 * (z0 * (z0 * (z0 + 10.) + 135.) + 1575.) + 11025.) /
                (z * (z * (z * (z + 10.) + 135.) + 1575.) + 11025.));
  if L == 5:
    return tf.sqrt(
        (z0 * (z0 * (z0 * (z0 * (z0 + 15.) + 315.) + 6300.) + 99225.) +
         893025.) /
        (z * (z * (z * (z * (z + 15.) + 315.) + 6300.) + 99225.) + 893025.));
  return 1.0


def barrier_factor(l,q,q0,d=3.0):
  ret = []
  for i in l:
    tmp = q**i * Bprime(i,q,q0,d)
    ret.append(tmp)
  return tf.stack(ret)

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

class ExpI_Cache(object):
  def __init__(self,phi,maxJ = 2):
    self.maxj = maxJ
    self.phi = phi
    a = tf.range(-maxJ,maxJ+1,1.0)
    a = tf.reshape(a,(-1,1))
    phi = tf.reshape(phi,(1,-1))
    mphi = tf.matmul(a,phi)
    self.sinphi = tf.sin(mphi)
    self.cosphi = tf.cos(mphi)
  def __call__(self,m):
    idx = m + self.maxj
    return complex(self.cosphi[idx],self.sinphi[idx])

class D_fun_Cache(object):
  def __init__(self,alpha,beta,gamma=0.0):
    self.alpha = ExpI_Cache(alpha)
    self.gamma = ExpI_Cache(gamma)
    self.beta = beta
    self.dfuncj = {}
  @functools.lru_cache()
  def __call__(self,j,m1=None,m2=None):
    if abs(m1) > j or abs(m2) > j:
      return 0.0
    if j not in self.dfuncj:
      self.dfuncj[j] = dfunctionJ(j)
      self.dfuncj[j].lazy_init(self.beta)
    d = self.dfuncj[j](m1,m2)
    return self.alpha(m1)*self.gamma(m2)*d

def Dfun_cos(j,m1,m2,alpha,cosbeta,gamma):
  tmp = complex(0.0,alpha * m1 + gamma * m2).exp() * dfunction(j, m1, m2, cosbeta)
  return tmp

def ExpI_all(maxJ,phi):
  a = tf.range(-maxJ,maxJ+1,1.0)
  a = tf.reshape(a,(-1,1))
  phi = tf.reshape(phi,(1,-1))
  if not isinstance(phi,float):
    a = tf.cast(a,phi.dtype)
  mphi = tf.matmul(a,phi)
  sinphi = tf.sin(mphi)
  cosphi = tf.cos(mphi)
  return tf.complex(cosphi,sinphi)

def Dfun_all(j,alpha,beta,gamma):
  d_fun = dfunctionJ(j)
  m = tf.range(-j,j+1)
  m1,m2=tf.meshgrid(m,m)
  d = d_fun(m1,m2,beta)
  expi_alpha = tf.reshape(ExpI_all(j,alpha),(2*j+1,1,-1))
  expi_gamma = tf.cast(tf.reshape(ExpI_all(j,gamma),(1,2*j+1,-1)),expi_alpha.dtype)
  #a = tf.tile(expi_alpha,[1,2*j+1,1])
  #b = tf.tile(expi_gamma,[2*j+1,1,1])
  return expi_alpha*expi_gamma * tf.complex(d,tf.zeros_like(d))

def delta_D_trans(j,la,lb,lc):
  """
  (ja,ja) -> (ja,jb,jc)
  """
  s = np.zeros(shape=(len(la),len(lb),len(lc),(2*j+1),(2*j+1)))
  for i_a in range(len(la)):
    for i_b in range(len(lb)):
      for i_c in range(len(lc)):
        delta = lb[i_b]-lc[i_c]
        if abs(delta) <= j:
          s[i_a][i_b][i_c][la[i_a]+j][delta+j] = 1.0
  return np.reshape(s,(len(la)*len(lb)*len(lc),(2*j+1)*(2*j+1)))
  

def Dfun_delta(ja,la,lb,lc,d):
  d = tf.reshape(d,((2*ja+1)*(2*ja+1),-1))
  t = delta_D_trans(ja,la,lb,lc)
  ret = tf.matmul(tf.cast(t,d.dtype),d)
  return tf.reshape(ret,(len(la),len(lb),len(lc),-1))

class D_Cache(object):
  def __init__(self,alpha,beta,gamma=0.0):
    self.alpha = alpha
    self.gamma = gamma
    self.beta = beta
    self.cachej = {}
  @functools.lru_cache()
  def __call__(self,j,m1=None,m2=None):
    if j not in self.cachej:
      self.cachej[j] = Dfun_all(j,self.alpha,self.beta,self.gamma)
    if m1 is None:
      return self.cachej[j]
    else :
      return self.cachej[m1+j][m2+j]

  def get_lambda(self,j,la,lb,lc):
    d = self(j)
    return Dfun_delta(j,la,lb,lc,d)

def fix_value(x):
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x,dtype=dtype)
    return x
  return f


class AllAmplitude(tf.keras.Model):
  def __init__(self,res):
    super(AllAmplitude,self).__init__()
    self.JA = 1;
    self.JB = 1;
    self.JC = 0;
    self.JD = 1;
    self.ParA = -1;
    self.ParB = -1;
    self.ParC = -1;
    self.ParD = -1;
    self.m0_A = 4.59925;
    self.m0_B = 2.01026;
    self.m0_C = 0.13957061;
    self.m0_D = 2.00685;
    self.A = Particle("A",self.m0_A,0,self.JA,self.ParA)
    self.B = Particle("B",self.m0_B,0,self.JB,self.ParB)
    self.C = Particle("C",self.m0_C,0,self.JC,self.ParC)
    self.D = Particle("D",self.m0_D,0,self.JD,self.ParD)
    self.add_var = Vars(self)
    self.res = res.copy()
    #if "Zc_4160" in self.res:
      #self.res["Zc_4160"]["m0"] = self.add_weight(name="Zc_4160_m0",initializer=fix_value(self.res["Zc_4160"]["m0"]))
    #
    self.res_decay = self.init_res_decay()
    self.coef = {}
    self.coef_norm = {}
    self.res_cache = {}
    self.init_res_param()
    
  def init_res_decay(self):
    ret = {}
    for i in self.res:
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      m0 = self.res[i]["m0"]
      g0 = self.res[i]["g0"]
      chain = self.res[i]["Chain"]
      tmp = Particle(i,m0,g0,J_reson,P_reson)
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
    if const_first:
      r = self.add_var(name=coef_head+"r",initializer=fix_value(1.0),trainable=True),
      i = self.add_var(name=head+"i",initializer=fix_value(0.0),trainable=False)
    else:
      r = self.add_var(name=coef_head+"r")
      i = self.add_var(name=head+"i")
    self.coef_norm[head] = [r,i]
    ls,arg = self.gen_coef(coef_head+"_",self.JA,config["J"],jc,self.ParA,config["Par"],-1,True)
    self.coef[head].append(arg)
    self.res_cache[head]["ls"].append(ls)
    ls,arg = self.gen_coef(coef_head+"_d_",config["J"],jd,je,config["Par"],-1,-1,True)
    self.coef[head].append(arg)
    self.res_cache[head]["ls"].append(ls)
    
  def gen_coef(self,head,ja,jb,jc,pa,pb,pc,const_first = False) :
    arg_list = []
    ls = []
    dl = 0 if pa*pb*pc == 1 else 1
    s_min = abs(jb-jc)
    s_max = jb + jc
    for s in range(s_min,s_max+1):
      for l in range(abs(ja-s),ja+s +1):
        if l%2 == dl :
          ls.append((l,s))
          name = "{head}BLS_{l}_{s}".format(head=head,l=l,s=s)
          if const_first:
            tmp_r = self.add_var(name=name+"r",initializer=fix_value(1.0),trainable=False)
            tmp_i = self.add_var(name=name+"i",initializer=fix_value(0.0),trainable=False)
            arg_list.append([tmp_r,tmp_i])
            const_first = False
          else :
            tmp_r = self.add_var(name=name+"r")
            tmp_i = self.add_var(name=name+"i")
            arg_list.append([tmp_r,tmp_i])
    return ls,arg_list
  
  def init_res_chain(self):
    for i in self.res:
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      self.res_cache[i]["cgls"] = []
      
      
  
  def Get_BWReson(self,m_BC,m_BD,m_CD):
    ret = {}
    for i in self.res:
      m = self.res[i]["m0"]
      g = self.res[i]["g0"]
      J_reson = self.res[i]["J"]
      P_reson = self.res[i]["Par"]
      chain = self.res[i]["Chain"]
      if (chain < 0) : # A->(DB)C
        p = Getp(self.m0_A, m_BD, self.m0_C)
        p0 = Getp(self.m0_A, m, self.m0_C)
        q = Getp(m_BD, self.m0_B, self.m0_D)
        q0 = Getp(m, self.m0_B, self.m0_D)
        l = GetMinL(J_reson, self.JB, self.JD,
                    P_reson, self.ParB, self.ParD)
        bw = BW(m_BD, m, g, q, q0, l, 3.0)
        ret[i] = [p,p0,q,q0,bw]
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        p = Getp(self.m0_A, m_BC, self.m0_D)
        p0 = Getp(self.m0_A, m, self.m0_D)
        q = Getp(m_BC, self.m0_B, self.m0_C)
        q0 = Getp(m, self.m0_B, self.m0_C)
        l = GetMinL(J_reson, self.JB, self.JC,
                    P_reson, self.ParB, self.ParC)
        bw = BW(m_BC, m, g, q, q0, l, 3.0)
        ret[i] = [p,p0,q,q0,bw]
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        p = Getp(self.m0_A, m_CD, self.m0_B)
        p0 = Getp(self.m0_A, m, self.m0_B)
        q = Getp(m_CD, self.m0_C, self.m0_D)
        q0 = Getp(m, self.m0_C, self.m0_D)
        l = GetMinL(J_reson, self.JC, self.JD,
                    P_reson, self.ParC, self.ParD)
        bw = BW(m_CD, m, g, q, q0, l, 3.0)
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
    cg_trans = tf.cast(self.res_decay[idx][layer].cg_matrix(),M_r.dtype)
    for l,s in self.res_cache[idx]["ls"][layer]:
      M = self.coef[idx][layer][ptr]
      ptr += 1
      ret = ret + M * \
               cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) * \
               cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c) * q**l * Bprime(l,q,q0,d) * tf.sqrt((2*l+1.0)/(2*ja+1.0))
    #print(tf.reshape(tf.matmul(tf.reshape(M_r,(1,-1)),tf.cast(self.res_decay[idx][layer].cg_matrix(),M_r.dtype)),(jb*2+1,jc*2+1,-1)))
    
    return ret
  
  def GetA2BC_LS_mat(self,idx,layer,q,q0,d):
    ja = self.res_decay[idx][layer].mother.J
    jb = self.res_decay[idx][layer].outs[0].J
    jc = self.res_decay[idx][layer].outs[1].J
    M_r = []
    M_i = []
    for r,i in self.coef[idx][layer]:
      M_r.append(r*tf.cos(i))
      M_i.append(r*tf.sin(i))
    M_r = tf.stack(M_r)
    M_i = tf.stack(M_i)
    l_s = self.res_decay[idx][layer].get_l_list()
    bf = barrier_factor(l_s,q,q0,d)
    norm = tf.sqrt((2*tf.cast(l_s,M_r.dtype)+1.0)/(2*ja+1.0))
    norm_r = tf.linalg.diag(M_r * norm)
    norm_i = tf.linalg.diag(M_i * norm)
    mdep_r = tf.matmul(norm_r,bf)
    mdep_i = tf.matmul(norm_i,bf)
    cg_trans = tf.cast(self.res_decay[idx][layer].cg_matrix(),M_r.dtype)
    H_r = tf.matmul(cg_trans,mdep_r)
    H_i = tf.matmul(cg_trans,mdep_i)
    ret = tf.reshape(tf.complex(H_r,H_i),(2*jb+1,2*jc+1,-1))
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
  
  def get_amp2s(self,m_BC, m_BD, m_CD, 
      cosTheta_BC,cosTheta_B_BC, phi_BC, phi_B_BC,
      cosTheta_BD,cosTheta_B_BD,phi_BD, phi_B_BD, 
      cosTheta_CD,cosTheta_D_CD, phi_CD,phi_D_CD,
      cosTheta_BD_B,cosTheta_BC_B,cosTheta_BD_D,cosTheta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2):
    D_fun_Cache = D_Cache
    res_cache = self.Get_BWReson(m_BC,m_BD,m_CD)
    ang_BD_B = D_fun_Cache(phi_BD_B,tf.acos(cosTheta_BD_B), phi_BD_B2)
    ang_BD_D = D_fun_Cache(phi_BD_D,tf.acos(cosTheta_BD_D), phi_BD_D2)
    ang_BD = D_fun_Cache(phi_BD,tf.acos(cosTheta_BD), 0.0)
    ang_B_BD = D_fun_Cache(phi_B_BD,tf.acos(cosTheta_B_BD), 0.0)
    ang_BC_B = D_fun_Cache(phi_BC_B, tf.acos(cosTheta_BC_B),phi_BC_B2)
    ang_BC = D_fun_Cache(phi_BC, tf.acos(cosTheta_BC),0.0)
    ang_B_BC = D_fun_Cache(phi_B_BC, tf.acos(cosTheta_B_BC),0.0)
    ang_CD_D = D_fun_Cache(phi_CD_D, tf.acos(cosTheta_CD_D),phi_CD_D2)
    ang_CD = D_fun_Cache(phi_CD, tf.acos(cosTheta_CD),0.0)
    ang_D_CD = D_fun_Cache(phi_D_CD, tf.acos(cosTheta_D_CD),0.0)
    d = 3.0
    sum_A = 0.1
    ret = []
    for i in self.res:
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
        abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        #s = tf.einsum("abcdi,bxi,dyi,i->axcyi",ad,aligned_B,aligned_D,res_cache[i][-1])
        ret.append(abcdi*res_cache[i][-1])
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
        #s = tf.einsum("ardi,rbci,bxi,i->axcdi",HD1,HD2,aligned_B,res_cache[i][-1])
        ret.append(abcdi*res_cache[i][-1])
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
        abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        #s = tf.einsum("arbi,rdci,dyi,i->abcyi",HD1,HD2,aligned_D,res_cache[i][-1])
        ret.append(abcdi*res_cache[i][-1])
      else:
        pass
        #std::cerr << "unknown chain" << std::endl
    ret = tf.stack(ret)
    amp = tf.reduce_sum(ret,axis=0)
    amp2s = tf.math.real(amp*tf.math.conj(amp))
    sum_A = tf.reduce_sum(amp2s,[0,1,2,3])
    return sum_A
  
  def cache_data(self,m_BC, m_BD, m_CD, 
      cosTheta_BC,cosTheta_B_BC, phi_BC, phi_B_BC,
      cosTheta_BD,cosTheta_B_BD,phi_BD, phi_B_BD, 
      cosTheta_CD,cosTheta_D_CD, phi_CD,phi_D_CD,
      cosTheta_BD_B,cosTheta_BC_B,cosTheta_BD_D,cosTheta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2,split=None):
    D_fun_Cache = D_Cache
    if split is None:
      ang_BD_B = D_fun_Cache(phi_BD_B,tf.acos(cosTheta_BD_B), phi_BD_B2)
      ang_BD_D = D_fun_Cache(phi_BD_D,tf.acos(cosTheta_BD_D), phi_BD_D2)
      ang_BD = D_fun_Cache(phi_BD,tf.acos(cosTheta_BD), 0.0)
      ang_B_BD = D_fun_Cache(phi_B_BD,tf.acos(cosTheta_B_BD), 0.0)
      ang_BC_B = D_fun_Cache(phi_BC_B, tf.acos(cosTheta_BC_B),phi_BC_B2)
      ang_BC = D_fun_Cache(phi_BC, tf.acos(cosTheta_BC),0.0)
      ang_B_BC = D_fun_Cache(phi_B_BC, tf.acos(cosTheta_B_BC),0.0)
      ang_CD_D = D_fun_Cache(phi_CD_D, tf.acos(cosTheta_CD_D),phi_CD_D2)
      ang_CD = D_fun_Cache(phi_CD, tf.acos(cosTheta_CD),0.0)
      ang_D_CD = D_fun_Cache(phi_D_CD, tf.acos(cosTheta_D_CD),0.0)
      return [m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D]
    else :
      data = [m_BC, m_BD, m_CD, 
      cosTheta_BC,cosTheta_B_BC, phi_BC, phi_B_BC,
      cosTheta_BD,cosTheta_B_BD,phi_BD, phi_B_BD, 
      cosTheta_CD,cosTheta_D_CD, phi_CD,phi_D_CD,
      cosTheta_BD_B,cosTheta_BC_B,cosTheta_BD_D,cosTheta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2]
      n = m_BC.shape[0]
      l = (n+split-1) // split
      ret = []
      for i in range(split):
        data_part = [ arg[i*l:min(i*l+l,n)] for arg in data]
        ret.append(self.cache_data(*data_part))
      return ret
  
  def get_amp2s_matrix(self,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D):
    d = 3.0
    res_cache = self.Get_BWReson(m_BC,m_BD,m_CD)
    sum_A = 0.1
    ret = []
    for i in self.res:
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
        abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
        abcdi = tf.reduce_sum(abcdyi,3)
        s = abcdi
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
        s = abcdi
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
        abcdyi = tf.reshape(abcdi,(2,3,1,3,1,-1))*tf.reshape(aligned_D,(1,1,1,3,3,-1))
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
