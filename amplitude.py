import tensorflow as tf
from cg import get_cg_coef
from d_function_new import d_function_cos
from complex_F import Complex_F

import os
import pysnooper
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
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
  denom = complex((m0 + m) * (m0 - m), -m0 * gamma)
  return denom.inverse()

def Gamma(m, gamma0, q, q0, L, m0,d):
  gammaM = gamma0 * (q / q0)**(2 * L + 1) * (m0 / m) * Bprime(L, q, q0, d)**2
  return gammaM

def Bprime(L, q, q0, d):
  z = (q * d)**2
  z0 = (q0 * d)**2
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

from dfun_tf import dfunctionJ
class ExpI_Cache(object):
  def __init__(self,phi,maxJ = 2):
    self.maxj = maxJ
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
  def __call__(self,j,m1,m2):
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
    self.coef = {}
    self.res = res.copy()
    self.res_cache = {}
    self.init_res_param()
    
    
  
  def init_res_param(self):
    const_first = True
    for i in self.res:
      self.init_res_param_sig(i,self.res[i],const_first=const_first)
      if const_first:
        const_first = False
    
  def init_res_param_sig(self,head,config,const_first=False):
    self.coef[head] = []
    chain = config["Chain"]
    if chain < 0:
        jc,jd,je = self.JC,self.JB,self.JD
    elif chain>0 and chain< 100:
        jc,jd,je = self.JD,self.JB,self.JC
    elif chain>100 :
        jc,jd,je = self.JB,self.JD,self.JC
    arg = self.gen_coef(head+"_",self.JA,config["J"],jc,self.ParA,config["Par"],-1,const_first)
    self.coef[head].append(arg)
    arg = self.gen_coef(head+"_d_",config["J"],jd,je,config["Par"],-1,-1,True)
    self.coef[head].append(arg)
    
  def gen_coef(self,head,ja,jb,jc,pa,pb,pc,const_first = False) :
    arg_list = []
    dl = 0 if pa*pb*pc == 1 else 1
    s_min = abs(jb-jc)
    s_max = jb + jc
    for s in range(s_min,s_max+1):
      for l in range(abs(ja-s),ja+s +1):
        if l%2 == dl :
          name = "{head}BLS_{l}_{s}".format(head=head,l=l,s=s)
          if const_first:
            tmp_r = self.add_weight(name=name+"r")
            tmp_i = self.add_weight(name=name+"i")
            arg_list.append(complex(tmp_r,tmp_i))
            const_first = False
          else :
            tmp_r = self.add_weight(name=name+"r")
            tmp_i = self.add_weight(name=name+"i")
            arg_list.append(complex(tmp_r,tmp_i))
    return arg_list
  
  def Get_BWReson(self,m_BC,m_BD,m_CD):
    ret ={}
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
        ret[i] = [p,p0,q,q0,BW(m_BD, m, g, q, q0, l, 3.0)]
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        p = Getp(self.m0_A, m_BC, self.m0_D)
        p0 = Getp(self.m0_A, m, self.m0_D)
        q = Getp(m_BC, self.m0_B, self.m0_C)
        q0 = Getp(m, self.m0_B, self.m0_C)
        l = GetMinL(J_reson, self.JB, self.JC,
                    P_reson, self.ParB, self.ParC)
        ret[i] = [p,p0,q,q0,BW(m_BC, m, g, q, q0, l, 3.0)]
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        p = Getp(self.m0_A, m_CD, self.m0_B)
        p0 = Getp(self.m0_A, m, self.m0_B)
        q = Getp(m_CD, self.m0_C, self.m0_D)
        q0 = Getp(m, self.m0_C, self.m0_D)
        l = GetMinL(J_reson, self.JC, self.JD,
                    P_reson, self.ParC, self.ParD)
        ret[i] = [p,p0,q,q0,BW(m_CD, m, g, q, q0, l, 3.0)]
      else :
        raise "unknown chain"
        ret[i]= complex(0, 0);
    return ret
  
  def GetA2BC_LS(self,idx,ja,jb,jc,pa,pb,pc,lambda_b,lambda_c,layer,q,q0,d):
    dl = 0 if pa * pb * pc == 1 else  1 # pa = pb * pc * (-1)^l
    s_min = abs(jb - jc);
    s_max = jb + jc;
    ns = s_max - s_min + 1
    ret = complex(0.0,0.0)
    ptr = 0
    for s in range(s_min,s_max+1):
      for l in range(abs(ja - s),ja + s +1 ):
        if l % 2 == dl :
          M = self.coef[idx][layer][ptr]
          ptr += 1
          
          ret = ret + M * \
               cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) * \
               cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c) * q**l * Bprime(l,q,q0,d) * tf.sqrt((2*l+1.0)/(2*ja+1.0))
    return ret
  
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
    sum_A= 0.0#tf.zeros(shape=m_BC.shape)
    for i_lambda_A in range(-self.JA,self.JA+1,2):
      sum_B = 0.0#tf.zeros(shape=m_BC.shape)
      for i_lambda_B in range(-self.JB,self.JB+1):
        sum_C = 0.0#tf.zeros(shape=m_BC.shape)
        for i_lambda_C in range(-self.JC,self.JC+1):
          sum_D = 0.0#tf.zeros(shape=m_BC.shape)
          for i_lambda_D in range( -self.JD,self.JD+1):
            amp = complex(0.0,0.0)#zeros.copy()
            for i in self.res:
              amp_reson = complex(0.0,0.0)
              # if(res[i]==0)continue;
              chain = self.res[i]["Chain"]
              if chain == 0:
                continue
              JReson = self.res[i]["J"]
              ParReson = self.res[i]["Par"]
              if chain < 0: # A->(DB)C
                for i_lambda_BD in range(-JReson,JReson+1):
                  for i_lambda_B_BD in range(-self.JB,self.JB+1):
                    for i_lambda_D_BD in range(-self.JD,self.JD+1):
                      angle_aligned = ang_BD_B(self.JB, i_lambda_B_BD, i_lambda_B) *\
                                      ang_BD_D(self.JD, i_lambda_D_BD, i_lambda_D)#, phi_BD_D,cosTheta_BD_D, phi_BD_D2)
                      H_A_DB_C = self.GetA2BC_LS(i, self.JA, JReson, self.JC, self.ParA, ParReson, self.ParC,
                                    i_lambda_BD, i_lambda_C, 0,res_cache[i][0],res_cache[i][1],3.0)
                      H_DB_D_B = self.GetA2BC_LS(i, JReson, self.JB, self.JD, ParReson, self.ParB, self.ParD,
                                    i_lambda_B_BD, i_lambda_D_BD, 1,res_cache[i][2],res_cache[i][3],3.0)
                      amp_reson = amp_reson + angle_aligned * \
                          H_A_DB_C * ang_BD(self.JA, i_lambda_A, i_lambda_BD - i_lambda_C) * \
                          H_DB_D_B * ang_B_BD(JReson, i_lambda_BD, i_lambda_B_BD - i_lambda_D_BD)#, phi_B_BD,cosTheta_B_BD,0.0)
              elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
                for i_lambda_B_BC in range(-self.JB,self.JB+1):
                  for i_lambda_BC in range(-JReson,JReson+1):
                    angle_aligned = ang_BC_B(self.JB, i_lambda_B_BC, i_lambda_B)#,phi_BC_B, cosTheta_BC_B,phi_BC_B2)

                    H_A_BC_D = self.GetA2BC_LS(i, self.JA, JReson, self.JD, self.ParA, ParReson, self.ParD,
                                  i_lambda_BC, i_lambda_D,0,res_cache[i][0],res_cache[i][1],3.0)
                    H_BC_B_C = self.GetA2BC_LS(i, JReson, self.JB, self.JC, ParReson, self.ParB, self.ParC,
                                  i_lambda_B_BC, i_lambda_C,1,res_cache[i][2],res_cache[i][3],3.0)
                    amp_reson = amp_reson + angle_aligned * \
                          H_A_BC_D * ang_BC(self.JA, i_lambda_A, i_lambda_BC - i_lambda_D) *\
                          H_BC_B_C * ang_B_BC(JReson, i_lambda_BC, i_lambda_B_BC - i_lambda_C)
                    #print(H_A_BC_D,Dfun_cos(self.JA, i_lambda_A, i_lambda_BC - i_lambda_D,phi_BC, cosTheta_BC,0.0))
              elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
                for i_lambda_CD in  range(-JReson,JReson+1):
                  for i_lambda_D_CD in range(-self.JD,self.JD+1):
                    angle_aligned = ang_CD_D(self.JD, i_lambda_D_CD, i_lambda_D)#,phi_CD_D, cosTheta_CD_D,phi_CD_D2)
                    H_A_CD_B = self.GetA2BC_LS(i, self.JA, JReson, self.JB, self.ParA, ParReson, self.ParB,
                                  i_lambda_CD, i_lambda_B, 0,res_cache[i][0],res_cache[i][1],3.0)
                    H_CD_C_D = self.GetA2BC_LS(i, JReson, self.JD, self.JC, ParReson, self.ParD, self.ParC,
                                  i_lambda_D_CD, i_lambda_C, 1,res_cache[i][2],res_cache[i][3],3.0)
                    amp_reson = amp_reson + angle_aligned * \
                          H_A_CD_B * ang_CD(self.JA, i_lambda_A, i_lambda_CD - i_lambda_B) * \
                          H_CD_C_D * ang_D_CD(JReson, i_lambda_CD, i_lambda_D_CD - i_lambda_C)#,phi_D_CD,cosTheta_D_CD,0.0)
              else:
                pass
                #std::cerr << "unknown chain" << std::endl;
              #print(i,amp_reson , res_cache[i][-1])
              amp = amp + amp_reson * res_cache[i][-1]
            amp2 = amp.rho2()
            sum_D = sum_D + amp2
          sum_C = sum_C + sum_D
        sum_B = sum_B + sum_C
      sum_A = sum_A + sum_B
    return sum_A
  
  def cache_data(self,m_BC, m_BD, m_CD, 
      cosTheta_BC,cosTheta_B_BC, phi_BC, phi_B_BC,
      cosTheta_BD,cosTheta_B_BD,phi_BD, phi_B_BD, 
      cosTheta_CD,cosTheta_D_CD, phi_CD,phi_D_CD,
      cosTheta_BD_B,cosTheta_BC_B,cosTheta_BD_D,cosTheta_CD_D,
      phi_BD_B,phi_BD_B2,phi_BC_B,phi_BC_B2,phi_BD_D,phi_BD_D2,phi_CD_D,phi_CD_D2,split=None):
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
  
  def get_amp2s_cache(self,m_BC, m_BD, m_CD,ang_BD,ang_B_BD,ang_BD_B,ang_BD_D,ang_BC,ang_B_BC,ang_BC_B,ang_CD,ang_D_CD,ang_CD_D):
    
    res_cache = self.Get_BWReson(m_BC,m_BD,m_CD)
    sum_A= 0.0#tf.zeros(shape=m_BC.shape)
    for i_lambda_A in range(-self.JA,self.JA+1,2):
      sum_B = 0.0#tf.zeros(shape=m_BC.shape)
      for i_lambda_B in range(-self.JB,self.JB+1):
        sum_C = 0.0#tf.zeros(shape=m_BC.shape)
        for i_lambda_C in range(-self.JC,self.JC+1):
          sum_D = 0.0#tf.zeros(shape=m_BC.shape)
          for i_lambda_D in range( -self.JD,self.JD+1):
            amp = complex(0.0,0.0)#zeros.copy()
            for i in self.res:
              amp_reson = complex(0.0,0.0)
              # if(res[i]==0)continue;
              chain = self.res[i]["Chain"]
              if chain == 0:
                continue
              JReson = self.res[i]["J"]
              ParReson = self.res[i]["Par"]
              if chain < 0: # A->(DB)C
                for i_lambda_BD in range(-JReson,JReson+1):
                  for i_lambda_B_BD in range(-self.JB,self.JB+1):
                    for i_lambda_D_BD in range(-self.JD,self.JD+1):
                      angle_aligned = ang_BD_B(self.JB, i_lambda_B_BD, i_lambda_B) *\
                                      ang_BD_D(self.JD, i_lambda_D_BD, i_lambda_D)#, phi_BD_D,cosTheta_BD_D, phi_BD_D2)
                      H_A_DB_C = self.GetA2BC_LS(i, self.JA, JReson, self.JC, self.ParA, ParReson, self.ParC,
                                    i_lambda_BD, i_lambda_C, 0,res_cache[i][0],res_cache[i][1],3.0)
                      H_DB_D_B = self.GetA2BC_LS(i, JReson, self.JB, self.JD, ParReson, self.ParB, self.ParD,
                                    i_lambda_B_BD, i_lambda_D_BD, 1,res_cache[i][2],res_cache[i][3],3.0)
                      amp_reson = amp_reson + angle_aligned * \
                          H_A_DB_C * ang_BD(self.JA, i_lambda_A, i_lambda_BD - i_lambda_C) * \
                          H_DB_D_B * ang_B_BD(JReson, i_lambda_BD, i_lambda_B_BD - i_lambda_D_BD)#, phi_B_BD,cosTheta_B_BD,0.0)
              elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
                for i_lambda_B_BC in range(-self.JB,self.JB+1):
                  for i_lambda_BC in range(-JReson,JReson+1):
                    angle_aligned = ang_BC_B(self.JB, i_lambda_B_BC, i_lambda_B)#,phi_BC_B, cosTheta_BC_B,phi_BC_B2)

                    H_A_BC_D = self.GetA2BC_LS(i, self.JA, JReson, self.JD, self.ParA, ParReson, self.ParD,
                                  i_lambda_BC, i_lambda_D,0,res_cache[i][0],res_cache[i][1],3.0)
                    H_BC_B_C = self.GetA2BC_LS(i, JReson, self.JB, self.JC, ParReson, self.ParB, self.ParC,
                                  i_lambda_B_BC, i_lambda_C,1,res_cache[i][2],res_cache[i][3],3.0)
                    amp_reson = amp_reson + angle_aligned * \
                          H_A_BC_D * ang_BC(self.JA, i_lambda_A, i_lambda_BC - i_lambda_D) *\
                          H_BC_B_C * ang_B_BC(JReson, i_lambda_BC, i_lambda_B_BC - i_lambda_C)
                    #print(H_A_BC_D,Dfun_cos(self.JA, i_lambda_A, i_lambda_BC - i_lambda_D,phi_BC, cosTheta_BC,0.0))
              elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
                for i_lambda_CD in  range(-JReson,JReson+1):
                  for i_lambda_D_CD in range(-self.JD,self.JD+1):
                    angle_aligned = ang_CD_D(self.JD, i_lambda_D_CD, i_lambda_D)#,phi_CD_D, cosTheta_CD_D,phi_CD_D2)
                    H_A_CD_B = self.GetA2BC_LS(i, self.JA, JReson, self.JB, self.ParA, ParReson, self.ParB,
                                  i_lambda_CD, i_lambda_B, 0,res_cache[i][0],res_cache[i][1],3.0)
                    H_CD_C_D = self.GetA2BC_LS(i, JReson, self.JD, self.JC, ParReson, self.ParD, self.ParC,
                                  i_lambda_D_CD, i_lambda_C, 1,res_cache[i][2],res_cache[i][3],3.0)
                    amp_reson = amp_reson + angle_aligned * \
                          H_A_CD_B * ang_CD(self.JA, i_lambda_A, i_lambda_CD - i_lambda_B) * \
                          H_CD_C_D * ang_D_CD(JReson, i_lambda_CD, i_lambda_D_CD - i_lambda_C)#,phi_D_CD,cosTheta_D_CD,0.0)
              else:
                pass
                #std::cerr << "unknown chain" << std::endl;
              #print(i,amp_reson , res_cache[i][-1])
              amp = amp + amp_reson * res_cache[i][-1]
            amp2 = amp.rho2()
            sum_D = sum_D + amp2
          sum_C = sum_C + sum_D
        sum_B = sum_B + sum_C
      sum_A = sum_A + sum_B
    return sum_A
  
  def call(self,x,cached=False):
    if cached:
      return self.get_amp2s_cache(*x)
    return self.get_amp2s(*x)
