import tensorflow as tf
from cg import get_cg_coef
from d_function import d_function_cos
from complex_F import Complex_F

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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
  num = complex(m0 * gamma, 0)
  denom = complex((m0 + m) * (m0 - m), -m0 * gamma)
  return (num / denom)

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
    self.m0_B = 2.01026;
    self.m0_C = 0.13957061;
    self.m0_D = 2.00685;
    self.coef = {}
    self.res = res.copy()
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
        jc,jd,je = 0,1,1
    elif chain>0 and chain< 100:
        jc,jd,je = 1,1,0
    elif chain>100 :
        jc,jd,je = 1,0,1
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
      chain = self.res[i]["Chain"]
      if (chain < 0) : # A->(DB)C
        q = Getp(m_BD, self.m0_B, self.m0_D)
        q0 = Getp(m, self.m0_B, self.m0_D)
        ret[i] = BW(m_BD, m, g, q, q0, 0, 3)
      elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
        q = Getp(m_BC, self.m0_B, self.m0_C)
        q0 = Getp(m, self.m0_B, self.m0_C)
        ret[i] = BW(m_BC, m, g, q, q0, 0, 3)
      elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
        q = Getp(m_CD, self.m0_C, self.m0_D)
        q0 = Getp(m, self.m0_C, self.m0_D)
        ret[i] = BW(m_CD, m, g, q, q0, 0, 1000)
      else :
        raise "unknown chain"
        ret[i]= complex(0, 0);
    return ret
  
  def GetA2BC_LS(self,idx,ja,jb,jc,pa,pb,pc,lambda_b,lambda_c,layer):
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
               cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
    return ret
  
  def get_amp2s(self,m_BC, m_BD, m_CD, cosTheta_BC,
      cosTheta_B_BC, phi_BC, phi_B_BC,
      cosTheta_BD, cosTheta_D_BD,
      phi_D_BD, cosTheta_CD,
      cosTheta_C_CD, phi_CD,
      phi_C_CD,cosTheta1, cosTheta2, phi1, phi2):
    
    v_BWReson = self.Get_BWReson(m_BC,m_BD,m_CD)
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
                for i_lambda_DB in range(-JReson,JReson+1):
                  H_A_DB_C = self.GetA2BC_LS(i, self.JA, JReson, self.JC, self.ParA, ParReson, self.ParC,
                                i_lambda_DB, i_lambda_C, 0)
                  H_DB_D_B = self.GetA2BC_LS(i, JReson, self.JD, self.JB, ParReson, self.ParD, self.ParB,
                                i_lambda_D, i_lambda_B, 1)
                  amp_reson = amp_reson +  H_A_DB_C * \
                      complex(0.0,-1.0 * i_lambda_DB * 0.0).exp() * \
                      dfunction(self.JA, i_lambda_A, i_lambda_DB - i_lambda_C, cosTheta_BD) * \
                      H_DB_D_B * \
                      complex(0.0,-1.0 * i_lambda_D * phi_D_BD).exp() * \
                      dfunction(JReson, i_lambda_DB, i_lambda_D - i_lambda_B, cosTheta_D_BD);
              elif (chain > 0 and chain < 100) : # A->(BC)D aligned B
                for i_lambda_BC in range(-JReson,JReson+1):
                  for i_lambda_B_BC in range(-self.JB,self.JB+1):
                    angle_aligned = complex(0.0,-1.0 * (i_lambda_B_BC * phi1)).exp() * \
                        dfunction(self.JB, i_lambda_B_BC, i_lambda_B, cosTheta1);

                    H_A_BC_D = self.GetA2BC_LS(i, self.JA, JReson, self.JD, self.ParA, ParReson, self.ParD,
                                  i_lambda_BC, i_lambda_D,0);
                    H_BC_B_C = self.GetA2BC_LS(i, JReson, self.JB, self.JC, ParReson, self.ParB, self.ParC,
                                  i_lambda_B_BC, i_lambda_C,1);
                    amp_reson = amp_reson + angle_aligned * H_A_BC_D * \
                        complex(0.0, -1.0 * (i_lambda_BC * phi_BC)).exp() * \
                        dfunction(self.JA, i_lambda_A, i_lambda_BC - i_lambda_D, cosTheta_BC) * \
                        H_BC_B_C * complex(0.0, -1.0 * (i_lambda_B_BC * phi_B_BC)).exp() * \
                        dfunction(JReson, i_lambda_BC, i_lambda_B_BC - i_lambda_C,
                          cosTheta_B_BC);
              elif (chain > 100 and chain < 200) : # A->B(CD) aligned D
                for i_lambda_CD in  range(-JReson,JReson+1):
                  for i_lambda_C_CD in range(-self.JD,self.JD+1):
                    angle_aligned = complex(0.0, -1.0 * (i_lambda_C_CD * phi2)).exp() * \
                        dfunction(self.JD, i_lambda_C_CD, i_lambda_D, cosTheta2)
                    H_A_CD_B = self.GetA2BC_LS(i, self.JA, JReson, self.JB, self.ParA, ParReson, self.ParB,
                                  i_lambda_CD, i_lambda_B, 0);
                    H_CD_C_D = self.GetA2BC_LS(i, JReson, self.JC, self.JD, ParReson, self.ParC, self.ParD,
                                  i_lambda_C, i_lambda_C_CD, 1);
                    amp_reson = amp_reson + angle_aligned * H_A_CD_B *complex(0.0, -1 * (i_lambda_CD * phi_CD)).exp() * \
                        dfunction(self.JA, i_lambda_A, i_lambda_CD - i_lambda_B, cosTheta_CD) * \
                         H_CD_C_D * complex(0.0, -1 * (i_lambda_C_CD * phi_C_CD)).exp() * \
                        dfunction(JReson, i_lambda_CD, i_lambda_C_CD - i_lambda_C,
                          cosTheta_C_CD)
              else:
                pass
                #std::cerr << "unknown chain" << std::endl;
              amp = amp + amp_reson * v_BWReson[i];
            amp2 = amp.rho2()
            sum_D = sum_D + amp2
          sum_C = sum_C + sum_D
        sum_B = sum_B + sum_C
      sum_A = sum_A + sum_B
    return sum_A
  
  def call(self,x):
    return self.get_amp2s(*x)
