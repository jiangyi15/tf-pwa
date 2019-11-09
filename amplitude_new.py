import tensorflow as tf
from cg import get_cg_coef
from d_function import d_function_cos

def cg_coef(j1,j2,m1,m2,j,m):
    return get_cg_coef(j1,j2,m1,m2,j,m)

def dfunction(j,m1,m2,cos):
    return d_function_cos(j,m1,m2)(cos)

def Dfunction(j,m1,m2,alpha,cos_beta,gamma):
    return dfunction(j,m1,m2,cos_beta) * tf.exp(tf.complex(0.,-m1*alpha-m2*gamma))

def Getq(m,m1,m2):
    ms = m1+m2
    md = m1-m2
    temp = (m+ms)*(m-ms)*(m+md)*(m-md)
    q = tf.sqrt(temp)/2/m
    return q

def BarrierFactor(L,z):
    if L==1:
        return 1.+z
    if L==2:
        return 9.+(3.+z)*z
    if L==3:
        return z*(z*(z+6.)+45.)+225.
    if L==4:
        return z*(z*(z*(z+10.)+135.)+1575.)+11025.
    if L==5:
        return z*(z*(z*(z*(z+15.)+315.)+6300.)+99225.)+893025.
    return 1.

def Bprime(L,q,q0,d):
    temp = q*d
    z = temp*temp
    temp = q0*d
    z0 = temp*temp
    Bpr = tf.sqrt( BarrierFactor(L,z0) / BarrierFactor(L,z) )
    return Bpr

def RunningWidth(m,m0,width0,L,q,q0,d):
    Bpr = Bprime(L,q,q0,d)
    width = width0 * (q/q0)**(2*L+1) * (m0/m) * Bpr*Bpr
    return width

def BreitWigner(m,m0,width0,L,q,q0,d):
    width = RunningWidth(m,m0,width0,L,q,q0,d)
    BW = 1/tf.complex(m*m-m0*m0,m0*width)
    return BW

class AllAmplitude(tf.keras.Model): # Model(data)
    def __init__(self,resons):
        super(AllAmplitude,self).__init__()
        self.Ecom = 4.59925
        self.JA = 1 # virtual gamma
        self.JB = 1 # D*-
        self.JC = 0 # pi+
        self.JD = 1 # D*0
        self.ParA = -1
        self.ParB = -1
        self.ParC = -1
        self.ParD = -1
        self.m0_B = 2.01026
        self.m0_C = 0.13957061
        self.m0_D = 2.00685
        self.resons = resons.copy() # resonance parameters
        self.coef = {} # fitting parameters
        self.init_coef() # initialize fitting parameters

    def init_coef(self):
        for reson in self.resons:
            self.coef[reson] = []
            chain = self.resons[reson]["Chain"]
            if chain<0:
                jb,jc,jd = self.JC,self.JD,self.JB # b denotes the third particle
                pb,pc,pd = self.ParC,self.ParD,self.ParB
            elif chain>0 and chain<100:
                jb,jc,jd = self.JD,self.JB,self.JC
                pb,pc,pd = self.ParD,self.ParB,self.ParC
            elif chain>100 and chain<200:
                jb,jc,jd = self.JB,self.JC,self.JD
                pb,pc,pd = self.ParB,self.ParC,self.ParD
            coef = self.gen_coef(reson+"_st_",
                                self.JA,self.resons[reson]["J"],jb,
                                self.ParA,self.resons[reson]["Par"],pb)
            self.coef[reson].append(coef)
            coef = self.gen_coef(reson+"_nd_",
                                self.resons[reson]["J"],jc,jd,
                                self.resons[reson]["Par"],pc,pd)
            self.coef[reson].append(coef)

    def gen_coef(self,header,J1,J2,J3,P1,P2,P3): # 1 -> 2 3
        coef_list = []
        dl = not (P1*P2*P3==1) # P1==P2*P3*(-1)^l
        s_min = abs(J2-J3)
        s_max = J2+J3
        for s in range(s_min,s_max+1):
            for l in range(abs(J1-s),J1+s+1):
                if l%2==dl:
                    coef_name = "{header}GLS_{l}_{s}".format(header=header,l=l,s=s)
                    temp_r = self.add_weight(name=coef_name+"_r")
                    temp_i = self.add_weight(name=coef_name+"_i")
                    coef_list.append(tf.complex(temp_r,temp_i))
        return coef_list

    def GetMinL(self,J1,J2,J3,P1,P2,P3):
        dl = not (P1*P2*P3==1)
        s_min = abs(J2-J3)
        s_max = J2+J3
        minL = 10000
        # faster? list_s = tf.range(s_min,s_max+1,1)
        for s in tf.range(s_min,s_max+1,1):
            for l in tf.range(abs(J1-s),J1+s+1,1):
                if l%2==dl:
                    minL = tf.minimum(l,minL)
        return minL

    def GetBWreson(self,m_BC,m_BD,m_CD):
        BWreson = {}
        for reson in self.resons:
            m0 = self.resons[reson]["m0"]
            width0 = self.resons[reson]["g0"]
            J_reson = self.resons[reson]["J"]
            P_reson = self.resons[reson]["Par"]
            chain = self.resons[reson]["Chain"]
            if (chain<0) : # A->(DB)C
                q = Getq(m_BD, self.m0_B, self.m0_D)
                q0 = Getq(m0, self.m0_B, self.m0_D)
                L = self.GetMinL(   J_reson, self.JB, self.JD,
                                    P_reson, self.ParB, self.ParD )
                BWreson[reson] = BreitWigner(m_BD,m0,width0,L,q,q0,d=3.)
            elif (chain>0 and chain<100) : # A->(BC)D
                q = Getq(m_BC, self.m0_B, self.m0_C)
                q0 = Getq(m0, self.m0_B, self.m0_C)
                L = self.GetMinL(   J_reson, self.JB, self.JC,
                                    P_reson, self.ParB, self.ParC )
                BWreson[reson] = BreitWigner(m_BC,m0,width0,L,q,q0,d=3.)
            elif (chain>100 and chain<200) : # A->B(CD)
                q = Getq(m_CD, self.m0_C, self.m0_D)
                q0 = Getq(m0, self.m0_C, self.m0_D)
                L = self.GetMinL(   J_reson, self.JC, self.JD,
                                    P_reson, self.ParC, self.ParD )
                BWreson[reson] = BreitWigner(m_CD,m0,width0,L,q,q0,d=3.)
            else :
                raise "Unknown Chain!"
                BWreson[reson] = tf.complex(0.,0.)
        return BWreson

    def GetDynamicF(self,reson,layer, J1,J2,J3,P1,P2,P3,lmd2,lmd3, q,q0,d):
        dl = not (P1*P2*P3==1)
        s_min = abs(J2-J3)
        s_max = J2+J3
        DynamicF = tf.complex(0.,0.)
        ptr = 0
        for s in tf.range(s_min,s_max+1,1):
            for l in tf.range(abs(J1-s),J1+s+1,1):
                if l%2==dl:
                    gls = self.coef[reson][layer][ptr] # 'gls' are the fitting parameters
                    ptr+=1
                    DynamicF+= gls * tf.sqrt((2*l+1)/(2*J1+1)) * cg_coef(J2,J3,lmd2,-lmd3,s,lmd2-lmd3) * cg_coef(l,s,0,lmd2-lmd3,J1,lmd2-lmd3) * q**l * Bprime(l,q,q0,d)
        return DynamicF

    def GetAmp_sq( self,
            m_BC, phi_BC, cos_BC, phi_B_BC, cos_B_BC, alpha_B_BC, cosbeta_B_BC, gamma_B_BC,
            m_BD, phi_BD, cos_BD, phi_B_BD, cos_B_BD, alpha_B_BD, cosbeta_B_BD, gamma_B_BD, alpha_D_BD, cosbeta_D_BD, gamma_D_BD,
            m_CD, phi_CD, cos_CD, phi_D_CD, cos_D_CD, alpha_D_CD, cosbeta_D_CD, gamma_D_CD ):
        
        # Amp_sq_tot = 0.
        BWreson = self.GetBWreson(m_BC,m_BD,m_CD)
        lmd_A = tf.range(-1,2,2.) # gamma only has -1 and 1
        lmd_B = tf.range(-self.JB,self.JB+1,1.)
        lmd_C = tf.range(-self.JC,self.JC+1,1.)
        lmd_D = tf.range(-self.JD,self.JD+1,1.)
        Amp_com = 0 # Amp inside the absolute value sign
        for reson in self.resons:
            chain = self.resons[reson]["Chain"]
            J_reson = self.resons[reson]["J"]
            P_reson = self.resons[reson]["Par"]
            lmd_reson = tf.range(-J_reson,J_reson+1,1.)
            
            if chain<0:
                Lmd_A,Lmd_B,Lmd_C,Lmd_D,Lmd_BD,Lmd_B_BD,Lmd_D_BD = tf.meshgrid(lmd_A,lmd_B,lmd_C,lmd_D,lmd_reson,lmd_B,lmd_D)
                q_rt = Getq(self.Ecom,self.m0_C,m_BD)
                q0_rt = Getq(self.Ecom,self.m0_C,m0)
                q_nd = Getq(m_BD, self.m0_B, self.m0_D)
                q0_nd = Getq(m0, self.m0_B, self.m0_C)
                Amp =   GetDynamicF(reson,0, self.JA,J_reson,self.JC, self.ParA,P_reson,self.ParC, Lmd_BD,Lmd_C, q_rt,q0_rt,d=3.) * Dfunction(self.JA,Lmd_A,Lmd_BD-Lmd_C, phi_BD,cos_BD,0.) * \
                        GetDynamicF(reson,1, J_reson,self.JB,self.JD, P_reson,self.ParB,self.ParD, Lmd_B_BD,Lmd_D_BD, q_nd,q0_nd,d=3.) * Dfunction(J_reson,Lmd_BD,Lmd_B_BD-Lmd_D_BD, phi_B_BD,cos_B_BD,0.) * \
                        Dfunction(self.JB,Lmd_B_BD,Lmd_B, alpha_B_BD,cosbeta_B_BD,gamma_B_BD) * \
                        Dfunction(self.JD,Lmd_D_BD,Lmd_D, alpha_D_BD,cosbeta_D_BD,gamma_D_BD)
                Amp_reson = tf.reduce_sum(Amp,[4,5,6])
            
            elif chain>0 and chain<100:
                Lmd_A,Lmd_B,Lmd_C,Lmd_D,Lmd_BC,Lmd_B_BC = tf.meshgrid(lmd_A,lmd_B,lmd_C,lmd_D,lmd_reson,lmd_B)
                q_rt = Getq(self.Ecom,self.m0_D,m_BC)
                q0_rt = Getq(self.Ecom,self.m0_D,m0)
                q_nd = Getq(m_BC, self.m0_B, self.m0_C)
                q0_nd = Getq(m0, self.m0_B, self.m0_C)
                Amp =   GetDynamicF(reson,0, self.JA,J_reson,self.JD, self.ParA,P_reson,self.ParD, Lmd_BC,Lmd_D, q_rt,q0_rt,d=3.) * Dfunction(self.JA,Lmd_A,Lmd_BC-Lmd_D, phi_BC,cos_BC,0.) * \
                        GetDynamicF(reson,1, J_reson,self.JB,self.JC, P_reson,self.ParB,self.ParC, Lmd_B_BC,0, q_nd,q0_nd,d=3.) * Dfunction(J_reson,Lmd_BC,Lmd_B_BC-0, phi_B_BC,cos_B_BC,0.) * \
                        Dfunction(self.JB,Lmd_B_BC,Lmd_B, alpha_B_BC,cosbeta_B_BC,gamma_B_BC)
                Amp_reson = tf.reduce_sum(Amp,[4,5])
            
            elif chain>100 and chain<200:
                Lmd_A,Lmd_B,Lmd_C,Lmd_D,Lmd_CD,Lmd_D_CD = tf.meshgrid(lmd_A,lmd_B,lmd_C,lmd_D,lmd_reson,lmd_D)
                q_rt = Getq(self.Ecom,self.m0_B,m_CD)
                q0_rt = Getq(self.Ecom,self.m0_B,m0)
                q_nd = Getq(m_CD, self.m0_C, self.m0_D)
                q0_nd = Getq(m0, self.m0_C, self.m0_D)
                Amp =   GetDynamicF(reson,0, self.JA,J_reson,self.JB, self.ParA,P_reson,self.ParB, Lmd_CD,Lmd_B, q_rt,q0_rt,d=3.) * Dfunction(self.JA,Lmd_A,Lmd_CD-Lmd_B, phi_CD,cos_CD,0.) * \
                        GetDynamicF(reson,1, J_reson,self.JC,self.JD, P_reson,self.ParC,self.ParD, Lmd_D_CD,0, q_nd,q0_nd,d=3.) * Dfunction(J_reson,Lmd_CD,Lmd_D_CD-0, phi_D_CD,cos_D_CD,0.) * \
                        Dfunction(self.JD,Lmd_D_CD,Lmd_D, alpha_D_CD,cosbeta_D_CD,gamma_D_CD)
                Amp_reson = tf.reduce_sum(Amp,[4,5])

            Amp_com += Amp_reson

        Amp_sq = tf.square(tf.abs(Amp_com))
        Amp_sq_tot = tf.reduce_sum(Amp_sq)
        return Amp_sq_tot

    def call(self,var):
        return self.GetAmp_sq(*var)

