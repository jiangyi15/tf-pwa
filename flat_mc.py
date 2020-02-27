#!/usr/bin/env python3
from tf_pwa.phasespace import  PhaseSpaceGenerator
import numpy as np

def flat_mc(number,outfile):
  a = PhaseSpaceGenerator(4.59925172,[2.01028,0.13957,2.00698]) #from data???
  flat_mc_data = a.generate(number)
  #print(w,flat_mc_data)
  #print(flat_mc_data[0].M())
  #print(flat_mc_data[1].M())
  #print(flat_mc_data[2].M())
  pb = flat_mc_data[0]
  pb_a = np.array([pb.T,pb.X,pb.Y,pb.Z]).reshape((4,-1))
  pc = flat_mc_data[1]
  pc_a = np.array([pc.T,pc.X,pc.Y,pc.Z]).reshape((4,-1))
  pd = flat_mc_data[2]
  pd_a = np.array([pd.T,pd.X,pd.Y,pd.Z]).reshape((4,-1))
  print(pd_a)
  
  pa = np.array([pd_a,pb_a,pc_a])
  print(pa.shape)
  pa = np.transpose(pa,(2,0,1)).reshape((-1,4))
  print(pa.shape)
  np.savetxt(outfile,pa)  # 一个不包含探测器效率的MC样本

if __name__=="__main__":
  flat_mc(300000,"data/flat_mc30w.dat")
