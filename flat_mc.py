from tf_pwa.phasespace import  PhaseSpaceGenerator
import numpy as np

def main():
  a = PhaseSpaceGenerator(4.59925,[2.01026,0.13957061,2.00685])
  flat_mc_data = a.generate(1000000)
  #print(w,flat_mc_data)
  #print(flat_mc_data[0].M())
  #print(flat_mc_data[1].M())
  #print(flat_mc_data[2].M())
  pb = flat_mc_data[0]
  pb_a = np.array([pb.T,pb.X,pb.Y,pb.Z]).reshape((4,-1))
  pc = flat_mc_data[1]
  pc_a = np.array([pc.T,pc.X,pc.Y,pc.Z]).reshape((4,-1))
  pd = flat_mc_data[2]
  pd_a = np.array([pd.T,pd.X,pd.Y,pd.Z])
  #print(pd_a)
  pd_a = pd_a.reshape((4,-1))
  
  pa = np.array([pd_a,pb_a,pc_a])
  #print(pa.shape)
  pa = np.transpose(pa,(2,0,1))
  np.savetxt("data/flat_mc.dat",pa.reshape((-1,4)))

if __name__=="__main__":
  main()
