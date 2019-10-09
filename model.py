import tensorflow as tf
from amplitude import AllAmplitude


class Model:
  def __init__(self,res,w_bkg = 0):
    self.Amp = AllAmplitude(res)
    self.w_bkg = w_bkg
    
  def nll(self,data,bg,mcdata):
    ln_data = tf.reduce_sum(tf.math.log(self.Amp(data)))
    ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg)))
    int_mc = tf.reduce_mean(self.Amp(mcdata))
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    return -(ln_data - self.w_bkg * ln_bg - (n_data - self.w_bkg*n_bg) * int_mc)
  

param_list = [
  "m_BC","m_BD","m_CD",
  "cosTheta_BC","cosTheta_B_BC",
  "phi_BC", "phi_B_BC",
  "cosTheta_BD","cosTheta_D_BD",
  "phi_D_BD",
  "cosTheta_CD","cosTheta_C_CD",
  "phi_CD","phi_C_CD",
  "cosTheta1","cosTheta2",
  "phi1","phi2"
]

config_list = {"D2_2460"
    :{
        "m0":2.4607,
        "m_min":2.4603,
        "m_max":2.4611,
        "g0":0.0475,
        "g_min":0.0464,
        "g_max":0.0486,
        "J":2,
        "Par":1,
        "Chain":21
    },
    "D2_2460p"
    :{
        "m0":2.4654,
        "m_min":2.4644,
        "m_max":2.4667,
        "g0":0.0467,
        "g_min":0.0455,
        "g_max":0.0479,
        "J":2,
        "Par":1,
        "Chain":121
    },
    "D1_2430"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.284,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":12
    },
    "D1_2430p"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.384,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":112
    },
    "D1_2420"
    :{
        "m0":2.4208,
        "m_min":2.4203,
        "m_max":2.4213,
        "g0":0.0317,
        "g_min":0.0292,
        "g_max":0.0342,
        "J":1,
        "Par":1,
        "Chain":11
    },
    "D1_2420p"
    :{
        "m0":2.4232,
        "m_min":2.4208,
        "m_max":2.4256,
        "g0":0.025,
        "g_min":0.019,
        "g_max":0.021,
        "J":1,
        "Par":1,
        "Chain":111
    },
    "Zc_4025"
    :{
        "m0":4.0263,
        "g0":0.0248,
        "J":1,
        "Par":1,
        "Chain":-1
    },
    "Zc_4160"
    :{
        "m0":4.1628,
        "g0":0.0701,
        "J":1,
        "Par":1,
        "Chain":-2
    }
}

def train_one_step(model, optimizer, data, bg,mc):
  with tf.GradientTape() as tape:
    loss = model.nll(data,bg,mc)

  grads = tape.gradient(loss, model.Amp.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  print(loss)
  return loss

def main():
  import json,time
  a = Model(config_list,0.8)
  data = []
  bg = []
  mcdata = []
  with open("./data/PHSP_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      mcdata.append(tmp_data)
  with open("./data/data_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      data.append(tmp_data)
  with open("./data/bg_ang.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      bg.append(tmp_data)
  #print(data,bg,mcdata)
  data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
  data_set = data_set.shuffle(10000).batch(800)
  data_set_it = iter(data_set)
  bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
  bg_set = bg_set.shuffle(10000).batch(340)
  bg_set_it = iter(bg_set)
  mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
  mc_set = mc_set.shuffle(10000).batch(2520)
  mc_set_it = iter(mc_set)
  now = time.time()
  with tf.device('/device:GPU:0'):
    print(a.nll(data,bg,mcdata))#.collect_params())
  optimizer = tf.keras.optimizers.Adagrad()
  for i in range(100):
    try :
      data_i = data_set_it.get_next()
      bg_i = bg_set_it.get_next()
      mcdata_i = mc_set_it.get_next()
      train_one_step(a,optimizer,data_i,bg_i,mcdata_i);
    except:
      data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
      data_set = data_set.shuffle(10000).batch(800)
      data_set_it = iter(data_set)
      bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
      bg_set = bg_set.shuffle(10000).batch(340)
      bg_set_it = iter(bg_set)
      mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
      mc_set = mc_set.shuffle(10000).batch(2520)
      mc_set_it = iter(mc_set)
      
  print(time.time()-now)
  #now = time.time()
  #with tf.device('/device:CPU:0'):
    #print(a(x))#.collect_params())
  #print(time.time()-now)
  with tf.device('/device:GPU:0'):
    print(a.nll(data,bg,mcdata))#.collect_params())
  print(a.Amp.trainable_variables)
  
if __name__=="__main__":
  main()
