import numpy as np
import matplotlib.pyplot as plt
from model_iminuit import *
import json
import time as t
import tensorflow as tf
import iminuit

def main():
	set_gpu_mem_growth()
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
	import iminuit 
	f = fcn(a,data,bg,mcdata,27648)# 1356*18
	args = {}
	args_name = []
	for i in a.Amp.trainable_variables:
	  args[i.name] = i.numpy()
	  args_name.append(i.name)
	  args["error_"+i.name] = 0.1
	m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,print_level=2,grad=f.grad,**args)
	t1 = t.time()
	with tf.device('/device:GPU:0'):
	  m.migrad()
	t2 = t.time()
	m.get_param_states()
	exit()
	data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
	#data_set = data_set.shuffle(10000).batch(800)
	data_set_it = iter(data_set)
	bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
	#bg_set = bg_set.shuffle(10000).batch(340)
	bg_set_it = iter(bg_set)
	mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
	#mc_set = mc_set.shuffle(10000).batch(2520)
	mc_set_it = iter(mc_set)
	t3 = t.time()
	with tf.device('/device:GPU:0'):
	  print("NLL0:",a.nll(data,bg,mcdata))#.collect_params())
	t4 = t.time()
	optimizer = tf.keras.optimizers.Adadelta(1.0)
	for i in range(100):
	  #try :
	    #data_i = data_set_it.get_next()
	    #bg_i = bg_set_it.get_next()
	    #mcdata_i = mc_set_it.get_next()
	  train_one_step(a,optimizer,data,bg,mcdata)
	  #except:
	    #data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
	    ##data_set = data_set.shuffle(10000).batch(800)
	    #data_set_it = iter(data_set)
	    #bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
	    ##bg_set = bg_set.shuffle(10000).batch(340)
	    #bg_set_it = iter(bg_set)
	    #mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
	    ##mc_set = mc_set.shuffle(10000).batch(2520)
	    #mc_set_it = iter(mc_set)
	t5 = t.time()
	#now = time.time()
	#with tf.device('/device:CPU:0'):
	  #print(a(x))#.collect_params())
	#print(time.time()-now)
	with tf.device('/device:GPU:0'):
	  print("NLL:",a.nll(data,bg,mcdata))#.collect_params())
	t6 = t.time()
	print("Time: Migrad:",t2-t1,"; NLL0:",t4-t3,"; Range:",t5-t4,"; NLL:",t6-t5)
	print("Variables:",a.Amp.trainable_variables)
	
	### plot
	ndata = 8065
	nbg = 3445
	bgw = 0.8
	n_reson = 8
	w = a.Amp(mcdata)
	weight = w.numpy()
	
	i = 1
	for reson in config_list:
		locals()["reson_"%i] = reson
		i = i+1
	reson_variables = []
	
	for i in range(1,n_reson+1):
	    exec("config_list_%s = {reson_%s:config_list[reson_%s]}" % (i,i,i))
	    locals()["a_%s"%i] = Model(locals()["config_list_%s"%i],0.8)
	    for j in locals()["a_%s"%i].Amp.trainable_variables:
	        reson_variables.append(j)
	
	for v in a.Amp.trainable_variables:
	    for u in reson_variables:
	        if u.name == v.name:
	            u.assign(v)
	
	# for m_BC variable
	var_name,var_num = "m_BC",0
	locals()[var_name] = mcdata[var_num].numpy()
	locals()[var_name+"_data"] = data[var_num].numpy()
	locals()[var_name+"_bg"] = bg[var_num].numpy()
	xbinmin,xbinmax = 2.15,2.65
	
	nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
	plt.title("Total "+var_name)
	plt.clf()
	nmcwei = sum(nn[0])
	
	for i in range(1,n_reson+1):
	    locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
	    locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
	    locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
	    plt.title(locals()["reson_%s"%i])
	    plt.clf()
	    
	xbin = []
	for i in range(50):
	    xbin.append((nn[1][i+1]+nn[1][i])/2)
	    
	plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
	(counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
	plt.hist(bins[:-1],bins,weights=bgw*counts)
	ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
	plt.plot(xbin,ybin)
	plt.title(var_name)
	
	for i in range(1,n_reson+1):
	    locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
	    plt.plot(xbin,locals()["ybin_%s"%i])
	
	plt.savefig(var_name)    
	plt.clf()
	
	# for m_BD variable
	var_name,var_num = "m_BD",1
	locals()[var_name] = mcdata[var_num].numpy()
	locals()[var_name+"_data"] = data[var_num].numpy()
	locals()[var_name+"_bg"] = bg[var_num].numpy()
	xbinmin,xbinmax = 4,4.5
	
	nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
	plt.title("Total "+var_name)
	plt.clf()
	nmcwei = sum(nn[0])
	
	for i in range(1,n_reson+1):
	    locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
	    locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
	    locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
	    plt.title(locals()["reson_%s"%i])
	    plt.clf()
	    
	xbin = []
	for i in range(50):
	    xbin.append((nn[1][i+1]+nn[1][i])/2)
	    
	plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
	(counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
	plt.hist(bins[:-1],bins,weights=bgw*counts)
	ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
	plt.plot(xbin,ybin)
	plt.title(var_name)
	
	for i in range(1,n_reson+1):
	    locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
	    plt.plot(xbin,locals()["ybin_%s"%i])
	
	plt.savefig(var_name)    
	plt.clf()
	
	# for m_CD variable
	var_name,var_num = "m_CD",2
	locals()[var_name] = mcdata[var_num].numpy()
	locals()[var_name+"_data"] = data[var_num].numpy()
	locals()[var_name+"_bg"] = bg[var_num].numpy()
	xbinmin,xbinmax = 2.15,2.65
	
	nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
	plt.title("Total "+var_name)
	plt.clf()
	nmcwei = sum(nn[0])
	
	for i in range(1,n_reson+1):
	    locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
	    locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
	    locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
	    plt.title(locals()["reson_%s"%i])
	    plt.clf()
	    
	xbin = []
	for i in range(50):
	    xbin.append((nn[1][i+1]+nn[1][i])/2)
	    
	plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
	(counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
	plt.hist(bins[:-1],bins,weights=bgw*counts)
	ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
	plt.plot(xbin,ybin)
	plt.title(var_name)
	
	for i in range(1,n_reson+1):
	    locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
	    plt.plot(xbin,locals()["ybin_%s"%i])
	
	plt.savefig(var_name)    
	plt.clf()
	
	# for cosTheta_BC variable
	var_name,var_num = "cosTheta_BC",3
	locals()[var_name] = mcdata[var_num].numpy()
	locals()[var_name+"_data"] = data[var_num].numpy()
	locals()[var_name+"_bg"] = bg[var_num].numpy()
	xbinmin,xbinmax = -1,1
	
	nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
	plt.title("Total "+var_name)
	plt.clf()
	nmcwei = sum(nn[0])
	
	for i in range(1,n_reson+1):
	    locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
	    locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
	    locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
	    plt.title(locals()["reson_%s"%i])
	    plt.clf()
	    
	xbin = []
	for i in range(50):
	    xbin.append((nn[1][i+1]+nn[1][i])/2)
	    
	plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
	(counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
	plt.hist(bins[:-1],bins,weights=bgw*counts)
	ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
	plt.plot(xbin,ybin)
	plt.title(var_name)
	
	for i in range(1,n_reson+1):
	    locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
	    plt.plot(xbin,locals()["ybin_%s"%i])
	
	plt.savefig(var_name)    
	plt.clf()
	print("Save Figure Done!")


if __name__=="__main__":
	t0 = t.time()
	print("Start-----")
	main()
	print("Elapsed Time:",t.time()-t0)


