import math

import tensorflow as F

d_fun_table_cos = {
  "0":{
    "0":{
      "0":lambda x:1.0,
    },
  },
  "1":{
    "1":{
      "1":lambda x:(1.0+x)/2,
      "0":lambda x:-F.sqrt(1.0-x*x)/math.sqrt(2),
      "-1":lambda x:(1.0-x)/2,
    },
    "0":{
      "0":lambda x:x,
    },
  },
  "2":{
    "2":{
      "2":lambda x:(1.0+x)**2/4,
      "1":lambda x:-F.sqrt(1.0-x*x)*(1.0+x)/2,
      "0":lambda x:math.sqrt(3/8)*F.sqrt(1.0-x*x)**2,
      "-1":lambda x:-F.sqrt(1.0-x*x)*(1.0-x)/2.0,
      "-2":lambda x:(1.0-x)**2/4.0,
    },
    "1":{
      "1":lambda x:(x*(2.0*x+1.0)-1.0)/2.0,
      "0":lambda x:-x*F.sqrt(1.0-x*x)*math.sqrt(3/2),
      "-1":lambda x:(x*(-2.0*x+1.0)+1.0)/2.0,
    },
    "0":{
      "0":lambda x:(3*x*x-1)/2,
    },
  }
}
    
def d_function_cos(j,m1,m2):
  assert j >=0 and j <= 2
  if m1 > j or m2 >j :
    return lambda x:0.0
  sign = 1.0
  if abs(m1) < abs(m2):
    if abs(m1-m2)%2 == 1:
      sign = -1
    m1,m2 = m2,m1
  if m1<0:
    m1,m2 = -m1,-m2
  try :
    return sign * d_fun_table_cos[str(j)][str(m1)][str(m2)]
  except:
    return lambda x:1.0
