import math

import tensorflow as F

d_fun_table_cos = {
  0:{
    0:{
      0:lambda x:1.0,
    },
  },
  1:{
    1:{
      1:lambda x:(1.0+x)/2,
      0:lambda x:-F.sqrt(1.0-x*x)/math.sqrt(2),
      -1:lambda x:(1.0-x)/2,
    },
    0:{
      1:lambda x:F.sqrt(1.0-x*x)/math.sqrt(2),
      0:lambda x:x,
      -1:lambda x:-F.sqrt(1.0-x*x)/math.sqrt(2),
    },
    -1:{
      1:lambda x:(1.0-x)/2,
      0:lambda x:F.sqrt(1.0-x*x)/math.sqrt(2),
      -1:lambda x:(1.0+x)/2,
    },
  },
  2:{
    2:{
      2:lambda x:(1.0+x)**2/4,
      1:lambda x:-F.sqrt(1.0-x*x)*(1.0+x)/2,
      0:lambda x:math.sqrt(3/8)*(1.0-x*x),
      -1:lambda x:-F.sqrt(1.0-x*x)*(1.0-x)/2.0,
      -2:lambda x:(1.0-x)**2/4.0,
    },
    1:{
      2:lambda x:F.sqrt(1.0-x*x)*(1.0+x)/2,
      1:lambda x:(x*(2.0*x+1.0)-1.0)/2.0,
      0:lambda x:-x*F.sqrt(1.0-x*x)*math.sqrt(3/2),
      -1:lambda x:(x*(-2.0*x+1.0)+1.0)/2.0,
      -2:lambda x:-F.sqrt(1.0-x*x)*(1.0-x)/2.0,
    },
    0:{
      2:lambda x:math.sqrt(3/8)*(1.0-x*x),
      1:lambda x:x*F.sqrt(1.0-x*x)*math.sqrt(3/2),
      0:lambda x:(3*x*x-1)/2,
      -1:lambda x:-x*F.sqrt(1.0-x*x)*math.sqrt(3/2),
      -2:lambda x:math.sqrt(3/8)*(1.0-x*x),
    },
    -1:{
      2:lambda x:F.sqrt(1.0-x*x)*(1.0-x)/2.0,
      1:lambda x:(x*(-2.0*x+1.0)+1.0)/2.0,
      0:lambda x:x*F.sqrt(1.0-x*x)*math.sqrt(3/2),
      -1:lambda x:(x*(2.0*x+1.0)-1.0)/2.0,
      -2:lambda x:-F.sqrt(1.0-x*x)*(1.0+x)/2,
    },
    -2:{
      2:lambda x:(1.0-x)**2/4.0,
      1:lambda x:F.sqrt(1.0-x*x)*(1.0-x)/2.0,
      0:lambda x:math.sqrt(3/8)*(1.0-x*x),
      -1:lambda x:F.sqrt(1.0-x*x)*(1.0+x)/2,
      -2:lambda x:(1.0+x)**2/4,
    },
  }
}
    
'''def d_function_cos(j,m1,m2,x):
  try:
    print("HERE",m1,m1)
    len(m1)
    len(m2)
    print("HERE len")
    arr = []
    for (m,n) in zip(m1,m2):
      print("HERE in for",m,n)
      temp = d_function_cos(j,m,n,x)
      arr.append(temp)
    return arr
  except:
    print("HERE Except",m1,m2)
    return d_fun_table_cos[j][m1][m2](x)'''

def d_function_cos(j,m1,m2):
  try:
    d_fun_table_cos[j][m1][m2]
    return d_fun_table_cos[j][m1][m2]
  except:
    return lambda x:0.
