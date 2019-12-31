import json
import os

has_sympy = True
try :
  from sympy.physics.quantum.cg import CG
except ImportError:
  has_sympy = False
  
_dirname = os.path.dirname(os.path.abspath(__file__))

with open(_dirname+"/cg_table.json") as f:
  cg_table = json.load(f)

def find_cg_table(j1,j2,m1,m2,j,m):
  try:
    return cg_table[str(j1)][str(j2)][str(m1)][str(m2)][str(j)][str(m)]
  except:
    return 0.0

def get_cg_coef(j1,j2,m1,m2,j,m):
  assert(m1 + m2 == m)
  assert(j1 >= 0)
  assert(j2 >= 0)
  assert(j >= 0)
  if j1 == 0 or j2 == 0:
    return 1.0
  sign = 1
  if j1 < j2:
    if (j1+j2-j)%2 == 1:
      sign = -1
    j1,j2 = j2,j1
    m1,m2 = m2,m1
  return sign * find_cg_table(j1,j2,m1,m2,j,m)

def cg_coef(jb,jc,mb,mc,ja,ma):
  if has_sympy:
    return CG(jb,mb,jc,mc,ja,ma).doit().evalf()
  else:
    return get_cg_coef(jb,jc,mb,mc,ja,ma)
