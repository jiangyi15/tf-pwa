import re
import numpy as np

frac_txt = """
B 1.0 ? 0.0
Ds2_2573p 0.018 +/- 0.005
Ds2_2573pxB -0.018 +/- 0.005
Ds1_2700p 0.410 +/- 0.029
Ds1_2700pxDs2_2573p 0.00074 +/- 0.00018
Ds1_2700pxB -0.410 +/- 0.029
NR(D*K)SP 0.132 +/- 0.015
NR(D*K)SPxDs1_2700p 0.110 +/- 0.007
NR(D*K)SPxDs2_2573p 0.0026 +/- 0.0006
NR(D*K)SPxB -0.132 +/- 0.015
NR(D*K)PS 0.183 +/- 0.024
NR(D*K)PSxNR(D*K)SP 0.0041 +/- 0.0020
NR(D*K)PSxDs1_2700p -0.108 +/- 0.011
NR(D*K)PSxDs2_2573p 0.0025 +/- 0.0004
NR(D*K)PSxB -0.183 +/- 0.024
NR(D*K)PD 0.037 +/- 0.009
NR(D*K)PDxNR(D*K)PS -0.0005 +/- 0.0004
NR(D*K)PDxNR(D*K)SP 0.0197 +/- 0.0024
NR(D*K)PDxDs1_2700p 0.070 +/- 0.008
NR(D*K)PDxDs2_2573p -0.0015 +/- 0.0006
NR(D*K)PDxB -0.037 +/- 0.009
NR(D*K)PP 0.0004 +/- 0.0010
NR(D*K)PPxNR(D*K)PD 0.000007 +/- 0.000011
NR(D*K)PPxNR(D*K)PS 0.000004 +/- 0.000012
NR(D*K)PPxNR(D*K)SP 0.0000020 +/- 0.0000032
NR(D*K)PPxDs1_2700p -0.001 +/- 0.006
NR(D*K)PPxDs2_2573p 0.000006 +/- 0.000008
NR(D*K)PPxB -0.0004 +/- 0.0010
NR(DK)DP 0.114 +/- 0.026
NR(DK)DPxNR(D*K)PP -0.000008 +/- 0.000018
NR(DK)DPxNR(D*K)PD 0.009 +/- 0.005
NR(DK)DPxNR(D*K)PS -0.0321 +/- 0.0033
NR(DK)DPxNR(D*K)SP 0.0072 +/- 0.0027
NR(DK)DPxDs1_2700p -0.032 +/- 0.013
NR(DK)DPxDs2_2573p 0.0000 +/- 0.0001
NR(DK)DPxB -0.114 +/- 0.026
NR(D*D)PD 0.18 +/- 0.04
NR(D*D)PDxNR(DK)DP -0.22 +/- 0.05
NR(D*D)PDxNR(D*K)PP 0.000003 +/- 0.000011
NR(D*D)PDxNR(D*K)PD 0.0009 +/- 0.0016
NR(D*D)PDxNR(D*K)PS 0.073 +/- 0.005
NR(D*D)PDxNR(D*K)SP -0.0005 +/- 0.0007
NR(D*D)PDxDs1_2700p 0.014 +/- 0.009
NR(D*D)PDxDs2_2573p 0.00114 +/- 0.00032
NR(D*D)PDxB -0.18 +/- 0.04
"""

def get_point(s):
    partten = re.compile(r"([^\s]+)\s+([+-.e1234567890]+)\s+")
    ret = {}
    for i in s.split("\n"):
        g = partten.match(i)
        #print(g)
        if g:
            name = g.group(1).split("x")
            #print(name)
            frac = float(g.group(2))
            if len(name) == 1:
                l, r = name*2
            elif len(name) == 2:
                l, r = name
            else:
                raise Exception("error {}".format(name))
            if l not in ret:
                ret[l] = {}
            ret[l][r] = frac
    return ret


def get_table(s):
    #print(s)
    idx = list(s)
    n_idx = len(idx)
    idx_map = dict(zip(idx, range(n_idx)))
    #print(idx_map)
    ret = []
    for i in range(n_idx):
        ret.append([0.0 for j in range(n_idx)])
    for i, k in s.items():
        for j, v in k.items():
            ret[idx_map[i]][idx_map[j]] = v
    return idx, ret


def frac_table(frac_txt):
    s = get_point(frac_txt)
    idx, table = get_table(s)
    import pprint
    #pprint.pprint(idx)
    #print(idx)
    #print(table)

    # remove D*
    #idx = idx[1:]
    #table = np.array(table)[1:,1:]

    ret = []
    for i, k in enumerate(table):
        tmp = []
        for j, v in enumerate(k):
            if i < j:
                tmp.append("-")
            else:
                tmp.append("{:.3f}".format(v))
        ret.append(tmp)
    #pprint.pprint(ret)
    for i, k in zip(idx, ret):
        print(i, end="\t")
        for v in k:
            print(v, end="\t")
        print()
    print("Total sum:", np.sum(table))
    print("Non-interference sum:", np.sum(np.diagonal(table)))


if __name__=="__main__":
    frac_table(frac_txt)
            
    
