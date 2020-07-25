#!/usr/bin/env python3
import sys
import os.path
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')


from tf_pwa.config_loader import ConfigLoader, MultiConfig
from pprint import pprint
from tf_pwa.utils import error_print
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np
from tf_pwa.data import data_index
from itertools import combinations


def get_data(config_file="config.yml", init_params="init_params.json"):

    config = ConfigLoader(config_file)
    try:
        config.set_params(init_params)
        print("using {}".format(init_params))
    except Exception as e:
        print("using RANDOM parameters")

    phsp = config.get_data("phsp")

    for i in config.full_decay:
        print(i)
        for j in i:
            print(j.get_ls_list())

    print("\n########### initial parameters")
    print(json.dumps(config.get_params(), indent=2))
    params = config.get_params()
    
    amp = config.get_amplitude()
    pw = amp.partial_weight(phsp)
    pw_if = amp.partial_weight_interference(phsp)
    weight = amp(phsp)
    print(weight)
    return config, amp, phsp, weight, pw, pw_if

def get_params():
    with open("test_results.json") as f:
        data = json.load(f)
    
    data_infer = [i["NR(1+)DxDs1(2700)"] for i in data["fracs"]]
    data_0 = [i["Ds1(2700)"] for i in data["fracs"]]
    data_1 = [i["NR(1+)S"] for i in data["fracs"]]
    data_2 = [i["NR(1+)D"] for i in data["fracs"]]

    idx = np.argmax(np.fabs(np.array(data_0)*np.array(data_1)*np.array(data_2)))
    print(idx, data_infer[idx])
    return data["prarms"][idx]


def test_plot(init_params):
    
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [10, 10] # for square canvas
    matplotlib.rcParams['figure.subplot.left'] = 0.05
    matplotlib.rcParams['figure.subplot.bottom'] = 0.05
    matplotlib.rcParams['figure.subplot.right'] = 0.95
    matplotlib.rcParams['figure.subplot.top'] = 0.95
    matplotlib.rcParams['font.size'] = 5
    config, amp, data, weight, pw, pw_if = get_data(init_params=init_params)#init_params=get_params())
    
    m_DsDst = data_index(data, config.plot_params.get_data_index("mass", "R_BC"))
    m_DsK = data_index(data, config.plot_params.get_data_index("mass", "R_BD"))
    m_DstK = data_index(data, config.plot_params.get_data_index("mass", "R_CD"))
    
    val = m_DsDst

    sw = np.sum(weight)/weight.shape[0]
    print(sw)
    # plt.hist(val, weights=[sw]*weight.shape[0], bins=100, label="PHSP")
    # plt.hist(val, weights=weight, bins=100, label="data")
    label =  amp.chains_particle()
    label = [str(i[0]) for i in label]#
    n_label = len(label)
    for i, w in zip(label, pw):
        print(i, np.sum(w)/sw)
    
    if n_label % 2 == 0:
        a, b = n_label//2, n_label - 1
    else:
        a, b = (n_label -1) //2, n_label
    fig = plt.figure(figsize=(4,3))
    for i, j in enumerate(combinations(range(n_label), 2)):
        # ax = plt.subplot(a,b,i+1)
        plt.clf()
        ax = plt.subplot(1,1,1)
        ax.hist(val, weights=pw[j[0]], bins=100, histtype="step", label=label[j[0]])
        ax.hist(val, weights=pw[j[1]], bins=100, histtype="step", label=label[j[1]])
        x, y, _ = ax.hist(val, weights=pw_if[j], bins=100, histtype="step", label=label[j[0]] +"+" + label[j[1]])
        ax.hist(val, weights=pw_if[j] - pw[j[0]] - pw[j[1]], bins=100, histtype="step", label=label[j[0]] +"x" + label[j[1]])
        ax.plot(y, np.zeros_like(y))
        ax.legend(loc="upper right")
        # ax.set_ylim((None, np.max(y) *1.2))
        ax.set_title("M(DsD*)")
        plt.savefig("fig/fig_{}_{}.pdf".format(*j))
    #plt.savefig("test_plot.png")
    #plt.show()
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("-i", default="init_params.json", dest="init_params")
    results = parser.parse_args()
    test_plot(results.init_params)


if __name__ == "__main__":
    main()
