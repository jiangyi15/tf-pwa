#!/usr/bin/env python3

# avoid using Xwindow
import matplotlib
matplotlib.use("agg")

from tf_pwa.config_loader import ConfigLoader, MultiConfig
from pprint import pprint
from tf_pwa.utils import error_print
import tensorflow as tf
import json


# examples of custom particle model
from tf_pwa.amp import Particle, regist_particle


@regist_particle("New")
class NewParticle(Particle):
    """example Particle model define, can be used in config.yml as `model: New`"""
    def init_params(self):
        # self.a = self.add_var("a")
        pass
    def get_amp(self, data, data_extra):
        # m = data["m"]
        # q = data_extra[self.outs[0]]["|q|"]
        # a = self.a()
        return 1.0


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s)


def fit(config_file="config.yml", init_params="init_params.json", method="BFGS"):
    """
    simple fit script 
    """
    # load config.yml
    config = ConfigLoader(config_file)
    
    # set initial parameters if have
    try:
        config.set_params(init_params)
        print("using {}".format(init_params))
    except Exception as e:
        if str(e) != "[Errno 2] No such file or directory: 'init_params.json'":
            print(e,"\nusing RANDOM parameters")
    
    print("\n########### initial parameters")
    json_print(config.get_params())

    # fit
    data, phsp, bg, inmc = config.get_all_data()
    fit_result = config.fit(batch=65000, method=method)
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # plot partial wave distribution
    config.plot_partial_wave(fit_result, plot_pull=True)

    # calculate parameters error
    fit_error = config.get_params_error(fit_result, batch=13000)
    fit_result.set_error(fit_error)
    fit_result.save_as("final_params.json")
    pprint(fit_error)

    print("\n########## fit results:")
    for k, v in config.get_params().items():
        print(k, error_print(v, fit_error.get(k, None)))

    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
        fit_frac_string += "{} {}\n".format(name, error_print(fit_frac[i], err_frac.get(i, None)))
    print(fit_frac_string)
    #from tf_pwa.utils import frac_table
    #frac_table(fit_frac_string)


def fit_combine(config_file=["config.yml"], init_params="init_params.json", method="BFGS", total_same=False):
    """fit with multiply config.yml"""
    config = MultiConfig(config_file, total_same=total_same)
    try:
        config.set_params(init_params)
        print("using {}".format(init_params))
    except Exception as e:
        print("using RANDOM parameters")
    
    print("\n########### initial parameters")
    pprint = lambda dic: print(json.dumps(dic, indent=2))
    pprint(config.get_params())
    
    fit_result = config.fit(method=method, batch=65000)
    
    pprint(fit_result.params)
    fit_result.save_as("final_params.json")
    for i, c in enumerate(config.configs):
        c.plot_partial_wave(fit_result, prefix="figure/s{}_".format(i))

    fit_error = config.get_params_error(fit_result, batch=13000)
    fit_result.set_error(fit_error)
    pprint(fit_error)
    
    print("\n########## fit results:")
    for k, v in fit_result.params.items():
        print(k, error_print(v, fit_error.get(k, None)))


def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse
    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument("--no-GPU", action="store_false", default=True, dest="has_gpu")
    parser.add_argument("-c", "--config", default="config.yml", dest="config")
    parser.add_argument("-i", "--init_params", default="init_params.json", dest="init")
    parser.add_argument("-m", "--method", default="BFGS", dest="method")
    parser.add_argument("-l", "--loop", type=int, default=1, dest="loop")
    parser.add_argument("--total-same", action="store_true", default=False, dest="total_same")
    results = parser.parse_args()
    config = results.config.split(",")
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        for i in range(results.loop):
            if len(config) > 1:
                fit_combine(config, results.init, results.method, results.total_same)
            else:
                fit(results.config, results.init, results.method)


if __name__ == "__main__":
    main()
