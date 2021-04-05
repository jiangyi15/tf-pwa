#!/usr/bin/env python3

import csv
import json
import time
from pprint import pprint

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

import tensorflow as tf
import numpy as np

# examples of custom particle model
from tf_pwa.amp import regist_particle, Particle, get_relative_p
from tf_pwa.breit_wigner import Bprime_num
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table


@regist_particle("exp2")
class ParticleExp(Particle):
    """
    .. math::
        R(m) = e^{-|a| m}

    """

    def init_params(self):
        self.a = self.add_var("a")

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        a = tf.abs(self.a())
        return tf.complex(tf.exp(-a * mass * mass), zeros)

@regist_particle("KmatrixDK")
class ParticleKmatrixDK(Particle):
    def init_params(self):
        self.mass1 = self.add_var("mass1", fix=True)
        self.mass2 = self.add_var("mass2", fix=True)
        #self.width1 = self.add_var("width1", fix=True) # is not used
        #self.width2 = self.add_var("width2", fix=True)

        # ratio 0.91+-0.18 sq 0.954+-0.094
        self.G1a = self.add_var("G1a") # 2700->DK 0.72357 0.57009 
        self.G1r = self.add_var("G1r") # 2700->D*K 0.69025 0.76551
        # ratio 1.10 +- 0.24 sq 1.049+-0.114
        self.G2a = self.add_var("G2a") # 2860->DK 0.69007 0.68775
        self.G2r = self.add_var("G2r") # 2860->D*K 0.72375 0.50498

        self.beta1 = self.add_var("beta1", is_complex=True)
        self.beta2 = self.add_var("beta2", is_complex=True)

        self.mdaughtera1 = 1.86483 # D0
        self.mdaughtera2 = 0.493677 # Kp
        self.mdaughterb1 = 2.00696 # D*0
        self.mdaughterb2 = self.mdaughtera2 # Kp
        self.d = 3.0


    def get_amp(self, data, data_c=None, **kwargs):
        m = data["m"]
        m1 = self.mass1()
        m2 = self.mass2()
        m1_m = m1**2 - m**2
        m2_m = m2**2 - m**2
        m_mlist = 1/tf.stack([m1_m, m2_m])
        
        D, rho = self.get_D_rho(m)
        K, P = self.get_K_P(m)

        rhoDD = rho*D*D
        iKrhoDD = tf.complex(np.float64(0), tf.einsum("ijs,jks->sik", K, rhoDD))
        KK = tf.eye(2, dtype=iKrhoDD.dtype) - iKrhoDD
        KK00 = KK[:,0,0]
        KK01 = KK[:,0,1]
        KK10 = KK[:,1,0]
        KK11 = KK[:,1,1]
        invdenom = KK00*KK11-KK01*KK10
        iKinv = tf.einsum("ijk->kij",tf.stack([[KK11/invdenom, -KK01/invdenom], [-KK10/invdenom, KK00/invdenom]]))
        #iKinv = tf.linalg.inv(KK)
        DiKinv = tf.einsum("ijs,sjk->iks", tf.cast(D,iKinv.dtype), iKinv)
        AK = tf.einsum("ijs,js->is", DiKinv, P)

        ret = AK[0]#*tf.cast(barrier, AK.dtype)
        return ret


    def get_D_rho(self, m):
        m1 = self.mass1()
        m2 = self.mass2()
        La = 1
        Lb = 1

        qa = get_relative_p(m, self.mdaughtera1, self.mdaughtera2)
        qb = get_relative_p(m, self.mdaughterb1, self.mdaughterb2)

        rhoa = 2*qa/m
        rhob = 2*qb/m
        zeros =tf.zeros_like(rhoa)
        rho = tf.stack([[rhoa, zeros],
                        [zeros, rhob]])

        Da = (qa / Bprime_num(La, qa, self.d))**La * self.d**La
        Db = (qb / Bprime_num(Lb, qb, self.d))**Lb * self.d**Lb
        D = tf.stack([[Da, zeros],
                      [zeros, Db]])
        return D, rho

    def get_K_P(self, m):
        m1 = self.mass1()
        m2 = self.mass2()
        u1 = m1**2 - m**2
        u2 = m2**2 - m**2

        G1a = tf.abs(self.G1a())
        G1r = tf.abs(self.G1r())
        G2a = tf.abs(self.G2a())
        G2r = tf.abs(self.G2r())
        q1a = get_relative_p(m1, self.mdaughtera1, self.mdaughtera2)
        q1b = get_relative_p(m1, self.mdaughterb1, self.mdaughterb2)
        q2a = get_relative_p(m2, self.mdaughtera1, self.mdaughtera2)
        q2b = get_relative_p(m2, self.mdaughterb1, self.mdaughterb2)
        g1afactor = m1*m1/2/q1a/q1a/q1a * (q1a*q1a+1/self.d/self.d)
        g1bfactor = m1*m1/2/q1b/q1b/q1b * (q1b*q1b+1/self.d/self.d)
        g2afactor = m2*m2/2/q2a/q2a/q2a * (q2a*q2a+1/self.d/self.d)
        g2bfactor = m2*m2/2/q2b/q2b/q2b * (q2b*q2b+1/self.d/self.d)

        g1a = tf.sqrt(G1a*g1afactor)
        g1b = tf.sqrt(G1a*G1r*g1bfactor)
        g2a = tf.sqrt(G2a*g2afactor)
        g2b = tf.sqrt(G2a*G2r*g2bfactor)

        beta1 = self.beta1()
        beta2 = self.beta2()

        Kaa = g1a*g1a/u1 + g2a*g2a/u2
        Kab = g1a*g1b/u1 + g2a*g2b/u2
        Kbb = g1b*g1b/u1 + g2b*g2b/u2
        Kmatrix = tf.stack([[Kaa, Kab],
                            [Kab, Kbb]])

        Pa = beta1 * tf.cast(g1a/u1,beta1.dtype) + beta2 * tf.cast(g2a/u2,beta2.dtype) # to DK
        Pb = beta1 * tf.cast(g1b/u1,beta1.dtype) + beta2 * tf.cast(g2b/u2,beta2.dtype) # to D*K
        Pvector = tf.stack([Pa, Pb])
        return Kmatrix, Pvector #tf.transpose(Pvector) # it'll be much slower if transpose here

def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


def load_config(config_file="config.yml", total_same=False):
    config_files = config_file.split(",")
    if len(config_files) == 1:
        return ConfigLoader(config_files[0])
    return MultiConfig(config_files, total_same=total_same)


def fit(config, init_params="", method="BFGS", loop=1, maxiter=500):
    """
    simple fit script
    """
    # load config.yml
    # config = ConfigLoader(config_file)

    # load data
    all_data = config.get_all_data()

    fit_results = []
    for i in range(loop):
        # set initial parameters if have
        if config.set_params(init_params):
            print("using {}".format(init_params))
        else:
            print("\nusing RANDOM parameters", flush=True)
        # try to fit
        try:
            fit_result = config.fit(
                batch=65000, method=method, maxiter=maxiter
            )
        except KeyboardInterrupt:
            config.save_params("break_params.json")
            raise
        except Exception as e:
            print(e)
            config.save_params("break_params.json")
            raise
        fit_results.append(fit_result)
        # reset parameters
        try:
            config.reinit_params()
        except Exception as e:
            print(e)

    fit_result = fit_results.pop()
    for i in fit_results:
        if i.success:
            if not fit_result.success or fit_result.min_nll > i.min_nll:
                fit_result = i

    config.set_params(fit_result.params)
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # calculate parameters error
    if maxiter is not 0:
        fit_error = config.get_params_error(fit_result, batch=13000)
        fit_result.set_error(fit_error)
        fit_result.save_as("final_params.json")
        pprint(fit_error)

        print("\n########## fit results:")
        print("Fit status: ", fit_result.success)
        print("Minimal -lnL = ", fit_result.min_nll)
        for k, v in config.get_params().items():
            print(k, error_print(v, fit_error.get(k, None)))

    return fit_result


def write_some_results(config, fit_result, save_root=False):
    # plot partial wave distribution
    config.plot_partial_wave(fit_result, plot_pull=True, save_root=save_root, smooth=False)

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
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    fit_frac_tmp = fit_frac_string.split("\n")[1:]
    fit_frac_string = ""
    for ff in fit_frac_tmp:
        if "xB" not in ff:
            fit_frac_string += f"{ff}\n"
    print(fit_frac_string)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)
    from frac_table import frac_table
    frac_table(fit_frac_string)
    # chi2, ndf = config.cal_chi2(mass=["R_BC", "R_CD"], bins=[[2,2]]*4)


def write_some_results_combine(config, fit_result, save_root=False):

    from tf_pwa.applications import fit_fractions

    for i, c in enumerate(config.configs):
        c.plot_partial_wave(
            fit_result, prefix="figure/s{}_".format(i), save_root=save_root
        )

    for it, config_i in enumerate(config.configs):
        print("########## fit fractions {}:".format(it))
        mcdata = config_i.get_phsp_noeff()
        fit_frac, err_frac = fit_fractions(
            config_i.get_amplitude(),
            mcdata,
            config.inv_he,
            fit_result.params,
        )
        fit_frac_string = ""
        for i in fit_frac:
            if isinstance(i, tuple):
                name = "{}x{}".format(*i)  # interference term
            else:
                name = i  # fit fraction
            fit_frac_string += "{} {}\n".format(
                name, error_print(fit_frac[i], err_frac.get(i, None))
            )
        print(fit_frac_string)
        save_frac_csv(f"fit_frac{it}.csv", fit_frac)
        save_frac_csv(f"fit_frac{it}_err.csv", err_frac)
    # from frac_table import frac_table
    # frac_table(fit_frac_string)


def save_frac_csv(file_name, fit_frac):
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)


def write_run_point():
    """ write time as a point of fit start"""
    with open(".run_start", "w") as f:
        localtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        f.write(localtime)


def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument(
        "--no-GPU", action="store_false", default=True, dest="has_gpu"
    )
    parser.add_argument("-c", "--config", default="config.yml", dest="config")
    parser.add_argument(
        "-i", "--init_params", default="init_params.json", dest="init"
    )
    parser.add_argument("-m", "--method", default="BFGS", dest="method")
    parser.add_argument("-l", "--loop", type=int, default=1, dest="loop")
    parser.add_argument("-x", "--maxiter", type=int, default=500, dest="maxiter")
    parser.add_argument("-r", "--save_root", default=False, dest="save_root")
    parser.add_argument(
        "--total-same", action="store_true", default=False, dest="total_same"
    )
    results = parser.parse_args()
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        config = load_config(results.config, results.total_same)
        fit_result = fit(
            config, results.init, results.method, results.loop, results.maxiter
        )
        if isinstance(config, ConfigLoader):
            write_some_results(config, fit_result, save_root=results.save_root)
        else:
            write_some_results_combine(
                config, fit_result, save_root=results.save_root
            )


if __name__ == "__main__":
    write_run_point()
    main()
