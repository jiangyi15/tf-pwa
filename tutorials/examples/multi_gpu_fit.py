"""
Fit with multi GPU or multi process when we have two same config.yml with different data
"""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import json
import os
import time
from multiprocessing import Pipe, Process

from scipy.optimize import minimize


def eval_nll(pipe, config, gpu=0):
    print("run on pid: ", os.getpid())
    # import in the child process
    import numpy as np
    import tensorflow as tf

    from tf_pwa.config_loader import ConfigLoader

    with tf.device("GPU:" + str(gpu)):
        config = ConfigLoader(config)
        fcn = config.get_fcn()
        while True:
            cmd, params = pipe.recv()
            if cmd == "eval":
                nll, grad = fcn.nll_grad(params)
                pipe.send([nll, grad])
            elif cmd == "get_x0":
                params = config.vm.get_all_dic()
                x0 = [params[i] for i in config.vm.trainable_vars]
                pipe.send(np.array(x0))
            elif cmd == "get_params":
                return pipe.send(config.get_params())
            elif cmd == "end":
                pipe.close()
                break


def main():
    # sample model, but different data
    config_list = ["config_a.yml", "config_b.yml", "config_c.yml"]
    n_gpu = 1
    pipes = []
    process = []
    for i, j in enumerate(config_list):
        a, b = Pipe()
        p1 = Process(target=eval_nll, args=(a, j, i % n_gpu))
        p1.start()
        pipes.append((a, b))
        process.append(p1)

    base_pipe = pipes[0][1]

    base_pipe.send(["get_x0", None])
    x0 = base_pipe.recv()

    def nll(params={}):
        # child
        start = time.perf_counter()
        for a, b in pipes:
            b.send(["eval", params])
        nll = 0
        grad = 0
        for a, b in pipes:
            nll_i, grad_i = b.recv()
            nll = nll + nll_i
            grad = grad + grad_i
        print("time cost: ", time.perf_counter() - start)
        return nll, grad

    fit_start = time.perf_counter()
    fit_result = minimize(nll, x0, jac=True)
    print(fit_result)
    print("fit time cost: ", time.perf_counter() - fit_start)

    base_pipe.send(["get_params", {}])
    params = base_pipe.recv()

    with open("final_params.json", "w") as f:
        json.dump(
            {
                "value": params,
                "error": {},
                "status": {
                    "NLL": fit_result.fun,
                    "success": fit_result.success,
                    "Ndf": len(x0),
                },
            },
            f,
            indent=2,
        )

    # end process
    for a, b in pipes:
        b.send(["end", {}])
    for p in process:
        p.join()


if __name__ == "__main__":
    main()
