import os.path
import re
import sys

import yaml

from tf_pwa.dec_parser import load_dec_file

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")


commands = {}
config = {"particle": {}, "decay": {}}
model_map = {"HELCOV": "default"}


def register_commands(name):
    def regist_(f):
        commands[name] = f

    return regist_


@register_commands("Particle")
def add_particle(params):
    particle = config["particle"]
    ret = {}
    params_name = ["mass", "width"]
    for k, v in zip(params_name, params["params"]):
        ret[k] = float(v)
    particle[params["name"]] = ret


@register_commands("RUNNINGWIDTH")
def add_running_width(params):
    for i in params:
        config["particle"][i]["running_width"] = True


@register_commands("Decay")
def add_decay(params):
    name = params["name"]
    finals = params["final"]
    ret = []
    for i in finals:
        kwargs = i.copy()
        outs = kwargs.pop("outs")
        for j in ["total"]:
            kwargs.pop(j)
        model = kwargs.get("model", "default")
        if model in model_map:
            kwargs["model"] = model_map[model]
        ret.append(outs + [kwargs])
    if ret:
        config["decay"][name] = ret


def load_parity_list(parity_list="_parity.list_"):
    patten = re.compile(r"([^\s]*)\s+([+-])(\d+)")
    with open(parity_list) as f:
        for i in f.readlines():
            grp = patten.match(i)
            if grp:
                name = grp[1]
                p = int(grp[2] + "1")
                j = int(grp[3])
                if name in config["particle"]:
                    config["particle"][name]["J"] = j
                    config["particle"][name]["P"] = p


def dec2yml(in_file, out_file, parity_list="_parity.list_"):
    with open("9R_c.dec") as f:
        dec = load_dec_file(f)
    for i in dec:
        # print(i)
        commands[i[0]](i[1])
    try:
        load_parity_list(parity_list)
    except FileNotFoundError:
        pass
    with open("config_test.yml", "w") as f:
        yaml.dump(config, f)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("-o", dest="out_file", default="config.yml")
    parser.add_argument(
        "--parity", dest="parity_file", default="_parity.list_"
    )
    results = parser.parse_args()
    dec2yml(results.in_file, results.out_file, results.parity_file)


if __name__ == "__main__":
    main()
