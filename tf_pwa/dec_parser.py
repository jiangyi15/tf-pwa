"""
module for parsing decay card *.dec file
"""
import re
import warnings


# from pysnooper import snoop

def load_dec(s):
    """
    load *.dec string
    """
    lines = split_lines(s)
    return process_decay_card(lines)


def load_dec_file(f):
    """
    load *.dec file
    """
    lines = f.readlines()
    return process_decay_card(lines)


def split_lines(s):
    """split each lines"""
    return re.split("\\s*\n\\s*", s)


def remove_comment(words):
    """remove comment string which starts with '#'."""
    ret = []
    for i in words:
        if len(i) <= 0:
            continue
        if i.startswith("#"):
            break
        if i.find("#") != -1:
            s = i.split("#")[0]
            ret.append(s)
            break
        ret.append(i)
    return ret


def get_words(lines):
    """get all words in a lines"""
    while True:
        line = next(lines)
        if len(line) <= 0:
            continue
        words_o = re.split(r"\s+", line)
        words = remove_comment(words_o)
        if len(words) >= 1:
            return words


commands = {}


def regist_command(name=None):
    """regist command function for command call"""
    def g(f):
        name1 = name
        if name1 is None:
            if hasattr(f, "__name__"):
                name1 = f.__name__
            elif hasattr(f, "name"):
                name1 = f.name
            else:
                raise Exception("name required")
        if name1 in commands:
            warnings.warn("override dec commands {}".format(name1))
        commands[name1] = f
        return f

    return g


def do_command(cmd, params, lines):
    """do command in commands"""
    if cmd in commands:
        fun = commands[cmd]
        return fun(params, lines)
    else:
        return cmd, params


def process_decay_card(lines):
    """process all the files as a generator"""
    lines = iter(lines)
    while True:
        words = get_words(lines)
        cmd = words[0]
        params = words[1:]
        if cmd == "End":
            break
        yield do_command(cmd, params, lines)


@regist_command(name="Decay")
def get_decay(words, lines):
    """parser decay command"""
    if len(words) <= 0:
        raise Exception("Decay need particles")
    core = words[0]
    ret = []
    while True:
        words = get_words(lines)
        if words[0] == "Enddecay":
            break
        if words[-1].endswith(";"):
            words[-1] = words[-1][:-1]
        s = sigle_decay(words)
        ret.append(s)
    return ("Decay", {"name": core, "final": ret})


models = {"HELCOV": None,
          "PHSP": None}


def sigle_decay(s):
    """do each decay line"""
    total = float(s[0])
    s = s[1:]
    outs = []
    model = "HELCOV"
    params = []
    is_particle = True
    for i in s:
        if i in models:
            model = i
            is_particle = False
        else:
            if is_particle:
                outs.append(i)
            else:
                params.append(i)
    return {"outs": outs,
            "model": model,
            "total": total,
            "params": params}


@regist_command(name="Particle")
def get_particles(words, _lines):
    """parser particles command"""
    if len(words) <= 0:
        raise Exception("Decay need particles")
    name = words[0]
    return ("Particle", {"name": name, "params": words[1:]})
