import argparse
import inspect
import warnings
import sys


__all__ = ["regist_subcommand"]


def _protect_dict(override_message=None):
    d = {}

    def get_dict():
        return d.copy()

    def set_var(name, var):
        if name in d:
            if override_message is not None:
                warnings.warn("override {}: {}".format(override_message, name))
        d[name] = var
    return get_dict, set_var


get_sub_cmd, set_sub_cmd = _protect_dict("sub commands")


def regist_subcommand(name=None, arg_fun=None, args=None):
    _sub_commands = get_sub_cmd()
    if args is None:
        args = {}

    def wrap(f):
        name_t = name
        if name_t is None:
            name_t = f.__name__
        if arg_fun is None:
            cmds = _build_arguments(f, args)
        else:
            cmds = arg_fun
        set_sub_cmd(name_t, cmds)
        return f
    return wrap


def _build_arguments(f, config_args):
    argspec = inspect.getfullargspec(f)

    def wrap_f(arg):
        args = []
        kwargs = {}
        for i in argspec.args:
            if hasattr(arg, i):
                var = getattr(arg, i)
                args.append(var)
        if argspec.kwonlyargs:
            for i in argspec.kwonlyargs:
                if hasattr(arg, i):
                    var = getattr(arg, i)
                    kwargs[i] = var
        return f(*args, **kwargs)
    wrap_f.__name__ = f.__name__

    ret = {"fun": wrap_f, "args": []}

    arg_defaults = {}
    if argspec.defaults:
        for i, j in zip(argspec.defaults[::-1], argspec.args[::-1]):
            arg_defaults[j] = i
    if argspec.kwonlydefaults:
        for i in argspec.kwonlydefaults:
            arg_defaults[i] = argspec.kwonlydefaults[i]

    arg_type = {}
    if hasattr(argspec, "annotations"):
        arg_type = argspec.annotations

    for i in config_args:
        ret["args"].append(config_args[i])

    for i in argspec.args:
        if i in config_args:
            break
        args = (i,)
        kwargs = {}
        if i in arg_defaults:
            kwargs["nargs"] = "?"
            kwargs["default"] = arg_defaults[i]
        if i in arg_type:
            kwargs["type"] = arg_type[i]
        ret["args"].append((args, kwargs))

    if not argspec.kwonlyargs:
        return ret

    for i in argspec.kwonlyargs:
        if i in config_args:
            break
        name = "--" + i
        args = (name,)
        kwargs = {}
        if i in arg_defaults:
            kwargs["default"] = arg_defaults[i]
        if i in arg_type:
            kwargs["type"] = arg_type[i]
        ret["args"].append((args, kwargs))

    return ret


@regist_subcommand(name="help")
def help_function():
    print("""
    using ```python -m tf_pwa [subprocess]```
    """)


def main(argv=None):
    parser = argparse.ArgumentParser("tf_pwa")
    subparsers = parser.add_subparsers()
    _sub_commands = get_sub_cmd()
    for i in _sub_commands:
        cmds = _sub_commands[i]
        pi = subparsers.add_parser(i)
        if callable(cmds):
            cmds(pi)
        else:
            pi.set_defaults(func=cmds["fun"])
            for args, kwargs in cmds["args"]:
                pi.add_argument(*args, **kwargs)
    rargs = parser.parse_args(argv)
    if hasattr(rargs, "func"):

        return rargs.func(rargs)
    else:
        help_function()
