import argparse
import os

from jinja2 import Template


def load_config(input_file, variable_file=""):
    """
    read input_file and render it with variable file
    """
    with open(input_file) as f:
        template = Template(f.read())

    variable = os.environ.copy()
    if variable_file != "":
        with open(variable_file) as f:
            exec(f.read(), variable)

    ret = template.render(**variable)
    return ret


def main():
    parser = argparse.ArgumentParser(description="Jinja")
    parser.add_argument("input_file")
    parser.add_argument("variable", nargs="?", default="")
    args = parser.parse_args()

    s = load_config(args.input_file, args.variable)
    print(s)


if __name__ == "__main__":
    main()
