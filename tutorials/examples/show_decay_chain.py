import os.path
import sys

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.vis import draw_decay_struct

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")




def main():
    config = ConfigLoader("config.yml")
    for i, dec in enumerate(config.get_decay()):
        draw_decay_struct(
            dec, filename="figure/fig_{}".format(i), format="png"
        )


if __name__ == "__main__":
    main()
