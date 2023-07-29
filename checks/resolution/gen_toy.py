import os

from tf_pwa.config_loader import ConfigLoader

os.makedirs("data", exist_ok=True)

config = ConfigLoader("config.yml")

config.set_params("toy_params.json")

toy = config.generate_toy(5000)
phsp = config.generate_phsp(500000)
toy.savetxt("data/toy.dat", config.get_dat_order())
phsp.savetxt("data/phsp.dat", config.get_dat_order())


config.plot_partial_wave(
    data=[toy], phsp=[phsp], prefix="figure/toy_", plot_pull=True
)
