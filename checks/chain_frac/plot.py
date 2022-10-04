from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader("config.yml")
config.set_params("a.json")
config.plot_partial_wave(prefix="figure/s_", plot_pull=True)
print(config.cal_fitfractions())
