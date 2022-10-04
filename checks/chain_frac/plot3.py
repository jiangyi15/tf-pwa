from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader("config3.yml")
config.set_params("a3.json")
config.plot_partial_wave(prefix="figure/s3_", plot_pull=True)
print(config.cal_fitfractions())
