from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader("config.yml")

config.set_params("a.json")

toy = config.generate_toy(10000)
phsp = config.generate_phsp(1000000)

config.plot_partial_wave(data=[toy], phsp=[phsp], plot_pull=True)

toy.savetxt("toy.dat", config.get_dat_order())
phsp.savetxt("phsp.dat", config.get_dat_order())


toy.savetxt("toy3.dat", ["pip", "pim", "(mum, mup)"])
phsp.savetxt("phsp3.dat", ["pip", "pim", "(mum, mup)"])
