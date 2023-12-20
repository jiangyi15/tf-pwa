from tf_pwa.config_loader import ConfigLoader

# import extra_amp

batch_size = 100000
config = ConfigLoader("config.yml")


def create_cached_file(d):
    d.prefetch = 0
    d.batch(batch_size)
    for i in d:
        # iter for Dataset
        pass


data, phsp, bg, *_ = config.get_all_data()

for i in range(len(data)):
    if bg is None or bg[i] is None:
        data_bg_merged = data[i]
    else:
        data_bg_merged = data[i].merge(bg[i])
    create_cached_file(data_bg_merged)
    create_cached_file(phsp[i])
