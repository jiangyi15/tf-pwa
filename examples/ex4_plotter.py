"""
Examples for Plotter class
----------------------------

Ploter is the new api for partial wave plots.

First, we can build a simple config.


"""

config_str = """

decay:
    A:
       - [R1, B]
       - [R2, C]
       - [R3, D]
    R1: [C, D]
    R2: [B, D]
    R3: [B, C]

particle:
    $top:
       A: { mass: 1.86, J: 0, P: -1}
    $finals:
       B: { mass: 0.494, J: 0, P: -1}
       C: { mass: 0.139, J: 0, P: -1}
       D: { mass: 0.139, J: 0, P: -1}
    R1: [ R1_a, R1_b ]
    R1_a: { mass: 0.7, width: 0.05, J: 1, P: -1}
    R1_b: { mass: 0.5, width: 0.05, J: 0, P: +1}
    R2: { mass: 0.824, width: 0.05, J: 0, P: +1}
    R3: { mass: 0.824, width: 0.05, J: 0, P: +1}


plot:
    mass:
        R1:
            display: "m(R1)"
        R2:
            display: "m(R2)"
"""

import matplotlib.pyplot as plt
import yaml

from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader(yaml.full_load(config_str))

# %%
# We set parameters to a blance value. And we can generate some toy data and calclute the weights
#

input_params = {
    "A->R1_a.BR1_a->C.D_total_0r": 6.0,
    "A->R1_b.BR1_b->C.D_total_0r": 1.0,
    "A->R2.CR2->B.D_total_0r": 2.0,
    "A->R3.DR3->B.C_total_0r": 1.0,
}
config.set_params(input_params)

data = config.generate_toy(1000)
phsp = config.generate_phsp(10000)


# %%
# plotter can be created directly from config

plotter = config.get_plotter(datasets={"data": [data], "phsp": [phsp]})

# %%
# Ploting all partial waves is simple.

plotter.plot_frame("m_R1")
plt.show()

# %%
# Also we can plot other variables in data

from tf_pwa.data import data_index

m2 = config.get_data_index("mass", "R2")
m3 = config.get_data_index("mass", "R3")


def f(data):
    return data_index(data, m2) - data_index(data, m3)


plt.clf()
plotter.plot_var(f)
plt.xlabel("m(R2)+m(R3)")
plt.show()

# %%
# There are 3 main parts in a Plotter
#
# 1. PlotAllData: datasets with weights
#    There is three level:
#    (1). idx: Datasets for combine fit
#    (2). type: data, mc, or bg
#    (3). observations and weights: weights are used for partial wave
#
# 2. Frame: function to get obsevations
#    It is samilar to RooFit's Frame.
#
# 3. Styles: Plot style for differenct componets
#
# The plot process is as follow:
#
# 1. Plotter.plot_item, extra_plot_item, and hidden_plot_item provide the list of histograms for plotting.
# 2. Loop over all data to get the observations though frame.
# 3. Frame provide the binning, weights from datas. Their combination is histogram
# 4. Plot on the axis with style
#
