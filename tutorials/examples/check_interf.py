"""
Check the interferance of partial waves.
If more than one group exists, we need fixed extra phase in fit.

For example in the following formula
|a+b|^2 + |a-b|^2 = 2(|a|^2 + |b|^2)
the phase of a and b arbitrary.

"""


import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.experimental import build_amp

config = ConfigLoader("config.yml")
phsp = config.get_data("data")[0]  # config.generate_phsp(100000)

config.set_params("final_params.json")
decay_list, amp_m = build_amp.build_angle_amp_matrix(config.get_decay(), phsp)

print("include decay chains")
for i in decay_list:
    print(i)
    for j in i:
        print(" ", j, "ls list: ", j.get_ls_list())

print("number of partial waves for each decay chains")
print([len(i) for i in amp_m])

a = []
for k in amp_m:
    for v in k:
        a.append(v)

inter_f = np.zeros((len(a), len(a)))
inter_f_have = []
epsilon = 1e-7
sum_axis = list(range(1, len(a[0].shape)))
print(sum_axis)
for i in range(len(a)):
    for j in range(i, len(a)):
        c = tf.reduce_sum(
            tf.math.real(a[i] * tf.math.conj(a[j])), axis=sum_axis
        )
        d = tf.reduce_sum(
            tf.math.imag(a[i] * tf.math.conj(a[j])), axis=sum_axis
        )
        inter_f[i, j] = tf.reduce_mean(c)
        inter_f[j, i] = tf.reduce_mean(c)
        if (
            i != j
            and float(tf.reduce_max(tf.abs(c))) > epsilon
            or float(tf.reduce_max(tf.abs(d))) > epsilon
        ):
            inter_f_have.append((i, j))
        print(
            i,
            j,
            float(tf.reduce_max(tf.abs(c))),
            float(tf.reduce_max(tf.abs(d))),
        )

print("pairs which have interferance")
print(inter_f_have)

group = []
sub_group = set([inter_f_have[0][0], inter_f_have[0][1]])
inter_f_have = inter_f_have[1:]
while len(inter_f_have) > 0:
    used_intef_f_have = []
    for k, v in inter_f_have:
        if k in sub_group:
            sub_group.add(v)
            used_intef_f_have.append((k, v))
        elif v in sub_group:
            sub_group.add(k)
            used_intef_f_have.append((k, v))
    if len(used_intef_f_have) == 0:
        group.append(sub_group)
        sub_group = set([inter_f_have[0][0], inter_f_have[0][1]])
        inter_f_have = inter_f_have[1:]
    else:
        inter_f_have = [i for i in inter_f_have if i not in used_intef_f_have]
        if len(inter_f_have) == 0:
            group.append(sub_group)

print("number of group: ", len(group))
for i, g in enumerate(group):
    print("group", i, ":", g)
