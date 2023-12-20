"""
Draw 3d plot of helicity angle with a chain boost.

The results can be seen in
https://agenda.infn.it/event/33110/contributions/198135/attachments/106337/149769/hadron2023_v5.pdf
Page 14.

The script requied mayavi for the 3d plot.

"""

import numpy as np
from mayavi import mlab

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index

config = ConfigLoader("config.yml")
data = config.get_data("data")[0]
index_of_data = 0  # the first data point


def draw_line(a, b, split=1):
    delta = b - a
    n = 2 * split - 1
    for i in range(n):
        if i % 2 == 1:
            continue
        mlab.plot3d(
            [
                a[0] + (i / (2 * split - 1)) * delta[0],
                a[0] + ((i + 1) / (2 * split - 1)) * delta[0],
            ],
            [
                a[1] + (i / (2 * split - 1)) * delta[1],
                a[1] + ((i + 1) / (2 * split - 1)) * delta[1],
            ],
            [
                a[2] + (i / (2 * split - 1)) * delta[2],
                a[2] + ((i + 1) / (2 * split - 1)) * delta[2],
            ],
        )


def draw_arc(o, z1, r, z2, diff=0, name=None):
    sinphi = np.dot(np.cross(z1, z2), r) / np.sqrt(np.sum(r**2))
    cosphi = np.dot(z1, z2)
    phi_all = np.arctan2(sinphi, cosphi)
    # print(phi_all)
    phi = np.linspace(0, phi_all, 64) + diff
    vx = z1
    v2 = np.cross(r, z1)
    v2 = v2 / np.sqrt(np.sum(v2**2)) * np.sqrt(np.sum(z2**2))
    x = o + z1 * np.cos(phi)[:, None] + v2 * np.sin(phi)[:, None]
    mlab.plot3d(x[:, 0], x[:, 1], x[:, 2], name=name)


def draw_plane(idx, x0, vz, vx, angle):
    """

     x - y          x - z
     \   phi        theta
      \              | /
    ---------        |/
        \            /
         \          /|
                   / |

    """

    rz = np.cos(angle["beta"])
    rx = np.sin(angle["beta"]) * np.cos(angle["alpha"])
    ry = np.sin(angle["beta"]) * np.sin(angle["alpha"])
    # print(angle["alpha"])

    vy = np.cross(vz, vx)
    vx = np.cross(vy, vz)

    vz = vz / np.sqrt(np.sum(vz**2))
    vx = vx / np.sqrt(np.sum(vx**2))
    vy = vy / np.sqrt(np.sum(vy**2))

    arrow = vz * rz + vx * rx + vy * ry
    arrow = arrow / np.sqrt(np.sum(arrow**2))
    # print(arrow, vx, vz, vy)

    center = x0 + vz
    left = vx * rx + vy * ry
    p1 = x0 - left
    p2 = x0 + left
    p3 = x0 + 2 * vz - left
    p4 = x0 + 2 * vz + left

    # mlab.points3d([center[0]], [center[1]], [center[2]])

    rz_v = np.cos(np.linspace(0, 2 * np.pi, 64))
    rx_v = np.sin(np.linspace(0, 2 * np.pi, 64)) * np.cos(angle["alpha"])
    ry_v = np.sin(np.linspace(0, 2 * np.pi, 64)) * np.sin(angle["alpha"])

    p = center + rz_v[:, None] * vz + ry_v[:, None] * vy + rx_v[:, None] * vx
    p2 = center + np.zeros_like(p)
    # print(p, p2)
    # mlab.points3d([center[0]], [center[1]], [center[2]], [0.3])
    draw_line(x0, center)
    draw_line(center, center + vz, 5)
    draw_line(center - arrow, center + arrow)
    draw_arc(
        center,
        0.5 * vz,
        0.5 * np.cross(vz, arrow),
        0.5 * arrow,
        name=f"theta{idx}",
    )
    new_vy = np.cross(vz, arrow)
    draw_arc(
        x0,
        0.5 * vy,
        vz,
        0.5 * new_vy / np.sqrt(np.sum(new_vy**2)),
        diff=-np.pi / 2,
        name=f"phi{idx}",
    )

    mlab.mesh([p[:, 0], p2[:, 0]], [p[:, 1], p2[:, 1]], [p[:, 2], p2[:, 2]])
    # print(([[p1[0], p2[0]],[p3[0],p4[0]]], [[p1[1], p2[1]],[ p3[1], p4[1]]], [[p1[2],p2[2]], [p3[2], p4[2]]]))
    # mlab.mesh([[p1[0], p2[0]],[p3[0],p4[0]]], [[p1[1], p2[1]],[ p3[1], p4[1]]], [[p1[2],p2[2]], [p3[2], p4[2]]])
    return center, arrow, np.cross(np.cross(vz, arrow), arrow)


decay_chain = config.get_decay(False)[1]
# print(decay_chain)
start_point = {decay_chain.top: np.array((0, 0, 0))}
start_arrow = {decay_chain.top: np.array((0, 0, 1))}
start_arrow2 = {decay_chain.top: np.array((1, 0, 0))}

plot_decay_chains = [
    config.get_decay(False)[0]
]  # , config.get_decay(False)[1], config.get_decay(False)[2]]

mlab.figure()
for decay_chain in plot_decay_chains:
    print(decay_chain)
    for idx, (level, decay) in enumerate(decay_chain.depth_first()):
        x0 = start_point[decay.core]
        vz = start_arrow[decay.core]
        vx = start_arrow2[decay.core]
        # load angle
        angle = data_index(
            data,
            config.get_data_index(
                "angle",
                "/".join(str(i.core) for i in decay_chain)
                + "/"
                + str(decay.outs[0]),
            ),
        )
        angle = {k: v[index_of_data] for k, v in angle.items()}

        # draw plane
        center, arrow, new_vx = draw_plane(idx, x0, vz, vx, angle)
        start_point[decay.outs[0]] = center + arrow
        start_point[decay.outs[1]] = center - arrow
        start_arrow[decay.outs[0]] = arrow
        start_arrow[decay.outs[1]] = -arrow
        start_arrow2[decay.outs[0]] = new_vx
        start_arrow2[decay.outs[1]] = -new_vx

# mlab.points3d([0], [0], [0],scale_factor=0.2)
# mlab.outline()
mlab.show()
