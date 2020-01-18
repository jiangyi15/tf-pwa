from .angle_tf import EularAngle, Vector3, LorentzVector
from .data import load_data, flatten_dict_data
import tensorflow as tf


def cal_angle(p_B, p_C, p_D):
    p_A = p_B + p_C + p_D
    p_B_A = p_A.rest_vector(p_B)
    p_C_A = p_A.rest_vector(p_C)
    p_D_A = p_A.rest_vector(p_D)
    return cal_angle_rest(p_B_A, p_C_A, p_D_A)


def cal_angle_rest(p4_B, p4_C, p4_D):
    p4_BD = p4_B + p4_D
    p4_BC = p4_B + p4_C
    p4_CD = p4_C + p4_D
    p4_B_BD = p4_BD.rest_vector(p4_B)
    p4_B_BC = p4_BC.rest_vector(p4_B)
    p4_D_CD = p4_CD.rest_vector(p4_D)

    zeros = tf.zeros_like(p4_B.e)
    ones = tf.ones_like(p4_B.e)
    u_z = Vector3(tf.transpose([zeros, zeros, ones], (1, 0)))
    u_x = Vector3(tf.transpose([ones, zeros, zeros], (1, 0)))
    ang_BC, x_BC = EularAngle.angle_zx_z_gety(
        u_z, u_x, p4_BC.vect())
    ang_B_BC, x_B_BC = EularAngle.angle_zx_z_gety(
        p4_BC.vect(), x_BC, p4_B_BC.vect())
    ang_BD, x_BD = EularAngle.angle_zx_z_gety(
        u_z, u_x, p4_BD.vect())
    ang_B_BD, x_B_BD = EularAngle.angle_zx_z_gety(
        p4_BD.vect(), x_BD, p4_B_BD.vect())
    ang_CD, x_CD = EularAngle.angle_zx_z_gety(
        u_z, u_x, p4_CD.vect())
    ang_D_CD, x_D_CD = EularAngle.angle_zx_z_gety(
        p4_CD.vect(), x_CD, p4_D_CD.vect())

    ang_BD_B = EularAngle.angle_zx_zx(
        p4_B_BD.vect(), x_B_BD, p4_B.vect(), x_CD)
    ang_BC_B = EularAngle.angle_zx_zx(
        p4_B_BC.vect(), x_B_BC, p4_B.vect(), x_CD)
    ang_BD_D = EularAngle.angle_zx_zx(-p4_B_BD.vect(),
                                      x_B_BD, p4_D.vect(), x_BC)
    ang_CD_D = EularAngle.angle_zx_zx(
        p4_D_CD.vect(), x_D_CD, p4_D.vect(), x_BC)

    return {
        "ang_BC": ang_BC,
        "ang_BD": ang_BD,
        "ang_CD": ang_CD,
        "ang_B_BC": ang_B_BC,
        "ang_B_BD": ang_B_BD,
        "ang_D_CD": ang_D_CD,
        "ang_BD_B": ang_BD_B,
        "ang_BD_D": ang_BD_D,
        "ang_BC_B": ang_BC_B,
        "ang_CD_D": ang_CD_D,
    }


def cal_ang_data(data):
    p = [LorentzVector(data[i]) for i in ["B", "C", "D"]]
    ret = cal_angle(*p)
    ret["m_BC"] = (p[0] + p[1]).M()
    ret["m_CD"] = (p[1] + p[2]).M()
    ret["m_BD"] = (p[2] + p[0]).M()
    ret["m_A"] = (p[0] + p[1] + p[2]).M()
    ret["m_B"] = (p[0]).M()
    ret["m_C"] = (p[1]).M()
    ret["m_D"] = (p[2]).M()
    return ret


def cal_ang_file(fname, dtype="float64", flatten=True):
    data = load_data(fname, ["D", "B", "C"])
    ret = cal_ang_data(data)
    if flatten:
        ret = flatten_dict_data(ret, fun=lambda x, y: y+x[3:])
    return ret


def load_dat_file(fname, dtype="float64", flatten=True):
    data = cal_ang_file(fname, dtype, flatten)
    ret = tf.data.Dataset.from_tensor_slices(data)
    return ret
