from .angle_tf import EularAngle, LorentzVector
from .data import load_dat_file, flatten_dict_data
from .tensorflow_wrapper import tf


def cal_angle(p_B, p_C, p_D):
    p_A = p_B + p_C + p_D
    p_B_A = LorentzVector.rest_vector(p_A, p_B)
    p_C_A = LorentzVector.rest_vector(p_A, p_C)
    p_D_A = LorentzVector.rest_vector(p_A, p_D)
    return cal_angle_rest(p_B_A, p_C_A, p_D_A)


def cal_angle_rest(p4_B, p4_C, p4_D):
    p4_BD = p4_B + p4_D
    p4_BC = p4_B + p4_C
    p4_CD = p4_C + p4_D
    p4_B_BD = LorentzVector.rest_vector(p4_BD, p4_B)
    p4_B_BC = LorentzVector.rest_vector(p4_BC, p4_B)
    p4_D_CD = LorentzVector.rest_vector(p4_CD, p4_D)

    zeros = tf.zeros_like(LorentzVector.get_e(p4_B))
    ones = tf.ones_like(zeros)
    u_z = tf.transpose([zeros, zeros, ones], (1, 0))
    u_x = tf.transpose([ones, zeros, zeros], (1, 0))
    ang_BC, x_BC = EularAngle.angle_zx_z_gety(
        u_z, u_x, LorentzVector.vect(p4_BC))
    ang_B_BC, x_B_BC = EularAngle.angle_zx_z_gety(
        LorentzVector.vect(p4_BC), x_BC, LorentzVector.vect(p4_B_BC))
    ang_BD, x_BD = EularAngle.angle_zx_z_gety(
        u_z, u_x, LorentzVector.vect(p4_BD))
    ang_B_BD, x_B_BD = EularAngle.angle_zx_z_gety(
        LorentzVector.vect(p4_BD), x_BD, LorentzVector.vect(p4_B_BD))
    ang_CD, x_CD = EularAngle.angle_zx_z_gety(
        u_z, u_x, LorentzVector.vect(p4_CD))
    ang_D_CD, x_D_CD = EularAngle.angle_zx_z_gety(
        LorentzVector.vect(p4_CD), x_CD, LorentzVector.vect(p4_D_CD))

    ang_BD_B = EularAngle.angle_zx_zx(
        LorentzVector.vect(p4_B_BD), x_B_BD, LorentzVector.vect(p4_B), x_CD)
    ang_BC_B = EularAngle.angle_zx_zx(
        LorentzVector.vect(p4_B_BC), x_B_BC, LorentzVector.vect(p4_B), x_CD)
    ang_BD_D = EularAngle.angle_zx_zx(-LorentzVector.vect(p4_B_BD),
                                      x_B_BD, LorentzVector.vect(p4_D), x_BC)
    ang_CD_D = EularAngle.angle_zx_zx(
        LorentzVector.vect(p4_D_CD), x_D_CD, LorentzVector.vect(p4_D), x_BC)

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
    p = [data[i] for i in ["B", "C", "D"]]
    ret = cal_angle(*p)
    ret["m_BC"] = LorentzVector.M(p[0] + p[1])
    ret["m_CD"] = LorentzVector.M(p[1] + p[2])
    ret["m_BD"] = LorentzVector.M(p[2] + p[0])
    ret["m_A"] = LorentzVector.M(p[0] + p[1] + p[2])
    ret["m_B"] = LorentzVector.M(p[0])
    ret["m_C"] = LorentzVector.M(p[1])
    ret["m_D"] = LorentzVector.M(p[2])
    return ret


def cal_ang_file(fname, _dtype="float64", flatten=True):
    data = load_dat_file(fname, ["D", "B", "C"])
    ret = cal_ang_data(data)
    if flatten:
        ret = flatten_dict_data(ret, fun=lambda x, y: y+x[3:])
    return ret


def load_dat_angle(fname, dtype="float64", flatten=True):
    data = cal_ang_file(fname, dtype, flatten)
    ret = tf.data.Dataset.from_tensor_slices(data)
    return ret
