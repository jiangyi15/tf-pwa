import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import yaml
import scipy.signal as signal


from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation
# import mplhep
# plt.style.use(mplhep.style.LHCb)

from tf_pwa.experimental.extra_amp import spline_matrix
from tf_pwa.config_loader import ConfigLoader

def vialid_name(s):
    return s.replace("+",".")


def polar_err(r, phi, r_e, phi_e):
    """polar errors for r and phi"""
    # print(r, phi, r_e, phi_e)
    dxdr = np.cos(phi)
    dxdphi = r*np.sin(phi)
    dydr = np.sin(phi)
    dydphi = - r * np.cos(phi)
    x_e = np.sqrt((dxdr*r_e)**2+(dxdphi*phi_e)**2)
    y_e = np.sqrt((dydr*r_e)**2+(dydphi*phi_e)**2)
    # print(x_e, y_e)
    return x_e, y_e


def dalitz_weight(s12, m0, m1, m2, m3):
    """phase space weight in dalitz plot"""
    m12 = np.sqrt(s12)
    m12 = np.where(m12 > (m1 + m2), m12,  m1 + m2)
    m12 = np.where(m12 < (m0 - m3), m12,  m0 - m3)
    # if(mz < (m_d+m_pi)) return 0;
    # if(mz > (m_b-m_pi)) return 0;
    E2st = 0.5*(m12*m12-m1*m1+m2*m2)/m12
    E3st = 0.5*(m0*m0-m12*m12-m3*m3)/m12
    
    p2st2 = E2st*E2st - m2*m2
    p3st2 = E3st*E3st - m3*m3
    p2st = np.sqrt(np.where(p2st2>0, p2st2, 0))
    p3st = np.sqrt(np.where(p3st2>0, p3st2, 0))
    return p2st * p3st


def load_params(config_file="config.yml", params="final_params.json", res="li(1+)S"):
    with open(params) as f:
        final_params = json.load(f)
    val = final_params["value"]
    err = final_params["error"]
    with open(config_file) as f:
        config = yaml.safe_load(f)
    xi = config["particle"][res].get("points", None)
    if xi is None:
        m_max = config["particle"][res].get("m_max", None)
        m_min = config["particle"][res].get("m_min", None)
        N = config["particle"][res].get("interp_N", None)
        dx = (m_max - m_min)/(N - 1)
        xi = [m_min + dx * i for i in range(N)]
    N = len(xi)
    head = "{}_point".format(vialid_name(res))
    r = np.array([0] + [val["{}_{}r".format(head, i)] for i in range(N-2)] +[0])
    phi = np.array([0] + [val["{}_{}i".format(head, i)] for i in range(N-2)] +[0])
    r_e = np.array([0, 0] + [err.get("{}_{}r".format(head, i), r[i]*0.1) for i in range(1,N-2)] + [0])
    phi_e = np.array([0, 0] + [err.get("{}_{}i".format(head, i), phi[i]*0.1) for i in range(1,N-2)] + [0])
    return np.array(xi), r, phi, r_e, phi_e


def trans_r2xy(r, phi, r_e, phi_e):
    """r,phi -> x,y """
    x = np.array(r) * np.cos(phi)
    y = np.array(r) * np.sin(phi)
    err = np.array([polar_err(i,j,k,l) for i,j,k,l in zip(r, phi, r_e, phi_e)])
    return x, y, err[:,0], err[:,1]


def plot_x_y(name, x, y, x_i, y_i, xlabel, ylabel, ylim=(None, None)):
    """plot x vs y"""
    plt.clf()
    plt.plot(x, y)
    plt.scatter(x_i, y_i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.savefig(name)


def plot_phi(name, m, phi, m_i, phi_i):
    """ plot phi and gradient of phi"""
    grad = (phi[2:] - phi[:-2])
    mask = (phi<3)&(phi>-3)
    grad_max = np.mean(np.abs(grad))
    # grad_max = np.max(grad[mask[1:-1]])

    idx, = signal.argrelextrema(grad,np.less)

    plt.clf()

    #plt.plot(m, pq/np.max(pq))# np.sqrt(x_new**2+y_new**2)**2)
    plt.plot(m[1:-1], grad/grad_max, label="$\\Delta \\phi$ ")
    plt.plot(m, phi, label="$\\phi$")# np.sqrt(x_new**2+y_new**2)**2)

    m_delta = m[idx+1]
    print("min Delta phi in mass:", m_delta)
    plt.scatter(m_delta, [-np.pi]*len(m_delta))
    plt.scatter(m_i, phi_i, label="points")
    plt.xlabel("mass")
    plt.ylabel("$\\phi$")
    plt.ylim((-np.pi,np.pi))
    plt.legend()
    plt.savefig(name)


def plot_x_y_err(name, x, y, x_e, y_e):
    """plot eror bar of x y"""
    plt.clf()
    plt.errorbar(x, y, xerr=x_e, yerr=y_e)
    plt.xlabel("real R(m)")
    plt.ylabel("imag R(m)")
    plt.savefig(name)


def plot3d_m_x_y(name, m, x, y):
    fig = plt.figure()
    axes3d = Axes3D(fig)
    axes3d.plot(m,x,y)
    axes3d.set_xlabel("m")
    axes3d.set_ylabel("real R(m)")
    axes3d.set_zlabel("imag R(m)")

    def update(frame):
        axes3d.view_init(elev=30,azim=frame)
        return None

    anim = animation.FuncAnimation(fig, update,interval=10, frames=range(0,360, 10))
    anim.save(name, writer='imagemagick')


def plot_all(res="MI(1+)S", config_file="config.yml", params="final_params.json", prefix="figure/"):
    """plot all figure"""
    config = ConfigLoader(config_file)
    config.set_params(params)
    particle = config.get_decay().get_particle(res)

    mi, r, phi_i, r_e, phi_e = load_params(config_file, params, res)
    x, y, x_e, y_e = trans_r2xy(r, phi_i, r_e, phi_e)

    m = np.linspace(mi[0], mi[-1], 1000)
    M_Kpm = 0.49368
    M_Dpm = 1.86961
    M_Dstar0 = 2.00685
    M_Bpm = 5.27926
    #x_new = interp1d(xi, x, "cubic")(m)
    #y_new = interp1d(xi, y, "cubic")(m)
    rm_new = particle.interp(m).numpy()
    x_new, y_new = rm_new.real, rm_new.imag

    pq = dalitz_weight(m*m, M_Bpm, M_Dstar0, M_Dpm, M_Kpm)
    pq_i = dalitz_weight(mi*mi, M_Bpm, M_Dstar0, M_Dpm, M_Kpm)

    phi = np.arctan2(y_new, x_new)
    r2 = x_new * x_new + y_new * y_new

    plot_phi(f"{prefix}phi.png", m, phi, mi, np.arctan2(y, x))
    plot_x_y(f"{prefix}r2.png", m, r2, mi, r*r, "mass", "$|R(m)|^2$", ylim=(0,None))
    plot_x_y(f"{prefix}x_y.png", x_new, y_new, x, y, "real R(m)", "imag R(m)")
    plot_x_y_err(f"{prefix}x_y_err.png", x[1:-1], y[1:-1], x_e[1:-1], y_e[1:-1])
    plot_x_y(f"{prefix}r2_pq.png", m, r2*pq, mi, r*r*pq_i, "mass", "$|R(m)|^2 p \cdot q$", ylim=(0,None))
    plot3d_m_x_y(f"{prefix}m_r.gif", m, x_new, y_new)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="plot interpolation")
    parser.add_argument("particle", type=str)
    parser.add_argument("-c", "--config", default="config.yml", dest="config")
    parser.add_argument("-i", "--params", default="final_params.json", dest="params")
    parser.add_argument("-p", "--prefix", default="figure/", dest="prefix")
    results = parser.parse_args()
    plot_all(results.particle,results.config, results.params, results.prefix)


if __name__=="__main__":
    main()
