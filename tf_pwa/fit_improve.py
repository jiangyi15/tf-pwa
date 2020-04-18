import numpy as np
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
from scipy.optimize import OptimizeResult


message_dict = {
    0: 'Optimization terminated successfully.',
    3: 'Maximum number of function evaluations has been exceeded.',
    2: 'Maximum number of iterations has been exceeded.',
    1: 'Desired error not necessarily achieved due to precision loss.',
    4: 'NaN result encountered.'
}


class Cached_FG:
    def __init__(self, f_g):
        self.f_g = f_g
        self.cached_fun = 0
        self.cached_grad = 0
        self.ncall = 0

    def __call__(self, x):
        f = self.fun(x)
        return f, self.cached_grad

    def fun(self, x):
        f, g = self.f_g(x)
        self.cached_x = x
        self.cached_grad = g
        self.cached_fun = f
        self.ncall += 1
        return f

    def grad(self, x):
        if not np.all(x == self.cached_x):
            self.fun(x)
        return self.cached_grad


class Seq:
    def __init__(self, size=5):
        self._cached = []
        self.size = size

    def get_max(self):
        return max(self._cached)

    def arg_max(self):
        max_item = self._cached[0]
        max_idx = 0
        for i, v in enumerate(self._cached):
            if max_item < v:
                max_idx = i
                max_item = v
        return max_idx

    def add(self, x):
        self._cached.append(x)
        if len(self._cached) > self.size:
            self._cached.pop(0)


def fmin_bfgs_f(f_g, x0, epsilon=1e-5, Delta=10.0, maxiter=2000, callback=None, norm_ord=np.Inf):
    fk, gk = f_g(x0)
    Bk = np.eye(len(x0))
    xk = x0
    norm = lambda x: np.linalg.norm(x, ord=norm_ord)
    theta = 0.9
    C = 0.5
    k = 0
    old_old_fval = fk + np.linalg.norm(gk) / 2
    f_s = Seq(1)
    f_s.add(fk)
    flag = 0
    re_search = 0
    for k in range(maxiter):
        if norm(gk) <= epsilon:
            break
        dki = - np.dot(np.linalg.pinv(Bk), gk)
        try:
            pk = dki
            f = f_g.fun
            myfprime = f_g.grad
            gfk = gk
            old_fval = f_s.get_max()  # fk
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                line_search_wolfe2(f, myfprime, xk, pk, gfk,
                                   old_fval, old_old_fval)
        except Exception as e:
            print(e)
            re_search += 1
            xk = xk + dki
            fk, gk = f_g(xk)
            old_fval, old_old_fval = fk, old_fval
            f_s.add(fk)
            continue
            if re_search > 2:
                flag = 1
                break
        if alpha_k is None:
            print("alpha is None")
            xk = xk + dki
            fk, gk = f_g(xk)
            old_fval, old_old_fval = fk, old_fval
            f_s.add(fk)
            continue
            re_search += 1
            if re_search > 2:
                flag = 1
                break
        dki = alpha_k * pk
        # fki, gki = f_g(xk + dki)
        fki, gki = old_fval, gfkp1
        Aredk = fk - fki
        Predk = - (np.dot(gk, dki) + 0.5 * np.dot(np.dot(Bk, dki), dki))
        rk = Aredk / Predk
        xk = xk+dki
        fk = fki
        yk = gki - gk
        tk = C + max(0, - np.dot(yk, dki)/norm(dki)**2) / norm(gk)
        ystark = (1-theta) * yk + theta * tk * norm(gk)*dki
        gk = gki
        bs = np.dot(Bk, dki)
        Bk = Bk + np.outer(yk, yk)/np.dot(yk, dki) - \
            np.outer(bs, bs)/np.dot(bs, dki)
        # Bk = Bk + np.outer(ystark, ystark)/np.dot(ystark, dki) - \
        #    np.outer(bs, bs)/np.dot(bs, dki)  # MBFGS
        f_s.add(fk)
        if callback is not None:
            callback(xk)
    else:
        print("maxiter")
        flag = 2
    # print("fit final: ", k, p, f_g.ncall)
    s = OptimizeResult()
    s.messgae = message_dict[flag]
    s.fun = float(fk)
    s.nit = k
    s.nfev = f_g.ncall
    s.njev = f_g.ncall
    s.status = flag
    s.x = np.array(xk)
    s.jac = np.array(gk)
    s.hess = np.array(Bk)
    s.success = (flag == 0)
    return s


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    # der += np.random.random(der.shape)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1], 1) - np.diag(400*x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H


@Cached_FG
def my_fit_fun(x):
    return rosen(x), rosen_der(x)


def minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    s = fmin_bfgs_f(Cached_FG(fun), x0, callback=callback)
    return s


if __name__ == "__main__":
    ret = fmin_bfgs_f(my_fit_fun, np.array([2.0, 1.3, 0.7, 0.8, 1.9, 1.2]))
    print(ret)
