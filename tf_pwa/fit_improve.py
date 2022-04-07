from warnings import warn

import numpy as np

# from numpy import xrange
# from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
from scipy.optimize import OptimizeResult


class LineSearchWarning(RuntimeWarning):
    pass


message_dict = {
    0: "Optimization terminated successfully.",
    3: "Maximum number of function evaluations has been exceeded.",
    2: "Maximum number of iterations has been exceeded.",
    1: "Desired error not necessarily achieved due to precision loss.",
    4: "NaN result encountered.",
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


def fmin_bfgs_f(
    f_g,
    x0,
    B0=None,
    M=2,
    gtol=1e-5,
    Delta=10.0,
    maxiter=None,
    callback=None,
    norm_ord=np.Inf,
    **_kwargs
):
    """test BFGS with nonmonote line search"""
    fk, gk = f_g(x0)
    if B0 is None:
        Bk = np.eye(len(x0))
    else:
        Bk = B0
    Hk = np.linalg.inv(Bk)
    maxiter = 200 * len(x0) if maxiter is None else maxiter
    xk = x0
    norm = lambda x: np.linalg.norm(x, ord=norm_ord)
    theta = 0.9
    C = 0.5
    k = 0
    old_old_fval = fk + np.linalg.norm(gk) / 2
    old_fval = fk
    f_s = Seq(M)
    f_s.add(fk)
    flag = 0
    re_search = 0
    for k in range(maxiter):
        if norm(gk) <= gtol:
            break
        dki = -np.dot(Hk, gk)
        try:
            pk = dki
            f = f_g.fun
            myfprime = f_g.grad
            gfk = gk
            old_fval = fk
            (
                alpha_k,
                fc,
                gc,
                old_fval,
                old_old_fval,
                gfkp1,
            ) = line_search_wolfe2(
                f, myfprime, xk, pk, gfk, f_s.get_max(), old_fval, old_old_fval
            )
        except Exception as e:
            print(e)
            re_search += 1
            xk = xk + dki
            fk, gk = f_g(xk)
            old_fval, old_old_fval = fk, old_fval
            f_s.add(fk)
            if re_search > 2:
                flag = 1
                break
            continue
        if alpha_k is None:
            print("alpha is None")
            xk = xk + dki
            fk, gk = f_g(xk)
            old_fval, old_old_fval = fk, old_fval
            f_s.add(fk)
            re_search += 1
            if re_search > 2:
                flag = 1
                break
            continue
        dki = alpha_k * pk
        # fki, gki = f_g(xk + dki)
        fki, gki = old_fval, gfkp1
        Aredk = fk - fki
        Predk = -(np.dot(gk, dki) + 0.5 * np.dot(np.dot(Bk, dki), dki))
        rk = Aredk / Predk
        xk = xk + dki
        fk = fki
        yk = gki - gk
        tk = C + max(0, -np.dot(yk, dki) / norm(dki) ** 2) / norm(gk)
        ystark = (1 - theta) * yk + theta * tk * norm(gk) * dki
        gk = gki
        bs = np.dot(Bk, dki)
        Bk = (
            Bk
            + np.outer(yk, yk) / np.dot(yk, dki)
            - np.outer(bs, bs) / np.dot(bs, dki)
        )
        # sk = dki
        # rhok = 1.0 / (np.dot(yk, sk))
        # A1 = 1 - np.outer(sk, yk) * rhok
        # A2 = 1 - np.outer(yk, sk) * rhok
        # Hk = np.dot(A2, np.dot(Hk, A1)) - (rhok * np.outer(sk, sk))
        # Bk = Bk + np.outer(ystark, ystark)/np.dot(ystark, dki) - \
        #    np.outer(bs, bs)/np.dot(bs, dki)  # MBFGS
        # print(np.dot(Hk, Bk))
        try:
            Hk = np.linalg.inv(Bk)
        except Exception:
            pass
        f_s.add(fk)
        if callback is not None:
            callback(xk)
    else:
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
    s.success = flag == 0
    return s


def minimize(
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
):
    options = options if options is not None else {}
    s = fmin_bfgs_f(Cached_FG(fun), x0, callback=callback, **options)
    return s


def line_search_nonmonote(
    f,
    myfprime,
    xk,
    pk,
    gfk=None,
    old_fval=None,
    fk=None,
    old_old_fval=None,
    args=(),
    c1=0.5,
    maxiter=10,
):
    alpha = max(-np.dot(gfk, pk) / np.dot(pk, pk), 1.0)
    phi_star = None
    print("init alpha", alpha, "\ngrad:", gfk)
    for i in range(maxiter):
        phi_star = f(xk + alpha * pk)
        if phi_star < old_fval + c1 * alpha * np.dot(gfk, pk):
            derphi_star = myfprime(xk + alpha * pk)
            return alpha, 0, 0, phi_star, old_fval, derphi_star
        alpha = c1 * alpha
    derphi_star = myfprime(xk + alpha * pk)
    print("not found")
    return alpha, 0, 0, phi_star, old_fval, derphi_star


# ------------------------------------------------------------------------------
# Pure-Python Wolfe line and scalar searches
# from scipy.optimize.linesearch
# ------------------------------------------------------------------------------


def line_search_wolfe2(
    f,
    myfprime,
    xk,
    pk,
    gfk=None,
    fk=None,
    old_fval=None,
    old_old_fval=None,
    args=(),
    c1=1e-4,
    c2=0.9,
    amax=None,
    extra_condition=None,
    maxiter=10,
):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)

    if isinstance(myfprime, tuple):

        def derphi(alpha):
            fc[0] += len(xk) + 1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f, eps) + args
            gval[0] = fprime(xk + alpha * pk, *newargs)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)

    else:
        fprime = myfprime

        def derphi(alpha):
            gc[0] += 1
            gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0], pk)

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    if extra_condition is not None:
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])

    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
        phi,
        derphi,
        old_fval,
        old_old_fval,
        derphi0,
        c1,
        c2,
        amax,
        extra_condition2,
        maxiter=maxiter,
    )

    if derphi_star is None:
        warn("The line search algorithm did not converge", LineSearchWarning)
        return line_search_nonmonote(
            f,
            myfprime,
            xk,
            pk,
            gfk,
            fk,
            old_fval,
            old_old_fval,
            args,
            c1,
            maxiter,
        )
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalar_search_wolfe2(
    phi,
    derphi,
    phi0=None,
    old_phi0=None,
    derphi0=None,
    c1=1e-4,
    c2=0.9,
    amax=None,
    extra_condition=None,
    maxiter=10,
):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform

    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star
    phi0 : float
        phi at 0
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """

    if phi0 is None:
        phi0 = phi(0.0)

    if derphi0 is None:
        derphi0 = derphi(0.0)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    # derphi_a1 = derphi(alpha1)  evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        extra_condition = lambda alpha, phi: True

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            if alpha1 == 0:
                msg = "Rounding errors prevent the line search from converging"
            else:
                msg = (
                    "The line search algorithm could not find a solution "
                    + "less than or equal to amax: %s" % amax
                )

            warn(msg, LineSearchWarning)
            break

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (
            (phi_a1 >= phi_a0) and (i > 1)
        ):
            alpha_star, phi_star, derphi_star = _zoom(
                alpha0,
                alpha1,
                phi_a0,
                phi_a1,
                derphi_a0,
                phi,
                derphi,
                phi0,
                derphi0,
                c1,
                c2,
                extra_condition,
            )
            break

        derphi_a1 = derphi(alpha1)
        if abs(derphi_a1) <= -c2 * derphi0:
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if derphi_a1 >= 0:
            alpha_star, phi_star, derphi_star = _zoom(
                alpha1,
                alpha0,
                phi_a1,
                phi_a0,
                derphi_a1,
                phi,
                derphi,
                phi0,
                derphi0,
                c1,
                c2,
                extra_condition,
            )
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn("The line search algorithm did not converge", LineSearchWarning)

    return alpha_star, phi_star, phi0, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found return None

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = np.dot(
                d1, np.asarray([fb - fa - C * db, fc - fa - C * dc]).flatten()
            )
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(
    a_lo,
    a_hi,
    phi_lo,
    phi_hi,
    derphi_lo,
    phi,
    derphi,
    phi0,
    derphi0,
    c1,
    c2,
    extra_condition,
):
    """
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if i > 0:
            cchk = delta1 * dalpha
            a_j = _cubicmin(
                a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
            )
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(
                a_j, phi_aj
            ):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star
