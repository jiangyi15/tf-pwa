"""
This module implements classes and methods to manage the variables in fitting.
"""
import tensorflow as tf
import numpy as np
import warnings
from .config import regist_config, get_config

'''
vm = VarsManager(dtype=tf.float64)

vm.set_fix(var_name,value)#var_name是实变量的name（复变量name的两个实分量分别叫namer，namei）
vm.set_bound({var_name:(a,b)},func="(b-a)*(sin(x)+1)/2+a")
vm.set_share_r([var_name1,var_name2])#var_name是复变量的name
vm.set_all(init_params)
vm.std_polar_all()
vm.trans_fcn(fcn,grad)#bound转换
'''


class VarsManager(object):
    """
    This class provides methods to operate the variables in fitting. Every variable is a 1-d **tf.Variable** of
    **dtype** (**tf.float64** by default).

    All variables are stored in a dictionary **self.variables**. The indices of the dictionary are the variables' names,
    so name property in **tf.Variable** does not matter. All methods intended to change the variables are operating
    **self.variables** directly.

    Besides, all trainable variables' names will be stored in a list **self.trainable_vars**.
    """

    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.polar = True
        self.variables = {}  # {name:tf.Variable,...}
        self.trainable_vars = []  # [name,...]
        self.complex_vars = {}  # {name:polar(bool),...}
        self.same_list = []  # [[name1,name2],...]

        # self.range = {}
        self.bnd_dic = {}  # {name:(a,b),...}

        self.var_head = {}  # {head:[name1,name2],...}

    def add_real_var(self, name, value=None, range_=None, trainable=True):
        """
        Add a real variable named **name** into **self.variables**. If **value** and **range_** are not provided, the initial value
        is set to be a uniform random number between 0 and 1.

        :param name: The name of the variable, the index of this variable in **self.variables**
        :param value: The initial value.
        :param range_: Length-2 array. It's useless if **value** is given. Otherwise the initial value is set to be a uniform random number between **range_[0]** and **range_[0]**.
        :param trainable: Boolean. If it's **True**, the variable is trainable while fitting.
        """
        if name in self.variables:  # not a new var
            if name in self.trainable_vars:
                self.trainable_vars.remove(name)
            # warnings.warn("Overwrite variable {}!".format(name))

        if value is None:
            if range_ is None:  # random [0,1]
                self.variables[name] = tf.Variable(tf.random.uniform(shape=[], minval=0., maxval=1., dtype=self.dtype),
                                                   trainable=trainable)
            else:  # random [a,b]
                self.variables[name] = tf.Variable(
                    tf.random.uniform(shape=[], minval=range_[0], maxval=range_[1], dtype=self.dtype),
                    trainable=trainable)
        else:  # constant value
            self.variables[name] = tf.Variable(value, dtype=self.dtype, trainable=trainable)

        if trainable:
            self.trainable_vars.append(name)

    def add_complex_var(self, name, polar=None, trainable=True, fix_vals=(1.0, 0.0)):
        """
        Add a complex variable. Two real variables named **name+'r'** and **name+'i'** will be added into
        **self.variables**. The initial values will be given automatically according to its form of coordinate.

        :param name: The name of the complex variable.
        :param polar: Boolean. If it's **True**, **name+'r'** and **name+'i'** are defined in polar coordinate; otherwise they are defined in Cartesian coordinate.
        :param trainable: Boolean. If it's **True**, real variables **name+'r'** and **name+'i'** will be trainable.
        :param fix_vals: Length-2 array. If **trainable=False**, the fixed values for **name+'r'** and **name+'i'** are **fix_vals[0]**, **fix_vals[1]** respectively.
        """
        if polar is None:
            polar = self.polar
        var_r = name + 'r'
        var_i = name + 'i'
        if trainable:
            if polar:
                self.add_real_var(name=var_r, range_=(0, 2.0))
                self.add_real_var(name=var_i, range_=(-np.pi, np.pi))
            else:
                self.add_real_var(name=var_r, range_=(-1, 1))
                self.add_real_var(name=var_i, range_=(-1, 1))
        else:
            self.add_real_var(name=var_r, value=fix_vals[0], trainable=False)
            self.add_real_var(name=var_i, value=fix_vals[1], trainable=False)
        self.complex_vars[name] = polar

    def remove_var(self, name, cplx=False):
        """
        Remove a variable from **self.variables**. More specifically, two variables (**name+'r'** and **name+'i'**)
        will be removed if **cplx=True**.

        :param name: The name of the variable
        :param cplx: Boolean. Users should indicate if this variable is complex or not.
        """
        if cplx:
            del self.complex_vars[name]
            name_r = name + 'r'
            name_i = name + 'i'
            if self.variables[name_r].trainable:
                if name_r in self.trainable_vars:
                    self.trainable_vars.remove(name_r)
            if self.variables[name_i].trainable:
                if name_i in self.trainable_vars:
                    self.trainable_vars.remove(name_i)
            for l in self.same_list:
                if name_r in l:
                    l.remove(name_r)
                if name_i in l:
                    l.remove(name_i)
            del self.variables[name_r]
            del self.variables[name_i]
        else:
            if self.variables[name].trainable:
                self.trainable_vars.remove(name)
            for l in self.same_list:
                if name in l:
                    l.remove(name)
            del self.variables[name]

    def refresh_vars(self, name):
        """
        Refresh all trainable variables (WIP)
        """
        cplx_vars = []
        for name in self.complex_vars:
            name_r = name + 'r'
            name_i = name + 'i'
            if self.complex_vars[name] == False:  # xy coordinate
                if name_r in self.trainable_vars:
                    cplx_vars.append(name_r)
                    self.variables[name_r].assign(tf.random.uniform(shape=[], minval=-1, maxval=1, dtype=self.dtype))
                if name_i in self.trainable_vars:
                    cplx_vars.append(name_i)
                    self.variables[name_i].assign(tf.random.uniform(shape=[], minval=-1, maxval=1, dtype=self.dtype))
            else:  # polar coordinate
                if name_r in self.trainable_vars:
                    cplx_vars.append(name_r)
                    self.variables[name_r].assign(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=self.dtype))
                if name_i in self.trainable_vars:
                    cplx_vars.append(name_i)
                    self.variables[name_i].assign(
                        tf.random.uniform(shape=[], minval=-np.pi, maxval=np.pi, dtype=self.dtype))
        real_vars = self.trainable_vars.copy()  # 实变量还没refresh
        for i in real_vars:
            if i in cplx_vars:
                real_vars.remove(i)

    def set_fix(self, name, value=None, unfix=False):
        """
        Fix or unfix a variable (change the trainability)
        :param name: The name of the variable
        :param value: The fixed value. It's useless if **unfix=True**.
        :param unfix: Boolean. If it's **True**, the variable will become trainable rather than be fixed.
        """
        if value == None:
            value = self.variables[name].value
        var = tf.Variable(value, dtype=self.dtype, trainable=unfix)
        self.variables[name] = var
        if unfix:
            if name in self.trainable_vars:
                warnings.warn("{} has been freed already!".format(name))
            else:
                self.trainable_vars.append(name)
        else:
            if name in self.trainable_vars:
                self.trainable_vars.remove(name)
            else:
                warnings.warn("{} has been fixed already!".format(name))

    def set_bound(self, bound_dic, func=None, overwrite=False):
        """
        Set boundary for the trainable variables. The variables will be constrained in their ranges while fitting.

        :param bound_dic: Dictionary. E.g. **{"name1":(-1.0,1.0), "name2":(None,1.0)}**. In this example, **None** means it has no lower limit.
        :param func: String. Users can provide a string to describe the transforming function. For details, refer to class **tf_pwa.variable.Bound**.
        :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
        """
        for name in bound_dic:
            if name in self.bnd_dic:
                if not overwrite:
                    warnings.warn("Overwrite bound of {}!".format(name))
            self.bnd_dic[name] = Bound(*bound_dic[name], func=func)

    def set_share_r(self, name_list):  # name_list==[name1,name2,...]
        """
        If some complex variables want to share their radia variable while their phase variable are still different.
        Users can set this type of constrain using this method.

        :param name_list: List of strings. Note the strings should be the name of the complex variables rather than of their radium parts.
        """
        self.xy2rp_all(name_list)
        name_r_list = [name + 'r' for name in name_list]
        self.set_same(name_r_list)
        for name in name_r_list:
            self.complex_vars[name[:-1]] = name_r_list

    def set_same(self, name_list, cplx=False):
        """
        Set some variables to be the same.

        :param name_list: List of strings. Name of the variables.
        :param cplx: Boolean. Whether the variables are complex or real.
        """
        tmp_list = []
        for name in name_list:
            for add_list in self.same_list:
                if name in add_list:
                    tmp_list += add_list
                    self.same_list.remove(add_list)
                    break
        for i in tmp_list:
            if i not in name_list:
                name_list.append(i)  # 去掉重复元素

        def same_real(name_list):
            var = self.variables[name_list[0]]
            for name in name_list[1:]:
                if name in self.trainable_vars:
                    self.trainable_vars.remove(name)
                else:
                    var = self.variables[name]  # if one is untrainable, the others will all be untrainable
                    if name_list[0] in self.trainable_vars:
                        self.trainable_vars.remove(name_list[0])
            for name in name_list:
                self.variables[name] = var

        if cplx:
            same_real([name + 'r' for name in name_list])
            same_real([name + 'i' for name in name_list])
        else:
            same_real(name_list)
        self.same_list.append(name_list)

    def get(self, name):
        """
        Get a real variable
        :param name: String
        :return: tf.Variable
        """
        if name not in self.variables:
            raise Exception("{} not found".format(name))
        return self.variables[name]  # tf.Variable

    def set(self, name, value):
        """
        Set value for a real variable

        :param name: String
        :param value: Real number
        """
        if name not in self.variables:
            raise Exception("{} not found".format(name))
        self.variables[name].assign(value)

    def rp2xy(self, name):
        """
        Transform a complex variable into Cartesian coordinate.
        :param name: String
        """
        if self.complex_vars[name] != True:  # if not polar (already xy)
            return
        r = self.variables[name + 'r']
        p = self.variables[name + 'i']
        x = r * tf.cos(p)
        y = r * tf.sin(p)
        self.variables[name + 'r'].assign(x)
        self.variables[name + 'i'].assign(y)
        self.complex_vars[name] = False
        for l in self.same_list:
            if name in l:
                for i in l:
                    self.complex_vars[i] = False
                break

    def xy2rp(self, name):
        """
        Transform a complex variable into polar coordinate.
        :param name: String
        """
        if self.complex_vars[name] != False:  # if already polar
            return
        x = self.variables[name + 'r']
        y = self.variables[name + 'i']
        r = tf.sqrt(x * x + y * y)
        p = tf.atan2(y, x)
        self.variables[name + 'r'].assign(r)
        self.variables[name + 'i'].assign(p)
        self.complex_vars[name] = True
        for l in self.same_list:
            if name in l:
                for i in l:
                    self.complex_vars[i] = True
                break

    @property
    def trainable_variables(self):
        """
        List of tf.Variable. It is similar to **tf.keras.Model.trainable_variables**.
        """
        vars_list = []
        for name in self.trainable_vars:
            vars_list.append(self.variables[name])
        return vars_list

    def get_all_val(self, after_trans=False):  # if bound transf var
        """
        Get the values of all trainable variables.

        :param after_trans: Boolean. If it's **True**, the values will be the ones post-boundary-transformation (the ones that are actually used in fitting).
        :return: List of real numbers.
        """
        vals = []
        if after_trans:
            for name in self.trainable_vars:
                yval = self.get(name).numpy()
                if name in self.bnd_dic:
                    xval = self.bnd_dic[name].get_y2x(yval)
                else:
                    xval = yval
                vals.append(xval)
        else:
            for name in self.trainable_vars:
                yval = self.get(name).numpy()
                vals.append(yval)
        return vals  # list (for list of tf.Variable use self.trainable_variables; for dict of all vars, use self.variables)

    def get_all_dic(self, trainable_only=False):
        """
        Get a dictionary of all variables.

        :param trainable_only: Boolean. If it's **True**, the dictionary only contains the trainable variables.
        :return: Dictionary
        """
        # self.std_polar_all()
        dic = {}
        if trainable_only:
            for i in self.trainable_vars:
                dic[i] = self.variables[i].numpy()
        else:
            for i in self.variables:
                dic[i] = self.variables[i].numpy()
        return dic

    def set_all(self, vals):  # use either dict or list
        """
        Set values for some variables.

        :param vals: It can be either a dictionary or a list of real numbers. If it's a list, the values correspond to all trainable variables in order.
        """
        if type(vals) == dict:
            for name in vals:
                self.set(name, vals[name])
        else:
            i = 0
            for name in self.trainable_vars:
                self.set(name, vals[i])
                i += 1

    def rp2xy_all(self, name_list=None):
        """
        If **name_list** is not provided, this method will transform all complex variables into Cartesian coordinate.

        :param name_list: List of names of complex variables
        """
        if not name_list:
            name_list = self.complex_vars
        for name in name_list:
            self.rp2xy(name)

    def xy2rp_all(self, name_list=None):
        """
        If **name_list** is not provided, this method will transform all complex variables into polar coordinate.

        :param name_list: List of names of complex variables
        """
        if not name_list:
            name_list = self.complex_vars
        for name in name_list:
            self.xy2rp(name)

    @staticmethod
    def _std_polar_angle(p, a=-np.pi, b=np.pi):
        twopi = b - a
        while p <= a:
            p.assign_add(twopi)
        while p >= b:
            p.assign_add(-twopi)

    def std_polar(self, name):
        """
        Transform a complex variable into standard polar coordinate, which mean its radium part is positive, and its
        phase part is between :math:`-\\pi` to :math:`\\pi`.
        :param name: String
        """
        self.xy2rp(name)
        r = self.variables[name + 'r']
        p = self.variables[name + 'i']
        if r < 0:
            r.assign(tf.abs(r))
            if type(self.complex_vars[name]) == list:
                for name_r in self.complex_vars[name]:
                    self.variables[name_r[:-1] + 'i'].assign_add(np.pi)
            else:
                p.assign_add(np.pi)
        self._std_polar_angle(p)

    def std_polar_all(self):  # std polar expression: r>0, -pi<p<pi
        """
        Transform all complex variables into standard polar coordinate.
        """
        for name in self.complex_vars:
            self.std_polar(name)

    def trans_params(self, polar):
        """
        Transform all complex variables into either polar coordinate or Cartesian coordinate.

        :param polar: Boolean
        """
        if polar:
            self.std_polar_all()
        else:
            self.rp2xy_all()

    def trans_fcn_grad(self, fcn_grad):  # bound transform fcn and grad
        """
        :math:`F(x)=F(y(x))`, :math:`G(x)=\\frac{dF}{dx}=\\frac{dF}{dy}\\frac{dy}{dx}`

        :param fcn_grad: The return of class **tf_pwa.model**???
        :return:
        """

        def fcn_t(xvals):
            xvals = np.array(xvals)
            yvals = xvals.copy()
            dydxs = []
            i = 0
            for name in self.trainable_vars:
                if name in self.bnd_dic:
                    yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
                    dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
                else:
                    dydxs.append(1)
                i += 1
            fcn, grad_yv = fcn_grad(yvals)
            grad = np.array(grad_yv) * dydxs
            return fcn, grad

        return fcn_t


import sympy as sy


class Bound(object):
    """
    This class provides methods to implement the boundary constraint for a variable.
    It has dependence on `SymPy <https://www.sympy.org/en/index.html>`_ .
    The boundary-transforming function can transform a variable *x* defined in the real domain to a variable *y* defined
    in a limited range *(a,b)*. *y* should be the physical parameter but *x* is the one used while fitting.

    :param a: Real number. The lower boundary
    :param b: Real number. The upper boundary
    :param func: String. The boundary-transforming function. By default, if neither **a** or **b** is **None**, **func** is **"(b-a)*(sin(x)+1)/2+a"**; else if only **a** is **None**, **func** is **"b+1-sqrt(x**2+1)"**; else if only **b** is **None**, **func** is **"a-1+sqrt(x**2+1)"**; else **func** is **"x"**.

    **a**, **b**, **func** can be refered by **self.lower**, **self.upper**, **self.func**.
    """

    def __init__(self, a=None, b=None, func=None):
        if a is not None and b is not None and a > b:
            raise Exception("Lower bound is larger than upper bound!")
        self.lower = a
        self.upper = b
        if func:
            self.func = func  # from R (x) to a limited range (y) #Note: y is gls but x is the var in fitting
        else:
            if a is None:
                if b is None:
                    self.func = "x"
                else:
                    self.func = "b+1-sqrt(x**2+1)"
            else:
                if b is None:
                    self.func = "a-1+sqrt(x**2+1)"
                else:
                    self.func = "(b-a)*(sin(x)+1)/2+a"
        self.f, self.df, self.inv = self.get_func()

    def get_func(self):  # init func string into sympy f(x) or f(y)
        """
        Initialize the function string into **sympy** objects.

        :return: **sympy** objects **f**, **df**, **inv**, which are the function, its derivative and its inverse function.
        """
        x, a, b, y = sy.symbols("x a b y")
        f = sy.sympify(self.func)
        f = f.subs({a: self.lower, b: self.upper})
        df = sy.diff(f, x)
        inv = sy.solve(f - y, x)
        if hasattr(inv, "__len__"):
            inv = inv[-1]
        return f, df, inv

    def get_x2y(self, val):  # var->gls
        """
        To derive *y* from *x*

        :param val: Real number *x*
        :return: Real number *y*
        """
        x = sy.symbols('x')
        return float(self.f.evalf(subs={x: val}))

    def get_y2x(self, val):  # gls->var
        """
        To derive *x* from *y*. *y* will be set to *a* if *y<a*, and *y* will be set to *b* if *y>b*.

        :param val: Real number *y*
        :return: Real number *x*
        """
        y = sy.symbols('y')
        if self.lower is not None and val < self.lower:
            val = self.lower
        elif self.upper is not None and val > self.upper:
            val = self.upper
        return float(self.inv.evalf(subs={y: val}))

    def get_dydx(self, val):  # gradient in fitting: dNLL/dx = dNLL/dy * dy/dx
        """
        To calculate the derivative :math:`\\frac{dy}{dx}`.

        :param val: Real number *x*
        :return: Real number :math:`\\frac{dy}{dx}`
        """
        x = sy.symbols('x')
        return float(self.df.evalf(subs={x: val}))


def _shape_func(f, shape, name, **kwargs):
    if not shape:
        f(name, **kwargs)
    else:
        for i in range(shape[0]):
            _shape_func(f, shape[1:], name + '_' + str(i), **kwargs)


regist_config("vm", VarsManager(dtype="float64"))


# vm = VarsManager(dtype=tf.float64)
class Variable(object):
    """
    This class has interface to **VarsManager**. It is convenient for users to define a group of real variables,
    since it may be more perceptually intuitive to define them together.

    By calling the instance of this class, it returns the value of this variable. The type is tf.Tensor.

    :param name: The name of the variable group
    :param shape: The shape of the group. E.g. for a 4*3*2 matrix, **shape** is **[4,3,2]**. By default, **shape** is [] for a real variable.
    :param cplx: Boolean. Whether the variable (or the variables) are complex or not.
    :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
    :param vm: VarsManager. It is by default the one automatically defined in the global scope by the program.
    :param kwargs: Other arguments that may be used when calling **self.real_var()** or **self.cplx_var()**
    """
    def __init__(self, name, shape=[], cplx=False, overwrite=False, vm=None, **kwargs):
        if vm is None:
            vm = get_config("vm")
        self.vm = vm
        self.name = name
        if name in self.vm.var_head:
            if not overwrite:
                warnings.warn("Overwrite Variable {}!".format(name))
            for i in self.vm.var_head[name]:
                self.vm.remove_var(i, cplx)
        self.vm.var_head[self.name] = []
        if type(shape) == int:
            shape = [shape]
        self.shape = shape
        self.cplx = cplx
        if cplx:
            self.cplx_var(**kwargs)
        else:
            self.real_var(**kwargs)
        self.bound = None

    def real_var(self, value=None, range_=None, fix=False):
        """
        It implements interface to ``VarsManager.add_real_var()``, but supports variables that are not of non-shape.

        :param value: Real number. The value of all real components.
        :param range_: Length-2 array. The length of all real components.
        :param fix: Boolean. Whether the variable is fixed.
        """
        trainable = not fix

        def func(name, **kwargs):
            self.vm.add_real_var(name, value, range_, trainable)
            self.vm.var_head[self.name].append(name)

        _shape_func(func, self.shape, self.name, value=value, range_=range_, trainable=trainable)

    def cplx_var(self, polar=True, fix=False, fix_which=0, fix_vals=(1.0, 0.0)):
        """
        It implements interface to ``VarsManager.add_complex_var()``, but supports variables that are not of non-shape.

        :param polar: Boolean. Whether the variable is defined in polar coordinate or in Cartesian coordinate.
        :param fix: Boolean. Whether the variable is fixed. It's enabled only if ``self.shape is None``.
        :param fix_which: Integer. Which complex component in the innermost layer of the variable is fixed. E.g. If ``self.shape==[2,3,4]`` and ``fix_which==1``, then Variable()[i][j][1] will be the fixed value. It's enabled only if ``self.shape is not None``.
        :param fix_vals: Length-2 tuple. The value of the fixed complex variable is ``fix_vals[0]+fix_vals[1]j``.
        """
        # fix_which = fix_which % self.shape[-1]
        def func(name, **kwargs):
            if self.shape:
                trainable = not (name[-2:] == '_' + str(fix_which))
            else:
                trainable = not fix
            self.vm.add_complex_var(name, polar, trainable, fix_vals)
            self.vm.var_head[self.name].append(name)

        _shape_func(func, self.shape, self.name, polar=polar, fix_which=fix_which, fix_vals=fix_vals)

    @property
    def value(self):
        """
        :return: Ndarray of ``self.shape``.
        """
        return tf.Variable(self()).numpy()

    @property
    def variables(self):
        """
        Names of the real variables contained in this Variable instance.

        :return: List of string.
        """
        return self.vm.var_head[self.name]

    def fixed(self, value):
        """
        Fix this Variable. Note only non-shape real Variable supports this method.

        :param value: Real number. The fixed value
        """
        if not self.shape:
            if self.cplx:
                value = complex(value)
                self.vm.set_fix(self.name + 'r', value.real)
                self.vm.set_fix(self.name + 'i', value.imag)
            else:
                self.vm.set_fix(self.name, value)
        else:
            raise Exception("Only shape==() real var supports 'fixed' method.")

    def freed(self):
        """
        Set free this Variable. Note only non-shape Variable supports this method.
        """
        if not self.shape:
            if self.cplx:
                self.vm.set_fix(self.name + 'r', unfix=True)
                self.vm.set_fix(self.name + 'i', unfix=True)
            else:
                self.vm.set_fix(self.name, unfix=True)
        else:
            raise Exception("Only shape==() var supports 'freed' method.")

    def set_bound(self, bound, func=None, overwrite=False):
        """
        Set boundary for this Variable. Note only non-shape real Variable supports this method.

        :param bound: Length-2 tuple.
        :param func: String. Refer to class **tf_pwa.variable.Bound**.
        :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
        """
        if not self.shape:
            self.vm.set_bound({self.name: bound}, func, overwrite=overwrite)
            self.bound = self.vm.bnd_dic[self.name]
        else:
            raise Exception("Only shape==() real var supports 'set_bound' method.")

    def r_shareto(self, Var):
        """
        Share the radium component to another Variable of the same shape. Only complex Variable supports this method.

        :param Var: Variable.
        """
        if self.shape != Var.shape:
            raise Exception("Shapes are not the same.")
        if not (self.cplx and Var.cplx):
            raise Exception("Type is not complex var.")

        def func(name, **kwargs):
            self.vm.set_share_r([self.name + name, Var.name + name])

        _shape_func(func, self.shape, '')

    def sameas(self, Var):
        """
        Set the Variable to be the same with another Variable of the same shape.

        :param Var: Variable.
        """
        if self.shape != Var.shape:
            raise Exception("Shapes are not the same.")
        if self.cplx != Var.cplx:
            raise Exception("Types (real or complex) are not the same.")

        def func(name, **kwargs):
            self.vm.set_same([self.name + name, Var.name + name], cplx=self.cplx)

        _shape_func(func, self.shape, '')

    def __call__(self):
        var_list = np.ones(shape=self.shape).tolist()
        if self.shape:
            def func(name, **kwargs):
                tmp = var_list
                idx_str = name.split('_')[-len(self.shape):]
                for i in idx_str[:-1]:
                    tmp = tmp[int(i)]
                if self.cplx:
                    if (name in self.vm.complex_vars) and self.vm.complex_vars[name]:
                        real = self.vm.variables[name + 'r'] * tf.cos(self.vm.variables[name + 'i'])
                        imag = self.vm.variables[name + 'r'] * tf.sin(self.vm.variables[name + 'i'])
                        tmp[int(idx_str[-1])] = tf.complex(real, imag)
                        # print("&&&&&pg",name)
                    else:
                        # print("$$$$$xg",name)
                        tmp[int(idx_str[-1])] = tf.complex(self.vm.variables[name + 'r'], self.vm.variables[name + 'i'])
                    # print(tmp[int(idx_str[-1])])
                else:
                    tmp[int(idx_str[-1])] = self.vm.variables[name]

            _shape_func(func, self.shape, self.name)
        else:
            if self.cplx:
                name = self.name
                if (name in self.vm.complex_vars) and self.vm.complex_vars[name]:
                    real = self.vm.variables[name + 'r'] * tf.cos(self.vm.variables[name + 'i'])
                    imag = self.vm.variables[name + 'r'] * tf.sin(self.vm.variables[name + 'i'])
                    var_list = tf.complex(real, imag)
                    # print("&&&pt",name)
                else:
                    # print("$$$xt",name)
                    var_list = tf.complex(self.vm.variables[self.name + 'r'], self.vm.variables[self.name + 'i'])
                # print(var_list)
            else:
                var_list = self.vm.variables[self.name]

        # return tf.stack(var_list)
        return var_list


if __name__ == "__main__":
    m = Variable("R_m", value=2.1)  # trainable is True by default
    g_ls = Variable("A2BR_H", shape=[3], cplx=True)
    fcr = Variable("R_total", cplx=True)
    m1 = Variable("R1_m", value=2.3)
    g_ls1 = Variable("A2BR1_H", shape=[3], cplx=True)
    fcr1 = Variable("R1_total", cplx=True)

    print(g_ls.value)
    print(g_ls())

    m.fixed(2.4)
    g_ls.sameas(g_ls1)
    fcr.r_shareto(fcr1)
