import numpy as np


class BWGenerator:
    def __init__(self, m0, gamma0, m_min, m_max):
        self.m0 = m0
        self.gamma0 = gamma0
        self.m_min = m_min
        self.m_max = m_max
        self.k = self.gamma0 / 2
        self.int_all = self.integral(m_max) - self.integral(m_min)
        self.kxmin = np.arctan((self.m_min - self.m0) / self.k)

    def __call__(self, x):
        return 1 / ((x - self.m0) ** 2 + self.gamma0**2 / 4)

    def integral(self, x):
        k = 1 / self.k
        return k * np.arctan(k * (x - self.m0))

    def solve(self, x):
        x = x * self.int_all
        y = self.k * np.tan(self.k * x + self.kxmin) + self.m0
        return y

    def generate(self, N):
        x = np.random.random(N)
        return self.solve(x)
