
class Complex_F:
  def __init__(self,F,r,i):
    self.F = F
    self.r = r
    self.i = i
  def __add__(self,other):
    if type(other) is Complex_F:
      return Complex_F(self.F,self.r+other.r,self.i+other.i)
    if type(other) is complex:
      return Complex_F(self.F,self.r+other.real,self.i+other.imag)
    return Complex_F(self.F,self.r+other,self.i)
  def __sub__(self,other):
    if type(other) is Complex_F:
      return Complex_F(self.F,self.r-other.r,self.i-other.i)
    if type(other) is complex:
      return Complex_F(self.F,self.r-other.real,self.i-other.imag)
    return Complex_F(self.F,self.r-other,self.i)
  def __mul__(self,other):
    if type(other) is Complex_F:
      r = self.r * other.r - self.i * other.i
      i = self.r * other.i + self.i * other.r
      return Complex_F(self.F,r,i)
    if type(other) is complex:
      r = self.r * other.real - self.i * other.imag
      i = self.r * other.imag + self.i * other.real
      return Complex_F(self.F,r,i)
    else :
      return Complex_F(self.F,self.r*other,self.i*other)
  def __truediv__(self,other):
    if type(other) is Complex_F:
      rho = other.rho2()
      r = self.r * other.r + self.i * other.i
      i = -self.r * other.i + self.i * other.r
      return Complex_F(self.F,r/rho,i/rho)
    if type(other) is complex:
      rho = other.rho2()
      r = self.r * other.real + self.i * other.imag
      i = self.r * other.imag - self.i * other.real
      return Complex_F(self.F,r/rho,i/rho)
    return Complex_F(self.F,self.r/other,self.i/other)
  def conj(self):
    return Complex_F(self.F,self.r,-self.i)
  def rho2(self):
    return self.r * self.r + self.i * self.i
  def inverse(self):
    rho = self.rho2()
    return Complex_F(self.F,self.r/rho,-self.i/rho)
  def exp(self):
    rho = self.F.exp(self.r)
    r = self.F.cos(self.i)
    i = self.F.sin(self.i)
    return Complex_F(self.F,rho * r,rho * i)
  def __repr__(self):
    return "("+self.r.__repr__() +","+ self.i.__repr__() +"j)"
