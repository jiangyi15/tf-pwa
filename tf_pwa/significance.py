from scipy.stats import chi2, norm

def prob(chi_2,ndf):
  """
  Computation of the probability for a certain Chi-squared (chi2)
  and number of degrees of freedom (ndf).

  Calculations are based on the incomplete gamma function P(a,x),
  where a=ndf/2 and x=chi2/2.
  
  P(a,x) represents the probability that the observed Chi-squared
  for a correct model should be less than the value chi2.

  The returned probability corresponds to 1-P(a,x),
  which denotes the probability that an observed Chi-squared exceeds
  the value chi2 by chance, even for a correct model.
  
  """
  if ndf <= 0.0: return 0.0 # Set CL to zero in case ndf<=0
  if chi_2 <= 0.0: 
    if chi_2 < 0.0: return 0.0
    else: return 1.0
  return chi2.sf(chi_2,ndf)

def erfc_inverse(x):
  # erfc-1(x) = - 1/sqrt(2) * normal_quantile( 0.5 * x)
  return - 0.70710678118654752440 * normal_quantile( 0.5 * x)

def normal_quantile(p):
  """
  Computes quantiles for standard normal distribution N(0, 1)
  at probability p

  """
  if p<=0 or p>=1 :
    raise Exception("probability outside (0, 1)")
    return 0
  return norm.ppf(p)

def significance(l1,l2,ndf):
  DeltaLL = 2 * abs(l1 - l2)
  p = prob(DeltaLL,ndf)
  # math.sqrt(2) * erfc_inverse(p)
  return - normal_quantile(p / 2)
