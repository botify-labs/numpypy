from random import Random
from numpy import zeros

_rand = Random()

def random_sample(size=None):
    if size is None:
        return _rand.random()
    ret = zeros(size)
    for x in xrange(ret.size):
        ret.flat[x] = _rand.random()
    return ret

def rand(*args):
    if len(args) == 0:
        args = None
    return random_sample(args)

def standard_normal(size=None):
    if size is None:
        return _rand.gauss(0., 1.)
    ret = zeros(size)
    for x in xrange(ret.size):
        ret.flat[x] = _rand.gauss(0., 1.)
    return ret

def randn(*args):
    if len(args) == 0:
        args = None
    return standard_normal(args)

def randint(low, high=None, size=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        return _rand.randint(low, high)
    ret = zeros(size, dtype='int')
    for x in xrange(ret.size):
        ret.flat[x] = _rand.randint(low, high)
    return ret

seed = _rand.seed

#from _numpypy.mtrand import *  # doesn't exist yet

for name in '''
beta binomial bytes chisquare exponential f gamma geometric get_state gumbel
hypergeometric laplace logistic lognormal logseries multinomial
multivariate_normal negative_binomial noncentral_chisquare noncentral_f normal
pareto permutation poisson power random_integers rayleigh set_state shuffle
standard_cauchy standard_exponential standard_gamma standard_t triangular
uniform vonmises wald weibull zipf
'''.split():
    if name not in globals():
        globals()[name] = None
