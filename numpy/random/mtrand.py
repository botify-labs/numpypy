from random import Random
from numpy import zeros

# This file is used in pypy's version of mtrand,
# cpython compiles a mtrand capi module.

# as functionality is added here, uncomment the function
# in __all__ located in numpy/random/info.py

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
