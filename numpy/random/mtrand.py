import os
from cffi import FFI
ffi = FFI()
import numpy as np

ffi.cdef('''
//#define RK_STATE_LEN 624

typedef struct rk_state_
{
    unsigned long key[624];
    int pos;
    int has_gauss; /* !=0: gauss contains a gaussian deviate */
    double gauss;

    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */

    int has_binomial; /* !=0: following parameters initialized for
                              binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;

}
rk_state;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} rk_error;

/* error strings */
extern char *rk_strerror[2]; //RK_ERR_MAX

/* Maximum generated random value */
//#define RK_MAX 0xFFFFFFFFUL

/*
 * Initialize the RNG state using the given seed.
 */
void rk_seed(unsigned long seed, rk_state *state);
/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */
extern rk_error rk_randomseed(rk_state *state);

/*
 * Returns a random unsigned long between 0 and RK_MAX inclusive
 */
extern unsigned long rk_random(rk_state *state);

/*
 * Returns a random long between 0 and LONG_MAX inclusive
 */
extern long rk_long(rk_state *state);

/*
 * Returns a random unsigned long between 0 and ULONG_MAX inclusive
 */
extern unsigned long rk_ulong(rk_state *state);

/*
 * Returns a random unsigned long between 0 and max inclusive.
 */
extern unsigned long rk_interval(unsigned long max, rk_state *state);

/*
 * Returns a random double between 0.0 and 1.0, 1.0 excluded.
 */
extern double rk_double(rk_state *state);

/*
 * fill the buffer with size random bytes
 */
extern void rk_fill(void *buffer, size_t size, rk_state *state);

/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
extern rk_error rk_devfill(void *buffer, size_t size, int strong);

/*
 * fill the buffer using rk_devfill if the random device is available and using
 * rk_fill if is is not
 * parameters have the same meaning as rk_fill and rk_devfill
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 */
extern rk_error rk_altfill(void *buffer, size_t size, int strong,
                            rk_state *state);

/*
 * return a random gaussian deviate with variance unity and zero mean.
 */
extern double rk_gauss(rk_state *state);

''')

# XXX this should open a shared object, not a extension module
import imp as __imp
suffixes = __imp.get_suffixes()
for suffix, mode, typ in suffixes:
    if typ == __imp.C_EXTENSION:
        break
_mtrand = ffi.dlopen(os.path.abspath(os.path.dirname(__file__)) + '/_mtrand' + suffix)

class RandomState(object):
    def __init__(self, seed=None):
        self.internal_state = ffi.new('rk_state *')
        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            _mtrand.rk_randomseed(self.internal_state)
            return
        if isinstance(seed, np.integer):
            seed = int(seed)
        if type(seed) is int:
            _mtrand.rk_seed(seed, self.internal_state)
        else:
            obj = np.array([seed], np.long)
            _mtrand.init_by_array(self.internal_state, obj.ctypes.data, 1)

    def get_state(self):
        state = np.array(list(self.internal_state.key))
        return ('MT19937', state, self.internal_state.pos,
            self.internal_state.has_gauss, self.internal_state.gauss)

    def set_state(self,state):
        algorithm_name = state[0]
        if algorithm_name != 'MT19937':
            raise ValueError("algorithm must be 'MT19937'")
        key, pos = state[1:3]
        if len(state) == 3:
            has_gauss = 0
            cached_gaussian = 0.0
        else:
            has_gauss, cached_gaussian = state[3:5]
        self.internal_state.key = key.tolist()
        self.internal_state.pos = pos
        self.internal_state.has_gauss = has_gauss
        self.internal_state.gauss = cached_gaussian

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)


    def random_sample(self, size=None):
        if size is None:
            return _mtrand.rk_double(self.internal_state)
        ret = np.zeros(size)
        for x in xrange(ret.size):
            ret.flat[x] = _mtrand.rk_double(self.internal_state)
        return ret

    def rand(self, *args):
        if len(args) == 0:
            args = None
        return self.random_sample(args)

    def standard_normal(self, size=None):
        if size is None:
            return _mtrand.rk_gauss(self.internal_state)
        ret = np.zeros(size)
        for x in xrange(ret.size):
            ret.flat[x] = _mtrand.rk_gauss(self.internal_state)
        return ret

    def randn(self, *args):
        if len(args) == 0:
            args = None
        return self.standard_normal(args)

    def randint(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        diff = high - low - 1
        if size is None:
            rv = low + _mtrand.rk_interval(diff, self. internal_state)
            return rv
        ret = np.zeros(size, dtype='int')
        for x in xrange(ret.size):
            rv = low + _mtrand.rk_interval(diff, self. internal_state)
            ret.flat[x] = rv
        return ret

seed = _mtrand.rk_seed

_rand = RandomState()

for attr in dir(_rand):
    if not attr.startswith('_'):
        globals()[attr] = getattr(_rand, attr)

