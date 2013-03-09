from numpypy.core.umath import *
"""(maximum,  minimum, absolute, equal, not_equal,
        isnan, isinf, isfinite, sin, cos, tan, invert, subtract, multiply, add,
        arctan2, less_equal, greater_equal, sqrt, exp, log, log10)"""
from numpypy import PINF, NAN, pi

frompyfunc = mod = None

def geterrobj():
    """Fake function: simply return defaults"""
    return [8192, 0, None]

def seterrobj(val):
    pass

# Constants:
SHIFT_DIVIDEBYZERO = 0
SHIFT_OVERFLOW = 3
SHIFT_UNDERFLOW = 6
SHIFT_INVALID = 9

ERR_IGNORE = 0
ERR_WARN = 1
ERR_RAISE = 2
ERR_CALL = 3
ERR_PRINT = 4
ERR_LOG = 5
ERR_DEFAULT2 = 521

UFUNC_BUFSIZE_DEFAULT = 8192

undef = '''hypot remainder
'''.split()
for name in undef:
    assert name not in globals()
    globals()[name] = None
