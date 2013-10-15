from _numpypy.umath import *

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

PINF = float('inf')
NAN = float('nan')
from math import pi

def geterrobj():
    return [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT2, None]

def seterrobj(val):
    pass

for name in '''
hypot remainder frompyfunc mod
'''.split():
    assert name not in globals()
    globals()[name] = None
