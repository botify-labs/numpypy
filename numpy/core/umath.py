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

ERR_DEFAULT = 521

UFUNC_BUFSIZE_DEFAULT = 8192

PZERO = float('0.0')
NZERO = float('-0.0')
PINF = float('inf')
NINF = float('-inf')
NAN = float('nan')
from math import e, pi

def geterrobj():
    return [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]

def seterrobj(val):
    pass

from _numpypy.umath import *

for name in '''
hypot remainder frompyfunc mod
'''.split():
    if name not in globals():
        globals()[name] = None
