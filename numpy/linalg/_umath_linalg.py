# A CFFI version of numpy/linalg/_umath_linalg.c.src
from warnings import warn

try:
    import cffi
    have_cffi = True
except ImportError:
    have_cffi = False


if 0 and have_cffi and '__pypy__' in sys.builtin_module_names:
    # TODO: use cffi versions with frompyfunc
    pass
else:
    try:
        from _umath_linalg_capi import *
    except:
        warn('no cffi linalg functions and no _umath_linalg_capi module, expect problems.')


def NotImplementedFunc(func):
    def tmp(*args, **kwargs):
        raise NotImplementedError("%s not implemented yet" % func)
    return tmp

for name in '''
eigvals eig eigh_lo cholesky_lo svd_n_f svd_m inv
'''.split():
    if name not in globals():
        globals()[name] = NotImplementedFunc(name)

