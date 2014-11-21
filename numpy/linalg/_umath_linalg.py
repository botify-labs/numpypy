# A CFFI version of numpy/linalg/_umath_linalg.c.src

try:
    import cffi
    have_cffi = True
except ImportError:
    have_cffi = False


if 0 and have_cffi and '__pypy__' in sys.builtin_module_names:
    # TODO: use cffi versions with frompyfunc
    pass
else:
    from _umath_linalg_capi import *


for name in ['svd_n_f']:
    if name not in locals():
        exec('def %(name)s(*args, **kwargs):\n    raise NotImplementedError("%(name)s")' % locals())

