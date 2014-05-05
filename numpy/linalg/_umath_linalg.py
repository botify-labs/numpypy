# A CFFI version of numpy/linalg/_umath_linalg.c.src


for name in ['svd_n_f']:
    if name not in locals():
        exec('def %(name)s(*args, **kwargs):\n    raise NotImplementedError("%(name)s")' % locals())

