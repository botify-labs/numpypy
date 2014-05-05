# A CFFI version of numpy/linalg/_umath_linalg.c.src

def svd_n_f():
    print 'h'

for name in ['svd_n_f']:
    if name not in locals():
        exec('def %(name)s(*args, **kwargs):\n    raise NotImplementedError("%(name)s")' % locals())

