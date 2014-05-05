# A CFFI version of numpy/linalg/lapack_module.c


for name in ['zheevd', 'dsyevd', 'dgeev', 'dgelsd', 'dgesv', 'dgessd',
             'dgetrf', 'dpotrf', 'dgeqrf', 'dorggr', 'zgeev', 'zgelsd',
            'zgesv', 'zgesdd', 'zgetrf', 'zpotrf', 'zgeqrf', 'zungqr',
            'xerbla']:
    if name not in locals():
        exec('def %(name)s(*args, **kwargs):\n    raise NotImplementedError("%(name)s")' % locals())
