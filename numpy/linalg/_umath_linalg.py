# A CFFI version of numpy/linalg/umath_linalg.c.src
# As opposed to the numpy version, the cffi version leaves broadcasting to the responsibility
# of the pypy extended frompyfunc, which removes the need for INIT_OUTER_LOOP*
from warnings import warn
import sys, os
import lapack_lite # either a cffi version or a cextension module

try:
    import cffi
    ffi = lapack_lite.ffi
    use_cffi = True
except ImportError:
    use_cffi = False
except AttributeError:
    use_cffi = False

if '__pypy__' in sys.builtin_module_names:
    import _numpypy.umath
    if 'frompyfunc' not in dir(_numpypy.umath):
        use_cffi = False
else:
    # since we use extended frompyfunc(),
    # skip the cffi stuff
    use_cffi = False

if use_cffi:
    # lapack_lite already imported the c functions from
    # somewhere (lapack_lite compiled shared object or blas,
    # but the name of the function may be a bit different.
    # XXX cache this instead of looking up each call
    def get_c_func(blas_ch='', name=''):
        name = blas_ch + lapack_lite.macros['pfx'] + name + lapack_lite.macros['sfx']
        return getattr(lapack_lite._C, name)


    import numpy as np

    # dtype has not been imported yet. Fake it.
    from numpy.core.multiarray import dtype
    class Dummy(object):
        pass
    nt = Dummy()
    nt.int32 = dtype('int32')
    nt.int8 = dtype('int8')
    nt.float32 = dtype('float32')
    nt.float64 = dtype('float64')
    nt.complex64 = dtype('complex64')
    nt.complex128 = dtype('complex128')

    from numpy.core.umath import frompyfunc
    __version__ = '0.1.4'

    toCtypeP = {nt.int32: 'int*', nt.float32: 'float*', nt.float64: 'double*',
               nt.complex64: 'f2c_complex*', nt.complex128: 'f2c_doublecomplex*',
               nt.int8: 'char *'}
    toCtypeA = {nt.int32: 'int[1]', nt.float32: 'float[1]', nt.float64: 'double[1]',
               nt.complex64: 'f2c_complex[1]', nt.complex128: 'f2c_doublecomplex[1]'}

    def toCptr(src):
        if src is None:
            return ffi.cast('void*', 0)
        pData = src.__array_interface__['data'][0]
        return ffi.cast(toCtypeP[src.dtype], pData)


    # Try to find hidden floatstatus functions. On Pypy, we should expose
    # these through umath. On CPython, they might have been exported from
    # the umath module shared object
    from numpy.core import umath
    npy_clear_floatstatus = None
    npy_set_floatstatus_invalid = None
    try:
        npy_clear_floatstatus = getattr(umath, 'clear_floatstatus')
        npy_set_floatstatus_invalid = getattr(umath, 'set_floatstatus_invalid')
    except AttributeError:
        sh_obj = umath.__file__
        ext = os.path.splitext(sh_obj)[1]
        if ext.startswith('.py'):
            warn('npy_clear_floatstatus, npy_set_floatstatus_invalid not found')
        else:
            try:
                umath_so = ffi.dlopen(sh_obj)
                if npy_clear_floatstatus is None:
                    npy_clear_floatstatus = getattr(umath_so, 'npy_clear_floatstatus')
                if npy_set_floatstatus_invalid is None:
                    npy_set_floatstatus_invalid = getattr(umath_so, 'npy_set_floatstatus_invalid')
            except OSError, AttributeError:
                warn('npy_clear_floatstatus, npy_set_floatstatus_invalid not found')
        def return0(*args):
            return 0
        if npy_clear_floatstatus is None:
            npy_clear_floatstatus = return0
        if npy_set_floatstatus_invalid is None:
            npy_set_floatstatus_invalid = return0
    def get_fp_invalid_and_clear():
        return bool(npy_clear_floatstatus() & np.FPE_INVALID)

    def set_fp_invalid_or_clear(error_occurred):
        if error_occurred:
            npy_set_floatstatus_invalid()
        else:
            npy_clear_floatstatus()

    base_vals = {'s':{}, 'c':{}, }
    base_vals['s']['one'] = 1
    base_vals['s']['zero'] = 0
    base_vals['s']['minus_one'] = -1
    base_vals['s']['ninf'] = -float('inf')
    base_vals['s']['nan'] = float('nan')
    base_vals['d'] = base_vals['s']
    base_vals['c']['one'] = complex(1, 0)
    base_vals['c']['zero'] = complex(0, 0)
    base_vals['c']['minus_one'] = complex(-1, 0)
    base_vals['c']['ninf'] = complex(-float('inf'), 0)
    base_vals['c']['nan'] = complex(float('nan'), float('nan'))
    base_vals['z'] = base_vals['c']

    class Params(object):
        params = ()
        def __init__(self, *args):
            for a, v in zip(self.params, args):
                setattr(self, a, v)

        def dump(self, name):
            print >> sys.stderr, name
            for p in self.params:
                v = getattr(self, p)
                try:
                    rep = v[0]
                except:
                    print >> sys.stderr, '\t%10s: %r' %(p, v)
                else:
                    if isinstance(v, np.ndarray):
                        print >> sys.stderr, '\t%10s: %r, %r' %(p, v.dtype, v.shape)
                    else:
                        print >> sys.stderr, '\t%10s: %r %r' %(p, v, rep)

    def linearize_matrix(dst, src):
        ''' in cpython numpy, dst, src are c-level pointers
            we (ab)use ndarrays instead
        '''
        if dst is None:
            raise ValueError('called with NULL input, should not happen')
        if src.dtype is not dst.dtype:
            raise ValueError('called with differing dtypes, should not happen')
        if len(src.shape) < 2:
            dst[:] = src
            return
        srcT = src.T
        if srcT.shape != dst.shape:
            raise ValueError('called with differing shapes, should not happen: %r != %r', (srcT.shape, dst.shape))
        dst[:] = srcT

    delinearize_matrix = linearize_matrix

    # --------------------------------------------------------------------------
    # Determinants

    def wrap_slogdet(typ, basetyp, cblas_typ):
        def slogdet_from_factored_diagonal(src, m, sign):
            sign_acc = sign
            logdet_acc = base_vals[cblas_typ]['zero']
            for i in range(m):
                abs_element = np.abs(src[i,i])
                sign_element = src[i, i] / abs_element
                sign_acc = sign_acc * sign_element
                logdet_acc += np.log(abs_element)
            return sign_acc, logdet_acc

        def slogdet_single_element(m, src, pivot):
            info = ffi.new('int[1]', [0])
            pM = ffi.new('int[1]', [m])
            get_c_func(cblas_typ, 'getrf')(pM, pM, toCptr(src), pM, toCptr(pivot), info)
            change_sign = 0
            if info[0] == 0:
                for i in range(m):
                    # fortran uses 1 based indexing
                    change_sign += (pivot[i] != (i + 1))
                if change_sign % 2:
                    sign = base_vals[cblas_typ]['minus_one']
                else:
                    sign = base_vals[cblas_typ]['one']
                sign, logdet = slogdet_from_factored_diagonal(src, m, sign)
            else:
                # if getrf fails, use - as sign and -inf as logdet
                sign = base_vals[cblas_typ]['zero']
                logdet = -float('inf')
            return sign, logdet

        def slogdet(in0):
            ''' notes:
             *   in must have shape [m, m], out[0] and out[1] are [m]
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = in0.shape[0]
            mbuffer = np.empty((m, m), typ)
            pivot = np.empty(m, nt.int32)

            # swapped steps to get matrix in FORTRAN order
            linearize_matrix(mbuffer, in0)
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            return sign, logdet

        def det(in0):
            ''' notes:
             *   in must have shape [m, m], out is scalar
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = in0.shape[0]
            mbuffer = np.empty((m, m), typ)
            linearize_matrix(mbuffer, in0)
            pivot = np.empty(m, nt.int32)

            # swapped steps to get matrix in FORTRAN order
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            if cblas_typ == 'c' or cblas_typ == 'z':
                raise NotImplementedError('cannot calc det(complex) yet')
                tmp = complex(np.exp(logdet), 0)
                retval = complex(sign.real * tmp.real - sign.imag * tmp.imag,
                                 sign.real * tmp.imag + sign.imag * tmp.real)
            else:
                retval = sign * np.exp(logdet)
            return retval
        return slogdet, det

    FLOAT_slogdet,   FLOAT_det   = wrap_slogdet(nt.float32,    nt.float32, 's')
    DOUBLE_slogdet,  DOUBLE_det  = wrap_slogdet(nt.float64,    nt.float64, 'd')
    CFLOAT_slogdet,  CFLOAT_det  = wrap_slogdet(nt.complex64,  nt.float32, 'c')
    CDOUBLE_slogdet, CDOUBLE_det = wrap_slogdet(nt.complex128, nt.float64, 'z')

    slogdet = frompyfunc([FLOAT_slogdet, DOUBLE_slogdet, CFLOAT_slogdet, CDOUBLE_slogdet],
                         1, 2, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.float32, nt.complex64,
                                       nt.complex128, nt.float64, nt.complex128],
                          signature='(m,m)->(),()', name='slogdet', stack_inputs=False,
                  doc="slogdet on the last two dimensions and broadcast on the rest. \n"\
                      "Results in two arrays, one with sign and the other with log of the"\
                      " determinants. \n"\
                      "    \"(m,m)->(),()\" \n",
              )

    det = frompyfunc([FLOAT_det, DOUBLE_det, CFLOAT_det, CDOUBLE_det],
                         1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.float32,
                                       nt.complex128, nt.float64],
                          doc="det on the last two dimensions and broadcast"\
                              " on the rest. \n    \"(m,m)->()\" \n",
                          signature='(m,m)->()', name='det', stack_inputs=False,
                     )

    # --------------------------------------------------------------------------
    # Eigh family

    class eigh_params(Params):
        params = ('A', 'W', 'WORK', 'RWORK', 'IWORK', 'N', 'LWORK',
                   'LRWORK', 'LIWORK', 'JOBZ', 'UPLO')

    def wrap_eigh_real(typ, cblas_typ):
        lapack_func = cblas_typ + 'syevd'
        def init_func(JOBZ, UPLO, N):
            query_work_size = ffi.new(toCtypeA[typ])
            query_iwork_size = ffi.new('int[1]')
            info = ffi.new('int[1]')
            lwork = ffi.new('int[1]', [-1])
            liwork = ffi.new('int[1]', [-1])
            a = np.empty([N, N], typ)
            w = np.empty([N], typ)
            pN = ffi.new('int[1]', [N])
            pJOBZ = ffi.new('char[1]', [JOBZ])
            pUPLO = ffi.new('char[1]', [UPLO])
            get_c_func('', lapack_func)(pJOBZ, pUPLO, pN, toCptr(a), pN, toCptr(w),
                     query_work_size, lwork,
                     query_iwork_size, liwork, info)
            if info[0] != 0:
                return None
            lwork[0] = int(query_work_size[0])
            liwork[0] = query_iwork_size[0]
            work = np.empty([lwork[0]], dtype=typ)
            iwork = np.empty([liwork[0]], dtype='int32')

            return eigh_params(a, w, work, None, iwork, pN, lwork, 0, liwork,
                               pJOBZ, pUPLO)

        def call_func(params):
            rv = ffi.new('int[1]')
            get_c_func('', lapack_func)(params.JOBZ, params.UPLO, params.N,
                        toCptr(params.A), params.N, toCptr(params.W),
                        toCptr(params.WORK), params.LWORK, toCptr(params.IWORK),
                        params.LIWORK, rv)
            return rv[0]
        return init_func, call_func

    def wrap_eigh_complex(typ, basetyp, cblas_typ):
        lapack_func = cblas_typ + 'heevd'
        def init_func(JOBZ, UPLO, N):
            query_work_size = ffi.new(toCtypeA[typ])
            query_rwork_size = ffi.new(toCtypeA[basetyp])
            query_iwork_size = ffi.new('int[1]')
            info = ffi.new('int[1]')
            lwork = ffi.new('int[1]', [-1])
            liwork = ffi.new('int[1]', [-1])
            lrwork = ffi.new('int[1]', [-1])
            a = np.empty([N, N], typ)
            w = np.empty([N], basetyp)
            pN = ffi.new('int[1]', [N])
            pJOBZ = ffi.new('char[1]', [JOBZ])
            pUPLO = ffi.new('char[1]', [UPLO])
            get_c_func('', lapack_func)(pJOBZ, pUPLO, pN, toCptr(a), pN, toCptr(w),
                     query_work_size, lwork, query_rwork_size, lrwork,
                     query_iwork_size, liwork, info)
            if info[0] != 0:
                return None
            lwork[0] = int(query_work_size[0].r)
            lrwork[0] = int(query_rwork_size[0])
            liwork[0] = query_iwork_size[0]
            work = np.empty([lwork[0]], dtype=typ)
            rwork = np.empty([lrwork[0]], dtype=basetyp)
            iwork = np.empty([liwork[0]], dtype='int32')

            return eigh_params(a, w, work, rwork, iwork, pN, lwork, lrwork,
                               liwork, pJOBZ, pUPLO)

        def call_func(params):
            rv = ffi.new('int[1]')
            get_c_func('', lapack_func)(params.JOBZ, params.UPLO, params.N,
                        toCptr(params.A), params.N, toCptr(params.W),
                        toCptr(params.WORK), params.LWORK, toCptr(params.RWORK),
                        params.LRWORK, toCptr(params.IWORK), params.LIWORK, rv)
            return rv[0]
        return init_func, call_func

    eigh_funcs = {}
    eigh_funcs['s'] = wrap_eigh_real(nt.float32, 's')
    eigh_funcs['d'] = wrap_eigh_real(nt.float64, 'd')
    eigh_funcs['c'] = wrap_eigh_complex(nt.complex64,  nt.float32, 'c')
    eigh_funcs['z'] = wrap_eigh_complex(nt.complex128, nt.float64, 'z')
    init_func = 0
    call_func = 1

    def wrap_eigh(typ, cblas_typ):
        def eigh_caller(JOBZ, UPLO, in0, *out):
            error_occurred = get_fp_invalid_and_clear()
            N = in0.shape[0]
            params = eigh_funcs[cblas_typ][init_func](JOBZ, UPLO, N)
            if params is None:
                return
            linearize_matrix(params.A, in0)
            not_ok = eigh_funcs[cblas_typ][call_func](params)
            if not_ok == 0:
                delinearize_matrix(out[0], params.W)
                if 'V' == JOBZ:
                    delinearize_matrix(out[1], params.A)
            else:
                out[0].fill(base_vals[cblas_typ]['nan'])
                if 'V' == JOBZ:
                    out[1].fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred)

        def eigh_lo(*args):
            return eigh_caller('V', 'L', *args)

        def eigh_up(*args):
            return eigh_caller('V', 'U', *args)

        def eigvalshlo(*args):
            return eigh_caller('N', 'L', *args)

        def eigvalshup(*args):
            return eigh_caller('N', 'U', *args)

        return eigh_lo, eigh_up, eigvalshlo, eigvalshup

    FLOAT_eigh_lo, FLOAT_eigh_up, FLOAT_eigvalshlo, FLOAT_eigvalshup = \
                                                    wrap_eigh(nt.float32, 's')
    DOUBLE_eigh_lo, DOUBLE_eigh_up, DOUBLE_eigvalshlo, DOUBLE_eigvalshup = \
                                                    wrap_eigh(nt.float64, 'd')
    CFLOAT_eigh_lo, CFLOAT_eigh_up, CFLOAT_eigvalshlo, CFLOAT_eigvalshup = \
                                                    wrap_eigh(nt.complex64, 'c')
    CDOUBLE_eigh_lo, CDOUBLE_eigh_up, CDOUBLE_eigvalshlo, CDOUBLE_eigvalshup = \
                                                    wrap_eigh(nt.complex128, 'z')

    eigh_lo = frompyfunc([FLOAT_eigh_lo, DOUBLE_eigh_lo, CFLOAT_eigh_lo, CDOUBLE_eigh_lo],
                        1, 2, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.float32, nt.complex64,
                                       nt.complex128, nt.float64, nt.complex128],
                        signature='(m,m)->(m),(m,m)', name='eigh_lo', stack_inputs=True,
                        doc = "eigh on the last two dimension and broadcast to the rest, using"\
                        " lower triangle \n"\
                        "Results in a vector of eigenvalues and a matrix with the"\
                        "eigenvectors. \n"\
                        "    \"(m,m)->(m),(m,m)\" \n",
                        )

    eigh_up = frompyfunc([FLOAT_eigh_up, DOUBLE_eigh_up, CFLOAT_eigh_up, CDOUBLE_eigh_up],
                        1, 2, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.float32, nt.complex64,
                                       nt.complex128, nt.float64, nt.complex128],
                        signature='(m,m)->(m),(m,m)', name='eigh_up', stack_inputs=True,
                        doc = "eigh on the last two dimension and broadcast to the rest, using"\
                        " upper triangle \n"\
                        "Results in a vector of eigenvalues and a matrix with the"\
                        "eigenvectors. \n"\
                        "    \"(m,m)->(m),(m,m)\" \n",
                        )

    eigvalsh_lo = frompyfunc([FLOAT_eigvalshlo, DOUBLE_eigvalshlo, CFLOAT_eigvalshlo, CDOUBLE_eigvalshlo],
                        1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.float32,
                                       nt.complex128, nt.float64],
                        signature='(m,m)->(m)', name='eigvaslh_lo', stack_inputs=True,
                        doc = "eigh on the last two dimension and broadcast to the rest, using"\
                        " lower triangle \n"\
                        "Results in a vector of eigenvalues. \n"\
                        "    \"(m,m)->(m)\" \n",
                        )

    eigvalsh_up = frompyfunc([FLOAT_eigvalshup, DOUBLE_eigvalshup, CFLOAT_eigvalshup, CDOUBLE_eigvalshup],
                        1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.float32,
                                       nt.complex128, nt.float64],
                        signature='(m,m)->(m)', name='eigvaslh_up', stack_inputs=True,
                        doc = "eigh on the last two dimension and broadcast to the rest, using"\
                        " upper triangle \n"\
                        "Results in a vector of eigenvalues. \n"\
                        "    \"(m,m)->(m)\" \n",
                        )

    # --------------------------------------------------------------------------
    # Solve family (includes inv)

    class gesv_params(Params):
        params = ('A', 'B', 'IPIV', 'N', 'NRHS', 'LDA', 'LDB')

    def wrap_solvers(typ, cblas_typ):
        def init_func(N, NRHS):
            A = np.empty([N, N], dtype=typ)
            if NRHS == 1:
                B = np.empty([N], dtype = typ)
            else:
                B = np.empty([NRHS, N], dtype = typ)
            ipiv = np.empty([N], dtype = nt.int32)
            pN = ffi.new('int[1]', [N])
            pNRHS = ffi.new('int[1]', [NRHS])
            return gesv_params(A, B, ipiv, pN, pNRHS, pN, pN)

        def call_func(params):
            rv = ffi.new('int[1]')
            get_c_func(cblas_typ, 'gesv')(params.N, params.NRHS, toCptr(params.A),
                                             params.LDA, toCptr(params.IPIV),
                                             toCptr(params.B), params.LDB, rv)
            return rv[0]

        def solve(in0, in1, out0):
            error_occurred = get_fp_invalid_and_clear()
            n = in0.shape[0]
            nrhs = in1.shape[1]
            params = init_func(n, nrhs)
            linearize_matrix(params.A, in0)
            linearize_matrix(params.B, in1)
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out0, params.B)
            else:
                error_occurred = 1
                out1.fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred);

        def solve1(in0, in1, out0):
            error_occurred = get_fp_invalid_and_clear()
            n = in0.shape[0]
            nrhs = 1
            params = init_func(n, nrhs)
            linearize_matrix(params.A, in0)
            linearize_matrix(params.B, in1)
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out0, params.B)
            else:
                error_occurred = 1
                out1.fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred)

        def identity_matrix(a):
            a[:] = base_vals[cblas_typ]['zero']
            for i in range(a.shape[0]):
                a[i,i] = base_vals[cblas_typ]['one']

        def inv(inarg, outarg):
            error_occurred = get_fp_invalid_and_clear()
            n = inarg.shape[0]
            params = init_func(n, n)
            linearize_matrix(params.A, inarg)
            if params.B.size < 2:
                params.B[:] = 1
            else:
                identity_matrix(params.B)
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(outarg, params.B)
            else:
                error_occurred = 1
                outarg.fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred)

        return solve, solve1, inv

    FLOAT_solve,   FLOAT_solve1, FLOAT_inv  = wrap_solvers(nt.float32, 's')
    DOUBLE_solve,  DOUBLE_solve1, DOUBLE_inv  = wrap_solvers(nt.float64, 'd')
    CFLOAT_solve,  CFLOAT_solve1, CFLOAT_inv  = wrap_solvers(nt.complex64, 'c')
    CDOUBLE_solve, CDOUBLE_solve1, CDOUBLE_inv = wrap_solvers(nt.complex128, 'z')

    solve = frompyfunc([FLOAT_solve, DOUBLE_solve, CFLOAT_solve, CDOUBLE_solve],
                         2, 1, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.complex64, nt.complex64,
                                       nt.complex128, nt.complex128, nt.complex128],
                          signature='(m,m),(m,n)->(m,n)', name='solve', stack_inputs=True,
                          doc = "solve the system a x = b, on the last two dimensions, broadcast"\
                                " to the rest. \n"\
                                "Results in a matrices with the solutions. \n"\
                                "    \"(m,m),(m,n)->(m,n)\" \n",
                        )

    solve1 = frompyfunc([FLOAT_solve1, DOUBLE_solve1, CFLOAT_solve1, CDOUBLE_solve1],
                         2, 1, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.complex64, nt.complex64,
                                       nt.complex128, nt.complex128, nt.complex128],
                          signature='(m,m),(m)->(m)', name='solve1', stack_inputs=True,
                          doc = "solve the system a x = b, for b being a vector, broadcast in"\
                                " the outer dimensions. \n"\
                                "Results in the vectors with the solutions. \n"\
                                "    \"(m,m),(m)->(m)\" \n",
                         )
    inv = frompyfunc([FLOAT_inv, DOUBLE_inv, CFLOAT_inv, CDOUBLE_inv],
                         1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.complex64,
                                       nt.complex128, nt.complex128],
                          signature='(m,m)->(m,m)', name='inv', stack_inputs=True,
                  doc="compute the inverse of the last two dimensions and broadcast "\
                      " to the rest. \n"\
                      "Results in the inverse matrices. \n"\
                      "    \"(m,m)->(m,m)\" \n",
              )

    # --------------------------------------------------------------------------
    # Cholesky decomposition
    class potr_params(Params):
        params = ('A', 'N', 'LDA', 'UPLO')

    def wrap_cholesky(typ, cblas_typ):
        def init_func(UPLO, N):
            a = np.empty([N, N], dtype=typ)
            pN = ffi.new('int[1]', [N])
            pUPLO = ffi.new('char[1]', [UPLO])
            return potr_params(a, pN, pN, pUPLO)

        def call_func(params, p_t):
            rv = ffi.new('int[1]')
            get_c_func(cblas_typ, 'potrf')(params.UPLO, params.N, toCptr(params.A),
                                              params.LDA, rv)
            return rv[0]

        def cholesky(uplo, in0, out0):
            error_occurred = get_fp_invalid_and_clear()
            n = inarg.shape[0]
            assert uplo == 'L'
            params = init_func(uplo, n)
            linearize_matrix(params.A, in0)
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out0, params.A)
            else:
                error_occurred = 1
                out0.fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred);

        def cholesky_lo(in0, out0):
            cholesky('L', in0, out0)

        return cholesky_lo


    FLOAT_cholesky_lo  = wrap_cholesky(nt.float32, 's')
    DOUBLE_cholesky_lo  = wrap_cholesky(nt.float64, 'd')
    CFLOAT_cholesky_lo  = wrap_cholesky(nt.complex64, 'c')
    CDOUBLE_cholesky_lo  = wrap_cholesky(nt.complex128, 'z')

    cholesky_lo = frompyfunc([FLOAT_cholesky_lo, DOUBLE_cholesky_lo, CFLOAT_cholesky_lo, CDOUBLE_cholesky_lo],
                         1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.float32,
                                       nt.complex128, nt.float64],
                         signature='(m,m)->(m,m)', name='cholesky_lo', stack_inputs=True,
                         doc = "cholesky decomposition of hermitian positive-definite matrices. \n"\
                               "Broadcast to all outer dimensions. \n"\
                               "    \"(m,m)->(m,m)\" \n",
                            )

    # --------------------------------------------------------------------------
    # eig family
    class geev_params_struct(Params):
        params = ('A', 'WR', 'WI', 'VLR', 'VRR', 'WORK', 'W', 'VL', 'VR', 'N',
                   'LDA', 'LDVL', 'LDVR', 'LWORK', 'JOBVL', 'JOBVR')

    def mk_complex_eigenvectors(c, r, i, n):
        ''' make the complex eigenvectors from the real array produced by sgeev/zgeev.
          * c is the array where the results will be left.
          * r is the source array of reals produced by sgeev/zgeev
          * i is the eigenvalue imaginary part produced by sgeev/zgeev
          * n is so that the order of the matrix is n by n
        '''
        # TODO: can this be optimized?
        iter = 0
        while iter < n:
            if i[n] == 0:
                # eigenvalue was real, eigenvectors as well
                c[iter, :].real = r[iter, :]
                c[iter, :].imag = 0
                iter += 1
            else:
                # eigenvalue was complex, generate a pair of eigenvectors
                c[iter, :].real = r[iter, :]
                c[iter, :].imag = r[iter + 1, :]
                c[iter + 1, :].real = r[iter, :]
                c[iter + 1, :].imag = -r[iter+1, :]
                iter += 2

    def wrap_geev_real(typ, complextyp, cblas_typ):
        def init_func(jobvl, jobvr, n):
            a = np.empty([n, n], typ)
            wr = np.empty([n], typ)
            wi = np.empty([n], typ)
            w  = np.empty([n], complextyp)
            if jobvl == 'V':
                vlr = np.empty([n, n], typ)
                vl = np.empty([n, n], complextyp)
            else:
                vlr = None
                vl = None
            if jobvr == 'V':
                vrr = np.empty([n, n], typ)
                vr = np.empty([n, n], complextyp)
            else:
                vrr = None
                vr = None
            work_size_query = ffi.new(toCtypeA[typ], [0])
            do_size_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]', [0])
            pN = ffi.new('int[1]', [n])
            get_c_func(cblas_typ, 'geev')(jobvl, jobvr, pN, toCptr(a), pN,
                                         toCptr(wr), toCptr(wi), toCptr(vlr),
                                         pN, toCptr(vrr), pN, work_size_query,
                                         do_size_query, rv)
            if rv[0] !=0:
                return None
            work_count = ffi.new('int[1]', [int(work_size_query[0])])
            work = np.empty([work_count[0] / 2], typ)
            return geev_params_struct(a, wr, wi, vlr, vrr, work, w, vl, vr,
                                      pN, pN, pN, pN, work_count, jobvl, jobvr)

        def call_func(params):
            rv = ffi.new('int[1]', [0])
            get_c_func(cblas_typ, 'geev')(params.JOBVL, params.JOBVR, params.N,
                    toCptr(params.A), params.LDA, toCptr(params.WR), toCptr(params.WI),
                    toCptr(params.VLR), params.LDVL, toCptr(params.VRR),
                    params.LDVR, toCptr(params.WORK), params.LWORK, rv)
            return rv[0]

        def process_results(params):
            ''' REAL versions of geev need the results to be translated
              * into complex versions. This is the way to deal with imaginary
              * results. In our gufuncs we will always return complex arrays!
            '''
            assert params.W.size == params.WR.size
            assert params.W.size == params.WI.size
            assert params.W.size == params.N[0]
            params.W.real = params.WR
            params.W.imag = params.WI
            if 'V' == params.JOBVL:
                mk_complex_eigenvectors(params.VL, params.VLR, params.WI, params.N)
            if 'V' == params.JOBVR:
                mk_complex_eigenvectors(params.VR, params.VRR, params.WI, params.N)
        return init_func, call_func, process_results

    def wrap_geev_complex(typ, realtyp, cblas_typ):
        def init_func(jobvl, jobvr, n):
            a = np.empty([n, n], typ)
            w = np.empty([n], typ)
            rwork = np.empty([2 * n], realtyp)
            if jobvl == 'V':
                vl = np.empty([n, n], typ)
            else:
                vl = None
            if jobvr == 'V':
                vr = np.empty([n, n], typ)
            else:
                vr = None
            work_size_query = ffi.new(toCtypeA[typ])
            work_size_query[0].r = 0
            do_size_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]', [0])
            pN = ffi.new('int[1]', [n])
            get_c_func(cblas_typ, 'geev')(jobvl, jobvr, pN, toCptr(a),
                                         pN, toCptr(w), toCptr(vl), pN, toCptr(vr),
                                         pN, work_size_query, do_size_query, toCptr(rwork), rv)
            if rv[0] !=0:
                return None
            work_count = ffi.new('int[1]', [int(work_size_query[0].r)])
            work = np.empty([work_count[0]], typ)
            return geev_params_struct(a, rwork, None, None, None, work, w, vl, vr,
                                      pN, pN, pN, pN, work_count, jobvl, jobvr)

        def call_func(params):
            rv = ffi.new('int[1]', [0])
            get_c_func(cblas_typ, 'geev')(params.JOBVL, params.JOBVR, params.N,
                    toCptr(params.A), params.LDA, toCptr(params.W),
                    toCptr(params.VL), params.LDVL, toCptr(params.VR), params.LDVR,
                    toCptr(params.WORK), params.LWORK,
                    toCptr(params.WR), # actually RWORK
                    rv)
            return rv[0]

        def process_results(params):
            ''' Nothing to do here, complex versions are ready to copy out
            '''
            pass

        return init_func, call_func, process_results

    geev_funcs={}
    geev_funcs['s'] = wrap_geev_real(nt.float32, nt.complex64, 's')
    geev_funcs['d'] = wrap_geev_real(nt.float64, nt.complex128, 'd')
    geev_funcs['c'] = wrap_geev_complex(nt.complex64, nt.float32, 'c')
    geev_funcs['z'] = wrap_geev_complex(nt.complex128, nt.float64, 'z')
    init_func = 0
    call_func = 1
    process_results = 2

    def wrap_eig(typ, cblas_typ):
        def eig_wrapper(JOBVL, JOBVR, in0, *out):
            op_count = 2
            error_occurred = get_fp_invalid_and_clear();
            assert(JOBVL == 'N')
            if 'V' == JOBVR:
                op_count += 1
            params = geev_funcs[cblas_typ][init_func](JOBVL, JOBVR, in0.shape[0])
            if params is None:
                return
            n = in0.shape[0]
            outcnt = 1
            linearize_matrix(params.A, in0)
            not_ok = geev_funcs[cblas_typ][call_func](params)
            if not_ok == 0:
                geev_funcs[cblas_typ][process_results](params)
                delinearize_matrix(out[0], params.W)
                outcnt = 1
                if 'V' == JOBVL:
                    delinearize_matrix(out[outcnt], params.VL)
                    outcnt += 1
                if 'V' == JOBVR:
                    delinearize_matrix(out[outcnt], params.VR)
            else:
                error_occurred = 1;
                for o in out:
                    o.fill(float('nan'))
            set_fp_invalid_or_clear(error_occurred)

        def eig(*args):
            return eig_wrapper('N', 'V', *args)

        def eigvals(*args):
            return eig_wrapper('N', 'N', *args)

        return eig, eigvals


    #  There are problems with eig in complex single precision.
    #  That kernel is disabled
    FLOAT_eig,     FLOAT_eigvals = wrap_eig(nt.float32,    's')
    DOUBLE_eig,   DOUBLE_eigvals = wrap_eig(nt.float64,    'd')
    CDOUBLE_eig, CDOUBLE_eigvals = wrap_eig(nt.complex128, 'z')

    eig = frompyfunc([FLOAT_eig, DOUBLE_eig, CDOUBLE_eig],
                      1, 2, dtypes=[nt.float32, nt.complex64, nt.complex64,
                                    nt.float64, nt.complex128, nt.complex128,
                                    nt.complex128, nt.complex128, nt.complex128],
                      signature='(m,m)->(m),(m,m)', name='eig', stack_inputs=True,
                      doc = "eig on the last two dimension and broadcast to the rest. \n"\
                            "Results in a vector with the  eigenvalues and a matrix with the"\
                            " eigenvectors. \n"\
                            "    \"(m,m)->(m),(m,m)\" \n",
                     )

    eigvals = frompyfunc([FLOAT_eigvals, DOUBLE_eigvals, CDOUBLE_eigvals],
                      1, 1, dtypes=[nt.float32, nt.complex64,
                                    nt.float64, nt.complex128,
                                    nt.complex128, nt.complex128],
                      signature='(m,m)->(m)', name='eig', stack_inputs=True,
                      doc = "eig on the last two dimension and broadcast to the rest. \n"\
                            "Results in a vector of eigenvalues. \n"\
                            "    \"(m,m)->(m)\" \n",
                     )

    # --------------------------------------------------------------------------
    # SVD family of singular value decomposition
    class gesdd_params(Params):
        params = ('A', 'S', 'U', 'VT', 'WORK', 'RWORK', 'IWORK',
                  'M', 'N', 'LDA', 'LDU', 'LDVT', 'LWORK', 'JOBZ')

    def compute_urows_vtcolumns(jobz, m, n):
        min_m_n = min(m, n)
        if jobz == 'N':
            return 0, 0
        elif jobz == 'A':
            return m, n
        elif jobz == 'S':
            return min_m_n, min_m_n
        else:
            return -1, -1

    def wrap_gesdd(typ, realtyp, cblas_typ):
        lapack_func = cblas_typ + 'gesdd'
        def init_func(jobz, m, n):
            min_m_n = min(m, n)
            u_size, vt_size = compute_urows_vtcolumns(jobz, m, n)
            if u_size < 0:
                return None
            a = np.empty([n, m], typ)
            s = np.empty([min_m_n], typ)
            if u_size == 0:
                u = None
            else:
                u = np.empty([u_size, m], typ)
            if vt_size == 0:
                vt = None
                pVt_column_count = ffi.new('int[1]', [1])
            else:
                vt = np.empty([n, vt_size], typ)
                pVt_column_count = ffi.new('int[1]', [vt_size])
            iwork = np.empty([16, min_m_n], 'int32')
            pN = ffi.new('int[1]', [n])
            pM = ffi.new('int[1]', [m])
            pjobz = ffi.new('char[1]', [jobz])
            pDo_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]')
            if realtyp is not None:
                pWork_size_query = ffi.new(toCtypeA[typ])
                pWork_size_query[0].r = -1
                pWork_size_query[0].i =  0
                s = np.empty([min_m_n], realtyp)
                if 'N'==jobz:
                    rwork_size = 7 * min_m_n
                else:
                    rwork_size = 5 * min_m_n * min_m_n + 5*min_m_n
                rwork = np.empty([rwork_size*8],'int8')
                get_c_func('', lapack_func)(pjobz, pM, pN, toCptr(a), pM, toCptr(s),
                    toCptr(u), pM, toCptr(vt), pVt_column_count,
                    pWork_size_query, pDo_query,
                    toCptr(rwork),
                    toCptr(iwork), rv)
                if rv[0] != 0:
                    return None
                work_count = ffi.new('int[1]', [int(pWork_size_query[0].r)])
            else:
                rwork = None
                s = np.empty([min_m_n], typ)
                pWork_size_query = ffi.new(toCtypeA[typ])
                get_c_func('', lapack_func)(pjobz, pM, pN, toCptr(a), pM, toCptr(s),
                    toCptr(u), pM, toCptr(vt), pVt_column_count,
                    pWork_size_query, pDo_query,
                    toCptr(iwork), rv)
                if rv[0] != 0:
                    return None
                work_count = ffi.new('int[1]', [int(pWork_size_query[0])])
            work = ffi.new('char[%d]' % (work_count[0] * typ.itemsize,))
            return gesdd_params(a, s, u, vt, work, rwork, iwork, pM, pN, pM, pM,
                        pVt_column_count, work_count, pjobz)

        if cblas_typ == 's' or cblas_typ == 'd':
            def call_func(params):
                rv = ffi.new('int[1]')
                get_c_func('', lapack_func)(params.JOBZ, params.M, params.N, toCptr(params.A),
                                    params.LDA, toCptr(params.S),
                                    toCptr(params.U), params.LDU, toCptr(params.VT),
                                    params.LDVT, ffi.cast('void*', params.WORK), params.LWORK,
                                    toCptr(params.IWORK), rv)
                return rv[0]
        else:
            def call_func(params):
                rv = ffi.new('int[1]')
                get_c_func('', lapack_func)(params.JOBZ, params.M, params.N, toCptr(params.A),
                                    params.LDA, toCptr(params.S),
                                    toCptr(params.U), params.LDU, toCptr(params.VT),
                                    params.LDVT, params.WORK, params.LWORK,
                                    toCptr(params.RWORK),
                                    toCptr(params.IWORK), rv)
                return rv[0]

        def svd(jobz, in0, *out):
            error_occurred = get_fp_invalid_and_clear()
            m, n = in0.shape[:2]
            params = init_func(jobz, m, n)
            if params is None:
                return None
            min_m_n = min(m, n)
            linearize_matrix(params.A, in0)
            not_ok = call_func(params)
            if not_ok == 0:
                if jobz == 'N':
                    delinearize_matrix(out[0], params.S)
                else:
                    delinearize_matrix(out[0], params.U)
                    delinearize_matrix(out[1], params.S)
                    delinearize_matrix(out[2], params.VT)
            else:
                error_occurred = 1;
                for o in out:
                    o.fill(float('nan'))
            set_fp_invalid_or_clear(error_occurred)

        def svd_N(*args):
            return svd('N', *args)

        def svd_S(*args):
            return svd('S', *args)

        def svd_A(*args):
            return svd('A', *args)

        return svd_N, svd_S, svd_A

    F_svd_N,   F_svd_S,  F_svd_A = wrap_gesdd(nt.float32,    None,       's')
    D_svd_N,   D_svd_S,  D_svd_A = wrap_gesdd(nt.float64,    None,       'd')
    CF_svd_N, CF_svd_S, CF_svd_A = wrap_gesdd(nt.complex64,  nt.float32, 'c')
    CD_svd_N, CD_svd_S, CD_svd_A = wrap_gesdd(nt.complex128, nt.float64, 'z')

    svd_m = frompyfunc([F_svd_N, D_svd_N, CF_svd_N, CD_svd_N],
                      1, 1, dtypes=[nt.float32, nt.float32,
                                    nt.float64, nt.float64,
                                    nt.complex64, nt.float32,
                                    nt.complex128, nt.float64],
                      signature='(m,n)->(m)', name='svd_m', stack_inputs=True,
                      doc = "svd when n>=m. ",
                     )

    svd_n = frompyfunc([F_svd_N, D_svd_N, CF_svd_N, CD_svd_N],
                      1, 1, dtypes=[nt.float32, nt.float32,
                                    nt.float64, nt.float64,
                                    nt.complex64, nt.float32,
                                    nt.complex128, nt.float64],
                      signature='(m,n)->(n)', name='svd_n', stack_inputs=True,
                      doc = "svd when n<=m. ",
                     )

    svd_m_s = frompyfunc([F_svd_S, D_svd_S, CF_svd_S, CD_svd_S],
                      1, 3, dtypes=[nt.float32, nt.float32, nt.float32, nt.float32,
                                    nt.float64, nt.float64, nt.float64, nt.float64,
                                    nt.complex64, nt.complex64, nt.float32, nt.complex64,
                                    nt.complex128, nt.complex128, nt.float64, nt.complex128],
                      signature='(m,n)->(m,n),(m),(m,n)', name='svd_m_s', stack_inputs=True,
                      doc = "svd when m>=n. ",
                     )

    svd_n_s = frompyfunc([F_svd_S, D_svd_S, CF_svd_S, CD_svd_S],
                      1, 3, dtypes=[nt.float32, nt.float32, nt.float32, nt.float32,
                                    nt.float64, nt.float64, nt.float64, nt.float64,
                                    nt.complex64, nt.complex64, nt.float32, nt.complex64,
                                    nt.complex128, nt.complex128, nt.float64, nt.complex128],
                      signature='(m,n)->(m,n),(n),(n,n)', name='svd_n_s', stack_inputs=True,
                      doc = "svd when m>=n. ",
                     )

    svd_m_f = frompyfunc([F_svd_A, D_svd_A, CF_svd_A, CD_svd_A],
                      1, 3, dtypes=[nt.float32, nt.float32, nt.float32, nt.float32,
                                    nt.float64, nt.float64, nt.float64, nt.float64,
                                    nt.complex64, nt.complex64, nt.float32, nt.complex64,
                                    nt.complex128, nt.complex128, nt.float64, nt.complex128],
                      signature='(m,n)->(m,m),(m),(n,n)', name='svd_m_f', stack_inputs=True,
                      doc = "svd when m>=n. ",
                     )

    svd_n_f = frompyfunc([F_svd_A, D_svd_A, CF_svd_A, CD_svd_A],
                      1, 3, dtypes=[nt.float32, nt.float32, nt.float32, nt.float32,
                                    nt.float64, nt.float64, nt.float64, nt.float64,
                                    nt.complex64, nt.complex64, nt.float32, nt.complex64,
                                    nt.complex128, nt.complex128, nt.float64, nt.complex128],
                      signature='(m,n)->(m,m),(n),(n,n)', name='svd_n_f', stack_inputs=True,
                      doc = "svd when m>=n. ",
                     )

    # --------------------------------------------------------------------------

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

