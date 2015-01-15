# A CFFI version of numpy/linalg/umath_linalg.c.src
# As opposed to the numpy version, the cffi version leaves broadcasting to the responsibility
# of the pypy extended frompyfunc, which removes the need for INIT_OUTER_LOOP*
from warnings import warn
import sys, os

try:
    import cffi
    use_cffi = True
except ImportError:
    use_cffi = False


if '__pypy__' in sys.builtin_module_names:
    import _numpypy.umath
    if 'frompyfunc' not in dir(_numpypy.umath):
        use_cffi = False
else:
    # Default on cpython is not to use cffi
    use_cffi = False 

if use_cffi:
    import numpy as np
    # dtype has not been imported yet
    from numpy.core.multiarray import dtype
    class Dummy(object):
        pass
    nt = Dummy()
    nt.int32 = dtype('int32')
    nt.float32 = dtype('float32')
    nt.float64 = dtype('float64')
    nt.complex64 = dtype('complex64')
    nt.complex128 = dtype('complex128')
    from numpy.core.umath import frompyfunc
    __version__ = '0.1.4'

    macros = {'sfx': '', 'pfx': ''}
    ffi = cffi.FFI()


    # The next section is a hack to find the lapack implementation, loosly based on
    # numpy.distutils.system_info.get_info. The idea is to use a site.cfg file to specify
    # where the shared object is located. Note that we need the lapack (high-level) interface,
    # they in turn call a low-level implementation maybe using blas or atlas.
    # This has not been tested on OSX
    __C = None
    from numpy.distutils import system_info
    # temporarily mess with environ
    env = os.environ.copy()
    if sys.platform == 'win32':
        ld_library_path = 'PATH'
    else:
        ld_library_path = 'LD_LIBRARY_PATH'
    for lapack, prefix, suffix in [ \
                    ['openblas_lapack', '', '_'], 
                    ['lapack_mkl', '', '_' ],
                    ['lapack', '', '_'],
                  ]:
        si = getattr(system_info, lapack + '_info')()
        libs = si.get_lib_dirs()
        if len(libs) > 0:
            os.environ[ld_library_path] = os.pathsep.join(libs + [os.environ.get(ld_library_path, '')])
        try:
            _C = ffi.dlopen(lapack)
            macros['sfx'] = suffix
            macros['pfx'] = prefix
            break
        except Exception as e:
            pass
    os.environ = env
    if _C is None:
        shared_name = os.path.abspath(os.path.dirname(__file__)) + '/' + prefix + 'lapack_lite' + suffix
        if not os.path.exists(shared_name):
            # cffi should support some canonical name formatting like 
            # distutils.ccompiler.library_filename()
            raise ValueError('could not find "%s", perhaps the name is slightly off' % shared_name)
        _C = ffi.dlopen(shared_name)
        warn('tuned lapack (openblas, atlas ...) not found, using lapack_lite')

    ffi.cdef('''
/*
 *                    LAPACK functions
 */

typedef struct {{ float r, i; }} f2c_complex;
typedef struct {{ double r, i; }} f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

extern int
{pfx}sgeev{sfx}(char *jobvl, char *jobvr, int *n,
             float a[], int *lda, float wr[], float wi[],
             float vl[], int *ldvl, float vr[], int *ldvr,
             float work[], int lwork[],
             int *info);
extern int
{pfx}dgeev{sfx}(char *jobvl, char *jobvr, int *n,
             double a[], int *lda, double wr[], double wi[],
             double vl[], int *ldvl, double vr[], int *ldvr,
             double work[], int lwork[],
             int *info);
extern int
{pfx}cgeev{sfx}(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);
extern int
{pfx}zgeev{sfx}(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);

extern int
{pfx}ssyevd{sfx}(char *jobz, char *uplo, int *n,
              float a[], int *lda, float w[], float work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
{pfx}dsyevd{sfx}(char *jobz, char *uplo, int *n,
              double a[], int *lda, double w[], double work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
{pfx}cheevd{sfx}(char *jobz, char *uplo, int *n,
              f2c_complex a[], int *lda,
              float w[], f2c_complex work[],
              int *lwork, float rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);
extern int
{pfx}zheevd{sfx}(char *jobz, char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              double w[], f2c_doublecomplex work[],
              int *lwork, double rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);

extern int
{pfx}dgelsd{sfx}(int *m, int *n, int *nrhs,
              double a[], int *lda, double b[], int *ldb,
              double s[], double *rcond, int *rank,
              double work[], int *lwork, int iwork[],
              int *info);
extern int
{pfx}zgelsd{sfx}(int *m, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              double s[], double *rcond, int *rank,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[],
              int *info);

extern int
{pfx}sgesv{sfx}(int *n, int *nrhs,
             float a[], int *lda,
             int ipiv[],
             float b[], int *ldb,
             int *info);
extern int
{pfx}dgesv{sfx}(int *n, int *nrhs,
             double a[], int *lda,
             int ipiv[],
             double b[], int *ldb,
             int *info);
extern int
{pfx}cgesv{sfx}(int *n, int *nrhs,
             f2c_complex a[], int *lda,
             int ipiv[],
             f2c_complex b[], int *ldb,
             int *info);
extern int
{pfx}zgesv{sfx}(int *n, int *nrhs,
             f2c_doublecomplex a[], int *lda,
             int ipiv[],
             f2c_doublecomplex b[], int *ldb,
             int *info);

extern int
{pfx}sgetrf{sfx}(int *m, int *n,
              float a[], int *lda,
              int ipiv[],
              int *info);
extern int
{pfx}dgetrf{sfx}(int *m, int *n,
              double a[], int *lda,
              int ipiv[],
              int *info);
extern int
{pfx}cgetrf{sfx}(int *m, int *n,
              f2c_complex a[], int *lda,
              int ipiv[],
              int *info);
extern int
{pfx}zgetrf{sfx}(int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              int ipiv[],
              int *info);

extern int
{pfx}spotrf{sfx}(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
{pfx}dpotrf{sfx}(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
{pfx}cpotrf{sfx}(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
{pfx}zpotrf{sfx}(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
{pfx}sgesdd{sfx}(char *jobz, int *m, int *n,
              float a[], int *lda, float s[], float u[],
              int *ldu, float vt[], int *ldvt, float work[],
              int *lwork, int iwork[], int *info);
extern int
{pfx}dgesdd{sfx}(char *jobz, int *m, int *n,
              double a[], int *lda, double s[], double u[],
              int *ldu, double vt[], int *ldvt, double work[],
              int *lwork, int iwork[], int *info);
extern int
{pfx}cgesdd{sfx}(char *jobz, int *m, int *n,
              f2c_complex a[], int *lda,
              float s[], f2c_complex u[], int *ldu,
              f2c_complex vt[], int *ldvt,
              f2c_complex work[], int *lwork,
              float rwork[], int iwork[], int *info);
extern int
{pfx}zgesdd{sfx}(char *jobz, int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              double s[], f2c_doublecomplex u[], int *ldu,
              f2c_doublecomplex vt[], int *ldvt,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[], int *info);

extern int
{pfx}spotrs{sfx}(char *uplo, int *n, int *nrhs,
              float a[], int *lda,
              float b[], int *ldb,
              int *info);
extern int
{pfx}dpotrs{sfx}(char *uplo, int *n, int *nrhs,
              double a[], int *lda,
              double b[], int *ldb,
              int *info);
extern int
{pfx}cpotrs{sfx}(char *uplo, int *n, int *nrhs,
              f2c_complex a[], int *lda,
              f2c_complex b[], int *ldb,
              int *info);
extern int
{pfx}zpotrs{sfx}(char *uplo, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              int *info);

extern int
{pfx}spotri{sfx}(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
{pfx}dpotri{sfx}(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
{pfx}cpotri{sfx}(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
{pfx}zpotri{sfx}(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
{pfx}scopy{sfx}(int *n,
             float *sx, int *incx,
             float *sy, int *incy);
extern int
{pfx}dcopy{sfx}(int *n,
             double *sx, int *incx,
             double *sy, int *incy);
extern int
{pfx}ccopy{sfx}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern int
{pfx}zcopy{sfx}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

extern float
{pfx}sdot{sfx}(int *n,
            float *sx, int *incx,
            float *sy, int *incy);
extern double
{pfx}ddot{sfx}(int *n,
            double *sx, int *incx,
            double *sy, int *incy);
extern f2c_complex
{pfx}cdotu{sfx}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern f2c_doublecomplex
{pfx}zdotu{sfx}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);
extern f2c_complex
{pfx}cdotc{sfx}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern f2c_doublecomplex
{pfx}zdotc{sfx}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

extern int
{pfx}sgemm{sfx}(char *transa, char *transb,
             int *m, int *n, int *k,
             float *alpha,
             float *a, int *lda,
             float *b, int *ldb,
             float *beta,
             float *c, int *ldc);
extern int
{pfx}dgemm{sfx}(char *transa, char *transb,
             int *m, int *n, int *k,
             double *alpha,
             double *a, int *lda,
             double *b, int *ldb,
             double *beta,
             double *c, int *ldc);
extern int
{pfx}cgemm{sfx}(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_complex *alpha,
             f2c_complex *a, int *lda,
             f2c_complex *b, int *ldb,
             f2c_complex *beta,
             f2c_complex *c, int *ldc);
extern int
{pfx}zgemm{sfx}(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, int *lda,
             f2c_doublecomplex *b, int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, int *ldc);

    '''.format(**macros))
    
    lapack_lite = Dummy()
    for name in ['sgeev', 'dgeev', 'cgeev', 'zgeev', 'ssyevd', 'dsyevd',
                 'cheevd', 'zheevd', 'dgelsd', 'zgelsd', 'sgesv', 'dgesv', 'cgesv', 'zgesv',
                 'sgetrf', 'dgetrf', 'cgetrf', 'zgetrf', 'spotrf', 'dpotrf', 'cpotrf', 'zpotrf',
                 'sgesdd', 'dgesdd', 'cgesdd', 'zgesdd', 'spotrs', 'dpotrs', 'cpotrs', 'zpotrs',
                 'spotri', 'dpotri', 'cpotri', 'zpotri', 'scopy', 'dcopy', 'ccopy', 'zcopy',
                 'sdot', 'ddot', 'cdotu', 'zdotu', 'cdotc', 'zdotc',
                 'sgemm', 'dgemm', 'cgemm', 'zgemm']:
        setattr(lapack_lite, name, getattr(_C, macros['pfx'] + name + macros['sfx']))
    lapack_lite.shared_object = _C

    toCtypeP = {nt.int32: 'int*', nt.float32: 'float*', nt.float64: 'double*',
               nt.complex64: 'f2c_complex*', nt.complex128: 'f2c_doublecomplex*'}
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
        try:
            umath_so = ffi.dlopen(sh_obj)
            if npy_clear_floatstatus is None:
                npy_clear_floatstatus = getattr(umath_so, 'npy_clear_floatstatus')
            if npy_set_floatstatus_invalid is None:
                npy_set_floatstatus_invalid = getattr(umath_so, 'npy_set_floatstatus_invalid')
        except OSError, AttributeError:
            def return0(*args):
                return 0
            warn('npy_clear_floatstatus, npy_set_floatstatus_invalid not found')
            if npy_clear_floatstatus is None:
                npy_clear_floatstatus = return0
            if npy_set_floatstatus_invalid is None:
                npy_clear_floatstatus = return0
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

    def lazy_init(*arg_names):
        ''' Decorate a class __init__ by assigning all the args to self.arg_names
        '''
        def ret(init):
            def __init__(self, *args):
                for a, v in zip(arg_names, args):
                    setattr(self, a, v)
            return __init__
        return ret

    class linearize_data(object):
        ''' contains information about how to linearize in a local buffer
           a matrix so that it can be used by blas functions.
           All strides are specified in number of elements (similar to what blas
           expects) rather than nelem*sizeof(elem) like in numpy
        '''
        @lazy_init('rows', 'columns', 'row_strides', 'column_strides')
        def __init__(*args):
            pass

    def linearize_matrix(dst, src, data, copy_func):
        ''' in cpython numpy, dst, src are c-level pointers
            we (ab)use ndarrays instead
        '''
        if dst is None:
            raise ValueError('called with NULL input, should not happen')
        if src.dtype is not dst.dtype:
            raise ValueError('called with differing dtypes, should not happen')
        psrc = toCptr(src)
        pdst = toCptr(dst)
        pcolumns = ffi.new('int[1]', [data.columns])
        pcolumn_strides = ffi.new('int[1]', [data.column_strides / src.dtype.itemsize])
        pone = ffi.new('int[1]', [1])
        for i in range(data.rows):
            psrc_void = ffi.cast("void*", psrc)
            pdst_void = ffi.cast("void*", pdst)
            if data.column_strides > 0:
                copy_func(pcolumns, psrc_void, pcolumn_strides, pdst_void, pone)
            elif data.column_strides < 0:
                copy_func(pcolumns, ff.cast("void*", psrc + (columns-1)*column_strides), 
                     pcolumn_strides, pdst_void, pone)
            else:
                # Zero strides has undefined behavior in some BLAS
                # implementations (e.g. OSX Accelerate), so do it
                # manually
                for j in range(columns):
                    ffi.memcpy(pdst+j, psrc, src.dtype.itemsize)
            psrc += data.row_strides / src.dtype.itemsize
            pdst += data.columns

    def delinearize_matrix(dst, src, data, copy_func):
        ''' in cpython numpy, dst, src are c-level pointers
            we (ab)use ndarrays instead
        '''
        if src is None:
            raise ValueError('called with NULL input, should not happen')
        if src.dtype is not dst.dtype:
            raise ValueError('called with differing dtypes, should not happen')
        psrc = toCptr(src)
        pdst = toCptr(dst)
        pcolumns = ffi.new('int [1]', [data.columns])
        pcolumn_strides = ffi.new('int[1]', [data.column_strides / src.dtype.itemsize])
        pone = ffi.new('int[1]', [1])
        for i in range(data.rows):
            psrc_void = ffi.cast("void*", psrc)
            pdst_void = ffi.cast("void*", pdst)
            if data.column_strides > 0:
                copy_func(pcolumns, psrc_void, pcolumn_strides, pdst_void, pone)
            elif data.column_strides < 0:
                copy_func(pcolumns, psrc_void, pcolumn_strides, 
                     ffi.cast("void*", pdst+(columns-1)*column_strides) , pone)
            else:
                # Zero strides has undefined behavior in some BLAS
                # implementations (e.g. OSX Accelerate), so do it
                # manually
                for j in range(columns):
                    ffi.memcpy(pdst, psrc+columns-1, src.dtype.itemsize)
            psrc += data.columns
            pdst += data.row_strides / src.dtype.itemsize

    # --------------------------------------------------------------------------
    # Determinants

    def wrap_slogdet(typ, basetyp, cblas_typ):
        def slogdet_from_factored_diagonal(src, m, sign):
            sign_acc = sign[0]
            logdet_acc = base_vals[cblas_typ]['zero']
            for i in range(m[0]):
                abs_element = np.abs(src[i,i])
                sign_element = src[i, i] / abs_element
                sign_acc = sign_acc * sign_element
                logdet_acc += np.log(abs_element)
            return sign_acc, logdet_acc    

        def slogdet_single_element(m, src, pivots):
            info = ffi.new('int[1]', [0])
            getattr(lapack_lite, cblas_typ + 'getrf')(m, m, src, m, pivots, info)
            if info[0] == 0:
                for i in range(m[0]):
                    # fortran uses 1 based indexing
                    change_sign += (pivots[i] != (i + 1))
                if change_sign % 2:
                    sign = base_vals[cblas_typ]['minus_one']
                else:
                    sign = base_vals[cblas_typ]['one']
                sign, logdet = slogdet_from_factored_diagonal(src, m, sign)
            else: 
                # if getrf fails, use - as sign and -inf as logdet
                sign = base_vals[cblas_typ]['zero']
                logdet = base_type(-float('inf'))
            return sign, logdet

        def slogdet(in0):
            ''' notes:
             *   in must have shape [m, m], out[0] and out[1] are [m]
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = in0.shape[0]
            mbuffer = np.empty((m, m), typ)
            pivot = np.empty(m, typ)
            
            # swapped steps to get matrix in FORTRAN order
            data = linearize_data(m, m, in0.strides[1], in0.strides[0]) 
            linearize_matrix(mbuffer, in0, data, getattr(lapack_lite, cblas_typ + 'copy'))
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            return sign, logdet

        def det(in0):
            ''' notes:
             *   in must have shape [m, m], out is scalar
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = in0.shape[0]
            mbuffer = np.empty((m, m), typ)
            pivot = np.empty(m, typ)
        
            # swapped steps to get matrix in FORTRAN order
            data = linearize_data(m, m, in0.strides[1], in0.strides[0]) 
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            return logdet
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

    class eigh_params(object):
        @lazy_init('A', 'W', 'WORK', 'RWORK', 'IWORK', 'N', 'LWORK',
                   'LRWORK', 'LIWORK', 'JOBZ', 'UPLO')
        def __init__(*args):
            pass

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
            getattr(lapack_lite, lapack_func)(pJOBZ, pUPLO, pN, toCptr(a), pN, toCptr(w),
                     query_work_size, lwork, 
                     query_iwork_size, liwork, info)
            if info[0] != 0:
                return None
            lwork[0] = int(query_work_size[0])
            liwork[0] = query_iwork_size[0]
            work = toCptr(np.empty([lwork[0]], dtype=typ))
            iwork = toCptr(np.empty([liwork[0]], dtype='int32'))
            
            return eigh_params(a, w, work, None, iwork, pN, lwork, 0, liwork,
                               pJOBZ, pUPLO)

        def call_func(params):
            rv = ffi.new('int[1]')
            getattr(lapack_lite, lapack_func)(params.JOBZ, params.UPLO, params.N,
                        toCptr(params.A), params.N, toCptr(params.W),
                        params.WORK, params.LWORK, params.IWORK, params.LIWORK, rv)
            return rv[0]
        return init_func, call_func

    def wrap_eigh_complex(typ, basetyp, cblas_typ):
        lapack_func = cblas_typ + 'heevd'
        def init_func(JOBZ, UPLO, N):
            query_work_size = ffi.new(toCtypeA[typ])
            query_rwork_size = ffi.new(toCtypeA[typ])
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
            getattr(lapack_lite, lapack_func)(pJOBZ, pUPLO, pN, toCptr(a), pN, toCptr(w),
                     query_work_size, lwork, query_rwork_size, lrwork,
                     query_iwork_size, liwork, info)
            if info[0] != 0:
                return None
            lwork[0] = query_work_size[0]
            lrwork[0] = query_rwork_size[0]
            liwork[0] = query_iwork_size[0]
            work = toCptr(np.empty([lwork[0]], dtype=typ))
            rwork = toCptr(np.empty([lrwork[0]], dtype=basetyp))
            iwork = toCptr(np.empty([liwork[0]], dtype='int32'))
            
            return eigh_params(a, w, work, rwork, iwork, pN, lwork, lrwork,
                               liwork, pJOBZ, pUPLO)

        def call_func(params):
            rv = ffi.new('int[1]')
            getattr(lapack_lite, lapack_func)(params.JOBZ, params.UPLO, params.N,
                        toCptr(params.A), params.N, toCptr(params.W),
                        params.WORK, params.LWORK, params.RWORK, params.LRWORK,
                        params.IWORK, paramw.LIWORK, rv)
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
            matrix_in_ld = linearize_data(N, N, in0.strides[1], in0.strides[0])
            eigval_out_ld = linearize_data(1, N, 0, out[0].strides[0])
            if 'V' == JOBZ:
                eigvec_out_ld = linearize_data(N, N, out[1].strides[1], out[1].strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, in0, matrix_in_ld, copy_func)
            not_ok = eigh_funcs[cblas_typ][call_func](params)
            if not_ok == 0:
                delinearize_matrix(out[0], params.W, eigval_out_ld, copy_func)
                if 'V' == JOBZ:
                    delinearize_matrix(out[1], params.A, eigvec_out_ld, copy_func)
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

    eighvalsh_lo = frompyfunc([FLOAT_eigvalshlo, DOUBLE_eigvalshlo, CFLOAT_eigvalshlo, CDOUBLE_eigvalshlo],
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

    eighvalsh_up = frompyfunc([FLOAT_eigvalshup, DOUBLE_eigvalshup, CFLOAT_eigvalshup, CDOUBLE_eigvalshup],
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

    class gesv_params(object):
        @lazy_init('A', 'B', 'IPIV', 'N', 'NRHS', 'LDA', 'LDB')
        def __init__(*args):
            pass

    def wrap_solvers(typ, cblas_typ):
        def init_func(N, NRHS):
            A = np.empty([N, N], dtype=typ)
            B = np.empty([N, NRHS], dtype = typ)
            ipiv = np.empty([N], dtype = nt.int32)
            pN = ffi.new('int[1]', [N])
            pNRHS = ffi.new('int[1]', [N])
            return gesv_params(A, B, ipiv, pN, pNRHS, pN, pN)

        def call_func(params):
            rv = ffi.new('int[1]')
            getattr(lapack_lite, cblas_typ + 'gesv')(params.N, params.NRHS, toCptr(params.A),
                                             params.LDA, toCptr(params.IPIV),
                                             toCptr(params.B), params.LDB, rv)
            return rv[0]
             
        def solve(inarg, out0, out1):
            error_occurred = get_fp_invalid_and_clear()
            n = inarg.shape[0]
            nrhs = inarg.shape[1]
            params = init_func(n, nrhs)
            a_in = linearize_data(n, n, inarg.strides[1], inarg.strides[0])
            b_in = linearize_data(nrhs, n, out0.strides[1], out0.strides[0])
            r_out = linearize_data(nrhs, n, out1.strides[1], out1.strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, inarg, a_in, copy_func) 
            linearize_matrix(params.B, out0, b_in, copy_func) 
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out1, params.B, r_out, copy_func)
            else:
                error_occurred = 1
                out1.fill(base_vals[cblas_typ]['nan'])
            set_fp_invalid_or_clear(error_occurred);
              
        def solve1(inarg, out0):
            error_occurred = get_fp_invalid_and_clear()
            n = inarg.shape[0]
            nrhs = 1
            params = init_func(n, nrhs)
            a_in = linearize_data(n, n, inarg.strides[1], inarg.strides[0])
            b_in = linearize_data(nrhs, n, 1, out0.strides[0])
            r_out = linearize_data(nrhs, n, 1, out1.strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, inarg, a_in, copy_func) 
            linearize_matrix(params.B, out0, b_in, copy_func) 
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out1, params.B, r_out, copy_func)
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
            a_in = linearize_data(n, n, inarg.strides[1], inarg.strides[0])
            r_out = linearize_data(n, n, outarg.strides[1], outarg.strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, inarg, a_in, copy_func) 
            identity_matrix(params.B)
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(outarg, params.B, r_out, copy_func)
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
                                       nt.complex64, nt.float32, nt.float32,
                                       nt.complex128, nt.float64, nt.float64],
                          signature='(m,m),(m,n)->(m,n)', name='solve', stack_inputs=True,
                          doc = "solve the system a x = b, on the last two dimensions, broadcast"\
                                " to the rest. \n"\
                                "Results in a matrices with the solutions. \n"\
                                "    \"(m,m),(m,n)->(m,n)\" \n",
                        )

    solve1 = frompyfunc([FLOAT_solve1, DOUBLE_solve1, CFLOAT_solve1, CDOUBLE_solve1],
                         2, 1, dtypes=[nt.float32, nt.float32, nt.float32,
                                       nt.float64, nt.float64, nt.float64,
                                       nt.complex64, nt.float32, nt.float32,
                                       nt.complex128, nt.float64, nt.float64],
                          signature='(m,m),(m)->(m)', name='solve1', stack_inputs=True,
                          doc = "solve the system a x = b, for b being a vector, broadcast in"\
                                " the outer dimensions. \n"\
                                "Results in the vectors with the solutions. \n"\
                                "    \"(m,m),(m)->(m)\" \n",
                         )
    inv = frompyfunc([FLOAT_inv, DOUBLE_inv, CFLOAT_inv, CDOUBLE_inv],
                         1, 1, dtypes=[nt.float32, nt.float32,
                                       nt.float64, nt.float64,
                                       nt.complex64, nt.float32,
                                       nt.complex128, nt.float64],
                          signature='(m,m)->(m,m)', name='inv', stack_inputs=True,
                  doc="compute the inverse of the last two dimensions and broadcast "\
                      " to the rest. \n"\
                      "Results in the inverse matrices. \n"\
                      "    \"(m,m)->(m,m)\" \n",
              )

    # --------------------------------------------------------------------------
    # Cholesky decomposition
    class potr_params(object):
        @lazy_init('A', 'N', 'LDA', 'UPLO')
        def __init__(self, *args):
            pass

    def wrap_cholesky(typ, cblas_typ):
        def init_func(UPLO, N):
            a = np.empty([N, N], dtype=typ)
            pN = ffi.new('int[1]', [N])
            pUPLO = ffi.new('char[1]', [UPLO])
            return potr_params(a, pN, pN, pUPLO)

        def call_func(params, p_t):
            rv = ffi.new('int[1]')
            getattr(lapack_lite, cblas_typ + 'potrf')(params.UPLO, params.N, toCptr(params.A),
                                              params.LDA, rv)
            return rv[0]

        def cholesky(uplo, in0, out0):
            error_occurred = get_fp_invalid_and_clear()
            n = inarg.shape[0]
            assert uplo == 'L'
            params = init_func(uplo, n)
            a_in = linearize_data(n, n, in0.strides[1], in0.strides[0])
            r_out = linearize_data(n, n, out0.strides[1], out0.strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, in0, a_in, copy_func) 
            not_ok = call_func(params)
            if not_ok == 0:
                delinearize_matrix(out0, params.A, r_out, copy_func) 
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
    class geev_params_struct(object):
        params = ('A', 'WR', 'WI', 'VLR', 'VRR', 'WORK', 'W', 'VL', 'VR', 'N',
                   'LDA', 'LDVL', 'LDVR', 'LWORK', 'JOBVL', 'JOBVR')
        @lazy_init(*params)
        def __init__(self, *args):
            pass

        def dump(self, name):
            print >> sys.stderr, name
            for p in self.params:
                v = getattr(self, p)
                if isinstance(v, np.ndarray):
                    print >> sys.stderr, '\t%10s: %r, %r' %(p, v.dtype, v.shape)
                elif isinstance(v, int):
                    print >> sys.stderr, '\t%10s: %r' %(p, v)
                else:
                    print >> sys.stderr, '\t%10s: %r %r' %(p, v, v[0])

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
                vl = np.empty([n, n*2], typ)
            else:
                vlr = None
                vl = None
            if jobvr == 'V':
                vrr = np.empty([n, n], typ)
                vr = np.empty([n, 2*n], typ)
            else:
                vrr = None
                vr = None
            work_size_query = ffi.new(toCtypeA[typ], [0])
            do_size_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]', [0])
            pN = ffi.new('int[1]', [n])
            getattr(lapack_lite, cblas_typ + 'geev')(jobvl, jobvr, pN, toCptr(a), pN,
                                         toCptr(wr), toCptr(wi), toCptr(vl),
                                         pN, toCptr(vr), pN, work_size_query,
                                         do_size_query, rv)
            if rv[0] !=0:
                return None
            work_count = ffi.new('int[1]', [int(work_size_query[0])])
            work = np.empty([work_count[0] / 2], typ)
            return geev_params_struct(a, wr, wi, vlr, vrr, work, w, vl, vr, 
                                      pN, pN, pN, pN, work_count, jobvl, jobvr)

        def call_func(params):
            rv = ffi.new('int[1]', [0])
            getattr(lapack_lite, cblas_typ + 'geev')(params.JOBVL, params.JOBVR, params.N,
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
            work_size_query = ffi.new(toCtypeA[typ], [0])
            do_size_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]', [0])
            pN = ffi.new('int[1]', [n])
            getattr(lapack_lite, cblas_typ + 'geev')(jobvl, jobvr, pN, toCptr(a),
                                         pN, toCptr(w), toCptr(vl), pN, toCptr(vr),
                                         pN, work_size_query, do_size_query, toCptr(rwork), rv)
            if rv[0] !=0:
                return None
            work_count = work_size_query[0][0]
            work = np.empty([work_count ], typ)
            return geev_params_struct(a, rwork, None, None, None, work, w, vl, vr, 
                                      pN, pN, pN, pN, work_count[0], jobvl, jobvr)

        def call_func(params):
            rv = ffi.new('int[1]', [0])
            getattr(lapack_lite, cblas_typ + 'geev')(params.JOBVL, params.JOBVR, params.N,
                    toCptr(params.A), params.LDA, toCptr(params.W), 
                    toCptr(params.VL), params.LDVL, toCptr(params.VR), params.LVDR,
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
            a_in = linearize_data(n, n, in0.strides[1], in0.strides[0])
            w_out = linearize_data(1, n, 0, out[0].strides[0])
            outcnt = 1
            if 'V' == JOBVL:
                vl_out = linearize_data(n, n, 
                                        out[outcnt].strides[1],
                                        out[outcnt].strides[0])
                outcnt += 1
            if 'V' == JOBVR:
                vr_out = linearize_data(n, n, 
                                        out[outcnt].strides[1],
                                        out[outcnt].strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, in0, a_in, copy_func)
            not_ok = geev_funcs[cblas_typ][call_func](params)
            if not_ok == 0:
                geev_funcs[cblas_typ][process_results](params)
                delinearize_matrix(out[0], params.W, w_out, copy_func) 
                outcnt = 1
                if 'V' == JOBVL:
                    delinearize_matrix(out[outcnt], params.VL, vl_out, copy_func)
                    outcnt += 1
                if 'V' == JOBVR:
                    delinearize_matrix(out[outcnt], params.VR, vr_out, copy_func)
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

    eigvals = frompyfunc([FLOAT_eig, DOUBLE_eig, CDOUBLE_eig],
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
    class gesdd_params(object):
        params = ('A', 'S', 'U', 'VT', 'WORK', 'RWORK', 'IWORK',
                  'M', 'N', 'LDA', 'LDU', 'LDVT', 'LWORK', 'JOBZ')
        @lazy_init(*params)
        def __init__(self, *args):
            pass

        def dump(self, name):
            print >> sys.stderr, name
            for p in self.params:
                v = getattr(self, p)
                if isinstance(v, np.ndarray):
                    print >> sys.stderr, '\t%10s: %r, %r' %(p, v.dtype, v.shape)
                elif isinstance(v, int):
                    print >> sys.stderr, '\t%10s: %r' %(p, v)
                else:
                    print >> sys.stderr, '\t%10s: %r %r' %(p, v, v[0])
    
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
            a = np.empty([m, n], typ)
            s = np.empty([min_m_n], typ)
            if u_size == 0:
                u = None
            else:
                u = np.empty([u_size], typ)
            if vt_size == 0:
                vt = None
                pVt_column_count = ffi.new('int[1]', [1])
            else:
                vt = np.empty([n, vt_size], typ) 
                pVt_column_count = ffi.new('int[1]', [vt_size])
            iwork = np.empty([8, min_m_n], 'int32')
            pN = ffi.new('int[1]', [n])
            pM = ffi.new('int[1]', [m])
            pjobz = ffi.new('char[1]', [jobz])
            pDo_query = ffi.new('int[1]', [-1])
            rv = ffi.new('int[1]')
            if realtyp is not None:
                pWork_size_query = ffi.new(toCtypeA[realtyp])
                s = np.empty([min_m_n], realtyp)
                if 'N'==jobz:
                    rwork_size = 7 * min_m_n
                else:
                    rwork_size = 5 * min_m_n * min_m_n + 5*min_m_n
                rwork = ffi.cast('void*', ffi.ffi.new('char[%d]' % rwork_size))
                getattr(lapack_lite, lapack_func)(pjobz, pM, pN, toCptr(a), pM, toCptr(s),
                    toCptr(u), pM, toCptr(vt), pVt_column_count,
                    pWork_size_query, pDo_query, 
                    toCPtr(rwork),
                    toCptr(iwork), rv)
            else:
                rwork = None
                s = np.empty([min_m_n], typ)
                pWork_size_query = ffi.new(toCtypeA[typ])
                getattr(lapack_lite, lapack_func)(pjobz, pM, pN, toCptr(a), pM, toCptr(s),
                    toCptr(u), pM, toCptr(vt), pVt_column_count,
                    pWork_size_query, pDo_query, 
                    toCptr(iwork), rv)

            if rv[0] != 0:
                return None
            work_count = ffi.new('int[1]', [int(pWork_size_query[0])])
            work = ffi.cast('void*', ffi.new('char[%d]' % work_count[0]))
            return gesdd_params(a, s, u, vt, work, rwork, iwork, pM, pN, pM, pM,
                        pVt_column_count, work_count, pjobz)  

        if cblas_typ == 's' or cblas_typ == 'd':
            def call_func(params):
                rv = ffi.new('int[1]')
                getattr(lapack_lite, lapack_func)(params.JOBZ, params.M, params.N, toCptr(params.A),
                                    params.LDA, toCptr(params.S), 
                                    toCptr(params.U), params.LDU, toCptr(params.VT),
                                    params.LDVT, params.WORK, params.LWORK,
                                    toCptr(params.IWORK), rv)
                return rv[0]
        else:
            def call_func(params):
                rv = ffi.new('int[1]')
                getattr(lapack_lite, lapack_func)(params.JOBZ, params.M, params.N, toCptr(params.A),
                                    params.LDA, toCptr(params.S), toCptr(params.S),
                                    toCptr(params.U), params.LDU, toCptr(params.VT),
                                    toCptr(params.LDVT), params.WORK, params.LWORK,
                                    params.RWORK,
                                    toCptr(params.IWORK), rv)
                return rv[0]

        def svd(jobz, in0, *out):
            error_occurred = get_fp_invalid_and_clear()
            m, n = in0.shape[:2]
            params = init_func(jobz, m, n)
            if params is None:
                return None
            min_m_n = min(m, n)
            a_in = linearize_data(n, m, in0.strides[1], in0.strides[0])
            if jobz == 'N':
                s_out = linearize_data(1, min_m_n, 0, out[0].strides[0])
            else:
                if 'S' == jobz:
                    ucols = min_m_n
                    vrows = min_m_n
                else:
                    ucols = m
                    vrows = n
                u_out = linearize_data(ucols, m, out[0].strides[1], out[0].strides[0])
                s_out = linearize_data(1, min_m_n, 0, out[1].strides[0])
                v_out = linearize_data(n, vrows, out[2].strides[1], out[2].strides[0])
            copy_func = getattr(lapack_lite, cblas_typ + 'copy')
            linearize_matrix(params.A, in0, a_in, copy_func)
            not_ok = call_func(params)
            if not_ok == 0:
                if jobz == 'N':
                    delinearize_matrix(out[0], params.S, s_out,copy_func)
                else:
                    delinearize_matrix(out[0], params.U, u_out,copy_func)
                    delinearize_matrix(out[1], params.S, s_out,copy_func)
                    delinearize_matrix(out[2], params.VT, v_out,copy_func)
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

