# A CFFI version of numpy/linalg/umath_linalg.c.src
# As opposed to the numpy version, the cffi version leaves broadcasting to the responsibility
# of the pypy extended frompyfunc, which removes the need for INIT_OUTER_LOOP*
from warnings import warn
import sys

try:
    import cffi
    have_cffi = True
except ImportError:
    have_cffi = False


if and have_cffi and '__pypy__' in sys.builtin_module_names:
    import numpy as np
    # numeric types have not been imported yet
    import numpy.core.numerictypes as nt
    __version__ = '0.1.4'

    NO_APPEND_FORTRAN=False
    macros = {'ul':'_'}
    if NO_APPEND_FORTRAN:
        macros['ul'] = ''

    ffi = cffi.FFI()
    ffi.cdef('''
/*
 *****************************************************************************
 *                    BLAS/LAPACK calling macros                             *
 *****************************************************************************
 */

typedef struct {{ float r, i; }} f2c_complex;
typedef struct {{ double r, i; }} f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

extern int
sgeev{ul}(char *jobvl, char *jobvr, int *n,
             float a[], int *lda, float wr[], float wi[],
             float vl[], int *ldvl, float vr[], int *ldvr,
             float work[], int lwork[],
             int *info);
extern int
dgeev{ul}(char *jobvl, char *jobvr, int *n,
             double a[], int *lda, double wr[], double wi[],
             double vl[], int *ldvl, double vr[], int *ldvr,
             double work[], int lwork[],
             int *info);
extern int
cgeev{ul}(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);
extern int
zgeev{ul}(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);

extern int
ssyevd{ul}(char *jobz, char *uplo, int *n,
              float a[], int *lda, float w[], float work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
dsyevd{ul}(char *jobz, char *uplo, int *n,
              double a[], int *lda, double w[], double work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
cheevd{ul}(char *jobz, char *uplo, int *n,
              f2c_complex a[], int *lda,
              float w[], f2c_complex work[],
              int *lwork, float rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);
extern int
zheevd{ul}(char *jobz, char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              double w[], f2c_doublecomplex work[],
              int *lwork, double rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);

extern int
dgelsd{ul}(int *m, int *n, int *nrhs,
              double a[], int *lda, double b[], int *ldb,
              double s[], double *rcond, int *rank,
              double work[], int *lwork, int iwork[],
              int *info);
extern int
zgelsd{ul}(int *m, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              double s[], double *rcond, int *rank,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[],
              int *info);

extern int
sgesv{ul}(int *n, int *nrhs,
             float a[], int *lda,
             int ipiv[],
             float b[], int *ldb,
             int *info);
extern int
dgesv{ul}(int *n, int *nrhs,
             double a[], int *lda,
             int ipiv[],
             double b[], int *ldb,
             int *info);
extern int
cgesv{ul}(int *n, int *nrhs,
             f2c_complex a[], int *lda,
             int ipiv[],
             f2c_complex b[], int *ldb,
             int *info);
extern int
zgesv{ul}(int *n, int *nrhs,
             f2c_doublecomplex a[], int *lda,
             int ipiv[],
             f2c_doublecomplex b[], int *ldb,
             int *info);

extern int
sgetrf{ul}(int *m, int *n,
              float a[], int *lda,
              int ipiv[],
              int *info);
extern int
dgetrf{ul}(int *m, int *n,
              double a[], int *lda,
              int ipiv[],
              int *info);
extern int
cgetrf{ul}(int *m, int *n,
              f2c_complex a[], int *lda,
              int ipiv[],
              int *info);
extern int
zgetrf{ul}(int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              int ipiv[],
              int *info);

extern int
spotrf{ul}(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
dpotrf{ul}(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
cpotrf{ul}(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
zpotrf{ul}(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
sgesdd{ul}(char *jobz, int *m, int *n,
              float a[], int *lda, float s[], float u[],
              int *ldu, float vt[], int *ldvt, float work[],
              int *lwork, int iwork[], int *info);
extern int
dgesdd{ul}(char *jobz, int *m, int *n,
              double a[], int *lda, double s[], double u[],
              int *ldu, double vt[], int *ldvt, double work[],
              int *lwork, int iwork[], int *info);
extern int
cgesdd{ul}(char *jobz, int *m, int *n,
              f2c_complex a[], int *lda,
              float s[], f2c_complex u[], int *ldu,
              f2c_complex vt[], int *ldvt,
              f2c_complex work[], int *lwork,
              float rwork[], int iwork[], int *info);
extern int
zgesdd{ul}(char *jobz, int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              double s[], f2c_doublecomplex u[], int *ldu,
              f2c_doublecomplex vt[], int *ldvt,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[], int *info);

extern int
spotrs{ul}(char *uplo, int *n, int *nrhs,
              float a[], int *lda,
              float b[], int *ldb,
              int *info);
extern int
dpotrs{ul}(char *uplo, int *n, int *nrhs,
              double a[], int *lda,
              double b[], int *ldb,
              int *info);
extern int
cpotrs{ul}(char *uplo, int *n, int *nrhs,
              f2c_complex a[], int *lda,
              f2c_complex b[], int *ldb,
              int *info);
extern int
zpotrs{ul}(char *uplo, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              int *info);

extern int
spotri{ul}(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
dpotri{ul}(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
cpotri{ul}(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
zpotri{ul}(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
scopy{ul}(int *n,
             float *sx, int *incx,
             float *sy, int *incy);
extern int
dcopy{ul}(int *n,
             double *sx, int *incx,
             double *sy, int *incy);
extern int
ccopy{ul}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern int
zcopy{ul}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

extern float
sdot{ul}(int *n,
            float *sx, int *incx,
            float *sy, int *incy);
extern double
ddot{ul}(int *n,
            double *sx, int *incx,
            double *sy, int *incy);
extern f2c_complex
cdotu{ul}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern f2c_doublecomplex
zdotu{ul}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);
extern f2c_complex
cdotc{ul}(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern f2c_doublecomplex
zdotc{ul}(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

extern int
sgemm{ul}(char *transa, char *transb,
             int *m, int *n, int *k,
             float *alpha,
             float *a, int *lda,
             float *b, int *ldb,
             float *beta,
             float *c, int *ldc);
extern int
dgemm{ul}(char *transa, char *transb,
             int *m, int *n, int *k,
             double *alpha,
             double *a, int *lda,
             double *b, int *ldb,
             double *beta,
             double *c, int *ldc);
extern int
cgemm{ul}(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_complex *alpha,
             f2c_complex *a, int *lda,
             f2c_complex *b, int *ldb,
             f2c_complex *beta,
             f2c_complex *c, int *ldc);
extern int
zgemm{ul}(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, int *lda,
             f2c_doublecomplex *b, int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, int *ldc);

    '''.format(**macros))

    def offset_ptr(ptr, offset):
        return ptr + offset

    def get_fp_invalid_and_clear():
        return bool(np.np_clear_floatstatus() & np.NPY_FPE_INVALID)

    def set_fp_invalid_and_clear(error_occured):
        if error_occured:
            np.npy_set_floatstatus_invalid()
        else:
            np.npy_clear_floatstatus()
    base_vals = {'s':{}, 'd':{}, 'c':{}, 'z':{}}
    base_vals['s']['one'] = nt.float32(1)
    base_vals['s']['zero'] = nt.float32(0)
    base_vals['s']['minus_one'] = nt.float32(1)
    base_vals['s']['ninf'] = nt.float32(-float('inf'))
    base_vals['s']['nan'] = nt.float32(float('nan'))
    base_vals['d']['one'] = nt.float64(1)
    base_vals['d']['zero'] = nt.float64(0)
    base_vals['d']['minus_one'] = nt.float64(1)
    base_vals['d']['ninf'] = nt.float64(-float('inf'))
    base_vals['d']['nan'] = nt.float64(float('nan'))
    base_vals['c']['one'] = nt.complex64(complex(1, 0))
    base_vals['c']['zero'] = nt.complex64(complex(0, 0))
    base_vals['c']['minus_one'] = nt.complex64(complex(-1, 0))
    base_vals['c']['ninf'] = nt.complex64(complex(-float('inf'), 0))
    base_vals['c']['nan'] = nt.complex64(complex(float('nan'), float('nan')))
    base_vals['z']['one'] = nt.complex128(complex(1, 0))
    base_vals['z']['zero'] = nt.complex128(complex(0, 0))
    base_vals['z']['minus_one'] = nt.complex128(complex(-1, 0))
    base_vals['z']['ninf'] = nt.complex128(complex(-float('inf'), 0))
    base_vals['z']['nan'] = nt.complex128(complex(float('nan'), float('nan')))

    class linearize_data(object):
        ''' contains information about how to linearize in a local buffer
           a matrix so that it can be used by blas functions.
           All strides are specified in number of elements (similar to what blas
           expects) rather than nelem*sizeof(elem) like in numpy
        '''
        def __init__(rows, columns, row_strides, column_strides):
            self.rows = rows
            self.columns = columns
            self.rows_strides = rows_strides
            self.columns_strides = columns_strides

    def wrapper_lm(func, p_t):
        def linearize_matrix(dst, src, data):
            ''' in cpython numpy, dst, src are c-level pointers
                we (ab)use ndarrays instead
            '''
            if dst is None:
                raise ValueError('called with NULL input, should not happen')
            if src.dtype is not dst.dtype:
                raise ValueError('called with differing dtypes, should not happen')
            psrc = ffi.cast(p_t, src.__array_interface__['data'])
            pdst = ffi.cast(p_t, dst.__array_interface__['data'])
            pcolumns = ffi.new('int [1]', [data.columns])
            pcolumn_strides = ffi.new('int[1]', [data.column_strides / src.dtypes.itemsize])
            pone = np.int32(ffi.new('int[1]', [1]))
            for i in range(data.rows):
                psrc_void = ffi.cast("void*", psrc)
                pdst_void = ffi.cast("void*", pdst)
                if column_strides > 0:
                    func(pcolumns, psrc_void, pcolumn_strides, pdst_void, pone)
                elif column_strides < 0:
                    func(pcolumns, ff.cast("void*", psrc + (columns-1)*column_strides), 
                         pcolumn_strides, pdst_void, pone)
                else:
                    # Zero stride has undefined behavior in some BLAS
                    # implementations (e.g. OSX Accelerate), so do it
                    # manually
                    for j in range(columns):
                        ffi.memcpy(pdst+j, psrc, src.dtype.itemsize)
                psrc += data.row_strides / src.dtype.itemsize
                pdst += data.columns
        return linearize_matrix
    linearize_float_matrix = wrapper_lm(C.scopy, "float*")
    linearize_double_matrix = wrapper_lm(C.dcopy, "double*")
    linearize_fcomplex_matrix = wrapper_lm(C.ccopy, "f2c_complex*")
    linearize_dcomplex_matrix = wrapper_lm(C.zcopy, "f2c_doublecomples*")

    def wrapper_dm(func, p_t):
        def delinearize_matrix(dst, src, data):
            ''' in cpython numpy, dst, src are c-level pointers
                we (ab)use ndarrays instead
            '''
            if src is None:
                raise ValueError('called with NULL input, should not happen')
            if src.dtype is not dst.dtype:
                raise ValueError('called with differing dtypes, should not happen')
            psrc = ffi.cast(p_t, src.__array_interface__['data'])
            pdst = ffi.cast(p_t, dst.__array_interface__['data'])
            pcolumns = ffi.new('int [1]', [data.columns])
            pcolumn_strides = ffi.new('int[1]', [data.column_strides / src.dtypes.itemsize])
            pone = np.int32(ffi.new('int[1]', [1]))
            for i in range(data.rows):
                psrc_void = ffi.cast("void*", psrc)
                pdst_void = ffi.cast("void*", pdst)
                if column_strides > 0:
                    func(pcolumns, psrc_void, pcolumn_strides, pdst_void, pone)
                elif column_strides < 0:
                    func(pcolumns, psrc_void, pcolumn_strides, 
                         ffi.cast("void*", pdst+(columns-1)*column_strides) , pone)
                else:
                    # Zero stride has undefined behavior in some BLAS
                    # implementations (e.g. OSX Accelerate), so do it
                    # manually
                    for j in range(columns):
                        ffi.memcpy(pdst, psrc+columns-1, src.dtype.itemsize)
                psrc += data.columns
                pdst += data.row_strides / src.dtype.itemsize

    delinearize_float_matrix = wrapper_dm(ffi.scopy, "float*")
    delinearize_double_matrix = wrapper_dm(ffi.dcopy, "double*")
    delinearize_fcomplex_matrix = wrapper_dm(ffi.ccopy, "f2c_complex*")
    delinearize_dcomplex_matrix = wrapper_dm(ffi.zcopy, "f2c_doublecomples*")

    def wrap_slogdet(typ, basetyp, cblas_type):
        def slogdet_from_factored_diagonal(src, m, sign):
            sign_acc = sign[0]
            logdet_acc = base_vals[cblas_type][0]
            for i in range(m[0]):
                abs_element = np.abs(src[i,i])
                sign_element = src[i, i] / abs_element
                sign_acc = sign_acc * sign_element
                logdet_acc += np.log(abs_element)
            return sign_acc, logdet_acc    

        def slogdet_single_element(m, src, pivots):
            info = ffi.new('int[1]', [0])
            getattr(C, cblas_type + 'getrf')(m, m, src, m, pivots, info)
            if info[0] == 0:
                for i in range(m[0]):
                    # fortran uses 1 based indexing
                    change_sign += (pivots[i] != (i + 1))
                if change_sign % 2:
                    sign = base_vals[cblas_type]['minus_one']
                else:
                    sign = base_vals[cblas_type]['one']
                sign, logdet = slogdet_from_factored_diagonal(src, m, sign)
            else: 
                # if getrf fails, use - as sign and -inf as logdet
                sign = base_vals[cblas_type]['zero']
                logdet = base_vals[cblas_type]['ninf']
            return sign, logdet

        def slogdet(inout):
            ''' notes:
             *   in must have shape [m, m], out[0] and out[1] are [m]
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = inout[0].shape[0]
            mbuffer = np.empty((m, m), typ)
            pivot = np.empty(m, typ)
            
            # swapped steps to get matrix in FORTRAN order
            data = linearize_data(m, m, inout[0].strides[1], inout[0].strides[0]) 
            linearize_matrix()
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            return sign, logdet

    FLOAT_slogdet = wrap_slogdet(nt.float32, nt.float32, 's')
    DOUBLE_slogdet = wrap_slogdet(nt.float64, nt.float64, 'd')
    CFLOAT_slogdet = wrap_slogdet(nt.complex64, nt.float32, 'c')
    CDOUBLE_slogdet = wrap_slogdet(nt.complex128, nt.float64, 'z')

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

    def wrap_det(typ, basetyp, cblas_type):
        def det(inout):
            ''' notes:
             *   in must have shape [m, m], out is scalar
             *   need to allocate memory for both, matrix_buffer and pivot buffer
            '''
            m = inout[0].shape[0]
            mbuffer = np.empty((m, m), typ)
            pivot = np.empty(m, typ)
        
            # swapped steps to get matrix in FORTRAN order
            data = linearize_data(m, m, inout[0].strides[1], inout[0].strides[0]) 
            sign, logdet = slogdet_single_element(m, mbuffer, pivot)
            return sign, logdet
            
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

