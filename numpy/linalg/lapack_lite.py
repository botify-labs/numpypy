# A CFFI version of numpy/linalg/lapack_module.c
import sys, os
import warnings
try:
    import cffi
    use_cffi = True
except ImportError:
    use_cffi = False

if use_cffi:
    import numpy as np
    # dtype has not been imported yet
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

    macros = {'sfx': '_', 'pfx': ''}
    ffi = cffi.FFI()


    # The next section is a hack to find the lapack implementation, loosly based on
    # numpy.distutils.system_info.get_info. The idea is to use a site.cfg file to specify
    # where the shared object is located. Note that we need the lapack (high-level) interface,
    # they in turn call a low-level implementation maybe using blas or atlas.
    # This has not been tested on OSX
    _C = None
    from numpy.distutils import system_info
    # temporarily mess with environ
    saved_environ = os.environ.copy()
    if sys.platform == 'win32':
        ld_library_path = 'PATH'
        so_prefix = ''
        so_suffix = 'dll'
    else:
        ld_library_path = 'LD_LIBRARY_PATH'
        so_prefix = 'lib'
        so_suffix = 'so'
    for lapack, prefix, suffix in [ \
                    ['openblas_lapack', '', '_'],
                    ['lapack_mkl', '', '_' ],
                    ['lapack', '', '_'],
                    ['lapack_lite', '', '_'],
                  ]:
        si = getattr(system_info, 'lapack_info')()
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
    # workaround for a distutils bugs where some env vars can
    # become longer and longer every time it is used
    for key, value in saved_environ.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
    if _C is None:
        shared_name = os.path.abspath(os.path.dirname(__file__)) + '/' + \
                            so_prefix + 'lapack_lite.' + so_suffix
        if not os.path.exists(shared_name):
            # cffi should support some canonical name formatting like
            # distutils.ccompiler.library_filename()
            raise ValueError('could not find "%s", perhaps the name is slightly off' % shared_name)
        try:
            _C = ffi.dlopen(shared_name)
            warnings.warn('tuned lapack (openblas, atlas ...) not found, using lapack_lite')
        except:
            warnings.warn("no lapack nor lapack_lite shared object available, will try cpyext version next")
            use_cffi = False

if not use_cffi:
    raise NotImplementedError("numpy installation failure: no lapack_lite compiled python module and no lapack shared object")

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

extern double
{pfx}sdot{sfx}(int *n,
            float *sx, int *incx,
            float *sy, int *incy);
extern double
{pfx}ddot{sfx}(int *n,
            double *sx, int *incx,
            double *sy, int *incy);
extern void
{pfx}cdotu{sfx}(f2c_complex *, int *,
       f2c_complex *, int *,
       f2c_complex *, int *);
extern void
{pfx}zdotu{sfx}(f2c_doublecomplex * ret_val, int *n,
	f2c_doublecomplex *zx, int *incx,
    f2c_doublecomplex *zy, int *incy);
extern void
{pfx}cdotc{sfx}(f2c_complex *, int *,
       f2c_complex *, int *,
       f2c_complex *, int *);
extern void
{pfx}zdotc{sfx}(f2c_doublecomplex * ret_val, int *n,
	f2c_doublecomplex *zx, int *incx,
    f2c_doublecomplex *zy, int *incy);

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

extern int
{pfx}dgeqrf{sfx}(int *, int *, double *, int *, double *,
	    double *, int *, int *);

extern int
{pfx}zgeqrf{sfx}(int *, int *, f2c_doublecomplex *, int *,
         f2c_doublecomplex *, f2c_doublecomplex *, int *, int *);
'''.format(**macros))

'''
Create a shared_object which maps the bare name to the one with pfx, sfx
'''

shared_object = Dummy()

for name in ['sgeev', 'dgeev', 'cgeev', 'zgeev', 'ssyevd', 'dsyevd',
             'cheevd', 'zheevd', 'dgelsd', 'zgelsd', 'sgesv', 'dgesv', 'cgesv', 'zgesv',
             'sgetrf', 'dgetrf', 'cgetrf', 'zgetrf', 'spotrf', 'dpotrf', 'cpotrf', 'zpotrf',
             'sgesdd', 'dgesdd', 'cgesdd', 'zgesdd', 'spotrs', 'dpotrs', 'cpotrs', 'zpotrs',
             'spotri', 'dpotri', 'cpotri', 'zpotri', 'scopy', 'dcopy', 'ccopy', 'zcopy',
             'sdot', 'ddot', 'cdotu', 'zdotu', 'cdotc', 'zdotc', 'dgeqrf', 'zgeqrf',
             'sgemm', 'dgemm', 'cgemm', 'zgemm']:
    setattr(shared_object, name, getattr(_C, macros['pfx'] + name + macros['sfx']))

'''
Since numpy expects to be able to call these functions with python objects,
create a mapping mechanism: 
  ndarray -> equivalent pointer to its data based on dtype
  numpy scalar -> equivalent pointer based on ffi.cast
  ffi.CData -> ready to be called
  arbitrary cpython type -> use ffi.new to create a pointer to a type
                            determined from the function signature
'''

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

def convert_arg(inarg, ffitype):
    '''
    try to convert the inarg to an appropriate c pointer
    '''
    if isinstance(inarg, np.ndarray):
        return toCptr(inarg)
    elif type(inarg) in toCtypeA:
        return ffi.cast(toCtypeA[inarg], inarg)
    elif isinstance(inarg, ffi.CData):
        return inarg
    # Hope for the best...
    ctyp_p = ffi.getctype(ffitype)
    ctyp = ctyp_p[:-2]
    return ffi.new( ctyp + '[1]', [inarg])

def call_func(name):
    def call_with_convert(*args):
        func = getattr(shared_object, name)
        fargs = ffi.typeof(func).args
        converted_args = [convert_arg(a,b) for a,b in zip(args, fargs)]
        res = func(*converted_args)
        retval = {'info':converted_args[-1][0]}
        # numpy expects a dictionary
        if 'gelsd' in name:
            # special case, the rank argument is returned as well
            retval['rank'] = converted_args[9][0]
        return retval
    return call_with_convert

def not_implemented(*args):
    raise NotImplementedError('function not found, does lapack_lite object exist?')

for name in ['sgeev', 'dgeev', 'cgeev', 'zgeev', 'ssyevd', 'dsyevd', 'cheevd',
         'zheevd', 'dgelsd', 'zgelsd', 'sgesv', 'dgesv', 'cgesv', 'zgesv',
         'sgetrf', 'dgetrf', 'cgetrf', 'zgetrf', 'spotrf', 'dpotrf', 'cpotrf',
         'zpotrf', 'sgesdd', 'dgesdd', 'cgesdd', 'zgesdd', 'spotrs', 'dpotrs',
         'cpotrs', 'zpotrs', 'spotri', 'dpotri', 'cpotri', 'zpotri', 'scopy',
         'dcopy', 'ccopy', 'zcopy', 'sdot', 'ddot', 'cdotu', 'zdotu', 'cdotc',
         'zdotc', 'sgemm', 'dgemm', 'cgemm', 'zgemm', 'dgessd', 'dgeqrf',
         'dorggr', 'zgeqrf', 'zungqr', 'xerbla']:
    if name in dir(shared_object):
        globals()[name] = call_func(name)
    else:
        globals()[name] = not_implemented
