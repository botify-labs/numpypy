/* Declare and export functions from lapack_lite for cffi
*/

#ifdef _WIN32
#pragma warning( disable : 4244)
#ifdef _LAPACK_LITE_DLL
#define EXPORTED __declspec(dllexport)
#else
#define EXPORTED __declspec(dllimport)
#endif
#else
#define EXPORTED extern
#endif



typedef complex f2c_complex;
typedef doublecomplex f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

EXPORTED int
sgeev_(char *jobvl, char *jobvr, int *n,
             float a[], int *lda, float wr[], float wi[],
             float vl[], int *ldvl, float vr[], int *ldvr,
             float work[], int lwork[],
             int *info);
EXPORTED int
dgeev_(char *jobvl, char *jobvr, int *n,
             double a[], int *lda, double wr[], double wi[],
             double vl[], int *ldvl, double vr[], int *ldvr,
             double work[], int lwork[],
             int *info);
EXPORTED int
cgeev_(char *jobvl, char *jobvr, int *n,
             f2c_complex a[], int *lda,
             f2c_complex w[],
             f2c_complex vl[], int *ldvl,
             f2c_complex vr[], int *ldvr,
             f2c_complex work[], int *lwork,
             float rwork[],
             int *info);
EXPORTED int
zgeev_(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);

EXPORTED int
ssyevd_(char *jobz, char *uplo, int *n,
              float a[], int *lda, float w[], float work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
EXPORTED int
dsyevd_(char *jobz, char *uplo, int *n,
              double a[], int *lda, double w[], double work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
EXPORTED int
cheevd_(char *jobz, char *uplo, int *n,
              f2c_complex a[], int *lda,
              float w[], f2c_complex work[],
              int *lwork, float rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);
EXPORTED int
zheevd_(char *jobz, char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              double w[], f2c_doublecomplex work[],
              int *lwork, double rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);

EXPORTED int
dgelsd_(int *m, int *n, int *nrhs,
              double a[], int *lda, double b[], int *ldb,
              double s[], double *rcond, int *rank,
              double work[], int *lwork, int iwork[],
              int *info);
EXPORTED int
zgelsd_(int *m, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              double s[], double *rcond, int *rank,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[],
              int *info);

EXPORTED int
sgesv_(int *n, int *nrhs,
             float a[], int *lda,
             int ipiv[],
             float b[], int *ldb,
             int *info);
EXPORTED int
dgesv_(int *n, int *nrhs,
             double a[], int *lda,
             int ipiv[],
             double b[], int *ldb,
             int *info);
EXPORTED int
cgesv_(int *n, int *nrhs,
             f2c_complex a[], int *lda,
             int ipiv[],
             f2c_complex b[], int *ldb,
             int *info);
EXPORTED int
zgesv_(int *n, int *nrhs,
             f2c_doublecomplex a[], int *lda,
             int ipiv[],
             f2c_doublecomplex b[], int *ldb,
             int *info);

EXPORTED int
sgetrf_(int *m, int *n,
              float a[], int *lda,
              int ipiv[],
              int *info);
EXPORTED int
dgetrf_(int *m, int *n,
              double a[], int *lda,
              int ipiv[],
              int *info);
EXPORTED int
cgetrf_(int *m, int *n,
              f2c_complex a[], int *lda,
              int ipiv[],
              int *info);
EXPORTED int
zgetrf_(int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              int ipiv[],
              int *info);

EXPORTED int
spotrf_(char *uplo, int *n,
              float a[], int *lda,
              int *info);
EXPORTED int
dpotrf_(char *uplo, int *n,
              double a[], int *lda,
              int *info);
EXPORTED int
cpotrf_(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
EXPORTED int
zpotrf_(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

EXPORTED int
sgesdd_(char *jobz, int *m, int *n,
              float a[], int *lda, float s[], float u[],
              int *ldu, float vt[], int *ldvt, float work[],
              int *lwork, int iwork[], int *info);
EXPORTED int
dgesdd_(char *jobz, int *m, int *n,
              double a[], int *lda, double s[], double u[],
              int *ldu, double vt[], int *ldvt, double work[],
              int *lwork, int iwork[], int *info);
EXPORTED int
cgesdd_(char *jobz, int *m, int *n,
              f2c_complex a[], int *lda,
              float s[], f2c_complex u[], int *ldu,
              f2c_complex vt[], int *ldvt,
              f2c_complex work[], int *lwork,
              float rwork[], int iwork[], int *info);
EXPORTED int
zgesdd_(char *jobz, int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              double s[], f2c_doublecomplex u[], int *ldu,
              f2c_doublecomplex vt[], int *ldvt,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[], int *info);

EXPORTED int
spotrs_(char *uplo, int *n, int *nrhs,
              float a[], int *lda,
              float b[], int *ldb,
              int *info);
EXPORTED int
dpotrs_(char *uplo, int *n, int *nrhs,
              double a[], int *lda,
              double b[], int *ldb,
              int *info);
EXPORTED int
cpotrs_(char *uplo, int *n, int *nrhs,
              f2c_complex a[], int *lda,
              f2c_complex b[], int *ldb,
              int *info);
EXPORTED int
zpotrs_(char *uplo, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              int *info);

EXPORTED int
spotri_(char *uplo, int *n,
              float a[], int *lda,
              int *info);
EXPORTED int
dpotri_(char *uplo, int *n,
              double a[], int *lda,
              int *info);
EXPORTED int
cpotri_(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
EXPORTED int
zpotri_(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

EXPORTED int
scopy_(int *n,
             float *sx, int *incx,
             float *sy, int *incy);
EXPORTED int
dcopy_(int *n,
             double *sx, int *incx,
             double *sy, int *incy);
EXPORTED int
ccopy_(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
EXPORTED int
zcopy_(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

EXPORTED double
sdot_(int *n,
            float *sx, int *incx,
            float *sy, int *incy);
EXPORTED double
ddot_(int *n,
            double *sx, int *incx,
            double *sy, int *incy);
EXPORTED VOID
cdotu_(complex *, integer *, 
       complex *, integer *, 
       complex *, integer *);
EXPORTED VOID
zdotu_(doublecomplex * ret_val, integer *n,
	doublecomplex *zx, integer *incx, 
    doublecomplex *zy, integer *incy);
EXPORTED VOID
cdotc_(complex *, integer *, 
       complex *, integer *, 
       complex *, integer *);
EXPORTED VOID
zdotc_(doublecomplex * ret_val, integer *n,
	doublecomplex *zx, integer *incx, 
    doublecomplex *zy, integer *incy);

EXPORTED int
sgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             float *alpha,
             float *a, int *lda,
             float *b, int *ldb,
             float *beta,
             float *c, int *ldc);
EXPORTED int
dgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             double *alpha,
             double *a, int *lda,
             double *b, int *ldb,
             double *beta,
             double *c, int *ldc);
EXPORTED int
cgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_complex *alpha,
             f2c_complex *a, int *lda,
             f2c_complex *b, int *ldb,
             f2c_complex *beta,
             f2c_complex *c, int *ldc);
EXPORTED int
zgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, int *lda,
             f2c_doublecomplex *b, int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, int *ldc);
EXPORTED int
dgeqrf_(int *, int *, double *, int *, double *,
	    double *, int *, int *);

EXPORTED int
zgeqrf_(int *, int *, f2c_doublecomplex *, int *,
         f2c_doublecomplex *, f2c_doublecomplex *, int *, int *);

