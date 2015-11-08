#define _CFFI_
#include <Python.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

/* See doc/misc/parse_c_type.rst in the source of CFFI for more information */

typedef void *_cffi_opcode_t;

#define _CFFI_OP(opcode, arg)   (_cffi_opcode_t)(opcode | (((uintptr_t)(arg)) << 8))
#define _CFFI_GETOP(cffi_opcode)    ((unsigned char)(uintptr_t)cffi_opcode)
#define _CFFI_GETARG(cffi_opcode)   (((intptr_t)cffi_opcode) >> 8)

#define _CFFI_OP_PRIMITIVE       1
#define _CFFI_OP_POINTER         3
#define _CFFI_OP_ARRAY           5
#define _CFFI_OP_OPEN_ARRAY      7
#define _CFFI_OP_STRUCT_UNION    9
#define _CFFI_OP_ENUM           11
#define _CFFI_OP_FUNCTION       13
#define _CFFI_OP_FUNCTION_END   15
#define _CFFI_OP_NOOP           17
#define _CFFI_OP_BITFIELD       19
#define _CFFI_OP_TYPENAME       21
#define _CFFI_OP_CPYTHON_BLTN_V 23   // varargs
#define _CFFI_OP_CPYTHON_BLTN_N 25   // noargs
#define _CFFI_OP_CPYTHON_BLTN_O 27   // O  (i.e. a single arg)
#define _CFFI_OP_CONSTANT       29
#define _CFFI_OP_CONSTANT_INT   31
#define _CFFI_OP_GLOBAL_VAR     33
#define _CFFI_OP_DLOPEN_FUNC    35
#define _CFFI_OP_DLOPEN_CONST   37
#define _CFFI_OP_GLOBAL_VAR_F   39

#define _CFFI_PRIM_VOID          0
#define _CFFI_PRIM_BOOL          1
#define _CFFI_PRIM_CHAR          2
#define _CFFI_PRIM_SCHAR         3
#define _CFFI_PRIM_UCHAR         4
#define _CFFI_PRIM_SHORT         5
#define _CFFI_PRIM_USHORT        6
#define _CFFI_PRIM_INT           7
#define _CFFI_PRIM_UINT          8
#define _CFFI_PRIM_LONG          9
#define _CFFI_PRIM_ULONG        10
#define _CFFI_PRIM_LONGLONG     11
#define _CFFI_PRIM_ULONGLONG    12
#define _CFFI_PRIM_FLOAT        13
#define _CFFI_PRIM_DOUBLE       14
#define _CFFI_PRIM_LONGDOUBLE   15

#define _CFFI_PRIM_WCHAR        16
#define _CFFI_PRIM_INT8         17
#define _CFFI_PRIM_UINT8        18
#define _CFFI_PRIM_INT16        19
#define _CFFI_PRIM_UINT16       20
#define _CFFI_PRIM_INT32        21
#define _CFFI_PRIM_UINT32       22
#define _CFFI_PRIM_INT64        23
#define _CFFI_PRIM_UINT64       24
#define _CFFI_PRIM_INTPTR       25
#define _CFFI_PRIM_UINTPTR      26
#define _CFFI_PRIM_PTRDIFF      27
#define _CFFI_PRIM_SIZE         28
#define _CFFI_PRIM_SSIZE        29
#define _CFFI_PRIM_INT_LEAST8   30
#define _CFFI_PRIM_UINT_LEAST8  31
#define _CFFI_PRIM_INT_LEAST16  32
#define _CFFI_PRIM_UINT_LEAST16 33
#define _CFFI_PRIM_INT_LEAST32  34
#define _CFFI_PRIM_UINT_LEAST32 35
#define _CFFI_PRIM_INT_LEAST64  36
#define _CFFI_PRIM_UINT_LEAST64 37
#define _CFFI_PRIM_INT_FAST8    38
#define _CFFI_PRIM_UINT_FAST8   39
#define _CFFI_PRIM_INT_FAST16   40
#define _CFFI_PRIM_UINT_FAST16  41
#define _CFFI_PRIM_INT_FAST32   42
#define _CFFI_PRIM_UINT_FAST32  43
#define _CFFI_PRIM_INT_FAST64   44
#define _CFFI_PRIM_UINT_FAST64  45
#define _CFFI_PRIM_INTMAX       46
#define _CFFI_PRIM_UINTMAX      47

#define _CFFI__NUM_PRIM         48
#define _CFFI__UNKNOWN_PRIM           (-1)
#define _CFFI__UNKNOWN_FLOAT_PRIM     (-2)
#define _CFFI__UNKNOWN_LONG_DOUBLE    (-3)


struct _cffi_global_s {
    const char *name;
    void *address;
    _cffi_opcode_t type_op;
    void *size_or_direct_fn;  // OP_GLOBAL_VAR: size, or 0 if unknown
                              // OP_CPYTHON_BLTN_*: addr of direct function
};

struct _cffi_getconst_s {
    unsigned long long value;
    const struct _cffi_type_context_s *ctx;
    int gindex;
};

struct _cffi_struct_union_s {
    const char *name;
    int type_index;          // -> _cffi_types, on a OP_STRUCT_UNION
    int flags;               // _CFFI_F_* flags below
    size_t size;
    int alignment;
    int first_field_index;   // -> _cffi_fields array
    int num_fields;
};
#define _CFFI_F_UNION         0x01   // is a union, not a struct
#define _CFFI_F_CHECK_FIELDS  0x02   // complain if fields are not in the
                                     // "standard layout" or if some are missing
#define _CFFI_F_PACKED        0x04   // for CHECK_FIELDS, assume a packed struct
#define _CFFI_F_EXTERNAL      0x08   // in some other ffi.include()
#define _CFFI_F_OPAQUE        0x10   // opaque

struct _cffi_field_s {
    const char *name;
    size_t field_offset;
    size_t field_size;
    _cffi_opcode_t field_type_op;
};

struct _cffi_enum_s {
    const char *name;
    int type_index;          // -> _cffi_types, on a OP_ENUM
    int type_prim;           // _CFFI_PRIM_xxx
    const char *enumerators; // comma-delimited string
};

struct _cffi_typename_s {
    const char *name;
    int type_index;   /* if opaque, points to a possibly artificial
                         OP_STRUCT which is itself opaque */
};

struct _cffi_type_context_s {
    _cffi_opcode_t *types;
    const struct _cffi_global_s *globals;
    const struct _cffi_field_s *fields;
    const struct _cffi_struct_union_s *struct_unions;
    const struct _cffi_enum_s *enums;
    const struct _cffi_typename_s *typenames;
    int num_globals;
    int num_struct_unions;
    int num_enums;
    int num_typenames;
    const char *const *includes;
    int num_types;
    int flags;      /* future extension */
};

struct _cffi_parse_info_s {
    const struct _cffi_type_context_s *ctx;
    _cffi_opcode_t *output;
    unsigned int output_size;
    size_t error_location;
    const char *error_message;
};

#ifdef _CFFI_INTERNAL
static int parse_c_type(struct _cffi_parse_info_s *info, const char *input);
static int search_in_globals(const struct _cffi_type_context_s *ctx,
                             const char *search, size_t search_len);
static int search_in_struct_unions(const struct _cffi_type_context_s *ctx,
                                   const char *search, size_t search_len);
#endif

/* this block of #ifs should be kept exactly identical between
   c/_cffi_backend.c, cffi/vengine_cpy.py, cffi/vengine_gen.py
   and cffi/_cffi_include.h */
#if defined(_MSC_VER)
# include <malloc.h>   /* for alloca() */
# if _MSC_VER < 1600   /* MSVC < 2010 */
   typedef __int8 int8_t;
   typedef __int16 int16_t;
   typedef __int32 int32_t;
   typedef __int64 int64_t;
   typedef unsigned __int8 uint8_t;
   typedef unsigned __int16 uint16_t;
   typedef unsigned __int32 uint32_t;
   typedef unsigned __int64 uint64_t;
   typedef __int8 int_least8_t;
   typedef __int16 int_least16_t;
   typedef __int32 int_least32_t;
   typedef __int64 int_least64_t;
   typedef unsigned __int8 uint_least8_t;
   typedef unsigned __int16 uint_least16_t;
   typedef unsigned __int32 uint_least32_t;
   typedef unsigned __int64 uint_least64_t;
   typedef __int8 int_fast8_t;
   typedef __int16 int_fast16_t;
   typedef __int32 int_fast32_t;
   typedef __int64 int_fast64_t;
   typedef unsigned __int8 uint_fast8_t;
   typedef unsigned __int16 uint_fast16_t;
   typedef unsigned __int32 uint_fast32_t;
   typedef unsigned __int64 uint_fast64_t;
   typedef __int64 intmax_t;
   typedef unsigned __int64 uintmax_t;
# else
#  include <stdint.h>
# endif
# if _MSC_VER < 1800   /* MSVC < 2013 */
   typedef unsigned char _Bool;
# endif
#else
# include <stdint.h>
# if (defined (__SVR4) && defined (__sun)) || defined(_AIX) || defined(__hpux)
#  include <alloca.h>
# endif
#endif

#ifdef __GNUC__
# define _CFFI_UNUSED_FN  __attribute__((unused))
#else
# define _CFFI_UNUSED_FN  /* nothing */
#endif

/**********  CPython-specific section  **********/
#ifndef PYPY_VERSION


#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong PyLong_FromLong
#endif

#define _cffi_from_c_double PyFloat_FromDouble
#define _cffi_from_c_float PyFloat_FromDouble
#define _cffi_from_c_long PyInt_FromLong
#define _cffi_from_c_ulong PyLong_FromUnsignedLong
#define _cffi_from_c_longlong PyLong_FromLongLong
#define _cffi_from_c_ulonglong PyLong_FromUnsignedLongLong

#define _cffi_to_c_double PyFloat_AsDouble
#define _cffi_to_c_float PyFloat_AsDouble

#define _cffi_from_c_int(x, type)                                        \
    (((type)-1) > 0 ? /* unsigned */                                     \
        (sizeof(type) < sizeof(long) ?                                   \
            PyInt_FromLong((long)x) :                                    \
         sizeof(type) == sizeof(long) ?                                  \
            PyLong_FromUnsignedLong((unsigned long)x) :                  \
            PyLong_FromUnsignedLongLong((unsigned long long)x)) :        \
        (sizeof(type) <= sizeof(long) ?                                  \
            PyInt_FromLong((long)x) :                                    \
            PyLong_FromLongLong((long long)x)))

#define _cffi_to_c_int(o, type)                                          \
    ((type)(                                                             \
     sizeof(type) == 1 ? (((type)-1) > 0 ? (type)_cffi_to_c_u8(o)        \
                                         : (type)_cffi_to_c_i8(o)) :     \
     sizeof(type) == 2 ? (((type)-1) > 0 ? (type)_cffi_to_c_u16(o)       \
                                         : (type)_cffi_to_c_i16(o)) :    \
     sizeof(type) == 4 ? (((type)-1) > 0 ? (type)_cffi_to_c_u32(o)       \
                                         : (type)_cffi_to_c_i32(o)) :    \
     sizeof(type) == 8 ? (((type)-1) > 0 ? (type)_cffi_to_c_u64(o)       \
                                         : (type)_cffi_to_c_i64(o)) :    \
     (Py_FatalError("unsupported size for type " #type), (type)0)))

#define _cffi_to_c_i8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[1])
#define _cffi_to_c_u8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[2])
#define _cffi_to_c_i16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[3])
#define _cffi_to_c_u16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[4])
#define _cffi_to_c_i32                                                   \
                 ((int(*)(PyObject *))_cffi_exports[5])
#define _cffi_to_c_u32                                                   \
                 ((unsigned int(*)(PyObject *))_cffi_exports[6])
#define _cffi_to_c_i64                                                   \
                 ((long long(*)(PyObject *))_cffi_exports[7])
#define _cffi_to_c_u64                                                   \
                 ((unsigned long long(*)(PyObject *))_cffi_exports[8])
#define _cffi_to_c_char                                                  \
                 ((int(*)(PyObject *))_cffi_exports[9])
#define _cffi_from_c_pointer                                             \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[10])
#define _cffi_to_c_pointer                                               \
    ((char *(*)(PyObject *, CTypeDescrObject *))_cffi_exports[11])
#define _cffi_get_struct_layout                                          \
    not used any more
#define _cffi_restore_errno                                              \
    ((void(*)(void))_cffi_exports[13])
#define _cffi_save_errno                                                 \
    ((void(*)(void))_cffi_exports[14])
#define _cffi_from_c_char                                                \
    ((PyObject *(*)(char))_cffi_exports[15])
#define _cffi_from_c_deref                                               \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[16])
#define _cffi_to_c                                                       \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[17])
#define _cffi_from_c_struct                                              \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[18])
#define _cffi_to_c_wchar_t                                               \
    ((wchar_t(*)(PyObject *))_cffi_exports[19])
#define _cffi_from_c_wchar_t                                             \
    ((PyObject *(*)(wchar_t))_cffi_exports[20])
#define _cffi_to_c_long_double                                           \
    ((long double(*)(PyObject *))_cffi_exports[21])
#define _cffi_to_c__Bool                                                 \
    ((_Bool(*)(PyObject *))_cffi_exports[22])
#define _cffi_prepare_pointer_call_argument                              \
    ((Py_ssize_t(*)(CTypeDescrObject *, PyObject *, char **))_cffi_exports[23])
#define _cffi_convert_array_from_object                                  \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[24])
#define _CFFI_NUM_EXPORTS 25

typedef struct _ctypedescr CTypeDescrObject;

static void *_cffi_exports[_CFFI_NUM_EXPORTS];

#define _cffi_type(index)   (                           \
    assert((((uintptr_t)_cffi_types[index]) & 1) == 0), \
    (CTypeDescrObject *)_cffi_types[index])

static PyObject *_cffi_init(const char *module_name, Py_ssize_t version,
                            const struct _cffi_type_context_s *ctx)
{
    PyObject *module, *o_arg, *new_module;
    void *raw[] = {
        (void *)module_name,
        (void *)version,
        (void *)_cffi_exports,
        (void *)ctx,
    };

    module = PyImport_ImportModule("_cffi_backend");
    if (module == NULL)
        goto failure;

    o_arg = PyLong_FromVoidPtr((void *)raw);
    if (o_arg == NULL)
        goto failure;

    new_module = PyObject_CallMethod(
        module, (char *)"_init_cffi_1_0_external_module", (char *)"O", o_arg);

    Py_DECREF(o_arg);
    Py_DECREF(module);
    return new_module;

  failure:
    Py_XDECREF(module);
    return NULL;
}

_CFFI_UNUSED_FN
static PyObject **_cffi_unpack_args(PyObject *args_tuple, Py_ssize_t expected,
                                    const char *fnname)
{
    if (PyTuple_GET_SIZE(args_tuple) != expected) {
        PyErr_Format(PyExc_TypeError,
                     "%.150s() takes exactly %zd arguments (%zd given)",
                     fnname, expected, PyTuple_GET_SIZE(args_tuple));
        return NULL;
    }
    return &PyTuple_GET_ITEM(args_tuple, 0);   /* pointer to the first item,
                                                  the others follow */
}

#endif
/**********  end CPython-specific section  **********/


#define _cffi_array_len(array)   (sizeof(array) / sizeof((array)[0]))

#define _cffi_prim_int(size, sign)                                      \
    ((size) == 1 ? ((sign) ? _CFFI_PRIM_INT8  : _CFFI_PRIM_UINT8)  :    \
     (size) == 2 ? ((sign) ? _CFFI_PRIM_INT16 : _CFFI_PRIM_UINT16) :    \
     (size) == 4 ? ((sign) ? _CFFI_PRIM_INT32 : _CFFI_PRIM_UINT32) :    \
     (size) == 8 ? ((sign) ? _CFFI_PRIM_INT64 : _CFFI_PRIM_UINT64) :    \
     _CFFI__UNKNOWN_PRIM)

#define _cffi_prim_float(size)                                          \
    ((size) == sizeof(float) ? _CFFI_PRIM_FLOAT :                       \
     (size) == sizeof(double) ? _CFFI_PRIM_DOUBLE :                     \
     (size) == sizeof(long double) ? _CFFI__UNKNOWN_LONG_DOUBLE :       \
     _CFFI__UNKNOWN_FLOAT_PRIM)

#define _cffi_check_int(got, got_nonpos, expected)      \
    ((got_nonpos) == (expected <= 0) &&                 \
     (got) == (unsigned long long)expected)

#ifdef __cplusplus
}
#endif

/************************************************************/


/*
 *                    LAPACK functions
 */

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

extern int
sgeev_(char *jobvl, char *jobvr, int *n,
             float a[], int *lda, float wr[], float wi[],
             float vl[], int *ldvl, float vr[], int *ldvr,
             float work[], int lwork[],
             int *info);
extern int
dgeev_(char *jobvl, char *jobvr, int *n,
             double a[], int *lda, double wr[], double wi[],
             double vl[], int *ldvl, double vr[], int *ldvr,
             double work[], int lwork[],
             int *info);
extern int
cgeev_(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);
extern int
zgeev_(char *jobvl, char *jobvr, int *n,
             f2c_doublecomplex a[], int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], int *ldvl,
             f2c_doublecomplex vr[], int *ldvr,
             f2c_doublecomplex work[], int *lwork,
             double rwork[],
             int *info);

extern int
ssyevd_(char *jobz, char *uplo, int *n,
              float a[], int *lda, float w[], float work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
dsyevd_(char *jobz, char *uplo, int *n,
              double a[], int *lda, double w[], double work[],
              int *lwork, int iwork[], int *liwork,
              int *info);
extern int
cheevd_(char *jobz, char *uplo, int *n,
              f2c_complex a[], int *lda,
              float w[], f2c_complex work[],
              int *lwork, float rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);
extern int
zheevd_(char *jobz, char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              double w[], f2c_doublecomplex work[],
              int *lwork, double rwork[], int *lrwork, int iwork[],
              int *liwork,
              int *info);

extern int
dgelsd_(int *m, int *n, int *nrhs,
              double a[], int *lda, double b[], int *ldb,
              double s[], double *rcond, int *rank,
              double work[], int *lwork, int iwork[],
              int *info);
extern int
zgelsd_(int *m, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              double s[], double *rcond, int *rank,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[],
              int *info);

extern int
sgesv_(int *n, int *nrhs,
             float a[], int *lda,
             int ipiv[],
             float b[], int *ldb,
             int *info);
extern int
dgesv_(int *n, int *nrhs,
             double a[], int *lda,
             int ipiv[],
             double b[], int *ldb,
             int *info);
extern int
cgesv_(int *n, int *nrhs,
             f2c_complex a[], int *lda,
             int ipiv[],
             f2c_complex b[], int *ldb,
             int *info);
extern int
zgesv_(int *n, int *nrhs,
             f2c_doublecomplex a[], int *lda,
             int ipiv[],
             f2c_doublecomplex b[], int *ldb,
             int *info);

extern int
sgetrf_(int *m, int *n,
              float a[], int *lda,
              int ipiv[],
              int *info);
extern int
dgetrf_(int *m, int *n,
              double a[], int *lda,
              int ipiv[],
              int *info);
extern int
cgetrf_(int *m, int *n,
              f2c_complex a[], int *lda,
              int ipiv[],
              int *info);
extern int
zgetrf_(int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              int ipiv[],
              int *info);

extern int
spotrf_(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
dpotrf_(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
cpotrf_(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
zpotrf_(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
sgesdd_(char *jobz, int *m, int *n,
              float a[], int *lda, float s[], float u[],
              int *ldu, float vt[], int *ldvt, float work[],
              int *lwork, int iwork[], int *info);
extern int
dgesdd_(char *jobz, int *m, int *n,
              double a[], int *lda, double s[], double u[],
              int *ldu, double vt[], int *ldvt, double work[],
              int *lwork, int iwork[], int *info);
extern int
cgesdd_(char *jobz, int *m, int *n,
              f2c_complex a[], int *lda,
              float s[], f2c_complex u[], int *ldu,
              f2c_complex vt[], int *ldvt,
              f2c_complex work[], int *lwork,
              float rwork[], int iwork[], int *info);
extern int
zgesdd_(char *jobz, int *m, int *n,
              f2c_doublecomplex a[], int *lda,
              double s[], f2c_doublecomplex u[], int *ldu,
              f2c_doublecomplex vt[], int *ldvt,
              f2c_doublecomplex work[], int *lwork,
              double rwork[], int iwork[], int *info);

extern int
spotrs_(char *uplo, int *n, int *nrhs,
              float a[], int *lda,
              float b[], int *ldb,
              int *info);
extern int
dpotrs_(char *uplo, int *n, int *nrhs,
              double a[], int *lda,
              double b[], int *ldb,
              int *info);
extern int
cpotrs_(char *uplo, int *n, int *nrhs,
              f2c_complex a[], int *lda,
              f2c_complex b[], int *ldb,
              int *info);
extern int
zpotrs_(char *uplo, int *n, int *nrhs,
              f2c_doublecomplex a[], int *lda,
              f2c_doublecomplex b[], int *ldb,
              int *info);

extern int
spotri_(char *uplo, int *n,
              float a[], int *lda,
              int *info);
extern int
dpotri_(char *uplo, int *n,
              double a[], int *lda,
              int *info);
extern int
cpotri_(char *uplo, int *n,
              f2c_complex a[], int *lda,
              int *info);
extern int
zpotri_(char *uplo, int *n,
              f2c_doublecomplex a[], int *lda,
              int *info);

extern int
scopy_(int *n,
             float *sx, int *incx,
             float *sy, int *incy);
extern int
dcopy_(int *n,
             double *sx, int *incx,
             double *sy, int *incy);
extern int
ccopy_(int *n,
             f2c_complex *sx, int *incx,
             f2c_complex *sy, int *incy);
extern int
zcopy_(int *n,
             f2c_doublecomplex *sx, int *incx,
             f2c_doublecomplex *sy, int *incy);

extern double
sdot_(int *n,
            float *sx, int *incx,
            float *sy, int *incy);
extern double
ddot_(int *n,
            double *sx, int *incx,
            double *sy, int *incy);
extern void
cdotu_(f2c_complex *, int *,
       f2c_complex *, int *,
       f2c_complex *, int *);
extern void
zdotu_(f2c_doublecomplex * ret_val, int *n,
	f2c_doublecomplex *zx, int *incx,
    f2c_doublecomplex *zy, int *incy);
extern void
cdotc_(f2c_complex *, int *,
       f2c_complex *, int *,
       f2c_complex *, int *);
extern void
zdotc_(f2c_doublecomplex * ret_val, int *n,
	f2c_doublecomplex *zx, int *incx,
    f2c_doublecomplex *zy, int *incy);

extern int
sgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             float *alpha,
             float *a, int *lda,
             float *b, int *ldb,
             float *beta,
             float *c, int *ldc);
extern int
dgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             double *alpha,
             double *a, int *lda,
             double *b, int *ldb,
             double *beta,
             double *c, int *ldc);
extern int
cgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_complex *alpha,
             f2c_complex *a, int *lda,
             f2c_complex *b, int *ldb,
             f2c_complex *beta,
             f2c_complex *c, int *ldc);
extern int
zgemm_(char *transa, char *transb,
             int *m, int *n, int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, int *lda,
             f2c_doublecomplex *b, int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, int *ldc);

extern int
dgeqrf_(int *, int *, double *, int *, double *,
	    double *, int *, int *);

extern int
zgeqrf_(int *, int *, f2c_doublecomplex *, int *,
         f2c_doublecomplex *, f2c_doublecomplex *, int *, int *);

extern int
dorgqr_(int *m, int *n, int *k, double a[], int *lda,
                          double tau[], double work[],
                          int *lwork, int *info);

extern int
zungqr_(int *m, int *n, int *k, f2c_doublecomplex a[],
                          int *lda, f2c_doublecomplex tau[],
                          f2c_doublecomplex work[], int *lwork, int *info);



/************************************************************/

static void *_cffi_types[] = {
/*  0 */ _CFFI_OP(_CFFI_OP_FUNCTION, 504), // double()(int *, double *, int *, double *, int *)
/*  1 */ _CFFI_OP(_CFFI_OP_POINTER, 508), // int *
/*  2 */ _CFFI_OP(_CFFI_OP_POINTER, 504), // double *
/*  3 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  4 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/*  5 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  6 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  7 */ _CFFI_OP(_CFFI_OP_FUNCTION, 504), // double()(int *, float *, int *, float *, int *)
/*  8 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  9 */ _CFFI_OP(_CFFI_OP_POINTER, 507), // float *
/* 10 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 11 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 12 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 13 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 14 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, double *, int *, double *, double *, double *, int *, double *, int *, double *, int *, int *)
/* 15 */ _CFFI_OP(_CFFI_OP_POINTER, 503), // char *
/* 16 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 17 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 18 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 19 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 20 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 21 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 22 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 23 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 24 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 25 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 26 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 27 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 28 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 29 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 30 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, double *, int *, double *, double *, int *, int *, int *, int *)
/* 31 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 32 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 33 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 34 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 35 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 36 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 37 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 38 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 39 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 40 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 41 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 42 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 43 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, f2c_complex *, int *, float *, f2c_complex *, int *, float *, int *, int *, int *, int *)
/* 44 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 45 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 46 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 47 */ _CFFI_OP(_CFFI_OP_POINTER, 505), // f2c_complex *
/* 48 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 49 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 50 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 51 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 52 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 53 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 54 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 55 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 56 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 57 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 58 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, f2c_doublecomplex *, int *, double *, f2c_doublecomplex *, int *, double *, int *, int *, int *, int *)
/* 59 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 60 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 61 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 62 */ _CFFI_OP(_CFFI_OP_POINTER, 506), // f2c_doublecomplex *
/* 63 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 64 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 65 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 66 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 67 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 68 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 69 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 70 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 71 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 72 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 73 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, double *, int *)
/* 74 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 75 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 76 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 77 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 78 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 79 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 80 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 81 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 82 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 83 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 84 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 85 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 86 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 87 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 88 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 89 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, float *, int *, float *, float *, float *, int *, float *, int *, float *, int *, int *)
/* 90 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 91 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 92 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 93 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 94 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 95 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 96 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 97 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 98 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 99 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 100 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 101 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 102 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 103 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 104 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 105 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, float *, int *, float *, float *, int *, int *, int *, int *)
/* 106 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 107 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 108 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 109 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 110 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 111 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 112 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 113 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 114 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 115 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 116 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 117 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 118 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *)
/* 119 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 120 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 121 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 122 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 123 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 124 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 125 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 126 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 127 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 128 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 129 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 130 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 131 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 132 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 133 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, int *, int *, f2c_complex *, f2c_complex *, int *, f2c_complex *, int *, f2c_complex *, f2c_complex *, int *)
/* 134 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 135 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 136 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 137 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 138 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 139 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 140 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 141 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 142 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 143 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 144 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 145 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 146 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 147 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 148 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, int *, int *, f2c_doublecomplex *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, f2c_doublecomplex *, int *)
/* 149 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 150 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 151 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 152 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 153 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 154 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 155 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 156 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 157 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 158 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 159 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 160 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 161 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 162 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 163 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *)
/* 164 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 165 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 166 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 167 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 168 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 169 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 170 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 171 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 172 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 173 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 174 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 175 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 176 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 177 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 178 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, double *, int *, int *)
/* 179 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 180 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 181 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 182 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 183 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 184 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 185 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, f2c_complex *, int *, int *)
/* 186 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 187 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 188 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 189 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 190 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 191 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 192 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, f2c_doublecomplex *, int *, int *)
/* 193 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 194 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 195 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 196 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 197 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 198 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 199 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, float *, int *, int *)
/* 200 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 201 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 202 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 203 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 204 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 205 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 206 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, double *, int *, double *, double *, int *, double *, int *, double *, int *, int *, int *)
/* 207 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 208 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 209 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 210 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 211 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 212 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 213 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 214 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 215 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 216 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 217 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 218 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 219 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 220 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 221 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 222 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, double *, int *, double *, int *, int *)
/* 223 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 224 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 225 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 226 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 227 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 228 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 229 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 230 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 231 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 232 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, f2c_complex *, int *, f2c_complex *, int *, int *)
/* 233 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 234 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 235 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 236 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 237 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 238 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 239 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 240 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 241 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 242 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, f2c_complex *, int *, float *, f2c_complex *, int *, f2c_complex *, int *, f2c_complex *, int *, float *, int *, int *)
/* 243 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 244 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 245 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 246 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 247 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 248 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 249 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 250 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 251 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 252 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 253 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 254 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 255 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 256 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 257 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 258 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 259 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, f2c_doublecomplex *, int *, double *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, double *, int *, int *)
/* 260 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 261 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 262 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 263 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 264 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 265 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 266 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 267 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 268 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 269 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 270 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 271 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 272 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 273 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 274 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 275 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 276 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, int *)
/* 277 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 278 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 279 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 280 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 281 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 282 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 283 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 284 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 285 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 286 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, float *, int *, float *, float *, int *, float *, int *, float *, int *, int *, int *)
/* 287 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 288 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 289 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 290 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 291 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 292 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 293 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 294 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 295 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 296 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 297 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 298 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 299 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 300 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 301 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 302 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(char *, int *, int *, float *, int *, float *, int *, int *)
/* 303 */ _CFFI_OP(_CFFI_OP_NOOP, 15),
/* 304 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 305 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 306 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 307 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 308 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 309 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 310 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 311 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 312 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, double *, int *, double *, int *)
/* 313 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 314 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 315 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 316 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 317 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 318 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 319 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, f2c_complex *, int *, f2c_complex *, int *)
/* 320 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 321 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 322 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 323 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 324 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 325 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 326 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *)
/* 327 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 328 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 329 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 330 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 331 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 332 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 333 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, float *, int *, float *, int *)
/* 334 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 335 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 336 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 337 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 338 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 339 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 340 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, double *, int *, double *, double *, int *, int *)
/* 341 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 342 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 343 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 344 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 345 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 346 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 347 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 348 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 349 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 350 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, double *, int *, int *, double *, int *, int *)
/* 351 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 352 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 353 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 354 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 355 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 356 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 357 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 358 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 359 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 360 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, double *, int *, int *, int *)
/* 361 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 362 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 363 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 364 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 365 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 366 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 367 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 368 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, f2c_complex *, int *, int *, f2c_complex *, int *, int *)
/* 369 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 370 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 371 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 372 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 373 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 374 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 375 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 376 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 377 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 378 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, f2c_complex *, int *, int *, int *)
/* 379 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 380 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 381 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 382 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 383 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 384 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 385 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 386 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, f2c_doublecomplex *, int *, int *)
/* 387 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 388 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 389 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 390 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 391 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 392 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 393 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 394 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 395 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 396 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, f2c_doublecomplex *, int *, int *, f2c_doublecomplex *, int *, int *)
/* 397 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 398 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 399 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 400 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 401 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 402 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 403 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 404 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 405 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 406 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, f2c_doublecomplex *, int *, int *, int *)
/* 407 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 408 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 409 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 410 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 411 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 412 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 413 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 414 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, float *, int *, int *, float *, int *, int *)
/* 415 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 416 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 417 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 418 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 419 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 420 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 421 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 422 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 423 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 424 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, float *, int *, int *, int *)
/* 425 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 426 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 427 */ _CFFI_OP(_CFFI_OP_NOOP, 9),
/* 428 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 429 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 430 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 431 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 432 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, int *, double *, int *, double *, double *, int *, int *)
/* 433 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 434 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 435 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 436 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 437 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 438 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 439 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 440 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 441 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 442 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 443 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, int *, double *, int *, double *, int *, double *, double *, int *, double *, int *, int *, int *)
/* 444 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 445 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 446 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 447 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 448 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 449 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 450 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 451 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 452 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 453 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 454 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 455 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 456 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 457 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 458 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 459 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, f2c_doublecomplex *, int *, int *)
/* 460 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 461 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 462 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 463 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 464 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 465 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 466 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 467 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 468 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 469 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 470 */ _CFFI_OP(_CFFI_OP_FUNCTION, 508), // int()(int *, int *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, double *, double *, int *, f2c_doublecomplex *, int *, double *, int *, int *)
/* 471 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 472 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 473 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 474 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 475 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 476 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 477 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 478 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 479 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 480 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 481 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 482 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 483 */ _CFFI_OP(_CFFI_OP_NOOP, 2),
/* 484 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 485 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 486 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 487 */ _CFFI_OP(_CFFI_OP_FUNCTION, 509), // void()(f2c_complex *, int *, f2c_complex *, int *, f2c_complex *, int *)
/* 488 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 489 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 490 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 491 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 492 */ _CFFI_OP(_CFFI_OP_NOOP, 47),
/* 493 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 494 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 495 */ _CFFI_OP(_CFFI_OP_FUNCTION, 509), // void()(f2c_doublecomplex *, int *, f2c_doublecomplex *, int *, f2c_doublecomplex *, int *)
/* 496 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 497 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 498 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 499 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 500 */ _CFFI_OP(_CFFI_OP_NOOP, 62),
/* 501 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 502 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 503 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 2), // char
/* 504 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14), // double
/* 505 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 0), // f2c_complex
/* 506 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 1), // f2c_doublecomplex
/* 507 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 13), // float
/* 508 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7), // int
/* 509 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 0), // void
};

_CFFI_UNUSED_FN
static void _cffi_checkfld_typedef_f2c_complex(f2c_complex *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  { float *tmp = &p->r; (void)tmp; }
  { float *tmp = &p->i; (void)tmp; }
}
struct _cffi_align_typedef_f2c_complex { char x; f2c_complex y; };

_CFFI_UNUSED_FN
static void _cffi_checkfld_typedef_f2c_doublecomplex(f2c_doublecomplex *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  { double *tmp = &p->r; (void)tmp; }
  { double *tmp = &p->i; (void)tmp; }
}
struct _cffi_align_typedef_f2c_doublecomplex { char x; f2c_doublecomplex y; };

static int _cffi_d_ccopy_(int * x0, f2c_complex * x1, int * x2, f2c_complex * x3, int * x4)
{
  return ccopy_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_ccopy_(PyObject *self, PyObject *args)
{
  int * x0;
  f2c_complex * x1;
  int * x2;
  f2c_complex * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "ccopy_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(47), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(47), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = ccopy_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_ccopy_ _cffi_d_ccopy_
#endif

static void _cffi_d_cdotc_(f2c_complex * x0, int * x1, f2c_complex * x2, int * x3, f2c_complex * x4, int * x5)
{
  cdotc_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cdotc_(PyObject *self, PyObject *args)
{
  f2c_complex * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  f2c_complex * x4;
  int * x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "cdotc_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(47), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(47), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { cdotc_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_cdotc_ _cffi_d_cdotc_
#endif

static void _cffi_d_cdotu_(f2c_complex * x0, int * x1, f2c_complex * x2, int * x3, f2c_complex * x4, int * x5)
{
  cdotu_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cdotu_(PyObject *self, PyObject *args)
{
  f2c_complex * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  f2c_complex * x4;
  int * x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "cdotu_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(47), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(47), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { cdotu_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_cdotu_ _cffi_d_cdotu_
#endif

static int _cffi_d_cgeev_(char * x0, char * x1, int * x2, f2c_doublecomplex * x3, int * x4, f2c_doublecomplex * x5, f2c_doublecomplex * x6, int * x7, f2c_doublecomplex * x8, int * x9, f2c_doublecomplex * x10, int * x11, double * x12, int * x13)
{
  return cgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cgeev_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  f2c_doublecomplex * x5;
  f2c_doublecomplex * x6;
  int * x7;
  f2c_doublecomplex * x8;
  int * x9;
  f2c_doublecomplex * x10;
  int * x11;
  double * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "cgeev_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(62), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(62), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (double *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(2), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cgeev_ _cffi_d_cgeev_
#endif

static int _cffi_d_cgemm_(char * x0, char * x1, int * x2, int * x3, int * x4, f2c_complex * x5, f2c_complex * x6, int * x7, f2c_complex * x8, int * x9, f2c_complex * x10, f2c_complex * x11, int * x12)
{
  return cgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cgemm_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  int * x3;
  int * x4;
  f2c_complex * x5;
  f2c_complex * x6;
  int * x7;
  f2c_complex * x8;
  int * x9;
  f2c_complex * x10;
  f2c_complex * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "cgemm_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(47), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(47), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(47), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(47), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(47), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cgemm_ _cffi_d_cgemm_
#endif

static int _cffi_d_cgesdd_(char * x0, int * x1, int * x2, f2c_complex * x3, int * x4, float * x5, f2c_complex * x6, int * x7, f2c_complex * x8, int * x9, f2c_complex * x10, int * x11, float * x12, int * x13, int * x14)
{
  return cgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cgesdd_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  f2c_complex * x3;
  int * x4;
  float * x5;
  f2c_complex * x6;
  int * x7;
  f2c_complex * x8;
  int * x9;
  f2c_complex * x10;
  int * x11;
  float * x12;
  int * x13;
  int * x14;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject *arg14;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 15, "cgesdd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];
  arg14 = aa[14];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(47), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(47), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(47), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(47), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (float *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(9), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg14, (char **)&x14);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x14 = (int *)alloca((size_t)datasize);
    memset((void *)x14, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x14, _cffi_type(1), arg14) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cgesdd_ _cffi_d_cgesdd_
#endif

static int _cffi_d_cgesv_(int * x0, int * x1, f2c_complex * x2, int * x3, int * x4, f2c_complex * x5, int * x6, int * x7)
{
  return cgesv_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cgesv_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  int * x4;
  f2c_complex * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "cgesv_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(47), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cgesv_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cgesv_ _cffi_d_cgesv_
#endif

static int _cffi_d_cgetrf_(int * x0, int * x1, f2c_complex * x2, int * x3, int * x4, int * x5)
{
  return cgetrf_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cgetrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  int * x4;
  int * x5;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "cgetrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cgetrf_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cgetrf_ _cffi_d_cgetrf_
#endif

static int _cffi_d_cheevd_(char * x0, char * x1, int * x2, f2c_complex * x3, int * x4, float * x5, f2c_complex * x6, int * x7, float * x8, int * x9, int * x10, int * x11, int * x12)
{
  return cheevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cheevd_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  f2c_complex * x3;
  int * x4;
  float * x5;
  f2c_complex * x6;
  int * x7;
  float * x8;
  int * x9;
  int * x10;
  int * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "cheevd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(47), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(47), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (float *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(9), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cheevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cheevd_ _cffi_d_cheevd_
#endif

static int _cffi_d_cpotrf_(char * x0, int * x1, f2c_complex * x2, int * x3, int * x4)
{
  return cpotrf_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cpotrf_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "cpotrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cpotrf_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cpotrf_ _cffi_d_cpotrf_
#endif

static int _cffi_d_cpotri_(char * x0, int * x1, f2c_complex * x2, int * x3, int * x4)
{
  return cpotri_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cpotri_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  f2c_complex * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "cpotri_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(47), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cpotri_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cpotri_ _cffi_d_cpotri_
#endif

static int _cffi_d_cpotrs_(char * x0, int * x1, int * x2, f2c_complex * x3, int * x4, f2c_complex * x5, int * x6, int * x7)
{
  return cpotrs_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_cpotrs_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  f2c_complex * x3;
  int * x4;
  f2c_complex * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "cpotrs_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(47), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(47), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_complex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(47), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = cpotrs_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_cpotrs_ _cffi_d_cpotrs_
#endif

static int _cffi_d_dcopy_(int * x0, double * x1, int * x2, double * x3, int * x4)
{
  return dcopy_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dcopy_(PyObject *self, PyObject *args)
{
  int * x0;
  double * x1;
  int * x2;
  double * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "dcopy_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (double *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(2), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dcopy_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dcopy_ _cffi_d_dcopy_
#endif

static double _cffi_d_ddot_(int * x0, double * x1, int * x2, double * x3, int * x4)
{
  return ddot_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_ddot_(PyObject *self, PyObject *args)
{
  int * x0;
  double * x1;
  int * x2;
  double * x3;
  int * x4;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "ddot_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (double *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(2), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = ddot_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_ddot_ _cffi_d_ddot_
#endif

static int _cffi_d_dgeev_(char * x0, char * x1, int * x2, double * x3, int * x4, double * x5, double * x6, double * x7, int * x8, double * x9, int * x10, double * x11, int * x12, int * x13)
{
  return dgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgeev_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  double * x6;
  double * x7;
  int * x8;
  double * x9;
  int * x10;
  double * x11;
  int * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "dgeev_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (double *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(2), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (double *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(2), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (double *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(2), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (double *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(2), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgeev_ _cffi_d_dgeev_
#endif

static int _cffi_d_dgelsd_(int * x0, int * x1, int * x2, double * x3, int * x4, double * x5, int * x6, double * x7, double * x8, int * x9, double * x10, int * x11, int * x12, int * x13)
{
  return dgelsd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgelsd_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  int * x6;
  double * x7;
  double * x8;
  int * x9;
  double * x10;
  int * x11;
  int * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "dgelsd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (double *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(2), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (double *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(2), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (double *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(2), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgelsd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgelsd_ _cffi_d_dgelsd_
#endif

static int _cffi_d_dgemm_(char * x0, char * x1, int * x2, int * x3, int * x4, double * x5, double * x6, int * x7, double * x8, int * x9, double * x10, double * x11, int * x12)
{
  return dgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgemm_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  int * x3;
  int * x4;
  double * x5;
  double * x6;
  int * x7;
  double * x8;
  int * x9;
  double * x10;
  double * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "dgemm_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (double *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(2), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (double *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(2), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (double *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(2), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (double *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(2), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgemm_ _cffi_d_dgemm_
#endif

static int _cffi_d_dgeqrf_(int * x0, int * x1, double * x2, int * x3, double * x4, double * x5, int * x6, int * x7)
{
  return dgeqrf_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgeqrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  double * x2;
  int * x3;
  double * x4;
  double * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "dgeqrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (double *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(2), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (double *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(2), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgeqrf_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgeqrf_ _cffi_d_dgeqrf_
#endif

static int _cffi_d_dgesdd_(char * x0, int * x1, int * x2, double * x3, int * x4, double * x5, double * x6, int * x7, double * x8, int * x9, double * x10, int * x11, int * x12, int * x13)
{
  return dgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgesdd_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  double * x6;
  int * x7;
  double * x8;
  int * x9;
  double * x10;
  int * x11;
  int * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "dgesdd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (double *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(2), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (double *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(2), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (double *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(2), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgesdd_ _cffi_d_dgesdd_
#endif

static int _cffi_d_dgesv_(int * x0, int * x1, double * x2, int * x3, int * x4, double * x5, int * x6, int * x7)
{
  return dgesv_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgesv_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  double * x2;
  int * x3;
  int * x4;
  double * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "dgesv_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (double *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(2), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgesv_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgesv_ _cffi_d_dgesv_
#endif

static int _cffi_d_dgetrf_(int * x0, int * x1, double * x2, int * x3, int * x4, int * x5)
{
  return dgetrf_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dgetrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  double * x2;
  int * x3;
  int * x4;
  int * x5;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "dgetrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (double *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(2), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dgetrf_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dgetrf_ _cffi_d_dgetrf_
#endif

static int _cffi_d_dorgqr_(int * x0, int * x1, int * x2, double * x3, int * x4, double * x5, double * x6, int * x7, int * x8)
{
  return dorgqr_(x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dorgqr_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  double * x6;
  int * x7;
  int * x8;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 9, "dorgqr_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (double *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(2), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dorgqr_(x0, x1, x2, x3, x4, x5, x6, x7, x8); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dorgqr_ _cffi_d_dorgqr_
#endif

static int _cffi_d_dpotrf_(char * x0, int * x1, double * x2, int * x3, int * x4)
{
  return dpotrf_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dpotrf_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  double * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "dpotrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (double *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(2), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dpotrf_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dpotrf_ _cffi_d_dpotrf_
#endif

static int _cffi_d_dpotri_(char * x0, int * x1, double * x2, int * x3, int * x4)
{
  return dpotri_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dpotri_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  double * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "dpotri_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (double *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(2), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dpotri_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dpotri_ _cffi_d_dpotri_
#endif

static int _cffi_d_dpotrs_(char * x0, int * x1, int * x2, double * x3, int * x4, double * x5, int * x6, int * x7)
{
  return dpotrs_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dpotrs_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "dpotrs_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dpotrs_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dpotrs_ _cffi_d_dpotrs_
#endif

static int _cffi_d_dsyevd_(char * x0, char * x1, int * x2, double * x3, int * x4, double * x5, double * x6, int * x7, int * x8, int * x9, int * x10)
{
  return dsyevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_dsyevd_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  double * x3;
  int * x4;
  double * x5;
  double * x6;
  int * x7;
  int * x8;
  int * x9;
  int * x10;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 11, "dsyevd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (double *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(2), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (double *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(2), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = dsyevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_dsyevd_ _cffi_d_dsyevd_
#endif

static int _cffi_d_scopy_(int * x0, float * x1, int * x2, float * x3, int * x4)
{
  return scopy_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_scopy_(PyObject *self, PyObject *args)
{
  int * x0;
  float * x1;
  int * x2;
  float * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "scopy_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (float *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(9), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = scopy_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_scopy_ _cffi_d_scopy_
#endif

static double _cffi_d_sdot_(int * x0, float * x1, int * x2, float * x3, int * x4)
{
  return sdot_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sdot_(PyObject *self, PyObject *args)
{
  int * x0;
  float * x1;
  int * x2;
  float * x3;
  int * x4;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "sdot_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (float *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(9), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sdot_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_sdot_ _cffi_d_sdot_
#endif

static int _cffi_d_sgeev_(char * x0, char * x1, int * x2, float * x3, int * x4, float * x5, float * x6, float * x7, int * x8, float * x9, int * x10, float * x11, int * x12, int * x13)
{
  return sgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sgeev_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  float * x3;
  int * x4;
  float * x5;
  float * x6;
  float * x7;
  int * x8;
  float * x9;
  int * x10;
  float * x11;
  int * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "sgeev_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (float *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(9), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (float *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(9), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (float *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(9), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (float *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(9), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_sgeev_ _cffi_d_sgeev_
#endif

static int _cffi_d_sgemm_(char * x0, char * x1, int * x2, int * x3, int * x4, float * x5, float * x6, int * x7, float * x8, int * x9, float * x10, float * x11, int * x12)
{
  return sgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sgemm_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  int * x3;
  int * x4;
  float * x5;
  float * x6;
  int * x7;
  float * x8;
  int * x9;
  float * x10;
  float * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "sgemm_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (float *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(9), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (float *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(9), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (float *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(9), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (float *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(9), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_sgemm_ _cffi_d_sgemm_
#endif

static int _cffi_d_sgesdd_(char * x0, int * x1, int * x2, float * x3, int * x4, float * x5, float * x6, int * x7, float * x8, int * x9, float * x10, int * x11, int * x12, int * x13)
{
  return sgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sgesdd_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  float * x3;
  int * x4;
  float * x5;
  float * x6;
  int * x7;
  float * x8;
  int * x9;
  float * x10;
  int * x11;
  int * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "sgesdd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (float *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(9), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (float *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(9), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (float *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(9), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_sgesdd_ _cffi_d_sgesdd_
#endif

static int _cffi_d_sgesv_(int * x0, int * x1, float * x2, int * x3, int * x4, float * x5, int * x6, int * x7)
{
  return sgesv_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sgesv_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  float * x2;
  int * x3;
  int * x4;
  float * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "sgesv_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (float *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(9), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sgesv_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_sgesv_ _cffi_d_sgesv_
#endif

static int _cffi_d_sgetrf_(int * x0, int * x1, float * x2, int * x3, int * x4, int * x5)
{
  return sgetrf_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_sgetrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  float * x2;
  int * x3;
  int * x4;
  int * x5;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "sgetrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (float *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(9), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = sgetrf_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_sgetrf_ _cffi_d_sgetrf_
#endif

static int _cffi_d_spotrf_(char * x0, int * x1, float * x2, int * x3, int * x4)
{
  return spotrf_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_spotrf_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  float * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "spotrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (float *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(9), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = spotrf_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_spotrf_ _cffi_d_spotrf_
#endif

static int _cffi_d_spotri_(char * x0, int * x1, float * x2, int * x3, int * x4)
{
  return spotri_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_spotri_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  float * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "spotri_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (float *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(9), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = spotri_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_spotri_ _cffi_d_spotri_
#endif

static int _cffi_d_spotrs_(char * x0, int * x1, int * x2, float * x3, int * x4, float * x5, int * x6, int * x7)
{
  return spotrs_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_spotrs_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  float * x3;
  int * x4;
  float * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "spotrs_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = spotrs_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_spotrs_ _cffi_d_spotrs_
#endif

static int _cffi_d_ssyevd_(char * x0, char * x1, int * x2, float * x3, int * x4, float * x5, float * x6, int * x7, int * x8, int * x9, int * x10)
{
  return ssyevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_ssyevd_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  float * x3;
  int * x4;
  float * x5;
  float * x6;
  int * x7;
  int * x8;
  int * x9;
  int * x10;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 11, "ssyevd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (float *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(9), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (float *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(9), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(9), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (float *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(9), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = ssyevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_ssyevd_ _cffi_d_ssyevd_
#endif

static int _cffi_d_zcopy_(int * x0, f2c_doublecomplex * x1, int * x2, f2c_doublecomplex * x3, int * x4)
{
  return zcopy_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zcopy_(PyObject *self, PyObject *args)
{
  int * x0;
  f2c_doublecomplex * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "zcopy_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(62), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zcopy_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zcopy_ _cffi_d_zcopy_
#endif

static void _cffi_d_zdotc_(f2c_doublecomplex * x0, int * x1, f2c_doublecomplex * x2, int * x3, f2c_doublecomplex * x4, int * x5)
{
  zdotc_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zdotc_(PyObject *self, PyObject *args)
{
  f2c_doublecomplex * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  f2c_doublecomplex * x4;
  int * x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "zdotc_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(62), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(62), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { zdotc_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_zdotc_ _cffi_d_zdotc_
#endif

static void _cffi_d_zdotu_(f2c_doublecomplex * x0, int * x1, f2c_doublecomplex * x2, int * x3, f2c_doublecomplex * x4, int * x5)
{
  zdotu_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zdotu_(PyObject *self, PyObject *args)
{
  f2c_doublecomplex * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  f2c_doublecomplex * x4;
  int * x5;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "zdotu_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(62), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(62), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { zdotu_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_zdotu_ _cffi_d_zdotu_
#endif

static int _cffi_d_zgeev_(char * x0, char * x1, int * x2, f2c_doublecomplex * x3, int * x4, f2c_doublecomplex * x5, f2c_doublecomplex * x6, int * x7, f2c_doublecomplex * x8, int * x9, f2c_doublecomplex * x10, int * x11, double * x12, int * x13)
{
  return zgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgeev_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  f2c_doublecomplex * x5;
  f2c_doublecomplex * x6;
  int * x7;
  f2c_doublecomplex * x8;
  int * x9;
  f2c_doublecomplex * x10;
  int * x11;
  double * x12;
  int * x13;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 14, "zgeev_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(62), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(62), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (double *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(2), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgeev_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgeev_ _cffi_d_zgeev_
#endif

static int _cffi_d_zgelsd_(int * x0, int * x1, int * x2, f2c_doublecomplex * x3, int * x4, f2c_doublecomplex * x5, int * x6, double * x7, double * x8, int * x9, f2c_doublecomplex * x10, int * x11, double * x12, int * x13, int * x14)
{
  return zgelsd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgelsd_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  f2c_doublecomplex * x5;
  int * x6;
  double * x7;
  double * x8;
  int * x9;
  f2c_doublecomplex * x10;
  int * x11;
  double * x12;
  int * x13;
  int * x14;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject *arg14;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 15, "zgelsd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];
  arg14 = aa[14];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (double *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(2), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (double *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(2), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(62), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (double *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(2), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg14, (char **)&x14);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x14 = (int *)alloca((size_t)datasize);
    memset((void *)x14, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x14, _cffi_type(1), arg14) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgelsd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgelsd_ _cffi_d_zgelsd_
#endif

static int _cffi_d_zgemm_(char * x0, char * x1, int * x2, int * x3, int * x4, f2c_doublecomplex * x5, f2c_doublecomplex * x6, int * x7, f2c_doublecomplex * x8, int * x9, f2c_doublecomplex * x10, f2c_doublecomplex * x11, int * x12)
{
  return zgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgemm_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  int * x3;
  int * x4;
  f2c_doublecomplex * x5;
  f2c_doublecomplex * x6;
  int * x7;
  f2c_doublecomplex * x8;
  int * x9;
  f2c_doublecomplex * x10;
  f2c_doublecomplex * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "zgemm_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(62), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(62), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(62), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgemm_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgemm_ _cffi_d_zgemm_
#endif

static int _cffi_d_zgeqrf_(int * x0, int * x1, f2c_doublecomplex * x2, int * x3, f2c_doublecomplex * x4, f2c_doublecomplex * x5, int * x6, int * x7)
{
  return zgeqrf_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgeqrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  f2c_doublecomplex * x4;
  f2c_doublecomplex * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "zgeqrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(62), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgeqrf_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgeqrf_ _cffi_d_zgeqrf_
#endif

static int _cffi_d_zgesdd_(char * x0, int * x1, int * x2, f2c_doublecomplex * x3, int * x4, double * x5, f2c_doublecomplex * x6, int * x7, f2c_doublecomplex * x8, int * x9, f2c_doublecomplex * x10, int * x11, double * x12, int * x13, int * x14)
{
  return zgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgesdd_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  double * x5;
  f2c_doublecomplex * x6;
  int * x7;
  f2c_doublecomplex * x8;
  int * x9;
  f2c_doublecomplex * x10;
  int * x11;
  double * x12;
  int * x13;
  int * x14;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject *arg13;
  PyObject *arg14;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 15, "zgesdd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];
  arg13 = aa[13];
  arg14 = aa[14];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(62), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(62), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (double *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(2), arg12) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg13, (char **)&x13);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x13 = (int *)alloca((size_t)datasize);
    memset((void *)x13, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x13, _cffi_type(1), arg13) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg14, (char **)&x14);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x14 = (int *)alloca((size_t)datasize);
    memset((void *)x14, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x14, _cffi_type(1), arg14) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgesdd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgesdd_ _cffi_d_zgesdd_
#endif

static int _cffi_d_zgesv_(int * x0, int * x1, f2c_doublecomplex * x2, int * x3, int * x4, f2c_doublecomplex * x5, int * x6, int * x7)
{
  return zgesv_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgesv_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  int * x4;
  f2c_doublecomplex * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "zgesv_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgesv_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgesv_ _cffi_d_zgesv_
#endif

static int _cffi_d_zgetrf_(int * x0, int * x1, f2c_doublecomplex * x2, int * x3, int * x4, int * x5)
{
  return zgetrf_(x0, x1, x2, x3, x4, x5);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zgetrf_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  int * x4;
  int * x5;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 6, "zgetrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (int *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(1), arg5) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zgetrf_(x0, x1, x2, x3, x4, x5); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zgetrf_ _cffi_d_zgetrf_
#endif

static int _cffi_d_zheevd_(char * x0, char * x1, int * x2, f2c_doublecomplex * x3, int * x4, double * x5, f2c_doublecomplex * x6, int * x7, double * x8, int * x9, int * x10, int * x11, int * x12)
{
  return zheevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zheevd_(PyObject *self, PyObject *args)
{
  char * x0;
  char * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  double * x5;
  f2c_doublecomplex * x6;
  int * x7;
  double * x8;
  int * x9;
  int * x10;
  int * x11;
  int * x12;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject *arg9;
  PyObject *arg10;
  PyObject *arg11;
  PyObject *arg12;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 13, "zheevd_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];
  arg9 = aa[9];
  arg10 = aa[10];
  arg11 = aa[11];
  arg12 = aa[12];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (char *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(15), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (double *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(2), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(2), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (double *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(2), arg8) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg9, (char **)&x9);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x9 = (int *)alloca((size_t)datasize);
    memset((void *)x9, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x9, _cffi_type(1), arg9) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg10, (char **)&x10);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x10 = (int *)alloca((size_t)datasize);
    memset((void *)x10, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x10, _cffi_type(1), arg10) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg11, (char **)&x11);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x11 = (int *)alloca((size_t)datasize);
    memset((void *)x11, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x11, _cffi_type(1), arg11) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg12, (char **)&x12);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x12 = (int *)alloca((size_t)datasize);
    memset((void *)x12, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x12, _cffi_type(1), arg12) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zheevd_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zheevd_ _cffi_d_zheevd_
#endif

static int _cffi_d_zpotrf_(char * x0, int * x1, f2c_doublecomplex * x2, int * x3, int * x4)
{
  return zpotrf_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zpotrf_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "zpotrf_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zpotrf_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zpotrf_ _cffi_d_zpotrf_
#endif

static int _cffi_d_zpotri_(char * x0, int * x1, f2c_doublecomplex * x2, int * x3, int * x4)
{
  return zpotri_(x0, x1, x2, x3, x4);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zpotri_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  f2c_doublecomplex * x2;
  int * x3;
  int * x4;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 5, "zpotri_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(62), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (int *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zpotri_(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zpotri_ _cffi_d_zpotri_
#endif

static int _cffi_d_zpotrs_(char * x0, int * x1, int * x2, f2c_doublecomplex * x3, int * x4, f2c_doublecomplex * x5, int * x6, int * x7)
{
  return zpotrs_(x0, x1, x2, x3, x4, x5, x6, x7);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zpotrs_(PyObject *self, PyObject *args)
{
  char * x0;
  int * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  f2c_doublecomplex * x5;
  int * x6;
  int * x7;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 8, "zpotrs_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(15), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (char *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(15), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (int *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(1), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zpotrs_(x0, x1, x2, x3, x4, x5, x6, x7); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zpotrs_ _cffi_d_zpotrs_
#endif

static int _cffi_d_zungqr_(int * x0, int * x1, int * x2, f2c_doublecomplex * x3, int * x4, f2c_doublecomplex * x5, f2c_doublecomplex * x6, int * x7, int * x8)
{
  return zungqr_(x0, x1, x2, x3, x4, x5, x6, x7, x8);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_zungqr_(PyObject *self, PyObject *args)
{
  int * x0;
  int * x1;
  int * x2;
  f2c_doublecomplex * x3;
  int * x4;
  f2c_doublecomplex * x5;
  f2c_doublecomplex * x6;
  int * x7;
  int * x8;
  Py_ssize_t datasize;
  int result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;
  PyObject *arg5;
  PyObject *arg6;
  PyObject *arg7;
  PyObject *arg8;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 9, "zungqr_");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];
  arg4 = aa[4];
  arg5 = aa[5];
  arg6 = aa[6];
  arg7 = aa[7];
  arg8 = aa[8];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (int *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (int *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (int *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(62), arg3) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x4 = (int *)alloca((size_t)datasize);
    memset((void *)x4, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x4, _cffi_type(1), arg4) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg5, (char **)&x5);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x5 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x5, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x5, _cffi_type(62), arg5) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(62), arg6, (char **)&x6);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x6 = (f2c_doublecomplex *)alloca((size_t)datasize);
    memset((void *)x6, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x6, _cffi_type(62), arg6) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg7, (char **)&x7);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x7 = (int *)alloca((size_t)datasize);
    memset((void *)x7, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x7, _cffi_type(1), arg7) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg8, (char **)&x8);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x8 = (int *)alloca((size_t)datasize);
    memset((void *)x8, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x8, _cffi_type(1), arg8) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = zungqr_(x0, x1, x2, x3, x4, x5, x6, x7, x8); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, int);
}
#else
#  define _cffi_f_zungqr_ _cffi_d_zungqr_
#endif

static const struct _cffi_global_s _cffi_globals[] = {
  { "ccopy_", (void *)_cffi_f_ccopy_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 319), (void *)_cffi_d_ccopy_ },
  { "cdotc_", (void *)_cffi_f_cdotc_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 487), (void *)_cffi_d_cdotc_ },
  { "cdotu_", (void *)_cffi_f_cdotu_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 487), (void *)_cffi_d_cdotu_ },
  { "cgeev_", (void *)_cffi_f_cgeev_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 73), (void *)_cffi_d_cgeev_ },
  { "cgemm_", (void *)_cffi_f_cgemm_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 133), (void *)_cffi_d_cgemm_ },
  { "cgesdd_", (void *)_cffi_f_cgesdd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 242), (void *)_cffi_d_cgesdd_ },
  { "cgesv_", (void *)_cffi_f_cgesv_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 368), (void *)_cffi_d_cgesv_ },
  { "cgetrf_", (void *)_cffi_f_cgetrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 378), (void *)_cffi_d_cgetrf_ },
  { "cheevd_", (void *)_cffi_f_cheevd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 43), (void *)_cffi_d_cheevd_ },
  { "cpotrf_", (void *)_cffi_f_cpotrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 185), (void *)_cffi_d_cpotrf_ },
  { "cpotri_", (void *)_cffi_f_cpotri_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 185), (void *)_cffi_d_cpotri_ },
  { "cpotrs_", (void *)_cffi_f_cpotrs_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 232), (void *)_cffi_d_cpotrs_ },
  { "dcopy_", (void *)_cffi_f_dcopy_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 312), (void *)_cffi_d_dcopy_ },
  { "ddot_", (void *)_cffi_f_ddot_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 0), (void *)_cffi_d_ddot_ },
  { "dgeev_", (void *)_cffi_f_dgeev_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 14), (void *)_cffi_d_dgeev_ },
  { "dgelsd_", (void *)_cffi_f_dgelsd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 443), (void *)_cffi_d_dgelsd_ },
  { "dgemm_", (void *)_cffi_f_dgemm_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 118), (void *)_cffi_d_dgemm_ },
  { "dgeqrf_", (void *)_cffi_f_dgeqrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 340), (void *)_cffi_d_dgeqrf_ },
  { "dgesdd_", (void *)_cffi_f_dgesdd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 206), (void *)_cffi_d_dgesdd_ },
  { "dgesv_", (void *)_cffi_f_dgesv_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 350), (void *)_cffi_d_dgesv_ },
  { "dgetrf_", (void *)_cffi_f_dgetrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 360), (void *)_cffi_d_dgetrf_ },
  { "dorgqr_", (void *)_cffi_f_dorgqr_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 432), (void *)_cffi_d_dorgqr_ },
  { "dpotrf_", (void *)_cffi_f_dpotrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 178), (void *)_cffi_d_dpotrf_ },
  { "dpotri_", (void *)_cffi_f_dpotri_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 178), (void *)_cffi_d_dpotri_ },
  { "dpotrs_", (void *)_cffi_f_dpotrs_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 222), (void *)_cffi_d_dpotrs_ },
  { "dsyevd_", (void *)_cffi_f_dsyevd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 30), (void *)_cffi_d_dsyevd_ },
  { "scopy_", (void *)_cffi_f_scopy_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 333), (void *)_cffi_d_scopy_ },
  { "sdot_", (void *)_cffi_f_sdot_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_sdot_ },
  { "sgeev_", (void *)_cffi_f_sgeev_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 89), (void *)_cffi_d_sgeev_ },
  { "sgemm_", (void *)_cffi_f_sgemm_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 163), (void *)_cffi_d_sgemm_ },
  { "sgesdd_", (void *)_cffi_f_sgesdd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 286), (void *)_cffi_d_sgesdd_ },
  { "sgesv_", (void *)_cffi_f_sgesv_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 414), (void *)_cffi_d_sgesv_ },
  { "sgetrf_", (void *)_cffi_f_sgetrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 424), (void *)_cffi_d_sgetrf_ },
  { "spotrf_", (void *)_cffi_f_spotrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 199), (void *)_cffi_d_spotrf_ },
  { "spotri_", (void *)_cffi_f_spotri_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 199), (void *)_cffi_d_spotri_ },
  { "spotrs_", (void *)_cffi_f_spotrs_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 302), (void *)_cffi_d_spotrs_ },
  { "ssyevd_", (void *)_cffi_f_ssyevd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 105), (void *)_cffi_d_ssyevd_ },
  { "zcopy_", (void *)_cffi_f_zcopy_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 326), (void *)_cffi_d_zcopy_ },
  { "zdotc_", (void *)_cffi_f_zdotc_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 495), (void *)_cffi_d_zdotc_ },
  { "zdotu_", (void *)_cffi_f_zdotu_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 495), (void *)_cffi_d_zdotu_ },
  { "zgeev_", (void *)_cffi_f_zgeev_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 73), (void *)_cffi_d_zgeev_ },
  { "zgelsd_", (void *)_cffi_f_zgelsd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 470), (void *)_cffi_d_zgelsd_ },
  { "zgemm_", (void *)_cffi_f_zgemm_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 148), (void *)_cffi_d_zgemm_ },
  { "zgeqrf_", (void *)_cffi_f_zgeqrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 386), (void *)_cffi_d_zgeqrf_ },
  { "zgesdd_", (void *)_cffi_f_zgesdd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 259), (void *)_cffi_d_zgesdd_ },
  { "zgesv_", (void *)_cffi_f_zgesv_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 396), (void *)_cffi_d_zgesv_ },
  { "zgetrf_", (void *)_cffi_f_zgetrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 406), (void *)_cffi_d_zgetrf_ },
  { "zheevd_", (void *)_cffi_f_zheevd_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 58), (void *)_cffi_d_zheevd_ },
  { "zpotrf_", (void *)_cffi_f_zpotrf_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 192), (void *)_cffi_d_zpotrf_ },
  { "zpotri_", (void *)_cffi_f_zpotri_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 192), (void *)_cffi_d_zpotri_ },
  { "zpotrs_", (void *)_cffi_f_zpotrs_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 276), (void *)_cffi_d_zpotrs_ },
  { "zungqr_", (void *)_cffi_f_zungqr_, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 459), (void *)_cffi_d_zungqr_ },
};

static const struct _cffi_field_s _cffi_fields[] = {
  { "r", offsetof(f2c_complex, r),
         sizeof(((f2c_complex *)0)->r),
         _CFFI_OP(_CFFI_OP_NOOP, 507) },
  { "i", offsetof(f2c_complex, i),
         sizeof(((f2c_complex *)0)->i),
         _CFFI_OP(_CFFI_OP_NOOP, 507) },
  { "r", offsetof(f2c_doublecomplex, r),
         sizeof(((f2c_doublecomplex *)0)->r),
         _CFFI_OP(_CFFI_OP_NOOP, 504) },
  { "i", offsetof(f2c_doublecomplex, i),
         sizeof(((f2c_doublecomplex *)0)->i),
         _CFFI_OP(_CFFI_OP_NOOP, 504) },
};

static const struct _cffi_struct_union_s _cffi_struct_unions[] = {
  { "$f2c_complex", 505, _CFFI_F_CHECK_FIELDS,
    sizeof(f2c_complex), offsetof(struct _cffi_align_typedef_f2c_complex, y), 0, 2 },
  { "$f2c_doublecomplex", 506, _CFFI_F_CHECK_FIELDS,
    sizeof(f2c_doublecomplex), offsetof(struct _cffi_align_typedef_f2c_doublecomplex, y), 2, 2 },
};

static const struct _cffi_typename_s _cffi_typenames[] = {
  { "f2c_complex", 505 },
  { "f2c_doublecomplex", 506 },
};

static const struct _cffi_type_context_s _cffi_type_context = {
  _cffi_types,
  _cffi_globals,
  _cffi_fields,
  _cffi_struct_unions,
  NULL,  /* no enums */
  _cffi_typenames,
  52,  /* num_globals */
  2,  /* num_struct_unions */
  0,  /* num_enums */
  2,  /* num_typenames */
  NULL,  /* no includes */
  510,  /* num_types */
  0,  /* flags */
};

#ifdef PYPY_VERSION
PyMODINIT_FUNC
_cffi_pypyinit__lapack_lite(const void *p[])
{
    p[0] = (const void *)0x2601;
    p[1] = &_cffi_type_context;
}
#  ifdef _MSC_VER
     PyMODINIT_FUNC
#  if PY_MAJOR_VERSION >= 3
     PyInit__lapack_lite(void) { return NULL; }
#  else
     init_lapack_lite(void) { }
#  endif
#  endif
#elif PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__lapack_lite(void)
{
  return _cffi_init("numpy.linalg._lapack_lite", 0x2601, &_cffi_type_context);
}
#else
PyMODINIT_FUNC
init_lapack_lite(void)
{
  _cffi_init("numpy.linalg._lapack_lite", 0x2601, &_cffi_type_context);
}
#endif
