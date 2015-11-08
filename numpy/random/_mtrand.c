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


#include "mtrand/initarray.h"
#include "mtrand/randomkit.h"
#include "mtrand/distributions.h"


/************************************************************/

static void *_cffi_types[] = {
/*  0 */ _CFFI_OP(_CFFI_OP_FUNCTION, 5), // double()(rk_state *)
/*  1 */ _CFFI_OP(_CFFI_OP_POINTER, 81), // rk_state *
/*  2 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  3 */ _CFFI_OP(_CFFI_OP_FUNCTION, 5), // double()(rk_state *, double)
/*  4 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  5 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14), // double
/*  6 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/*  7 */ _CFFI_OP(_CFFI_OP_FUNCTION, 5), // double()(rk_state *, double, double)
/*  8 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/*  9 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 10 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 11 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 12 */ _CFFI_OP(_CFFI_OP_FUNCTION, 5), // double()(rk_state *, double, double, double)
/* 13 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 14 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 15 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 16 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 17 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 18 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // long()(rk_state *)
/* 19 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 20 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 21 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // long()(rk_state *, double)
/* 22 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 23 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 24 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 25 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // long()(rk_state *, double, double)
/* 26 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 27 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 28 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 29 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 30 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // long()(rk_state *, long, double)
/* 31 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 32 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 9), // long
/* 33 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 14),
/* 34 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 35 */ _CFFI_OP(_CFFI_OP_FUNCTION, 32), // long()(rk_state *, long, long, long)
/* 36 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 37 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 9),
/* 38 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 9),
/* 39 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 9),
/* 40 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 41 */ _CFFI_OP(_CFFI_OP_FUNCTION, 80), // rk_error()(rk_state *)
/* 42 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 43 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 44 */ _CFFI_OP(_CFFI_OP_FUNCTION, 80), // rk_error()(void *, size_t, int)
/* 45 */ _CFFI_OP(_CFFI_OP_POINTER, 84), // void *
/* 46 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 28), // size_t
/* 47 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7), // int
/* 48 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 49 */ _CFFI_OP(_CFFI_OP_FUNCTION, 80), // rk_error()(void *, size_t, int, rk_state *)
/* 50 */ _CFFI_OP(_CFFI_OP_NOOP, 45),
/* 51 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 28),
/* 52 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 7),
/* 53 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 54 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 55 */ _CFFI_OP(_CFFI_OP_FUNCTION, 59), // unsigned long()(rk_state *)
/* 56 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 57 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 58 */ _CFFI_OP(_CFFI_OP_FUNCTION, 59), // unsigned long()(unsigned long, rk_state *)
/* 59 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 10), // unsigned long
/* 60 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 61 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 62 */ _CFFI_OP(_CFFI_OP_FUNCTION, 84), // void()(rk_state *, unsigned long *, intptr_t)
/* 63 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 64 */ _CFFI_OP(_CFFI_OP_POINTER, 59), // unsigned long *
/* 65 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 25), // intptr_t
/* 66 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 67 */ _CFFI_OP(_CFFI_OP_FUNCTION, 84), // void()(unsigned long, rk_state *)
/* 68 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 10),
/* 69 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 70 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 71 */ _CFFI_OP(_CFFI_OP_FUNCTION, 84), // void()(void *, size_t, rk_state *)
/* 72 */ _CFFI_OP(_CFFI_OP_NOOP, 45),
/* 73 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 28),
/* 74 */ _CFFI_OP(_CFFI_OP_NOOP, 1),
/* 75 */ _CFFI_OP(_CFFI_OP_FUNCTION_END, 0),
/* 76 */ _CFFI_OP(_CFFI_OP_POINTER, 79), // char *
/* 77 */ _CFFI_OP(_CFFI_OP_ARRAY, 76), // char *[2]
/* 78 */ (_cffi_opcode_t)(2),
/* 79 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 2), // char
/* 80 */ _CFFI_OP(_CFFI_OP_ENUM, 0), // rk_error
/* 81 */ _CFFI_OP(_CFFI_OP_STRUCT_UNION, 0), // rk_state
/* 82 */ _CFFI_OP(_CFFI_OP_ARRAY, 59), // unsigned long[624]
/* 83 */ (_cffi_opcode_t)(624),
/* 84 */ _CFFI_OP(_CFFI_OP_PRIMITIVE, 0), // void
};

static int _cffi_const_RK_NOERR(unsigned long long *o)
{
  int n = (RK_NOERR) <= 0;
  *o = (unsigned long long)((RK_NOERR) << 0);  /* check that RK_NOERR is an integer */
  return n;
}

static int _cffi_const_RK_ENODEV(unsigned long long *o)
{
  int n = (RK_ENODEV) <= 0;
  *o = (unsigned long long)((RK_ENODEV) << 0);  /* check that RK_ENODEV is an integer */
  return n;
}

static int _cffi_const_RK_ERR_MAX(unsigned long long *o)
{
  int n = (RK_ERR_MAX) <= 0;
  *o = (unsigned long long)((RK_ERR_MAX) << 0);  /* check that RK_ERR_MAX is an integer */
  return n;
}

static void _cffi_d_init_by_array(rk_state * x0, unsigned long * x1, intptr_t x2)
{
  init_by_array(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_init_by_array(PyObject *self, PyObject *args)
{
  rk_state * x0;
  unsigned long * x1;
  intptr_t x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "init_by_array");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(64), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (unsigned long *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(64), arg1) < 0)
      return NULL;
  }

  x2 = _cffi_to_c_int(arg2, intptr_t);
  if (x2 == (intptr_t)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { init_by_array(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_init_by_array _cffi_d_init_by_array
#endif

static rk_error _cffi_d_rk_altfill(void * x0, size_t x1, int x2, rk_state * x3)
{
  return rk_altfill(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_altfill(PyObject *self, PyObject *args)
{
  void * x0;
  size_t x1;
  int x2;
  rk_state * x3;
  Py_ssize_t datasize;
  rk_error result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_altfill");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(45), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(45), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, size_t);
  if (x1 == (size_t)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg3, (char **)&x3);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x3 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x3, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x3, _cffi_type(1), arg3) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_altfill(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(80));
}
#else
#  define _cffi_f_rk_altfill _cffi_d_rk_altfill
#endif

static double _cffi_d_rk_beta(rk_state * x0, double x1, double x2)
{
  return rk_beta(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_beta(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_beta");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_beta(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_beta _cffi_d_rk_beta
#endif

static long _cffi_d_rk_binomial(rk_state * x0, long x1, double x2)
{
  return rk_binomial(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_binomial(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  double x2;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_binomial");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_binomial(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_binomial _cffi_d_rk_binomial
#endif

static long _cffi_d_rk_binomial_btpe(rk_state * x0, long x1, double x2)
{
  return rk_binomial_btpe(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_binomial_btpe(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  double x2;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_binomial_btpe");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_binomial_btpe(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_binomial_btpe _cffi_d_rk_binomial_btpe
#endif

static long _cffi_d_rk_binomial_inversion(rk_state * x0, long x1, double x2)
{
  return rk_binomial_inversion(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_binomial_inversion(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  double x2;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_binomial_inversion");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_binomial_inversion(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_binomial_inversion _cffi_d_rk_binomial_inversion
#endif

static double _cffi_d_rk_chisquare(rk_state * x0, double x1)
{
  return rk_chisquare(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_chisquare(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_chisquare");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_chisquare(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_chisquare _cffi_d_rk_chisquare
#endif

static rk_error _cffi_d_rk_devfill(void * x0, size_t x1, int x2)
{
  return rk_devfill(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_devfill(PyObject *self, PyObject *args)
{
  void * x0;
  size_t x1;
  int x2;
  Py_ssize_t datasize;
  rk_error result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_devfill");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(45), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(45), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, size_t);
  if (x1 == (size_t)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, int);
  if (x2 == (int)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_devfill(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(80));
}
#else
#  define _cffi_f_rk_devfill _cffi_d_rk_devfill
#endif

static double _cffi_d_rk_double(rk_state * x0)
{
  return rk_double(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_double(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  double result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_double(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_double _cffi_d_rk_double
#endif

static double _cffi_d_rk_exponential(rk_state * x0, double x1)
{
  return rk_exponential(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_exponential(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_exponential");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_exponential(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_exponential _cffi_d_rk_exponential
#endif

static double _cffi_d_rk_f(rk_state * x0, double x1, double x2)
{
  return rk_f(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_f(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_f");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_f(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_f _cffi_d_rk_f
#endif

static void _cffi_d_rk_fill(void * x0, size_t x1, rk_state * x2)
{
  rk_fill(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_fill(PyObject *self, PyObject *args)
{
  void * x0;
  size_t x1;
  rk_state * x2;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_fill");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(45), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (void *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(45), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, size_t);
  if (x1 == (size_t)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg2, (char **)&x2);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x2 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x2, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x2, _cffi_type(1), arg2) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { rk_fill(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_rk_fill _cffi_d_rk_fill
#endif

static double _cffi_d_rk_gamma(rk_state * x0, double x1, double x2)
{
  return rk_gamma(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_gamma(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_gamma");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_gamma(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_gamma _cffi_d_rk_gamma
#endif

static double _cffi_d_rk_gauss(rk_state * x0)
{
  return rk_gauss(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_gauss(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  double result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_gauss(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_gauss _cffi_d_rk_gauss
#endif

static long _cffi_d_rk_geometric(rk_state * x0, double x1)
{
  return rk_geometric(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_geometric(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_geometric");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_geometric(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_geometric _cffi_d_rk_geometric
#endif

static long _cffi_d_rk_geometric_inversion(rk_state * x0, double x1)
{
  return rk_geometric_inversion(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_geometric_inversion(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_geometric_inversion");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_geometric_inversion(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_geometric_inversion _cffi_d_rk_geometric_inversion
#endif

static long _cffi_d_rk_geometric_search(rk_state * x0, double x1)
{
  return rk_geometric_search(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_geometric_search(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_geometric_search");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_geometric_search(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_geometric_search _cffi_d_rk_geometric_search
#endif

static double _cffi_d_rk_gumbel(rk_state * x0, double x1, double x2)
{
  return rk_gumbel(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_gumbel(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_gumbel");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_gumbel(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_gumbel _cffi_d_rk_gumbel
#endif

static long _cffi_d_rk_hypergeometric(rk_state * x0, long x1, long x2, long x3)
{
  return rk_hypergeometric(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_hypergeometric(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  long x2;
  long x3;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_hypergeometric");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, long);
  if (x2 == (long)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, long);
  if (x3 == (long)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_hypergeometric(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_hypergeometric _cffi_d_rk_hypergeometric
#endif

static long _cffi_d_rk_hypergeometric_hrua(rk_state * x0, long x1, long x2, long x3)
{
  return rk_hypergeometric_hrua(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_hypergeometric_hrua(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  long x2;
  long x3;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_hypergeometric_hrua");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, long);
  if (x2 == (long)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, long);
  if (x3 == (long)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_hypergeometric_hrua(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_hypergeometric_hrua _cffi_d_rk_hypergeometric_hrua
#endif

static long _cffi_d_rk_hypergeometric_hyp(rk_state * x0, long x1, long x2, long x3)
{
  return rk_hypergeometric_hyp(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_hypergeometric_hyp(PyObject *self, PyObject *args)
{
  rk_state * x0;
  long x1;
  long x2;
  long x3;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_hypergeometric_hyp");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = _cffi_to_c_int(arg1, long);
  if (x1 == (long)-1 && PyErr_Occurred())
    return NULL;

  x2 = _cffi_to_c_int(arg2, long);
  if (x2 == (long)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, long);
  if (x3 == (long)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_hypergeometric_hyp(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_hypergeometric_hyp _cffi_d_rk_hypergeometric_hyp
#endif

static unsigned long _cffi_d_rk_interval(unsigned long x0, rk_state * x1)
{
  return rk_interval(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_interval(PyObject *self, PyObject *args)
{
  unsigned long x0;
  rk_state * x1;
  Py_ssize_t datasize;
  unsigned long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_interval");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, unsigned long);
  if (x0 == (unsigned long)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_interval(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned long);
}
#else
#  define _cffi_f_rk_interval _cffi_d_rk_interval
#endif

static double _cffi_d_rk_laplace(rk_state * x0, double x1, double x2)
{
  return rk_laplace(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_laplace(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_laplace");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_laplace(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_laplace _cffi_d_rk_laplace
#endif

static double _cffi_d_rk_logistic(rk_state * x0, double x1, double x2)
{
  return rk_logistic(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_logistic(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_logistic");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_logistic(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_logistic _cffi_d_rk_logistic
#endif

static double _cffi_d_rk_lognormal(rk_state * x0, double x1, double x2)
{
  return rk_lognormal(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_lognormal(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_lognormal");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_lognormal(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_lognormal _cffi_d_rk_lognormal
#endif

static long _cffi_d_rk_logseries(rk_state * x0, double x1)
{
  return rk_logseries(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_logseries(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_logseries");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_logseries(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_logseries _cffi_d_rk_logseries
#endif

static long _cffi_d_rk_long(rk_state * x0)
{
  return rk_long(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_long(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  long result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_long(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_long _cffi_d_rk_long
#endif

static long _cffi_d_rk_negative_binomial(rk_state * x0, double x1, double x2)
{
  return rk_negative_binomial(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_negative_binomial(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_negative_binomial");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_negative_binomial(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_negative_binomial _cffi_d_rk_negative_binomial
#endif

static double _cffi_d_rk_noncentral_chisquare(rk_state * x0, double x1, double x2)
{
  return rk_noncentral_chisquare(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_noncentral_chisquare(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_noncentral_chisquare");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_noncentral_chisquare(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_noncentral_chisquare _cffi_d_rk_noncentral_chisquare
#endif

static double _cffi_d_rk_noncentral_f(rk_state * x0, double x1, double x2, double x3)
{
  return rk_noncentral_f(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_noncentral_f(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  double x3;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_noncentral_f");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  x3 = (double)_cffi_to_c_double(arg3);
  if (x3 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_noncentral_f(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_noncentral_f _cffi_d_rk_noncentral_f
#endif

static double _cffi_d_rk_normal(rk_state * x0, double x1, double x2)
{
  return rk_normal(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_normal(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_normal");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_normal(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_normal _cffi_d_rk_normal
#endif

static double _cffi_d_rk_pareto(rk_state * x0, double x1)
{
  return rk_pareto(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_pareto(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_pareto");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_pareto(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_pareto _cffi_d_rk_pareto
#endif

static long _cffi_d_rk_poisson(rk_state * x0, double x1)
{
  return rk_poisson(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_poisson(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_poisson");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_poisson(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_poisson _cffi_d_rk_poisson
#endif

static long _cffi_d_rk_poisson_mult(rk_state * x0, double x1)
{
  return rk_poisson_mult(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_poisson_mult(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_poisson_mult");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_poisson_mult(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_poisson_mult _cffi_d_rk_poisson_mult
#endif

static long _cffi_d_rk_poisson_ptrs(rk_state * x0, double x1)
{
  return rk_poisson_ptrs(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_poisson_ptrs(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_poisson_ptrs");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_poisson_ptrs(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_poisson_ptrs _cffi_d_rk_poisson_ptrs
#endif

static double _cffi_d_rk_power(rk_state * x0, double x1)
{
  return rk_power(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_power(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_power");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_power(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_power _cffi_d_rk_power
#endif

static unsigned long _cffi_d_rk_random(rk_state * x0)
{
  return rk_random(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_random(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  unsigned long result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_random(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned long);
}
#else
#  define _cffi_f_rk_random _cffi_d_rk_random
#endif

static rk_error _cffi_d_rk_randomseed(rk_state * x0)
{
  return rk_randomseed(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_randomseed(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  rk_error result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_randomseed(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_deref((char *)&result, _cffi_type(80));
}
#else
#  define _cffi_f_rk_randomseed _cffi_d_rk_randomseed
#endif

static double _cffi_d_rk_rayleigh(rk_state * x0, double x1)
{
  return rk_rayleigh(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_rayleigh(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_rayleigh");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_rayleigh(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_rayleigh _cffi_d_rk_rayleigh
#endif

static void _cffi_d_rk_seed(unsigned long x0, rk_state * x1)
{
  rk_seed(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_seed(PyObject *self, PyObject *args)
{
  unsigned long x0;
  rk_state * x1;
  Py_ssize_t datasize;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_seed");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  x0 = _cffi_to_c_int(arg0, unsigned long);
  if (x0 == (unsigned long)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg1, (char **)&x1);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x1 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x1, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x1, _cffi_type(1), arg1) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { rk_seed(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  Py_INCREF(Py_None);
  return Py_None;
}
#else
#  define _cffi_f_rk_seed _cffi_d_rk_seed
#endif

static double _cffi_d_rk_standard_cauchy(rk_state * x0)
{
  return rk_standard_cauchy(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_standard_cauchy(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  double result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_standard_cauchy(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_standard_cauchy _cffi_d_rk_standard_cauchy
#endif

static double _cffi_d_rk_standard_exponential(rk_state * x0)
{
  return rk_standard_exponential(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_standard_exponential(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  double result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_standard_exponential(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_standard_exponential _cffi_d_rk_standard_exponential
#endif

static double _cffi_d_rk_standard_gamma(rk_state * x0, double x1)
{
  return rk_standard_gamma(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_standard_gamma(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_standard_gamma");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_standard_gamma(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_standard_gamma _cffi_d_rk_standard_gamma
#endif

static double _cffi_d_rk_standard_t(rk_state * x0, double x1)
{
  return rk_standard_t(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_standard_t(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_standard_t");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_standard_t(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_standard_t _cffi_d_rk_standard_t
#endif

static double _cffi_d_rk_triangular(rk_state * x0, double x1, double x2, double x3)
{
  return rk_triangular(x0, x1, x2, x3);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_triangular(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  double x3;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 4, "rk_triangular");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];
  arg3 = aa[3];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  x3 = (double)_cffi_to_c_double(arg3);
  if (x3 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_triangular(x0, x1, x2, x3); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_triangular _cffi_d_rk_triangular
#endif

static unsigned long _cffi_d_rk_ulong(rk_state * x0)
{
  return rk_ulong(x0);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_ulong(PyObject *self, PyObject *arg0)
{
  rk_state * x0;
  Py_ssize_t datasize;
  unsigned long result;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_ulong(x0); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, unsigned long);
}
#else
#  define _cffi_f_rk_ulong _cffi_d_rk_ulong
#endif

static double _cffi_d_rk_uniform(rk_state * x0, double x1, double x2)
{
  return rk_uniform(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_uniform(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_uniform");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_uniform(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_uniform _cffi_d_rk_uniform
#endif

static double _cffi_d_rk_vonmises(rk_state * x0, double x1, double x2)
{
  return rk_vonmises(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_vonmises(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_vonmises");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_vonmises(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_vonmises _cffi_d_rk_vonmises
#endif

static double _cffi_d_rk_wald(rk_state * x0, double x1, double x2)
{
  return rk_wald(x0, x1, x2);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_wald(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  double x2;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 3, "rk_wald");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];
  arg2 = aa[2];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_wald(x0, x1, x2); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_wald _cffi_d_rk_wald
#endif

static double _cffi_d_rk_weibull(rk_state * x0, double x1)
{
  return rk_weibull(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_weibull(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  double result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_weibull");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_weibull(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_double(result);
}
#else
#  define _cffi_f_rk_weibull _cffi_d_rk_weibull
#endif

static long _cffi_d_rk_zipf(rk_state * x0, double x1)
{
  return rk_zipf(x0, x1);
}
#ifndef PYPY_VERSION
static PyObject *
_cffi_f_rk_zipf(PyObject *self, PyObject *args)
{
  rk_state * x0;
  double x1;
  Py_ssize_t datasize;
  long result;
  PyObject *arg0;
  PyObject *arg1;
  PyObject **aa;

  aa = _cffi_unpack_args(args, 2, "rk_zipf");
  if (aa == NULL)
    return NULL;
  arg0 = aa[0];
  arg1 = aa[1];

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg0, (char **)&x0);
  if (datasize != 0) {
    if (datasize < 0)
      return NULL;
    x0 = (rk_state *)alloca((size_t)datasize);
    memset((void *)x0, 0, (size_t)datasize);
    if (_cffi_convert_array_from_object((char *)x0, _cffi_type(1), arg0) < 0)
      return NULL;
  }

  x1 = (double)_cffi_to_c_double(arg1);
  if (x1 == (double)-1 && PyErr_Occurred())
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = rk_zipf(x0, x1); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  return _cffi_from_c_int(result, long);
}
#else
#  define _cffi_f_rk_zipf _cffi_d_rk_zipf
#endif

_CFFI_UNUSED_FN
static void _cffi_checkfld__rk_state(rk_state *p)
{
  /* only to generate compile-time warnings or errors */
  (void)p;
  { unsigned long(*tmp)[624] = &p->key; (void)tmp; }
  (void)((p->pos) << 1);  /* check that 'rk_state.pos' is an integer */
  (void)((p->has_gauss) << 1);  /* check that 'rk_state.has_gauss' is an integer */
  { double *tmp = &p->gauss; (void)tmp; }
  (void)((p->has_binomial) << 1);  /* check that 'rk_state.has_binomial' is an integer */
  { double *tmp = &p->psave; (void)tmp; }
  (void)((p->nsave) << 1);  /* check that 'rk_state.nsave' is an integer */
  { double *tmp = &p->r; (void)tmp; }
  { double *tmp = &p->q; (void)tmp; }
  { double *tmp = &p->fm; (void)tmp; }
  (void)((p->m) << 1);  /* check that 'rk_state.m' is an integer */
  { double *tmp = &p->p1; (void)tmp; }
  { double *tmp = &p->xm; (void)tmp; }
  { double *tmp = &p->xl; (void)tmp; }
  { double *tmp = &p->xr; (void)tmp; }
  { double *tmp = &p->c; (void)tmp; }
  { double *tmp = &p->laml; (void)tmp; }
  { double *tmp = &p->lamr; (void)tmp; }
  { double *tmp = &p->p2; (void)tmp; }
  { double *tmp = &p->p3; (void)tmp; }
  { double *tmp = &p->p4; (void)tmp; }
}
struct _cffi_align__rk_state { char x; rk_state y; };

static char *(*_cffi_var_rk_strerror(void))[2]
{
  return &(rk_strerror);
}

static const struct _cffi_global_s _cffi_globals[] = {
  { "RK_ENODEV", (void *)_cffi_const_RK_ENODEV, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "RK_ERR_MAX", (void *)_cffi_const_RK_ERR_MAX, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "RK_NOERR", (void *)_cffi_const_RK_NOERR, _CFFI_OP(_CFFI_OP_ENUM, -1), (void *)0 },
  { "init_by_array", (void *)_cffi_f_init_by_array, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 62), (void *)_cffi_d_init_by_array },
  { "rk_altfill", (void *)_cffi_f_rk_altfill, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 49), (void *)_cffi_d_rk_altfill },
  { "rk_beta", (void *)_cffi_f_rk_beta, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_beta },
  { "rk_binomial", (void *)_cffi_f_rk_binomial, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 30), (void *)_cffi_d_rk_binomial },
  { "rk_binomial_btpe", (void *)_cffi_f_rk_binomial_btpe, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 30), (void *)_cffi_d_rk_binomial_btpe },
  { "rk_binomial_inversion", (void *)_cffi_f_rk_binomial_inversion, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 30), (void *)_cffi_d_rk_binomial_inversion },
  { "rk_chisquare", (void *)_cffi_f_rk_chisquare, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_chisquare },
  { "rk_devfill", (void *)_cffi_f_rk_devfill, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 44), (void *)_cffi_d_rk_devfill },
  { "rk_double", (void *)_cffi_f_rk_double, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 0), (void *)_cffi_d_rk_double },
  { "rk_exponential", (void *)_cffi_f_rk_exponential, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_exponential },
  { "rk_f", (void *)_cffi_f_rk_f, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_f },
  { "rk_fill", (void *)_cffi_f_rk_fill, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 71), (void *)_cffi_d_rk_fill },
  { "rk_gamma", (void *)_cffi_f_rk_gamma, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_gamma },
  { "rk_gauss", (void *)_cffi_f_rk_gauss, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 0), (void *)_cffi_d_rk_gauss },
  { "rk_geometric", (void *)_cffi_f_rk_geometric, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_geometric },
  { "rk_geometric_inversion", (void *)_cffi_f_rk_geometric_inversion, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_geometric_inversion },
  { "rk_geometric_search", (void *)_cffi_f_rk_geometric_search, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_geometric_search },
  { "rk_gumbel", (void *)_cffi_f_rk_gumbel, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_gumbel },
  { "rk_hypergeometric", (void *)_cffi_f_rk_hypergeometric, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 35), (void *)_cffi_d_rk_hypergeometric },
  { "rk_hypergeometric_hrua", (void *)_cffi_f_rk_hypergeometric_hrua, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 35), (void *)_cffi_d_rk_hypergeometric_hrua },
  { "rk_hypergeometric_hyp", (void *)_cffi_f_rk_hypergeometric_hyp, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 35), (void *)_cffi_d_rk_hypergeometric_hyp },
  { "rk_interval", (void *)_cffi_f_rk_interval, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 58), (void *)_cffi_d_rk_interval },
  { "rk_laplace", (void *)_cffi_f_rk_laplace, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_laplace },
  { "rk_logistic", (void *)_cffi_f_rk_logistic, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_logistic },
  { "rk_lognormal", (void *)_cffi_f_rk_lognormal, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_lognormal },
  { "rk_logseries", (void *)_cffi_f_rk_logseries, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_logseries },
  { "rk_long", (void *)_cffi_f_rk_long, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 18), (void *)_cffi_d_rk_long },
  { "rk_negative_binomial", (void *)_cffi_f_rk_negative_binomial, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 25), (void *)_cffi_d_rk_negative_binomial },
  { "rk_noncentral_chisquare", (void *)_cffi_f_rk_noncentral_chisquare, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_noncentral_chisquare },
  { "rk_noncentral_f", (void *)_cffi_f_rk_noncentral_f, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 12), (void *)_cffi_d_rk_noncentral_f },
  { "rk_normal", (void *)_cffi_f_rk_normal, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_normal },
  { "rk_pareto", (void *)_cffi_f_rk_pareto, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_pareto },
  { "rk_poisson", (void *)_cffi_f_rk_poisson, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_poisson },
  { "rk_poisson_mult", (void *)_cffi_f_rk_poisson_mult, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_poisson_mult },
  { "rk_poisson_ptrs", (void *)_cffi_f_rk_poisson_ptrs, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_poisson_ptrs },
  { "rk_power", (void *)_cffi_f_rk_power, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_power },
  { "rk_random", (void *)_cffi_f_rk_random, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 55), (void *)_cffi_d_rk_random },
  { "rk_randomseed", (void *)_cffi_f_rk_randomseed, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 41), (void *)_cffi_d_rk_randomseed },
  { "rk_rayleigh", (void *)_cffi_f_rk_rayleigh, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_rayleigh },
  { "rk_seed", (void *)_cffi_f_rk_seed, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 67), (void *)_cffi_d_rk_seed },
  { "rk_standard_cauchy", (void *)_cffi_f_rk_standard_cauchy, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 0), (void *)_cffi_d_rk_standard_cauchy },
  { "rk_standard_exponential", (void *)_cffi_f_rk_standard_exponential, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 0), (void *)_cffi_d_rk_standard_exponential },
  { "rk_standard_gamma", (void *)_cffi_f_rk_standard_gamma, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_standard_gamma },
  { "rk_standard_t", (void *)_cffi_f_rk_standard_t, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_standard_t },
  { "rk_strerror", (void *)_cffi_var_rk_strerror, _CFFI_OP(_CFFI_OP_GLOBAL_VAR_F, 77), (void *)0 },
  { "rk_triangular", (void *)_cffi_f_rk_triangular, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 12), (void *)_cffi_d_rk_triangular },
  { "rk_ulong", (void *)_cffi_f_rk_ulong, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_O, 55), (void *)_cffi_d_rk_ulong },
  { "rk_uniform", (void *)_cffi_f_rk_uniform, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_uniform },
  { "rk_vonmises", (void *)_cffi_f_rk_vonmises, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_vonmises },
  { "rk_wald", (void *)_cffi_f_rk_wald, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 7), (void *)_cffi_d_rk_wald },
  { "rk_weibull", (void *)_cffi_f_rk_weibull, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 3), (void *)_cffi_d_rk_weibull },
  { "rk_zipf", (void *)_cffi_f_rk_zipf, _CFFI_OP(_CFFI_OP_CPYTHON_BLTN_V, 21), (void *)_cffi_d_rk_zipf },
};

static const struct _cffi_field_s _cffi_fields[] = {
  { "key", offsetof(rk_state, key),
           sizeof(((rk_state *)0)->key),
           _CFFI_OP(_CFFI_OP_NOOP, 82) },
  { "pos", offsetof(rk_state, pos),
           sizeof(((rk_state *)0)->pos),
           _CFFI_OP(_CFFI_OP_NOOP, 47) },
  { "has_gauss", offsetof(rk_state, has_gauss),
                 sizeof(((rk_state *)0)->has_gauss),
                 _CFFI_OP(_CFFI_OP_NOOP, 47) },
  { "gauss", offsetof(rk_state, gauss),
             sizeof(((rk_state *)0)->gauss),
             _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "has_binomial", offsetof(rk_state, has_binomial),
                    sizeof(((rk_state *)0)->has_binomial),
                    _CFFI_OP(_CFFI_OP_NOOP, 47) },
  { "psave", offsetof(rk_state, psave),
             sizeof(((rk_state *)0)->psave),
             _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "nsave", offsetof(rk_state, nsave),
             sizeof(((rk_state *)0)->nsave),
             _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "r", offsetof(rk_state, r),
         sizeof(((rk_state *)0)->r),
         _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "q", offsetof(rk_state, q),
         sizeof(((rk_state *)0)->q),
         _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "fm", offsetof(rk_state, fm),
          sizeof(((rk_state *)0)->fm),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "m", offsetof(rk_state, m),
         sizeof(((rk_state *)0)->m),
         _CFFI_OP(_CFFI_OP_NOOP, 32) },
  { "p1", offsetof(rk_state, p1),
          sizeof(((rk_state *)0)->p1),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "xm", offsetof(rk_state, xm),
          sizeof(((rk_state *)0)->xm),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "xl", offsetof(rk_state, xl),
          sizeof(((rk_state *)0)->xl),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "xr", offsetof(rk_state, xr),
          sizeof(((rk_state *)0)->xr),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "c", offsetof(rk_state, c),
         sizeof(((rk_state *)0)->c),
         _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "laml", offsetof(rk_state, laml),
            sizeof(((rk_state *)0)->laml),
            _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "lamr", offsetof(rk_state, lamr),
            sizeof(((rk_state *)0)->lamr),
            _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "p2", offsetof(rk_state, p2),
          sizeof(((rk_state *)0)->p2),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "p3", offsetof(rk_state, p3),
          sizeof(((rk_state *)0)->p3),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
  { "p4", offsetof(rk_state, p4),
          sizeof(((rk_state *)0)->p4),
          _CFFI_OP(_CFFI_OP_NOOP, 5) },
};

static const struct _cffi_struct_union_s _cffi_struct_unions[] = {
  { "rk_state_", 81, _CFFI_F_CHECK_FIELDS,
    sizeof(rk_state), offsetof(struct _cffi_align__rk_state, y), 0, 21 },
};

static const struct _cffi_enum_s _cffi_enums[] = {
  { "$rk_error", 80, _cffi_prim_int(sizeof(rk_error), ((rk_error)-1) <= 0),
    "RK_NOERR,RK_ENODEV,RK_ERR_MAX" },
};

static const struct _cffi_typename_s _cffi_typenames[] = {
  { "rk_error", 80 },
  { "rk_state", 81 },
};

static const struct _cffi_type_context_s _cffi_type_context = {
  _cffi_types,
  _cffi_globals,
  _cffi_fields,
  _cffi_struct_unions,
  _cffi_enums,
  _cffi_typenames,
  55,  /* num_globals */
  1,  /* num_struct_unions */
  1,  /* num_enums */
  2,  /* num_typenames */
  NULL,  /* no includes */
  85,  /* num_types */
  0,  /* flags */
};

#ifdef PYPY_VERSION
PyMODINIT_FUNC
_cffi_pypyinit__mtrand(const void *p[])
{
    p[0] = (const void *)0x2601;
    p[1] = &_cffi_type_context;
}
#  ifdef _MSC_VER
     PyMODINIT_FUNC
#  if PY_MAJOR_VERSION >= 3
     PyInit__mtrand(void) { return NULL; }
#  else
     init_mtrand(void) { }
#  endif
#  endif
#elif PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__mtrand(void)
{
  return _cffi_init("numpy.random._mtrand", 0x2601, &_cffi_type_context);
}
#else
PyMODINIT_FUNC
init_mtrand(void)
{
  _cffi_init("numpy.random._mtrand", 0x2601, &_cffi_type_context);
}
#endif
