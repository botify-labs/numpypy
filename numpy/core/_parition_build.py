import cffi
import itertools

PARTITION_DEFS = """
// copy from pyport.h // not sure if it's correct
typedef uintptr_t   Py_uintptr_t;
typedef intptr_t    Py_intptr_t;

// Copy from  pypy/pypy/module/cpyext/include/numpy/npy_common.h
// with "PY_LONG_LONG" replaced by "long long"
typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;
typedef long long npy_longlong;
typedef unsigned long long npy_ulonglong;
typedef unsigned char npy_bool;
typedef long npy_int32;
typedef unsigned long npy_uint32;
typedef unsigned long npy_ucs4;
typedef long npy_int64;
typedef unsigned long npy_uint64;
typedef unsigned char npy_uint8;

typedef signed char npy_byte;
typedef unsigned char npy_ubyte;
typedef unsigned short npy_ushort;
typedef unsigned int npy_uint;
typedef unsigned long npy_ulong;

/* These are for completeness */
typedef char npy_char;
typedef short npy_short;
typedef int npy_int;
typedef long npy_long;
typedef float npy_float;
typedef double npy_double;

typedef struct { float real, imag; } npy_cfloat;
typedef struct { double real, imag; } npy_cdouble;
typedef npy_cdouble npy_complex128;

// copy from npy_common.h of numpypy
typedef long double npy_longdouble;

extern int get_max_pivot_stack();
"""

_str_suff = """bool, byte, ubyte, short, ushort,
int, uint, long, ulong,
longlong, ulonglong, half, float, double, longdouble,"""

_str_type = """npy_bool, npy_byte, npy_ubyte, npy_short, npy_ushort,
npy_int, npy_uint, npy_long, npy_ulong,
npy_longlong, npy_ulonglong,
npy_ushort, npy_float, npy_double, npy_longdouble, """

_function_template = '''
int introselect_@suff@(@type@ *v, npy_intp num,
                                             npy_intp kth,
                                             npy_intp * pivots,
                                             npy_intp * npiv,
                                             void *NOT_USED);
'''


def parse_types(str_types):
    list_types = map(lambda s: s.strip(), str_types.split(','))
    list_types = filter(None, list_types)
    return list_types

list_suff = parse_types(_str_suff)
list_type = parse_types(_str_type)


def generate_declarations():
    """
    Generates function declarations based on template and lists of types
    Returns
        function declarations as string
    """

    list_declaration_functions = []
    for _suff, _type in itertools.izip(list_suff, list_type):
        list_declaration_functions.append(_function_template.replace("@suff@", _suff).replace("@type@", _type))
    return ''.join(list_declaration_functions)


PARTITION_DEFS += '\n' + generate_declarations()

ffi = cffi.FFI()
ffi.cdef(PARTITION_DEFS)
ffi.set_source("numpy.core._partition", PARTITION_DEFS)
