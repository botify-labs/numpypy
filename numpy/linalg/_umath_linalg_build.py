import sys, os
import cffi

ffi = cffi.FFI()
ffi.cdef('''
    void init_constants(void);
    int _npy_clear_floatstatus(void);
    int _npy_set_floatstatus_invalid(void);
''')

ufunc_cdef = 'void %s(char **args, intptr_t * dimensions, intptr_t * steps, void*);'
all_four = ['FLOAT_', 'DOUBLE_', 'CFLOAT_', 'CDOUBLE_']
three = ['FLOAT_', 'DOUBLE_', 'CDOUBLE_']
base_four_names = [
    'slogdet', 'det', 'inv', 'solve1', 'solve', 'eighup', 'eighlo',
    'eigvalshlo', 'eigvalshup', 'cholesky_lo', 'svd_A', 'svd_S', 'svd_N']
base_three_names = ['eig', 'eigvals']
names = []
for name in base_four_names:
    names += [pre + name for pre in all_four]
for name in base_three_names:
    names += [pre + name for pre in three]
for name in names:
    ffi.cdef(ufunc_cdef % name)

# TODO macos?
if sys.platform == 'win32':
    so_name = '/umath_linalg_cffi.dll'
else:
    so_name = '/libumath_linalg_cffi.so'
umath_linalg_capi = ffi.dlopen(os.path.dirname(__file__) + so_name)
