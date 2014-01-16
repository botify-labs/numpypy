# is it a good idea to put it on a top-level here?.. maybe some singleton would be nicer or something...
from cffi import FFI
ffi = FFI()


ffi.cdef('''
//typedef ... Treal; // damn, stuff like this doesn't work with cffi yet... structs are fine, but not the basic types :(
// so, we stick to 'double' here:
void cffti(int n, double wsave[]);
void cfftf(int n, double c[], double wsave[]);
''')


C = ffi.verify('''
// yes, we know .c files are not for #include, but we won't paste it here -- it is even bigger evil than that!
// fftpack.c is wget'ed from https://bitbucket.org/pypy/numpy/raw/74707b00761ec326657c8f39d2c2133bfefbe521/numpy/fft/fftpack.c
#include "fftpack.c" 
''', libraries=[])



def cffti(n):
    '''
    void cffti(int n, Treal wsave[])
    '''
    wsave_cdata = ffi.new('double[]', 4*n+15) # don't even ask me why 4 and why 15. Just took them from C-ext wrapper "as is".
    C.cffti(n, wsave_cdata)
    return wsave_cdata


def cfftf(a, wsave):
    '''
    void cfftf(int n, Troeal c[], Treal wsave[]);
    '''
    n = ffi.cast('int', len(a))
    a_copy = a.copy()
    a_copy_cdata = ffi.cast('double*', a_copy.__array_interface__['data'][0])
    C.cfftf(n, a_copy_cdata, wsave)
    return a_copy
