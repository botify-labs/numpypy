from cffi import FFI
ffi = FFI()


ffi.cdef('''
void cffti(int n, double wsave[]);
void rffti(int n, double wsave[]);
void cfftf(int n, double c[], double wsave[]);
void cfftb(int n, double c[], double wsave[]);
void rfftf(int n, double r[], double wsave[]);
void rfftb(int n, double r[], double wsave[]);
''')


ffi.set_source('numpy.fft._fft_cffi', '#include "fftpack.h"')
