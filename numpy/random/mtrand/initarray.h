
#ifdef _MTRAND_DLL
#include <stddef.h>
typedef intptr_t npy_intp;
#else
  #include "Python.h"
  #define NO_IMPORT_ARRAY
  #include "numpy/arrayobject.h"
#endif
#include "randomkit.h"

EXPORTED void
init_by_array(rk_state *self, unsigned long init_key[],
              npy_intp key_length);
