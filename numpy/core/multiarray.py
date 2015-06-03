from _numpypy.multiarray import *
from _numpypy.multiarray import _reconstruct

def _fastCopyAndTranspose(a):
    return a.T.copy()

def copyto(dst, src, casting='same_kind', where=None):
    src = array(src)
    if not can_cast(src.dtype, dst.dtype, casting=casting):
        raise TypeError('Cannot cast from %s to %s according to the rule %s' % (
                        src.dtype, dst.dtype, casting))
    src = src.astype(dst.dtype)
    if where is None and src.size < 2:
        dst.fill(src)
    elif where is None:
        dst[:] = src
    elif src.size > 1:
        dst[where] = src[where]
    else:
        dst[where] = src

def format_longfloat(x, precision):
    return "%%.%df" % precision % x

def set_typeDict(d):
    pass

def may_share_memory(a, b):
    """
    Determine if two arrays can share memory

    The memory-bounds of a and b are computed.  If they overlap then
    this function returns True.  Otherwise, it returns False.

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Parameters
    ----------
    a, b : ndarray

    Returns
    -------
    out : bool

    Examples
    --------
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False

    """
    from ..lib import byte_bounds
    a_low, a_high = byte_bounds(a)
    b_low, b_high = byte_bounds(b)
    if b_low >= a_high or a_low >= b_high:
        return False
    return True

def NotImplementedFunc(func):
    def tmp(*args, **kwargs):
        raise NotImplementedError("%s not implemented yet" % func)
    return tmp

for name in '''
CLIP WRAP RAISE MAXDIMS ALLOW_THREADS BUFSIZE
'''.split():
    if name not in globals():
        globals()[name] = None
 
for name in '''
nditer nested_iters
broadcast empty_like fromiter fromfile frombuffer newbuffer getbuffer
int_asbuffer set_numeric_ops can_cast promote_types
min_scalar_type lexsort compare_chararrays putmask einsum inner
_vec_string datetime_data correlate correlate2
datetime_as_string busday_offset busday_count is_busday busdaycalendar
_flagdict flagsobj
'''.split():
    if name not in globals():
        globals()[name] = NotImplementedFunc(name)
