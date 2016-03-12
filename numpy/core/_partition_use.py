from _partition import lib, ffi

from _parition_build import list_suff, list_type
from numpy import apply_along_axis
from numpy import partition as numpy_partition
from numpy import sort
from numpy.core.multiarray import dtype

_type_to_suff = dict(zip(list_type, list_suff))
_dtype_to_cffi_type = {dtype('int32'): 'npy_int',
                       dtype('int64'): 'npy_longlong',
                       dtype('uint32'): 'npy_uint',
                       dtype('uint64'): 'npy_ulonglong',
                       dtype('float64'): 'npy_double',
                       dtype('float32'): 'npy_float',
                       }


def _cffi_type(dtype_input):
    return _dtype_to_cffi_type.get(dtype_input)


def _partition_for_1d(a, kth, kind='introselect', order=None):
    """
    Performs in-place partition on 1D array.

    Parameters
    ----------
    a
        input 1D array
    kth
        array of kth
    kind
    order

    Returns None
    -------

    """
    assert a.ndim == 1
    if kind != 'introselect':
        raise NotImplementedError("kind == '{}' is not implemented yet".format(kind))
    if order is not None:
        raise NotImplementedError("Only order == None is implemented")

    str_dst_type = _cffi_type(a.dtype)
    if str_dst_type is None:
        raise NotImplementedError("Partition for type '{}' is not implemented yet".format(a.dtype))

    def get_pointer(np_arr):
        p_data = np_arr.__array_interface__['data'][0]
        dd = ffi.cast(str_dst_type + ' *', p_data)
        return dd

    try:
        iter_kth = iter(kth)
    except TypeError:
        iter_kth = iter((kth,))

    pivots = ffi.new("intptr_t []", lib.get_max_pivot_stack())
    npiv = ffi.new("intptr_t * ", 0)

    function_introselect = getattr(lib, 'introselect_' + _type_to_suff[str_dst_type])

    for single_kth in iter_kth:
        res = function_introselect(get_pointer(a), len(a), single_kth, pivots, npiv, ffi.NULL)
        if res != 0:
            raise RuntimeError("Something goes wrong in partition")


def partition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Performs partition inplace.
    See numpy.partition documentation for parameters description.

    Parameters
    ----------
    a
        input array
    kth
    axis
    kind
    order

    Returns None
    -------

    """

    if a.size == 0:
        return None

    try:
        if (axis == -1 or axis == a.ndim - 1) and a.ndim == 1:
            return _partition_for_1d(a, kth, kind, order)
        else:
            return apply_along_axis(numpy_partition, axis=axis, arr=a, kth=kth, order=order)
    except NotImplementedError:
        sort(a, axis=axis, order=order)
