from __future__ import print_function

from _partition import lib, ffi

from _parition_build import list_suff, list_type
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


class IndexesOverAxis(object):
    """
    Class for iterating over an array along one axis. Similar functionality is implemented in numpy.apply_along_axis.

    >>> indexes = IndexesOverAxis((2,3,3), 1)
    >>> list(indexes)
    [(0, slice(None, None, None), 0), (0, slice(None, None, None), 1), (0, slice(None, None, None), 2), (1, slice(None, None, None), 0), (1, slice(None, None, None), 1), (1, slice(None, None, None), 2)]
    >>> indexes = IndexesOverAxis((2,2,3), 2)
    >>> list(indexes)
    [(0, 0, slice(None, None, None)), (0, 1, slice(None, None, None)), (1, 0, slice(None, None, None)), (1, 1, slice(None, None, None))]
    """

    def __init__(self, shape, axis):
        len_shape = len(shape)
        if len_shape <= 0:
            raise ValueError("Shape must have at least one dimension")

        if axis < 0:
            axis += len_shape

        if not (0 <= axis < len_shape):
            raise IndexError("Axis must be in 0..{}. Current value {}".format(len_shape, axis))

        self.axis = axis
        self.limits = list(shape)
        self.limits[axis] = 0
        self.current_index_slice = [0] * len_shape

    @staticmethod
    def _generate_next(array, limits):
        """
        Performs per digit (per element) increment with overflow processing
        Assuming len(array) == len(limits), limits[x] >= 0 for each x.

        Parameters
        ----------
        array current state of array
        limits limits for each element.

        Returns
        -------
        """
        i = len(array) - 1
        array[i] += 1  # increment the last "digit"
        while array[i] >= limits[i]:  # while overflow
            if i <= 0:  # overflow in the last "digit" -> exit
                return False
            array[i] = 0
            array[i - 1] += 1
            i -= 1  # move to next "digit"
        return True

    def _get_output(self):
        output = self.current_index_slice[:]  # copy
        output[self.axis] = slice(None)
        return tuple(output)

    def __iter__(self):
        while True:
            yield self._get_output()
            if not self._generate_next(self.current_index_slice, self.limits):
                return


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


def _apply_inplace_along_axis(func1d, axis, arr, args=(), kwargs={}):
    for indexes in IndexesOverAxis(arr.shape, axis):
        extracted_axis = arr[indexes].copy()
        func1d(extracted_axis, *args, **kwargs)
        arr[indexes] = extracted_axis


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
            _partition_for_1d(a, kth, kind, order)
        else:
            _apply_inplace_along_axis(_partition_for_1d, axis=axis, arr=a, args=(),
                                      kwargs=dict(kth=kth, order=order, kind=kind))
    except NotImplementedError:
        a.sort(axis=axis, order=order)
