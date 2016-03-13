from numpy.core.multiarray import dtype
from numpy.core.multiarray import array


def digitize(x, bins, right=False):

    x = array(x, dtype=dtype('float'))
    bins = array(bins, dtype=dtype('float'))

    if len(bins) == 0:
        raise ValueError("bins must have non-zero length")

    monotonic = check_monotonic(bins)
    if monotonic == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")

    if monotonic == -1:
        bins = bins[::-1]
    result = bins.searchsorted(x, side='right' if not right else 'left')
    if monotonic == -1:
        result = len(bins) - result

    return result


def check_monotonic(a):
    """

    Parameters
    ----------
    a
        input array

    Returns
        -1 -- for monotonic, non-increasing
        0  -- for non-monotonic
        1  -- for monotonic, non-decreasing
    -------

    >>> check_monotonic([1,2,3])
    1
    >>> check_monotonic([3,2,1])
    -1
    >>> check_monotonic([3,1,2])
    0
    >>> check_monotonic([1, 1, 1, 3, 100])
    1
    >>> check_monotonic([1, 1, 1, 0, -1])
    -1
    >>> check_monotonic([1, 1, 1, 3, 2])
    0
    >>> check_monotonic([1123123])
    1
    """
    len_a = len(a)
    assert len_a > 0

    last = a[0]
    i = 1
    while i < len_a and a[0] == a[i]:
        i += 1

    if i == len_a:
        return 1

    next = a[i]
    i += 1
    if last < next:
        while i < len_a:
            last = next
            next = a[i]
            if last > next:
                return 0
            i += 1
        return 1
    else:
        while i < len_a:
            last = next
            next = a[i]
            if last < next:
                return 0
            i += 1
        return -1
