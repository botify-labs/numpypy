from numpy.core.multiarray import dtype
from numpy.core.multiarray import zeros
from numpy.core.multiarray import array


def bincount(x, weights=None, minlength=None):
    if minlength is None:
        minlength = 0
    else:
        if not isinstance(minlength, (int, long)):
            raise TypeError("an integer is required")
        if minlength <= 0:
            raise ValueError("minlength must be positive")

    x = array(x)
    len_output = minlength
    if len(x) > 0:
        if x.min() < 0:
            raise ValueError("x must not be negative")
        len_output = max(len_output, x.max() + 1)

    if x.dtype.kind not in 'ui':
        raise ValueError("x must be integer")

    if weights is None:
        output = zeros(len_output, dtype=dtype('int'))
        for elem in x:
            output[elem] += 1
    else:
        if len(weights) != len(x):
            raise ValueError("x and weights arrays must have the same size")
        output = zeros(len_output, dtype=dtype('float'))
        for i in range(len(x)):
            output[x[i]] += weights[i]
    return output

