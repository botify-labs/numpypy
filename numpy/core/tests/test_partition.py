from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import TestCase
import sys


def is_pypy():
    return '__pypy__' in sys.builtin_module_names


class TestPartition(TestCase):
    pivot = 2
    pivot_index = 2

    array = np.array([[[5, 0, 3, 3],
        [7, 3, 5, 2],
        [4, 7, 6, 8]],

       [[8, 1, 6, 7],
        [7, 8, 1, 5],
        [8, 4, 3, 0]]])

    def test_uint16(self):
        arr = np.arange(10, -1, -1, dtype='int16')

        partition = np.partition(arr, self.pivot_index)
        self._check_partition(partition, self.pivot_index)
        self._check_content_along_axis(arr, partition, -1)


    def test_uint32(self):
        arr = np.arange(10, -1, -1, dtype='int32')

        partition = np.partition(arr, self.pivot_index)

        self._check_partition(partition, self.pivot_index)
        self._check_content_along_axis(arr, partition, -1)

    def test_uint64(self):
        arr = np.arange(10, -1, -1, dtype='int64')

        partition = np.partition(arr, self.pivot_index)

        self._check_partition(partition, self.pivot_index)
        self._check_content_along_axis(arr, partition, -1)

    def _check_partition(self, partition, pivot_index):
        pivot = np.sort(partition)[pivot_index]
        self.assertTrue(np.all(partition[:pivot_index] <= partition[pivot_index]))
        self.assertTrue(np.all(partition[pivot_index:] >= partition[pivot_index]))
        self.assertTrue(partition[pivot_index] == pivot)
        return 0

    def _check_multidimensional_partition(self, partition, axis, pivot_index):
        np.apply_along_axis(lambda x: self._check_partition(x, pivot_index), axis=axis, arr=partition)

    def test_numpy_partition_doesnt_change_array(self):
        arr = self.array.copy()

        np.partition(arr, 1)

        self.assertTrue(np.array_equal(arr, self.array))

    def test_multidimensional_axis_default(self):
        self._test_for_axis(axis=-1)

    def test_multidimensional(self):
        self._test_for_axis(axis=0)
        self._test_for_axis(axis=1)
        self._test_for_axis(axis=2)

    def _test_for_axis(self, axis):
        arr = self.array.copy()
        pivot_index = 1
        res = np.partition(arr, kth=pivot_index, axis=axis)
        self._check_content_along_axis(self.array, res, axis)
        self._check_multidimensional_partition(res, axis=axis, pivot_index=pivot_index)

    def _check_content_along_axis(self, source, array, axis):
        self.assertTrue(np.array_equal(np.sort(source, axis=axis), np.sort(array, axis=axis)))
