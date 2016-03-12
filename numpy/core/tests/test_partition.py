from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import TestCase
import sys


def is_pypy():
    return '__pypy__' in sys.builtin_module_names


class TestPartition(TestCase):
    pivot = 2
    pivot_index = 2

    def test_uint16(self):
        arr = np.arange(10, -1, -1, dtype='int16')

        if is_pypy():
            with self.assertRaises(NotImplementedError):
                np.partition(arr, self.pivot_index)
        else:
            partition = np.partition(arr, self.pivot_index)
            self._check_partition(partition, self.pivot_index, self.pivot)

    def test_uint32(self):
        arr = np.arange(10, -1, -1, dtype='int32')

        partition = np.partition(arr, self.pivot_index)

        self._check_partition(partition, self.pivot_index, self.pivot)

    def test_uint64(self):
        arr = np.arange(10, -1, -1, dtype='int64')

        partition = np.partition(arr, self.pivot_index)

        self._check_partition(partition, self.pivot_index, self.pivot)

    def _check_partition(self, partition, pivot_index, pivot):
        self.assertTrue(np.all(partition[:pivot_index] < partition[pivot_index]))
        self.assertTrue(np.all(partition[pivot_index:] >= partition[pivot_index]))
        self.assertTrue(partition[pivot_index] == pivot)
