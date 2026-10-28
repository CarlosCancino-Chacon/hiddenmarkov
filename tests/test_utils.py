#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for hiddenmarkov.utils: softmax, inverted_softmax, create_prob_models.
"""

import unittest
import numpy as np

from hiddenmarkov.utils import inverted_softmax, softmax, create_prob_models


class TestSoftmax(unittest.TestCase):

    def test_sums_to_one(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result.sum(), 1.0, places=12)
        self.assertTrue(np.all(result > 0))

    def test_monotone_increasing(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(result[0] < result[1] < result[2])


class TestInvertedSoftmax(unittest.TestCase):

    def test_sums_to_one(self):
        result = inverted_softmax(np.array([0.1, 0.5, 1.0]))
        self.assertAlmostEqual(result.sum(), 1.0, places=12)
        self.assertTrue(np.all(result > 0))

    def test_monotone_decreasing(self):
        """Lower distance → higher probability."""
        result = inverted_softmax(np.array([0.1, 0.5, 1.0]))
        self.assertTrue(result[0] > result[1] > result[2])


class TestCreateProbModels(unittest.TestCase):

    def test_shape(self):
        models = create_prob_models(no_global_states=10, window_size=3)
        self.assertEqual(len(models), 10)
        self.assertEqual(len(models[0]), 3)

    def test_entries_are_callable_int_pairs(self):
        models = create_prob_models(no_global_states=5, window_size=2)
        prob_fn, global_idx = models[0][0]
        self.assertTrue(callable(prob_fn))
        self.assertIsInstance(global_idx, int)

    def test_prob_fn_indicator(self):
        """Prob function returns 1 for matching reference, 0 otherwise."""
        models = create_prob_models(no_global_states=5, window_size=1)
        prob_fn, idx = models[2][0]
        self.assertEqual(prob_fn(idx), 1)
        self.assertEqual(prob_fn(idx + 1), 0)

    def test_global_idx_clamped(self):
        """Global indices must not exceed no_global_states - 1."""
        models = create_prob_models(no_global_states=3, window_size=5)
        for _, idx in models[-1]:
            self.assertLessEqual(idx, 2)


if __name__ == "__main__":
    unittest.main()
