#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for HMM model classes: TransitionModel, ObservationModel,
ConstantTransitionModel, CategoricalObservationModel, WindowedObservationModel.
"""

import unittest
import numpy as np

from hiddenmarkov import (
    HMM,
    TransitionModel,
    ObservationModel,
    ConstantTransitionModel,
    CategoricalObservationModel,
    WindowedObservationModel,
    viterbi_algorithm,
)
from hiddenmarkov.utils import create_prob_models


# ---------------------------------------------------------------------------
# Base-class abstract contracts
# ---------------------------------------------------------------------------

class TestBaseClassContracts(unittest.TestCase):

    def test_transition_model_call_raises(self):
        tm = TransitionModel()
        with self.assertRaises(NotImplementedError):
            tm()

    def test_observation_model_call_raises(self):
        om = ObservationModel()
        with self.assertRaises(NotImplementedError):
            om("x")


# ---------------------------------------------------------------------------
# ConstantTransitionModel — normalization
# ---------------------------------------------------------------------------

class TestConstantTransitionModelNormalization(unittest.TestCase):

    def test_normalize_init_probabilities(self):
        """Raw init probs must sum to 1; log probs must equal log of raw."""
        tm = ConstantTransitionModel(
            np.eye(3),
            init_probabilities=np.array([1.0, 2.0, 3.0]),
            normalize_init_probabilities=True,
        )
        np.testing.assert_allclose(tm._init_dist.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(tm._init_dist, [1 / 6, 2 / 6, 3 / 6], atol=1e-12)
        np.testing.assert_allclose(tm._log_init_dist, np.log(tm._init_dist), atol=1e-12)

    def test_normalize_init_probabilities_log_mode_off(self):
        """Normalization should work identically when use_log_probabilities=False."""
        tm = ConstantTransitionModel(
            np.eye(3),
            init_probabilities=np.array([1.0, 2.0, 3.0]),
            normalize_init_probabilities=True,
            use_log_probabilities=False,
        )
        np.testing.assert_allclose(tm._init_dist.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(tm._init_dist, [1 / 6, 2 / 6, 3 / 6], atol=1e-12)

    def test_normalize_transition_probabilities(self):
        """Each row of the raw transition matrix must sum to 1."""
        raw = np.array([[3.0, 1.0], [1.0, 4.0]])
        tm = ConstantTransitionModel(
            raw,
            normalize_transition_probabilities=True,
        )
        row_sums = tm._transition_prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            tm._log_transition_prob, np.log(tm._transition_prob), atol=1e-12
        )

    def test_normalize_both(self):
        """Both normalize flags active at the same time."""
        tm = ConstantTransitionModel(
            np.array([[3.0, 1.0], [1.0, 4.0]]),
            init_probabilities=np.array([2.0, 8.0]),
            normalize_init_probabilities=True,
            normalize_transition_probabilities=True,
        )
        np.testing.assert_allclose(tm._init_dist.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(tm._transition_prob.sum(axis=1), [1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# ConstantTransitionModel — indexed access
# ---------------------------------------------------------------------------

class TestConstantTransitionModelIndexing(unittest.TestCase):

    def setUp(self):
        trans = np.array([[0.7, 0.3], [0.4, 0.6]])
        self.tm_log = ConstantTransitionModel(trans, use_log_probabilities=True)
        self.tm_raw = ConstantTransitionModel(trans, use_log_probabilities=False)

    def test_row_access_log(self):
        row = self.tm_log(i=0)
        np.testing.assert_allclose(row, np.log([0.7, 0.3]), atol=1e-12)

    def test_row_access_raw(self):
        row = self.tm_raw(i=1)
        np.testing.assert_allclose(row, [0.4, 0.6], atol=1e-12)

    def test_col_access_log(self):
        col = self.tm_log(j=1)
        np.testing.assert_allclose(col, np.log([0.3, 0.6]), atol=1e-12)

    def test_col_access_raw(self):
        col = self.tm_raw(j=0)
        np.testing.assert_allclose(col, [0.7, 0.4], atol=1e-12)

    def test_scalar_access(self):
        val_raw = self.tm_raw(i=0, j=1)
        self.assertAlmostEqual(float(val_raw), 0.3)
        val_log = self.tm_log(i=0, j=1)
        self.assertAlmostEqual(float(val_log), np.log(0.3), places=10)

    def test_full_matrix_access(self):
        mat = self.tm_raw()
        self.assertEqual(mat.shape, (2, 2))


# ---------------------------------------------------------------------------
# CategoricalObservationModel
# ---------------------------------------------------------------------------

class TestCategoricalObservationModelAutoLabels(unittest.TestCase):

    def test_auto_labels_created(self):
        obs_probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
        model = CategoricalObservationModel(obs_probs)  # no observations arg
        self.assertEqual(model.observations, ["0", "1", "2"])
        self.assertIn("0", model.observation_indices)

    def test_auto_labels_callable_raw(self):
        obs_probs = np.array([[0.8, 0.2], [0.4, 0.6]])
        model = CategoricalObservationModel(obs_probs, use_log_probabilities=False)
        result = model("0")
        np.testing.assert_allclose(result, [0.8, 0.2], atol=1e-12)

    def test_auto_labels_callable_log(self):
        obs_probs = np.array([[0.5, 0.5], [0.3, 0.7]])
        model = CategoricalObservationModel(obs_probs, use_log_probabilities=True)
        result = model("1")
        np.testing.assert_allclose(result, np.log([0.3, 0.7]), atol=1e-12)


# ---------------------------------------------------------------------------
# WindowedObservationModel
# ---------------------------------------------------------------------------

class TestWindowedObservationModelInvertedProbs(unittest.TestCase):

    def _make_hmm(self, use_inverted_probs, use_log_probabilities):
        prob_models = create_prob_models(no_global_states=20, window_size=2)
        obs_model = WindowedObservationModel(
            prob_models,
            use_inverted_probs=use_inverted_probs,
            use_log_probabilities=use_log_probabilities,
        )
        trans_probs = np.array([[0.7, 0.3], [0.001, 0.999]])
        init_probs = np.array([0.999, 0.001])
        trans_model = ConstantTransitionModel(
            trans_probs,
            init_probabilities=init_probs,
            use_log_probabilities=use_log_probabilities,
        )
        return HMM(obs_model, trans_model)

    def test_inverted_probs_valid_distribution_log(self):
        hmm = self._make_hmm(use_inverted_probs=True, use_log_probabilities=True)
        probs, idx = hmm.observation_model(5, 5)
        np.testing.assert_allclose(np.exp(probs).sum(), 1.0, atol=1e-12)
        self.assertEqual(len(idx), 2)

    def test_inverted_probs_valid_distribution_raw(self):
        hmm = self._make_hmm(use_inverted_probs=True, use_log_probabilities=False)
        probs, idx = hmm.observation_model(5, 5)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    def test_inverted_probs_viterbi_runs(self):
        obs = np.arange(15)
        hmm = self._make_hmm(use_inverted_probs=True, use_log_probabilities=True)
        path, _ = hmm.find_best_sequence(obs, log_probabilities=True, viterbi="windowed")
        self.assertEqual(len(path), len(obs))

    def test_inverted_vs_softmax_complete_without_error(self):
        """Both softmax variants must decode without error."""
        obs = np.arange(10)
        path_inv, _ = self._make_hmm(True, True).find_best_sequence(
            obs, log_probabilities=True, viterbi="windowed"
        )
        path_sft, _ = self._make_hmm(False, True).find_best_sequence(
            obs, log_probabilities=True, viterbi="windowed"
        )
        self.assertEqual(len(path_inv), len(obs))
        self.assertEqual(len(path_sft), len(obs))


class TestWindowedObservationModelLogModes(unittest.TestCase):
    """Windowed model correctness across log/raw and outfunc update."""

    def _make_windowed_hmm(self, use_log_probabilities):
        prob_models = create_prob_models(no_global_states=20, window_size=2)
        obs_model = WindowedObservationModel(
            prob_models,
            use_inverted_probs=False,
            use_log_probabilities=use_log_probabilities,
        )
        trans_probs = np.array([[0.7, 0.3], [0.001, 0.999]])
        init_probs = np.array([0.999, 0.001])
        trans_model = ConstantTransitionModel(
            trans_probs,
            init_probabilities=init_probs,
            use_log_probabilities=use_log_probabilities,
        )
        return HMM(obs_model, trans_model)

    def test_log_and_raw_give_same_path(self):
        """Windowed Viterbi in log and raw mode must decode the same path."""
        obs = np.arange(15)
        path_log, _ = self._make_windowed_hmm(True).find_best_sequence(
            obs, log_probabilities=True, viterbi="windowed"
        )
        path_raw, _ = self._make_windowed_hmm(False).find_best_sequence(
            obs, log_probabilities=False, viterbi="windowed"
        )
        np.testing.assert_array_equal(path_log, path_raw)

    def test_outfunc_updates_after_construction(self):
        """A model built with log=True must produce correct results when run with log=False."""
        obs = np.arange(10)
        path_native, _ = self._make_windowed_hmm(False).find_best_sequence(
            obs, log_probabilities=False, viterbi="windowed"
        )
        # Build with log=True but decode with log_probabilities=False
        path_mixed, _ = self._make_windowed_hmm(True).find_best_sequence(
            obs, log_probabilities=False, viterbi="windowed"
        )
        np.testing.assert_array_equal(path_mixed, path_native)


if __name__ == "__main__":
    unittest.main()
