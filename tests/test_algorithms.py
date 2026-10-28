#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for HMM decoding and inference algorithms:
Viterbi (optimized, naive, windowed) and forward algorithm.
"""

import unittest
import numpy as np

import hiddenmarkov
from hiddenmarkov import (
    HMM,
    ConstantTransitionModel,
    CategoricalObservationModel,
    WindowedObservationModel,
    forward_algorithm_step,
    viterbi_algorithm,
    viterbi_algorithm_naive,
)
from hiddenmarkov.utils import create_prob_models


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_wikipedia_hmm():
    """
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
    """
    obs_labels = ("normal", "cold", "dizzy")
    obs_probs = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
    trans_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    init_probs = np.array([0.6, 0.4])
    obs_model = CategoricalObservationModel(obs_probs, obs_labels)
    trans_model = ConstantTransitionModel(trans_probs, init_probs)
    hmm = HMM(obs_model, trans_model, state_space=("Healthy", "Fever"))
    observations = ("normal", "cold", "dizzy")
    return hmm, observations


_WIKIPEDIA_EXPECTED_PROB = 0.01512
_WIKIPEDIA_EXPECTED_PATH = np.array(["Healthy", "Healthy", "Fever"])


# ---------------------------------------------------------------------------
# Viterbi — optimized (vectorized NumPy / Cython)
# ---------------------------------------------------------------------------

class TestViterbiOptimized(unittest.TestCase):

    def test_prob_wikipedia(self):
        hmm, obs = _make_wikipedia_hmm()
        path, prob = viterbi_algorithm(hmm, obs, log_probabilities=False)
        np.testing.assert_array_equal(path, _WIKIPEDIA_EXPECTED_PATH)
        self.assertTrue(np.isclose(prob, _WIKIPEDIA_EXPECTED_PROB, atol=1e-5))

    def test_log_wikipedia(self):
        hmm, obs = _make_wikipedia_hmm()
        path, prob = viterbi_algorithm(hmm, obs, log_probabilities=True)
        np.testing.assert_array_equal(path, _WIKIPEDIA_EXPECTED_PATH)
        self.assertTrue(np.isclose(prob, np.log(_WIKIPEDIA_EXPECTED_PROB), atol=1e-5))

    def test_upenn(self):
        """
        Example from https://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf
        """
        obs_labels = ("A", "C", "G", "T")
        obs_probs = np.array([[0.2, 0.3], [0.3, 0.2], [0.3, 0.2], [0.2, 0.3]])
        trans_probs = np.array([[0.5, 0.5], [0.4, 0.6]])
        init_probs = np.array([0.5, 0.5])
        obs_model = CategoricalObservationModel(obs_probs, obs_labels)
        trans_model = ConstantTransitionModel(trans_probs, init_probs)
        hmm = HMM(obs_model, trans_model, state_space=("H", "L"))

        observations = ["G", "G", "C", "A", "C", "T", "G", "A", "A"]
        path, prob = viterbi_algorithm(hmm, observations, log_probabilities=False)
        self.assertTrue(all(path == ["H", "H", "H", "L", "L", "L", "L", "L", "L"]))
        self.assertTrue(np.isclose(prob, 4.25e-8))


# ---------------------------------------------------------------------------
# Viterbi — naive (pure Python loop)
# ---------------------------------------------------------------------------

class TestViterbiNaive(unittest.TestCase):

    def test_prob_wikipedia(self):
        hmm, obs = _make_wikipedia_hmm()
        path, prob = viterbi_algorithm_naive(hmm, obs, log_probabilities=False)
        np.testing.assert_array_equal(path, _WIKIPEDIA_EXPECTED_PATH)
        self.assertTrue(np.isclose(prob, _WIKIPEDIA_EXPECTED_PROB, atol=1e-5))

    def test_log_wikipedia(self):
        hmm, obs = _make_wikipedia_hmm()
        path, prob = viterbi_algorithm_naive(hmm, obs, log_probabilities=True)
        np.testing.assert_array_equal(path, _WIKIPEDIA_EXPECTED_PATH)
        self.assertTrue(np.isclose(prob, np.log(_WIKIPEDIA_EXPECTED_PROB), atol=1e-5))


# ---------------------------------------------------------------------------
# HMM.find_best_sequence — routing to all Viterbi variants
# ---------------------------------------------------------------------------

class TestFindBestSequence(unittest.TestCase):

    def _check(self, hmm, obs, viterbi, log_probabilities):
        path, prob = hmm.find_best_sequence(
            observations=obs,
            log_probabilities=log_probabilities,
            viterbi=viterbi,
        )
        expected_prob = np.log(_WIKIPEDIA_EXPECTED_PROB) if log_probabilities else _WIKIPEDIA_EXPECTED_PROB
        np.testing.assert_array_equal(path, _WIKIPEDIA_EXPECTED_PATH)
        self.assertTrue(np.isclose(prob, expected_prob, atol=1e-5))

    def test_optimized_log(self):
        hmm, obs = _make_wikipedia_hmm()
        self._check(hmm, obs, "optimized", True)

    def test_optimized_raw(self):
        hmm, obs = _make_wikipedia_hmm()
        self._check(hmm, obs, "optimized", False)

    def test_naive_log(self):
        hmm, obs = _make_wikipedia_hmm()
        self._check(hmm, obs, "naive", True)

    def test_naive_raw(self):
        hmm, obs = _make_wikipedia_hmm()
        self._check(hmm, obs, "naive", False)

    def test_invalid_viterbi_raises(self):
        hmm, obs = _make_wikipedia_hmm()
        with self.assertRaises(ValueError):
            hmm.find_best_sequence(obs, log_probabilities=False, viterbi="other")


# ---------------------------------------------------------------------------
# Log vs raw consistency across Viterbi implementations
# ---------------------------------------------------------------------------

class TestViterbiLogRawConsistency(unittest.TestCase):

    def test_optimized_log_vs_raw(self):
        hmm, obs = _make_wikipedia_hmm()
        path_log, _ = viterbi_algorithm(hmm, obs, log_probabilities=True)
        path_raw, _ = viterbi_algorithm(hmm, obs, log_probabilities=False)
        np.testing.assert_array_equal(path_log, path_raw)

    def test_naive_log_vs_raw(self):
        hmm, obs = _make_wikipedia_hmm()
        path_log, _ = viterbi_algorithm_naive(hmm, obs, log_probabilities=True)
        path_raw, _ = viterbi_algorithm_naive(hmm, obs, log_probabilities=False)
        np.testing.assert_array_equal(path_log, path_raw)

    def test_optimized_vs_naive_log(self):
        hmm, obs = _make_wikipedia_hmm()
        path_opt, prob_opt = viterbi_algorithm(hmm, obs, log_probabilities=True)
        path_nai, prob_nai = viterbi_algorithm_naive(hmm, obs, log_probabilities=True)
        np.testing.assert_array_equal(path_opt, path_nai)
        np.testing.assert_allclose(prob_opt, prob_nai, atol=1e-10)


# ---------------------------------------------------------------------------
# NumPy fallback path (when Cython extension is absent)
# ---------------------------------------------------------------------------

class TestViterbiNumPyFallback(unittest.TestCase):
    """Force _HAS_CYTHON = False and verify results match the Cython path."""

    def _run(self, log_probabilities):
        hmm, obs = _make_wikipedia_hmm()
        original = hiddenmarkov._HAS_CYTHON
        try:
            hiddenmarkov._HAS_CYTHON = True
            path_cy, prob_cy = viterbi_algorithm(hmm, obs, log_probabilities=log_probabilities)
            hiddenmarkov._HAS_CYTHON = False
            path_np, prob_np = viterbi_algorithm(hmm, obs, log_probabilities=log_probabilities)
        finally:
            hiddenmarkov._HAS_CYTHON = original
        return path_cy, prob_cy, path_np, prob_np

    def test_fallback_log_matches(self):
        path_cy, prob_cy, path_np, prob_np = self._run(True)
        np.testing.assert_array_equal(path_cy, path_np)
        np.testing.assert_allclose(prob_cy, prob_np, atol=1e-12)

    def test_fallback_raw_matches(self):
        path_cy, prob_cy, path_np, prob_np = self._run(False)
        np.testing.assert_array_equal(path_cy, path_np)
        np.testing.assert_allclose(prob_cy, prob_np, atol=1e-12)


# ---------------------------------------------------------------------------
# Windowed Viterbi
# ---------------------------------------------------------------------------

class TestViterbiWindowed(unittest.TestCase):

    def _make_windowed_hmm(self, no_global_states=100, use_log_probabilities=True):
        prob_models = create_prob_models(no_global_states=no_global_states, window_size=2)
        obs_model = WindowedObservationModel(
            prob_models, use_inverted_probs=False, use_log_probabilities=use_log_probabilities
        )
        trans_probs = np.array([[0.7, 0.3], [0.001, 0.999]])
        init_probs = np.array([0.999, 0.001])
        trans_model = ConstantTransitionModel(
            trans_probs, init_probabilities=init_probs, use_log_probabilities=use_log_probabilities
        )
        return HMM(obs_model, trans_model)

    def test_windowed_regression(self):
        """Full windowed Viterbi regression test with a long observation sequence."""
        hmm = self._make_windowed_hmm(100, True)
        obs = np.array([
            0, 1, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 9, 10, 10, 10,
            11, 11, 12, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
            19, 19, 20, 20, 20, 21, 22, 22, 23, 24, 25, 25, 26, 26, 27, 27,
            28, 28, 28, 29, 30, 31, 31, 31, 31, 32, 33, 33, 34, 35, 35, 35,
            36, 36, 36, 37, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 41, 41,
            41, 41, 41, 41, 41, 42, 42, 42, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 52, 53, 54, 54, 55, 55, 55, 55, 56, 57, 58, 58, 59,
            60, 60, 60, 61, 62, 63, 64, 64, 64, 65, 65, 65, 66, 66, 67, 68,
            69, 69, 69, 69, 70, 71, 72, 73, 73, 74, 74, 74, 74, 75, 75, 75,
            75, 75, 75, 75, 76, 77, 78, 78, 78, 78, 79, 79, 80, 81, 81, 81,
            82, 83, 83, 83, 83, 83, 83, 84, 85, 85, 86, 87, 88, 88, 88, 89,
            90, 91, 91, 92, 92, 92, 93, 93, 94, 95, 96, 96, 96, 97, 97, 98,
            98, 98, 99, 99,
        ])
        path, prob = hmm.find_best_sequence(obs, log_probabilities=True, viterbi="windowed")
        self.assertTrue(np.all(path == obs))
        self.assertTrue(np.isclose(prob, -217.51414002612017, atol=1e-5))

    def test_windowed_log_vs_raw(self):
        """Windowed Viterbi in log and raw mode must decode the same path."""
        obs = np.arange(15)
        path_log, _ = self._make_windowed_hmm(20, True).find_best_sequence(
            obs, log_probabilities=True, viterbi="windowed"
        )
        path_raw, _ = self._make_windowed_hmm(20, False).find_best_sequence(
            obs, log_probabilities=False, viterbi="windowed"
        )
        np.testing.assert_array_equal(path_log, path_raw)


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------

class TestForwardAlgorithm(unittest.TestCase):

    def _run_forward(self, log_probabilities):
        """Return the (exponentiated) forward variable after all observations."""
        hmm, obs = _make_wikipedia_hmm()
        hmm.forward_variable = None
        for o in obs:
            hmm.forward_algorithm_step(o, log_probabilities=log_probabilities)
        fv = hmm.forward_variable
        return np.exp(fv) if log_probabilities else fv

    def test_log_and_raw_agree(self):
        """Log-space and raw-space forward variables must represent the same distribution."""
        fv_log = self._run_forward(log_probabilities=True)
        fv_raw = self._run_forward(log_probabilities=False)
        np.testing.assert_allclose(fv_log, fv_raw, atol=1e-10)

    def test_forward_variable_is_normalised(self):
        """The forward variable must sum to 1."""
        for log_prob in (True, False):
            fv = self._run_forward(log_probabilities=log_prob)
            self.assertAlmostEqual(fv.sum(), 1.0, places=10)

    def test_argmax_matches_viterbi(self):
        """The most probable state at the final step should match Viterbi's last state."""
        hmm, obs = _make_wikipedia_hmm()
        path, _ = viterbi_algorithm(hmm, obs, log_probabilities=True)
        expected_last = path[-1]
        for log_prob in (True, False):
            fv = self._run_forward(log_probabilities=log_prob)
            best = hmm.state_space[np.argmax(fv)]
            self.assertEqual(best, expected_last)

    def test_standalone_forward_step_log_vs_raw(self):
        """Standalone forward_algorithm_step: log and raw modes must agree."""
        hmm, obs = _make_wikipedia_hmm()
        om = hmm.observation_model
        tm = hmm.transition_model
        fv_log = None
        fv_raw = None
        for o in obs:
            fv_log = forward_algorithm_step(om, tm, o, fv_log, log_probabilities=True)
            fv_raw = forward_algorithm_step(om, tm, o, fv_raw, log_probabilities=False)
        np.testing.assert_allclose(np.exp(fv_log), fv_raw, atol=1e-10)

    def test_forward_step_first_observation(self):
        """First step (forward_variable=None) must give correct probabilities."""
        hmm, _ = _make_wikipedia_hmm()
        om = hmm.observation_model
        tm = hmm.transition_model

        fv_raw = forward_algorithm_step(om, tm, "normal", None, log_probabilities=False)
        expected_unnorm = np.array([0.5 * 0.6, 0.1 * 0.4])
        expected = expected_unnorm / expected_unnorm.sum()
        np.testing.assert_allclose(fv_raw, expected, atol=1e-12)

        fv_log = forward_algorithm_step(om, tm, "normal", None, log_probabilities=True)
        np.testing.assert_allclose(np.exp(fv_log), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
