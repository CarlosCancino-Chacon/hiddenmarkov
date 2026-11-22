#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the optimized (sparsity-exploiting) forward algorithm step:
numerical equivalence with the standard implementation, selection through
the ``forward`` argument of the HMM, and speed.
"""

import time
import unittest

import numpy as np

from hiddenmarkov import (
    HMM,
    CategoricalObservationModel,
    ConstantTransitionModel,
    TransitionModel,
    forward_algorithm_step,
    forward_algorithm_step_optimized,
)


def _banded_transitions(n_states, max_jump=2, rng=None):
    """Left-to-right transition matrix: state i -> i, i+1, ..., i+max_jump."""
    rng = rng or np.random.default_rng(0)
    T = np.zeros((n_states, n_states))
    for i in range(n_states):
        hi = min(i + max_jump + 1, n_states)
        w = rng.random(hi - i) + 0.1
        T[i, i:hi] = w / w.sum()
    return T


def _make_hmm(forward, n_states=50, n_obs=5, T=None, seed=0):
    rng = np.random.default_rng(seed)
    if T is None:
        T = _banded_transitions(n_states, rng=rng)
    n_states = len(T)
    obs_probs = rng.random((n_obs, n_states)) + 0.05
    obs_model = CategoricalObservationModel(obs_probs, list(range(n_obs)))
    init = np.zeros(n_states)
    init[0] = 1.0
    with np.errstate(divide="ignore"):
        trans_model = ConstantTransitionModel(T, init)
    return HMM(obs_model, trans_model, forward=forward), rng


def _run(hmm, observations, log_probabilities):
    hmm.forward_variable = None
    out = []
    with np.errstate(divide="ignore"):
        for o in observations:
            hmm.forward_algorithm_step(o, log_probabilities=log_probabilities)
            fv = hmm.forward_variable
            out.append(np.exp(fv) if log_probabilities else fv.copy())
    return np.array(out)


class TestForwardSelection(unittest.TestCase):

    def test_default_is_optimized(self):
        hmm, _ = _make_hmm(forward="optimized")
        default = HMM(hmm.observation_model, hmm.transition_model)
        self.assertEqual(default.forward, "optimized")
        self.assertIs(default._forward_step, forward_algorithm_step_optimized)

    def test_standard_selection(self):
        hmm, _ = _make_hmm(forward="standard")
        self.assertIs(hmm._forward_step, forward_algorithm_step)

    def test_invalid_value(self):
        hmm, _ = _make_hmm(forward="standard")
        with self.assertRaises(ValueError):
            HMM(hmm.observation_model, hmm.transition_model, forward="fast")

    def test_other_transition_models_use_standard(self):
        class MyTransitionModel(TransitionModel):
            n_states = 2

        hmm, _ = _make_hmm(forward="standard")
        hmm2 = HMM(hmm.observation_model, MyTransitionModel())
        self.assertIs(hmm2._forward_step, forward_algorithm_step)


class TestForwardEquivalence(unittest.TestCase):

    def _compare(self, T, n_steps=60, seed=0):
        for log_probabilities in (False, True):
            h_std, rng = _make_hmm("standard", T=T, seed=seed)
            h_opt, _ = _make_hmm("optimized", T=T, seed=seed)
            obs = rng.integers(0, 5, size=n_steps)
            fv_std = _run(h_std, obs, log_probabilities)
            fv_opt = _run(h_opt, obs, log_probabilities)
            np.testing.assert_allclose(fv_opt, fv_std, rtol=1e-10, atol=1e-12)
            np.testing.assert_array_equal(
                fv_opt.argmax(axis=1), fv_std.argmax(axis=1)
            )

    def test_banded_matrix(self):
        self._compare(_banded_transitions(80, max_jump=2))

    def test_wide_band(self):
        self._compare(_banded_transitions(80, max_jump=15))

    def test_dense_matrix(self):
        rng = np.random.default_rng(1)
        T = rng.random((30, 30))
        T /= T.sum(axis=1, keepdims=True)
        self._compare(T)

    def test_lower_triangular_and_zero_row(self):
        # Backward transitions and a row without any transition.
        rng = np.random.default_rng(2)
        T = np.tril(rng.random((20, 20)))
        T[5] = 0.0
        self._compare(T, n_steps=20)

    def test_standalone_function(self):
        h, rng = _make_hmm("standard")
        om, tm = h.observation_model, h.transition_model
        fv_std = fv_opt = None
        with np.errstate(divide="ignore"):
            for o in rng.integers(0, 5, size=40):
                fv_std = forward_algorithm_step(om, tm, o, fv_std, False)
                fv_opt = forward_algorithm_step_optimized(om, tm, o, fv_opt, False)
        np.testing.assert_allclose(fv_opt, fv_std, rtol=1e-10, atol=1e-12)

    def test_updating_transition_probabilities(self):
        """The cached column bounds must follow the transition matrix."""
        h_opt, rng = _make_hmm("optimized")
        h_std, _ = _make_hmm("standard")
        T_new = _banded_transitions(50, max_jump=6, rng=np.random.default_rng(5))
        for h in (h_opt, h_std):
            with np.errstate(divide="ignore"):
                h.transition_model.transition_probabilities = T_new
        obs = rng.integers(0, 5, size=30)
        np.testing.assert_allclose(
            _run(h_opt, obs, False), _run(h_std, obs, False), rtol=1e-10, atol=1e-12
        )


class TestForwardSpeed(unittest.TestCase):

    @staticmethod
    def _time(hmm, obs, repeats=5):
        best = np.inf
        for _ in range(repeats):
            hmm.forward_variable = None
            t0 = time.perf_counter()
            for o in obs:
                hmm.forward_algorithm_step(o, log_probabilities=False)
            best = min(best, time.perf_counter() - t0)
        return best

    def test_optimized_is_faster_on_large_banded_model(self):
        n_states = 3000
        T = _banded_transitions(n_states, max_jump=2)
        h_std, rng = _make_hmm("standard", T=T)
        h_opt, _ = _make_hmm("optimized", T=T)
        obs = rng.integers(0, 5, size=100)

        # warm-up
        self._time(h_std, obs[:5], repeats=1)
        self._time(h_opt, obs[:5], repeats=1)

        t_std = self._time(h_std, obs)
        t_opt = self._time(h_opt, obs)
        print(
            f"\nforward step ({n_states} states, 100 steps): "
            f"standard {t_std * 1e3:.1f} ms, optimized {t_opt * 1e3:.1f} ms, "
            f"speed-up x{t_std / t_opt:.1f}"
        )
        # Conservative threshold to avoid flakiness on shared CI runners.
        self.assertLess(t_opt * 2, t_std)


if __name__ == "__main__":
    unittest.main()
