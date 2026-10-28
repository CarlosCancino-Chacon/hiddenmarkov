#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np


def inverted_softmax(array: np.ndarray) -> np.ndarray:
    """Convert distances to probabilities: lower distance → higher probability."""
    e = np.exp(-array)
    return e / e.sum()


def softmax(array: np.ndarray) -> np.ndarray:
    """Convert scores to probabilities: higher score → higher probability."""
    e = np.exp(array)
    return e / e.sum()


def create_prob_models(
    no_global_states: int = 100,
    window_size: int = 5,
) -> List[List[Tuple[Callable[..., Any], int]]]:
    """
    Build example windowed probability models for use with
    :class:`~hiddenmarkov.WindowedObservationModel` and
    :func:`~hiddenmarkov.viterbi_algorithm_windowed`.

    Parameters
    ----------
    no_global_states : int
        Total number of global states in the HMM.
    window_size : int
        Number of states visible at any one time.

    Returns
    -------
    list of lists of (callable, int) pairs
        ``result[i]`` is the window of ``(prob_model, global_index)`` pairs
        centred at global state ``i``.
    """
    models_per_state: List[List[Tuple[Callable[..., Any], int]]] = []
    for i in range(no_global_states):
        models: List[Tuple[Callable[..., Any], int]] = []
        for j in range(window_size):
            def probf(obs: Any, ref: int = i + j) -> int:
                return int(obs == ref)
            models.append((probf, max(0, min(no_global_states - 1, i + j))))
        models_per_state.append(models)
    return models_per_state
