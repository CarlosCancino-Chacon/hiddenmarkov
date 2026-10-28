# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-accelerated inner loop for the Viterbi algorithm.

Why Cython helps here but not in the forward step
--------------------------------------------------
The Viterbi step needs a row-wise argmax over the N×N matrix
    omega[t-1, i] + log_T[i, j]   for all (i, j).
NumPy handles this by broadcasting a length-N vector over the whole N×N
transition matrix, creating a fresh N×N temporary allocation at *every*
time step.  The Cython version replaces that with a doubly-nested C loop:
no temporaries, working set stays in L1/L2 cache for moderate N.

The forward step is different: its inner operation is a plain matrix-vector
multiply (np.dot), which NumPy hands to BLAS.  BLAS is already SIMD-
vectorised and for N ≳ 64 will outperform any pure-C scalar loop.  We
therefore keep the forward step in NumPy.
"""

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY

np.import_array()


def viterbi_loop(
    double[:, ::1] omega,
    np.int64_t[:, ::1] omega_idx,
    const double[:, ::1] obs_probs,
    const double[:, ::1] trans_mat,
    const double[:] init_probs,
    bint log_probabilities,
):
    """Fill the Viterbi trellis on pre-computed probability arrays.

    The caller is responsible for pre-computing *obs_probs* and *trans_mat*
    from the HMM's Python model objects, and for the backtracking step
    after this function returns.

    Parameters
    ----------
    omega : (T, N) float64, written in-place
        Viterbi trellis.
    omega_idx : (T, N) int64, written in-place
        Backtracking table; omega_idx[t, j] = best predecessor state for
        state j at time t.
    obs_probs : (T, N) float64
        Observation (log-)probabilities for every time step and state.
    trans_mat : (N, N) float64
        Transition (log-)probabilities; trans_mat[i, j] = P(j | i).
    init_probs : (N,) float64
        Initial state (log-)probabilities.
    log_probabilities : bool
        When True all arrays contain log-probabilities and additions replace
        multiplications; when False they contain raw probabilities.
    """
    cdef Py_ssize_t T = obs_probs.shape[0]
    cdef Py_ssize_t N = obs_probs.shape[1]
    cdef Py_ssize_t t, i, j
    cdef double best_val, val

    if log_probabilities:
        for j in range(N):
            omega[0, j] = obs_probs[0, j] + init_probs[j]
        for t in range(1, T):
            for j in range(N):
                best_val = -INFINITY
                omega_idx[t, j] = 0
                for i in range(N):
                    val = omega[t - 1, i] + trans_mat[i, j]
                    if val > best_val:
                        best_val = val
                        omega_idx[t, j] = i
                omega[t, j] = obs_probs[t, j] + best_val
    else:
        for j in range(N):
            omega[0, j] = obs_probs[0, j] * init_probs[j]
        for t in range(1, T):
            for j in range(N):
                best_val = 0.0
                omega_idx[t, j] = 0
                for i in range(N):
                    val = omega[t - 1, i] * trans_mat[i, j]
                    if val > best_val:
                        best_val = val
                        omega_idx[t, j] = i
                omega[t, j] = obs_probs[t, j] * best_val
