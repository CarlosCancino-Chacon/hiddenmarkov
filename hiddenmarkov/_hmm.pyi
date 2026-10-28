import numpy as np

def viterbi_loop(
    omega: np.ndarray,
    omega_idx: np.ndarray,
    obs_probs: np.ndarray,
    trans_mat: np.ndarray,
    init_probs: np.ndarray,
    log_probabilities: bool,
) -> None: ...
