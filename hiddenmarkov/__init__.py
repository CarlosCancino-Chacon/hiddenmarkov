#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hidden Markov Models

This module contains case classes to define Hidden Markov Models.
"""
import warnings
import numpy as np
from importlib.metadata import version, PackageNotFoundError

from collections import defaultdict
from typing import Any, Iterable, Optional, Tuple, ClassVar, List

from scipy.special import logsumexp

from .utils import inverted_softmax, softmax


# define a version variable
try:
    __version__ = version("python-hiddenmarkov")
except PackageNotFoundError:
    __version__ = "unknown"


class TransitionModel(object):
    """
    Base class for implementing a Transition Model
    """

    init_probabilities: ClassVar[np.ndarray]
    n_states: int

    def __init__(self, use_log_probabilities: bool = True) -> None:
        self.use_log_probabilities: bool = use_log_probabilities

    def __call__(
        self,
        i: Optional[int] = None,
        j: Optional[int] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


class ObservationModel(object):
    """
    Base class for implementing an Observation Model
    """

    def __init__(self, use_log_probabilities: bool = True) -> None:
        self.use_log_probabilities = use_log_probabilities

    def __call__(self, observation: Any, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class HiddenMarkovModel(object):
    """
    Hidden Markov Model

    Parameters
    ----------
    observation_model : ObservationModel
        A model for computiong the observation (emission) probabilities.
    transition_model: TransitionModel
        A model for computing the transition probabilities.
    state_space: iterable (optional)
        Labels of the states (e.g., a list of strings containing
        the names of each state).

    Attributes
    ----------
    observation_model: ObservationModel
    transition_model: TransitionModel
    n_states: int
        Number of states
    state_space: np.array

    """

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        state_space: Optional[Iterable] = None,
    ) -> None:
        super().__init__()
        self.observation_model: ObservationModel = observation_model
        self.transition_model: TransitionModel = transition_model
        self.n_states: int = self.transition_model.n_states
        if state_space is not None:
            self.state_space: np.ndarray = np.asarray(state_space)
        else:
            self.state_space: np.ndarray = np.arange(self.n_states)

        self.forward_variable: Optional[np.ndarray] = None

    def find_best_sequence(
        self,
        observations: Iterable[Any],
        log_probabilities: bool = True,
        viterbi: str = "optimized",
    ) -> Tuple[np.ndarray, float]:
        """
        Find the best sequence of hidden states given a sequence of
        observations using the viterbi algorithm

        Parameters
        ----------
        observations : Iterable
            The sequence of observations.
        log_probabilities : bool
            If True, log  probabilities will be used.
        viterbi : {"optimized", "windowed", "naive"}
            The implementation of the viterbi algorithm.

        Returns
        -------
        best_sequence : np.ndarray
            The best sequence of hidden states
        sequence_likelihood : float
            The probability (or log probability if  `log_probabilities=True`)
            of the sequence of observations.
        """
        if viterbi == "optimized":
            viterbi_fun = viterbi_algorithm
        elif viterbi == "windowed":
            viterbi_fun = viterbi_algorithm_windowed
        elif viterbi == "naive":
            viterbi_fun = viterbi_algorithm_naive
        else:
            raise ValueError(
                "`viterbi` needs to be 'optimized', 'windowed', or 'naive' "
                f"but is {viterbi}"
            )
            # warnings.warn("viterbi needs to be 'optimized', 'windowed', or 'naive'")
            # return
        best_sequence, sequence_likelihood = viterbi_fun(
            hmm=self,
            observations=observations,
            log_probabilities=log_probabilities,
        )
        return best_sequence, sequence_likelihood

    def forward_algorithm_step(
        self,
        observation: Any,
        log_probabilities: bool = False,
    ) -> int:
        """
        Find the hidden state that has the maximal probability
        given the current observation.

        Parameters
        ----------
        observation : Any
            The current observation
        log_probabilities: bool
            If True, log  probabilities will be used.

        Returns
        -------
        current_state : int
            Index of the current hidden state.
        """
        self.forward_variable = forward_algorithm_step(
            observation_model=self.observation_model,
            transition_model=self.transition_model,
            observation=observation,
            forward_variable=self.forward_variable,
            log_probabilities=log_probabilities,
        )
        current_state: int = np.argmax(self.forward_variable)

        return current_state
        # pass


# alias
HMM = HiddenMarkovModel


def viterbi_algorithm_naive(
    hmm: HiddenMarkovModel,
    observations: Iterable,
    log_probabilities: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Find the most probable sequence of latent variables given
    a sequence of observations

    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.

    Note
    ----
    This is a naïve implementation, mostly for educational purposes!
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize dictionary for tracking the best paths
    path = defaultdict(lambda: list())

    # Initiate for i == 0
    obs_prob = hmm.observation_model(observations[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_probabilities
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_probabilities

    # Viterbi recursion
    for i, obs in enumerate(observations[1:], 1):
        obs_prob = hmm.observation_model(obs)
        for j in range(hmm.n_states):
            if log_probabilities:
                prob, state = max(
                    [
                        (omega[i - 1, k] + hmm.transition_model(k, j), k)
                        for k in range(hmm.n_states)
                    ],
                    key=lambda x: x[0],
                )
                omega[i, j] = obs_prob[j] + prob

            else:
                prob, state = max(
                    [
                        (omega[i - 1, k] * hmm.transition_model(k, j), k)
                        for k in range(hmm.n_states)
                    ],
                    key=lambda x: x[0],
                )
                omega[i, j] = obs_prob[j] * prob
            # keep track of the best state
            path[j].append(state)

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]
    # follow the best path backwards
    seq = [best_sequence_idx]
    for s in range(len(path[best_sequence_idx])):
        best_sequence_idx = path[best_sequence_idx][-(s + 1)]
        seq.append(best_sequence_idx)
    # invert the path
    best_sequence = np.array(seq[::-1], dtype=int)

    if hmm.state_space is not None:
        best_sequence = hmm.state_space[best_sequence]

    return best_sequence, path_likelihood


def viterbi_algorithm(
    hmm: HiddenMarkovModel,
    observations: Iterable[Any],
    log_probabilities: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Find the most probable sequence of latent variables given
    a sequence of observations

    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize matrix for holding the best sub-sequence idx
    omega_idx = np.zeros((len(observations), hmm.n_states), dtype=int)

    # Initiate for i == 0
    obs_prob = hmm.observation_model(observations[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_probabilities
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_probabilities

    omega_idx[0, :] = 0

    # Viterbi recursion
    if log_probabilities:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob = hmm.observation_model(obs)
            # omega slice is a row vector, transition_model is a matrix
            # of prob from state id_row to state id_column
            prob_of_jump_to_state = omega[i - 1, :] + hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis=1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states), state]
            omega[i, :] = obs_prob + prob
            omega_idx[i, :] = state

    else:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob = hmm.observation_model(obs)
            # omega slice is a row vector, transition_model is a matrix
            # of prob from state id_row to state id_column
            prob_of_jump_to_state = omega[i - 1, :] * hmm.transition_model().T
            state = np.argmax(prob_of_jump_to_state, axis=1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states), state]
            omega[i, :] = obs_prob * prob
            omega_idx[i, :] = state

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]
    # Get best path (backtracking!)
    seq = [best_sequence_idx]
    for s in range(len(observations) - 1):
        best_sequence_idx = omega_idx[-(s + 1), best_sequence_idx]
        seq.append(best_sequence_idx)
    best_sequence = np.array(seq[::-1], dtype=int)
    if hmm.state_space is not None:
        best_sequence = hmm.state_space[best_sequence]

    return best_sequence, path_likelihood


def viterbi_algorithm_windowed(
    hmm: HiddenMarkovModel,
    observations: Iterable[Any],
    log_probabilities: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Find the most probable sequence of latent variables given
    a sequence of observations.

    !!! This version uses a transition model with fixed window of
    states (see WindowedHiddenMarkiovModel). This window represents
    the currently "visible" states from any of an underlying
    longer sequence of global states.

    The viterbi uses the current best path at each step to
    update the global path. This effectively limits the path
    to corridor in the states. The observation model is called
    with a global state id, so the true, underlying states
    can have different probabilities.


    Parameters
    ----------
    observations: iterable
       An iterable containing observations. The type of each
       element depends on input types accepted by the
       `hmm.observation_model`
    log_probabilities: Bool (optional)
       If True, uses log probabilities to compute the Viterbi
       recursion (better for numerical stability). Default is True.

    Returns
    -------
    path: np.ndarray
        The most probable sequence of latent variables
    likelihood: float
        The likelihood (either the probability or the
        log proability if `log_probabilities` is True)
        of the best sequence.
    """
    # Set whether to use log probabilities in transition and
    # observation models
    hmm.transition_model.use_log_probabilities = log_probabilities
    hmm.observation_model.use_log_probabilities = log_probabilities
    # Initialize matrix for holding the best sub-sequence
    # (log-)likelihood
    omega = np.zeros((len(observations), hmm.n_states))
    # Initialize matrix for holding the best sub-sequence idx
    omega_idx = np.zeros((len(observations), hmm.n_states), dtype=int)

    # Initialize vector for holding the current best idx
    # Current window id; start at state 0 and keep the state at the
    # start of the current window
    current_window_idx = np.zeros((len(observations) + 1), dtype=int)

    omega_idx[0, :] = 0
    current_window_idx[0] = 0  # zero is a dummy state just for jump size
    current_window_idx[1] = 0  # zero is the actual first window idx

    # Initiate for i == 0
    obs_prob, _ = hmm.observation_model(observations[0], current_window_idx[0])

    if log_probabilities:
        omega[0, :] = obs_prob + hmm.transition_model.init_probabilities
    else:
        omega[0, :] = obs_prob * hmm.transition_model.init_probabilities

    # Viterbi recursion
    if log_probabilities:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob, glob_ref_idx = hmm.observation_model(obs, current_window_idx[i])
            # omega slice is a row vector, transition_model is a matrix
            # of prob from state id_row to state id_column
            # use only the slice of omega that is shifted by the previous jump
            previous_jump = current_window_idx[i] - current_window_idx[i - 1]
            prob_of_jump_to_state = (
                np.concatenate(
                    (omega[i - 1, previous_jump:], np.ones(previous_jump) * -np.inf)
                )
                + hmm.transition_model().T
            )
            state = np.argmax(prob_of_jump_to_state, axis=1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states), state]
            omega[i, :] = obs_prob + prob
            omega_idx[i, :] = state
            # slide the window to the current best
            current_best_sequence_idx = omega[i, :].argmax()
            current_window_idx[i + 1] = glob_ref_idx[current_best_sequence_idx]

    else:
        for i, obs in enumerate(observations[1:], 1):
            obs_prob, glob_ref_idx = hmm.observation_model(obs, current_window_idx[i])
            # omega slice is a row vector, transition_model is a matrix
            # of prob from state id_row to state id_column
            # use only the slice of omega that is shifted by the previous jump
            previous_jump = current_window_idx[i] - current_window_idx[i - 1]
            prob_of_jump_to_state = (
                np.concatenate((omega[i - 1, previous_jump:], np.zeros(previous_jump)))
                * hmm.transition_model().T
            )
            state = np.argmax(prob_of_jump_to_state, axis=1)
            prob = prob_of_jump_to_state[np.arange(hmm.n_states), state]
            omega[i, :] = obs_prob * prob
            omega_idx[i, :] = state
            # slide the window to the current best
            current_best_sequence_idx = omega[i, :].argmax()
            current_window_idx[i + 1] = glob_ref_idx[current_best_sequence_idx]

    # Get best path (backtracking!)
    # Get index of the best state
    best_sequence_idx = omega[-1, :].argmax()
    # likelihood of the path
    path_likelihood = omega[-1, best_sequence_idx]

    return current_window_idx[1:], path_likelihood


def forward_algorithm_step(
    observation_model: ObservationModel,
    transition_model: TransitionModel,
    observation: Any,
    forward_variable: Optional[np.ndarray] = None,
    log_probabilities: bool = False,
) -> np.ndarray:
    """
    Step of the forward algorithm.

    Parameters
    ----------
    observation_model : ObservationModel
        An observation model.
    transition_model: TransitionModel
        A transition model.
    observation: Any
        An observation. The observation needs to be of the same
        type as specified by the observation model.
    forward_variable : np.ndarray
        The forward variable of the HMM. If none is given, the
        initial probabilities specified by the transition model
        will be used.
    log_probabilities : bool
        If true, use log probabilities.

    Returns
    -------
    forward_variable : np.ndarray
        The updated forward variable.
    """
    # since computing the log of the matrix vector multiplication is not
    # trivial (TODO: see if there is a clever way to do this multiplication
    # efficiently)
    transition_model.use_log_probabilities = False
    observation_model.use_log_probabilities = log_probabilities
    if forward_variable is None:
        transition_prob: np.ndarray = transition_model.init_probabilities
    else:
        transition_prob: np.ndarray = np.dot(transition_model().T, forward_variable)
    observation_prob: np.ndarray = observation_model(observation)
    if log_probabilities:
        forward_variable = observation_prob + np.log(transition_prob)
        forward_variable -= logsumexp(forward_variable)
    else:
        forward_variable = observation_prob * transition_prob
        forward_variable /= max(forward_variable.sum(), 1e-6)

    return forward_variable


class ConstantTransitionModel(TransitionModel):
    """
    Constant Transition Model

    This transition model represents the case were the
    transition proabilities do not change over time (i.e.,
    they are static). In this case, the transition probabilities
    can be represented by a transition matrix

    Parameters
    ----------
    transition_probabilities: np.ndarray
        A (n_states, n_states) matrix where component
        [i, j] represents the probability of going to state j
        coming from state i.
    init_probabilities: np.ndarray or None (optional)
        A 1D vector of length n_states defining the initial
        probabilities of each state
    normalize_init_probabilities: bool (optional)
        If True, the initial distribution will be normalized.
        Default is False.
    use_log_probabilities: bool (optional)
        If True, use log proabilities instead of norm proabilities
        (better for numerical stability)
    """

    def __init__(
        self,
        transition_probabilities: np.ndarray,
        init_probabilities: Optional[np.ndarray] = None,
        normalize_init_probabilities: bool = False,
        normalize_transition_probabilities: bool = False,
        use_log_probabilities: bool = True,
    ) -> None:
        super().__init__(use_log_probabilities=use_log_probabilities)
        self.transition_probabilities = transition_probabilities
        self.n_states = len(transition_probabilities)

        if init_probabilities is None:
            self.init_probabilities = (
                1.0 / float(self.n_states) * np.ones(self.n_states, dtype=float)
            )
        else:
            self.init_probabilities = init_probabilities

        if normalize_init_probabilities:
            # Normalize initial distribution
            self.init_probabilities /= np.maximum(
                np.sum(self.init_probabilities), 1e-10
            )

        if normalize_transition_probabilities:
            self.transition_probabilities /= np.sum(
                self.transition_probabilities, 1, keepdims=True
            )

    @property
    def init_probabilities(self) -> np.ndarray:
        if self.use_log_probabilities:
            return self._log_init_dist
        else:
            return self._init_dist

    @init_probabilities.setter
    def init_probabilities(self, init_probabilities: np.ndarray) -> None:
        self._init_dist = init_probabilities
        self._log_init_dist = np.log(self._init_dist)

    @property
    def transition_probabilities(self) -> np.ndarray:
        if self.use_log_probabilities:
            return self._log_transition_prob
        else:
            return self._transition_prob

    @transition_probabilities.setter
    def transition_probabilities(self, transition_probabilities) -> None:
        self._transition_prob = transition_probabilities
        self._log_transition_prob = np.log(self._transition_prob)

    def __call__(self, i=None, j=None) -> np.ndarray:
        if i is None and j is None:
            return self.transition_probabilities
        elif i is not None and j is None:
            return self.transition_probabilities[i, :]
        elif i is None and j is not None:
            return self.transition_probabilities[:, j]
        else:
            return self.transition_probabilities[i, j]


class CategoricalObservationModel(ObservationModel):
    """
    A Categorical observation model.

    Parameters
    ----------
    observation_probabilities : np.ndarray
        A table of probabilities for each observation in each state.
    observations : Iterable
        A list of the observations
    use_log_probabilities : bool
        If True, use log probabilities.
    """

    def __init__(
        self,
        observation_probabilities: np.ndarray,
        observations: Iterable[Any] = None,
        use_log_probabilities: bool = True,
    ):
        super().__init__(use_log_probabilities=use_log_probabilities)

        self.observation_probabilities = observation_probabilities

        if observations is not None:
            self.observations = list(observations)
        else:
            self.observations = [str(i) for i in range(len(observation_probabilities))]

        self.observation_indices = dict(
            [(obs, i) for obs, i in zip(self.observations, range(len(observations)))]
        )

    @property
    def observation_probabilities(self):
        if self.use_log_probabilities:
            return self._log_obs_prob
        else:
            return self._obs_prob

    @observation_probabilities.setter
    def observation_probabilities(self, observation_probabilities):
        self._obs_prob = observation_probabilities
        self._log_obs_prob = np.log(self._obs_prob)

    def __call__(self, observation, *args, **kwargs):
        idx = self.observation_indices[observation]
        # idx = self.observations.index(observation)
        return self.observation_probabilities[idx]


# deprecated alias
CategoricalStringObservationModel = CategoricalObservationModel


class WindowedObservationModel(ObservationModel):
    """
    Windowed Observation Model
    Uses a list of state-specific probability models
    as observation model. Probability models can also be distances,
    in which case the different distances are inverted and softmaxed.

    Only works with viterbi_algorithm_windowed

    Parameters
    ----------
    prob_models_at_state : list of tuples
        A list of tuples (probability_model,prob_model_state_id)
        indexed by state_id

    Attributes
    ----------
    prob_models_at_state: list of tuples

    """

    def __init__(
        self,
        prob_models_at_state: List[tuple],
        use_inverted_probs: bool = True,
        use_log_probabilities: bool = True,
    ):
        super().__init__(use_log_probabilities=use_log_probabilities)
        self.use_log_probabilities = use_log_probabilities
        self.use_inverted_probs = use_inverted_probs
        self.prob_models_at_state = prob_models_at_state
        self.state_number = len(prob_models_at_state)
        if self.use_inverted_probs:
            if self.use_log_probabilities:
                self.outfunc = lambda d: np.log(inverted_softmax(np.array(d)))
            else:
                self.outfunc = lambda d: inverted_softmax(np.array(d))
        else:
            if self.use_log_probabilities:
                self.outfunc = lambda d: np.log(softmax(np.array(d)))
            else:
                self.outfunc = lambda d: softmax(np.array(d))

    def __call__(self, observation, current_state, *args, **kwargs):
        # give the current state pick the right states
        prob_models = self.prob_models_at_state[
            max(0, min(current_state, self.state_number - 1))
        ]
        # compute the probability for all models
        dists = list()
        glob_ref_idx = list()

        for prob_model, idx in prob_models:
            dists.append(prob_model(observation))
            glob_ref_idx.append(idx)

        return self.outfunc(dists), np.array(glob_ref_idx)
