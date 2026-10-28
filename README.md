# hiddenmarkov

[![PyPI](https://img.shields.io/pypi/v/python-hiddenmarkov)](https://pypi.org/project/python-hiddenmarkov/)
[![CI](https://github.com/CarlosCancino-Chacon/hiddenmarkov/actions/workflows/ci.yml/badge.svg)](https://github.com/CarlosCancino-Chacon/hiddenmarkov/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CarlosCancino-Chacon/hiddenmarkov/branch/main/graph/badge.svg)](https://codecov.io/gh/CarlosCancino-Chacon/hiddenmarkov)

A simple Python library for Hidden Markov Models.

This library was created for educational purposes and does not aim to have the most efficient implementation.

## Setup

Simple installation

```
pip install python-hiddenmarkov
```

Develop version
```bash
git clone https://github.com/CarlosCancino-Chacon/hiddenmarkov.git
cd hiddenmarkov
pip install -e .
```

## Usage

Example from [Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm#Example)

```python
import numpy as np
from hiddenmarkov import CategoricalObservationModel, ConstantTransitionModel, HMM

obs = ("normal", "cold", "dizzy")
observations = ("normal", "cold", "dizzy")
states = ("Healthy", "Fever")
observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])

observation_model = CategoricalObservationModel(
	observation_probabilities, obs
)

init_distribution = np.array([0.6, 0.4])

transition_model = ConstantTransitionModel(
	transition_probabilities, init_distribution
)

hmm = HMM(observation_model, transition_model, state_space=states)

path, prob = hmm.find_best_sequence(observations, log_probabilities=False)
print("Example Wikipedia")
print("Best sequence", path)
print("Expected Sequence", ["Healthy", "Healthy", "Fever"])
print("Sequence probability", prob)
print("Expected probability", 0.01512)
```

## Licence

The code in this package is licensed under the MIT Licence. For details,
please see the [LICENSE](https://github.com/CarlosCancino-Chacon/hiddenmarkov/blob/main/LICENSE) file.
