# Release Notes

## Version 0.2.0

* Fix log-probability bugs in `ConstantTransitionModel` (normalization operated on log values instead of raw probabilities), `forward_algorithm_step` (dot product used log forward variable), and `WindowedObservationModel` (output function not updated when `use_log_probabilities` changed after construction).
* Add optional Cython extension (`_hmm.pyx`) for a ~2× speedup of the Viterbi inner loop; falls back gracefully to NumPy when the extension is not available.
* Distribute pre-built wheels for Linux, macOS, and Windows via cibuildwheel; pre-generated `_hmm.c` included in the sdist so source builds do not require Cython.
* Add full type annotations (`from __future__ import annotations`, PEP 561 `py.typed` marker, `_hmm.pyi` stub).
* Reorganize test suite into topic-based modules (`test_models`, `test_algorithms`, `test_utils`); achieve 100% coverage on the public package.
* Add CI workflow (`.github/workflows/ci.yml`) that runs on every push and pull request across Linux, macOS, and Windows for Python 3.9–3.13, including an sdist smoke-test job.

## Version 0.1.4 (prerelease)

* Migrate package setup to `pyproject.toml`.
* Replace deprecated `pkg_resources` with `importlib.metadata`.

## Version 0.1.3 (prerelease)

* Rename `CategoricalStringObservationModel` to `CategoricalObservationModel`. The old name is now an alias but will be removed in a future release.
* Cleanup documentation and add typing hints. 
* Consistently declare encoding (utf-8) in all python files.
* Add more tests to increase coverage (not yet 100%).

## Version 0.1.2 (prerelease)

Code cleanup and iron things for PyPI.

## Version 0.1.1 (prerelease)

Add forward algorithm.

## Version 0.1.0 (prerelease)

This is the first prerelease of the package. In this release, the implementation of the forward algorithm is missing.
