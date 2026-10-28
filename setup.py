import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy as np

# ---------------------------------------------------------------------------
# Optional Cython compilation
#
# Strategy:
#   1. If Cython is available, compile from the .pyx source (ensures the
#      generated C code is always fresh).
#   2. If Cython is absent, fall back to the pre-generated _hmm.c that is
#      bundled in the source distribution — no Cython required for source
#      installs.
#   3. If compilation fails for any reason (no C compiler, unsupported
#      platform, …), the extension is silently skipped.  The package still
#      installs and works correctly; only the Cython speed-up is absent.
# ---------------------------------------------------------------------------

try:
    from Cython.Build import cythonize as _cythonize
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False

_PYX = "hiddenmarkov/_hmm.pyx"
_C   = "hiddenmarkov/_hmm.c"

_source = _PYX if _USE_CYTHON else _C
_can_build = _USE_CYTHON or os.path.exists(_C)

_extension = Extension(
    "hiddenmarkov._hmm",
    [_source],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

if _USE_CYTHON:
    _ext_modules = _cythonize(
        [_extension],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
elif _can_build:
    _ext_modules = [_extension]
else:
    _ext_modules = []


class _OptionalBuildExt(_build_ext):
    """Build the C extension, but don't fail the install if it can't compile."""

    def run(self):
        try:
            super().run()
        except Exception:
            _warn_no_extension()

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            _warn_no_extension()


def _warn_no_extension():
    import warnings
    warnings.warn(
        "Could not compile the Cython extension for hiddenmarkov._hmm. "
        "The package will work correctly but without the Viterbi speed-up. "
        "Install a C compiler (e.g. gcc or clang) and re-install to get it.",
        stacklevel=2,
    )


setup(
    ext_modules=_ext_modules,
    cmdclass={"build_ext": _OptionalBuildExt},
)
