"""Warning category and helper for the weights subpackage."""

from __future__ import annotations

import os
import sys
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_MARKER = f"{os.sep}tests{os.sep}"


class WeightWarning(UserWarning):
    """Warning emitted by participant weighting.

    A dedicated subclass of :class:`UserWarning`, so weight warnings can be
    filtered on their own::

        import warnings
        from cntmosaic.weights import WeightWarning

        warnings.filterwarnings("ignore", category=WeightWarning)
    """


def _external_stacklevel() -> int:
    """Stack level of the first frame outside the cntmosaic package (test
    modules count as outside), so warnings point at the user's call."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        inside = filename.startswith(_PACKAGE_DIR) and _TESTS_MARKER not in filename
        if not inside:
            return level
        frame = frame.f_back
        level += 1
    return level


def warn(message: str) -> None:
    """Emit a :class:`WeightWarning` attributed to the caller's code."""
    warnings.warn(message, WeightWarning, stacklevel=_external_stacklevel())
