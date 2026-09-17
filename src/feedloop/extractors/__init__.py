"""Optional learned extractors behind the ``[extract]`` extra. Importing this package
loads nothing learned; each extractor imports its runtime when ``load()`` runs."""
from __future__ import annotations

import importlib

EXTRA_HINT = "pip install 'feedloop[extract]'"


class MissingExtra(ImportError):
    """A learned runtime is not installed; metadata mode keeps working without it."""


def require(module: str, *, purpose: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(f"{purpose} needs the optional module '{module}'; install the extractor extra with: {EXTRA_HINT}") from exc


__all__ = ["MissingExtra", "require", "EXTRA_HINT"]
