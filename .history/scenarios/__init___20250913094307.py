# core package

"""scenarios package — convenience imports for submodules.

This lets you write:
    from scenarios import lump, annual, auto as auto_scn, housing
"""

from . import lump
from . import annual
from . import auto as auto_scn
from . import housing

__all__ = ["lump", "annual", "auto_scn", "housing"]