"""QDGeo: Quick and dirty molecular geometry conformation optimization"""

try:
    import _qdgeo
except ImportError:
    # For development: try adding build directory to path
    import sys
    import os
    _build = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
    if os.path.exists(_build):
        for item in os.listdir(_build):
            if item.startswith('lib.'):
                sys.path.insert(0, os.path.join(_build, item))
                break
    import _qdgeo

from _qdgeo import Bond, Angle, Dihedral, Optimizer, optimize
from .optimize_mol import optimize_mol

__all__ = ["Bond", "Angle", "Dihedral", "Optimizer", "optimize", "optimize_mol"]
