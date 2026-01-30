"""QDGeo: Quick & dirty molecular geometry construction"""

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

from _qdgeo import MoleculeBuilder, build_molecule
from .build_mol import build_mol

__all__ = ["MoleculeBuilder", "build_molecule", "build_mol"]
