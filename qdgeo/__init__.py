"""QDGeo: Quick and dirty molecular geometry conformation optimization"""

import sys
import os

# Import compiled C++ extension (_qdgeo.so)
# For editable installs, add build directory to path
_build = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
if os.path.exists(_build):
    for item in os.listdir(_build):
        if item.startswith('lib.'):
            path = os.path.join(_build, item)
            if path not in sys.path:
                sys.path.insert(0, path)
            break

import _qdgeo

# Re-export extension API
Bond = _qdgeo.Bond
Angle = _qdgeo.Angle
Optimizer = _qdgeo.Optimizer
optimize = _qdgeo.optimize

from .optimize_mol import optimize_mol

__all__ = ["Bond", "Angle", "Optimizer", "optimize", "optimize_mol"]
