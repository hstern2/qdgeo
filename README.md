# QDGeo: Quick and Dirty Molecular Geometry Optimization

Optimizes molecular geometries using L-BFGS minimization with harmonic bond/angle terms, dihedral constraints, and van der Waals repulsion.

## Installation

```bash
uv pip install .  # requires python 3.10-3.12
```

## Quick Start

```python
from rdkit import Chem
import qdgeo

mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
coords = qdgeo.optimize_mol(mol)
```

## API

### `qdgeo.optimize_mol(mol, ...)`

**Parameters:**
- `mol`: RDKit molecule
- `bond_k`, `angle_k`, `dihedral_k`: Force constants (defaults: 1.5, 2.0, 5.0)
- `dihedral`: Dict of `{(i,j,k,l): angle_degrees}` constraints
- `repulsion_k`: van der Waals repulsion strength (default: 0.01)
- `n_starts`: Random starting points (default: 10)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `maxeval`: Maximum evaluations (default: 5000)
- `template`: RDKit molecule with conformer for MCS-based coordinate restraints
- `template_coordinate_k`: Template restraint strength (default: 10.0)

**Returns:** NumPy array of shape `(n_atoms, 3)`

### `qdgeo.optimize(n_atoms, bonds, angles, ...)`

Low-level function returning `(coords, converged, energy)`.

## Testing

```bash
pytest
```

Generates SDF files in `test_output/` for visual inspection.
