# QDGeo: Quick & Dirty Molecular Geometry Construction

Builds molecular 3D geometries using ideal bond lengths and angles with specified torsion angles. No optimization - just direct coordinate construction for maximum speed.

## Installation

```bash
uv pip install .  # requires python 3.10-3.12
```

## Quick Start

```python
from rdkit import Chem
import qdgeo

# Build with default (anti/180°) torsions
mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
coords = qdgeo.build_mol(mol)

# Build with specific torsion angles
c_atoms = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == 'C']
coords = qdgeo.build_mol(mol, torsions={
    (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0  # gauche
})
```

## API

### `qdgeo.build_mol(mol, torsions=None, verbose=0)`

Build 3D coordinates for an RDKit molecule.

**Parameters:**
- `mol`: RDKit molecule (with implicit or explicit hydrogens)
- `torsions`: Dict mapping `(i, j, k, l)` atom indices to dihedral angles in degrees.
  The torsion is defined around the j-k bond. Unspecified torsions default to 180° (anti).
- `verbose`: Verbosity level (0=silent, 1=info)

**Returns:** NumPy array of shape `(n_atoms, 3)` with Cartesian coordinates in Angstroms

### `qdgeo.build_molecule(n_atoms, bonds, angles, torsions, rings)`

Low-level function for direct coordinate construction.

**Parameters:**
- `n_atoms`: Number of atoms
- `bonds`: List of `(atom1, atom2, ideal_length)` tuples
- `angles`: List of `(atom1, center, atom2, angle_rad)` tuples
- `torsions`: List of `(a1, a2, a3, a4, angle_rad)` tuples
- `rings`: List of ring atom index lists

**Returns:** NumPy array of shape `(n_atoms, 3)`

### `qdgeo.MoleculeBuilder`

Class-based API for fine-grained control:

```python
builder = qdgeo.MoleculeBuilder(n_atoms=4)
builder.add_bond(0, 1, 1.54)
builder.add_bond(1, 2, 1.54)
builder.add_bond(2, 3, 1.54)
builder.set_angle(0, 1, 2, 1.91)  # ~109.5° in radians
builder.set_angle(1, 2, 3, 1.91)
builder.set_torsion(0, 1, 2, 3, 1.047)  # 60° in radians
coords = builder.build()
```

## Testing

```bash
pytest
```

Generates SDF files in `test_output/` for visual inspection.

## Performance

The rigid-body approach is deterministic and extremely fast - typically microseconds per molecule regardless of size, since it's just coordinate transforms with no iteration or optimization.
