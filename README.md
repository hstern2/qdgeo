# QDGeo: Quick and Dirty Molecular Geometry Optimization

QDGeo is a library for optimizing molecular geometries using conjugate gradient minimization. It takes a set of atoms, bonds (with ideal lengths), and angles (with ideal angles) and finds the 3D coordinates that minimize a harmonic energy function. You can also give it an RDKit Mol object.

## Features

- Fast conjugate gradient minimization using Polak-Riviere algorithm
- Harmonic energy function for bonds and angles
- Automatic generation of random initial coordinates
- Python interface with NumPy arrays
- RDKit integration for molecular input
- Efficient C++ backend

## Installation

```bash
uv pip install .
```

## Usage

### Using RDKit Mol object

```python
from rdkit import Chem
import qdgeo

# Create molecule from SMILES
mol = Chem.MolFromSmiles('CCO')  # Ethanol
mol = Chem.AddHs(mol)

# Optimize geometry
coords = qdgeo.optimize_mol(mol, verbose=1)
print(f"Optimized coordinates shape: {coords.shape}")
```

The `optimize_mol` function automatically extracts bonds and angles from the RDKit molecule and assigns reasonable bond lengths and angles based on atom types and hybridization.

### Low-Level API

```python
import numpy as np
import qdgeo

# Define bonds and angles
bonds = [(0, 1, 1.0), (1, 2, 1.0)]  # (atom1, atom2, ideal_length)
angles = [(0, 1, 2, np.pi / 3)]      # (atom1, atom2, atom3, ideal_angle_in_radians)

# Optimize
coords, converged, energy = qdgeo.optimize(
    n_atoms=3,
    bonds=bonds,
    angles=angles,
    bond_force_constant=1.0,
    angle_force_constant=1.0
)

print(f"Converged: {converged}, Energy: {energy}")
```

## API Reference

### `qdgeo.optimize(n_atoms, bonds, angles, ...)`

Low-level optimization function.

**Parameters:**
- `n_atoms`: Number of atoms
- `bonds`: List of `(atom1, atom2, ideal_length)` tuples
- `angles`: List of `(atom1, atom2, atom3, ideal_angle)` tuples (angles in radians)
- `bond_force_constant`: Force constant for bonds (default: 1.0)
- `angle_force_constant`: Force constant for angles (default: 1.0)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `maxeval`: Maximum function evaluations (default: 1000)
- `verbose`: Verbosity level (default: 0)

**Returns:** `(coords, converged, energy)` tuple

### `qdgeo.optimize_mol(mol, ...)`

High-level function for optimizing RDKit molecules.

**Parameters:**
- `mol`: RDKit molecule object
- `bond_k`: Bond force constant (default: 1.5)
- `angle_k`: Angle force constant (default: 1.0)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `maxeval`: Maximum function evaluations (default: 3000)
- `verbose`: If > 0, prints bond lengths and angles (default: 0)

**Returns:** Optimized coordinates array, shape `(n_atoms, 3)`
