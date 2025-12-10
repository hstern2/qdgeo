# QDGeo: Quick and Dirty Molecular Geometry Optimization

QDGeo is a library for optimizing molecular geometries using conjugate gradient minimization. It takes a set of atoms, bonds (with ideal lengths), angles (with ideal angles), and optionally dihedral constraints, then finds the 3D coordinates that minimize a harmonic energy function. You can also give it an RDKit Mol object.

## Features

- Fast conjugate gradient minimization using Polak-Riviere algorithm
- Harmonic energy function for bonds and angles
- Torsional restraints with cos²-based dihedral constraints
- Non-bonded repulsion to prevent atom overlap
- Automatic exclusion list for 1-2, 1-3, and 1-4 interactions
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

### Using Dihedral Constraints

```python
from rdkit import Chem
import qdgeo

# Create molecule
mol = Chem.MolFromSmiles('CCCC')  # Butane
mol = Chem.AddHs(mol)

# Find carbon atoms
c_atoms = [i for i in range(mol.GetNumAtoms()) 
           if mol.GetAtomWithIdx(i).GetSymbol() == 'C']

# Constrain dihedral angle to 60 degrees
dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}

# Optimize with dihedral constraint and repulsion
coords = qdgeo.optimize_mol(
    mol, 
    dihedral=dihedral_dict,
    dihedral_k=5.0,           # Dihedral force constant
    repulsion_k=0.1,          # Non-bonded repulsion
    repulsion_cutoff=3.0      # Repulsion cutoff in Angstroms
)
```

Dihedral constraints use a cos²-based harmonic term. The dictionary keys are tuples of 4 atom indices `(i, j, k, l)` and values are target angles in degrees. Ring dihedrals are automatically excluded.

### Low-Level API

```python
import numpy as np
import qdgeo

# Define bonds and angles
bonds = [(0, 1, 1.0), (1, 2, 1.0)]  # (atom1, atom2, ideal_length)
angles = [(0, 1, 2, np.pi / 3)]      # (atom1, atom2, atom3, ideal_angle_in_radians)
dihedrals = [(0, 1, 2, 3, np.deg2rad(60.0))]  # (atom1, atom2, atom3, atom4, ideal_dihedral_in_radians)

# Optimize with repulsion and dihedral constraints
coords, converged, energy = qdgeo.optimize(
    n_atoms=4,
    bonds=bonds,
    angles=angles,
    dihedrals=dihedrals,
    bond_force_constant=1.0,
    angle_force_constant=1.0,
    dihedral_force_constant=5.0,
    repulsion_force_constant=0.1,    # Non-bonded repulsion
    repulsion_cutoff=3.0             # Cutoff in Angstroms
)

print(f"Converged: {converged}, Energy: {energy}")
```

Non-bonded repulsion uses a 1/r¹² potential. The exclusion list automatically excludes 1-2 (bonded), 1-3 (angle), and 1-4 (dihedral) interactions from repulsion.

The optimizer tries multiple random starting points (10 by default) and returns the lowest energy conformer, helping find better local minima.

## API Reference

### `qdgeo.optimize(n_atoms, bonds, angles, ...)`

Low-level optimization function.

**Parameters:**
- `n_atoms`: Number of atoms
- `bonds`: List of `(atom1, atom2, ideal_length)` tuples
- `angles`: List of `(atom1, atom2, atom3, ideal_angle)` tuples (angles in radians)
- `bond_force_constant`: Force constant for bonds (default: 1.0)
- `angle_force_constant`: Force constant for angles (default: 1.0)
- `dihedrals`: List of `(atom1, atom2, atom3, atom4, ideal_dihedral)` tuples in radians (default: empty)
- `dihedral_force_constant`: Force constant for dihedrals (default: 1.0)
- `repulsion_force_constant`: Non-bonded repulsion force constant (default: 0.0, disabled)
- `repulsion_cutoff`: Repulsion cutoff distance in Angstroms (default: 3.0)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `linesearch_tolerance`: Line search tolerance (default: 0.5)
- `maxeval`: Maximum function evaluations (default: 1000)
- `verbose`: Verbosity level (default: 0)
- `n_starts`: Number of random starting points to try (default: 10)

**Returns:** `(coords, converged, energy)` tuple

### `qdgeo.optimize_mol(mol, ...)`

High-level function for optimizing RDKit molecules.

**Parameters:**
- `mol`: RDKit molecule object
- `bond_k`: Bond force constant (default: 1.5)
- `angle_k`: Angle force constant (default: 2.0)
- `dihedral`: Optional dictionary of dihedral constraints. Keys are tuples of 4 atom indices `(i, j, k, l)`, values are target angles in degrees (default: None)
- `dihedral_k`: Dihedral force constant (default: 5.0)
- `repulsion_k`: Non-bonded repulsion force constant (default: 0.0, disabled)
- `repulsion_cutoff`: Repulsion cutoff distance in Angstroms (default: 3.0)
- `n_starts`: Number of random starting points to try (default: 10)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `maxeval`: Maximum function evaluations (default: 5000)
- `verbose`: If > 0, prints bond lengths, angles, and dihedrals (default: 0)

**Returns:** Optimized coordinates array, shape `(n_atoms, 3)`

**Note:** Dihedral constraints are only applied to non-ring bonds. Ring dihedrals are automatically excluded.
