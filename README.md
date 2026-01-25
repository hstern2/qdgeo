# QDGeo: Quick and Dirty Molecular Geometry Optimization

QDGeo is a library for optimizing molecular geometries using conjugate gradient minimization. It uses a harmonic energy function for bonds and angles, with optional dihedral constraints and non-bonded repulsion.

## Installation

```bash
uv pip install .
```

## Quick Start

```python
from rdkit import Chem
import qdgeo

# Create molecule from SMILES
mol = Chem.MolFromSmiles('CCO')  # Ethanol
mol = Chem.AddHs(mol)

# Optimize geometry
coords = qdgeo.optimize_mol(mol)
```

## Usage Examples

### Dihedral Constraints

```python
from rdkit import Chem
import qdgeo

mol = Chem.MolFromSmiles('CCCC')  # Butane
mol = Chem.AddHs(mol)

# Find carbon atoms
c_atoms = [i for i in range(mol.GetNumAtoms()) 
           if mol.GetAtomWithIdx(i).GetSymbol() == 'C']

# Constrain dihedral angle to 60 degrees
dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}

coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0)
```

### Template Restraints

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import qdgeo

# Create template with known geometry
template = Chem.MolFromSmiles('CCCC')
template = Chem.AddHs(template)
AllChem.EmbedMolecule(template)

# Target molecule
mol = Chem.MolFromSmiles('CCCCC')  # Pentane
mol = Chem.AddHs(mol)

# Optimize with template restraints
coords = qdgeo.optimize_mol(mol, template=template, template_k=5.0)
```

## API

### `qdgeo.optimize_mol(mol, ...)`

Main function for optimizing RDKit molecules.

**Parameters:**
- `mol`: RDKit molecule object
- `bond_k`: Bond force constant (default: 1.5)
- `angle_k`: Angle force constant (default: 2.0)
- `dihedral`: Dict of `{(i,j,k,l): angle_degrees}` constraints (default: None)
- `dihedral_k`: Dihedral force constant (default: 5.0)
- `repulsion_k`: Non-bonded repulsion force constant (default: 0.1)
- `repulsion_cutoff`: Repulsion cutoff distance in Å (default: 3.0)
- `n_starts`: Number of random starting points (default: 10)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `maxeval`: Maximum evaluations (default: 5000)
- `template`: RDKit molecule with conformer for restraints (default: None)
- `template_k`: Template dihedral force constant (default: 5.0)

**Returns:** `coords` - NumPy array of shape `(n_atoms, 3)`

### `qdgeo.optimize(n_atoms, bonds, angles, ...)`

Low-level optimization function. See code for full parameters.

**Returns:** `(coords, converged, energy)` tuple

## Testing

Run the test suite:

```bash
cd ~/qdgeo
source ~/venvs/qdgeo/bin/activate
pytest
```

**Output:** Generates 23 `.sdf` files in `test_output/` covering:
- Simple molecules (water, ethane, propane)
- Rings (cyclopropane, cyclohexane)
- Aromatics (benzene)
- Dihedral constraints (butane, pentane, hexane, butadiene)
- Template-based optimization
- Error handling

View the `.sdf` files in PyMOL, Avogadro, or any molecular viewer to verify geometry quality.

## Features

- Fast conjugate gradient minimization (Polak-Riviere algorithm)
- Harmonic bond and angle constraints
- Cos²-based dihedral constraints
- Non-bonded repulsion with 1/r¹² potential
- Multiple random starting points for better convergence
- Template-based restraints via substructure matching
- Efficient C++ backend with Python interface
