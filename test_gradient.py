#!/usr/bin/env python3
"""Test gradient calculation for coordinate constraints."""

import numpy as np
from rdkit import Chem
import qdgeo

# Simple test: water molecule with coordinate constraint
mol = Chem.MolFromSmiles('O')
mol = Chem.AddHs(mol)
print(f"Molecule: {mol.GetNumAtoms()} atoms")

# Create a simple coordinate constraint - fix one H atom
coords = qdgeo.optimize_mol(mol, verbose=0, maxeval=100)
print(f"Initial coords shape: {coords.shape}")

# Test gradient numerically
def numerical_gradient(coords, constraint_pos, k, atom_idx, coord_idx, eps=1e-6):
    """Calculate numerical gradient."""
    coords_plus = coords.copy()
    coords_plus[atom_idx, coord_idx] += eps
    
    # Energy with coordinate constraint
    diff_orig = coords[atom_idx] - constraint_pos
    e_orig = 0.5 * k * np.sum(diff_orig**2)
    
    diff_plus = coords_plus[atom_idx] - constraint_pos
    e_plus = 0.5 * k * np.sum(diff_plus**2)
    
    return (e_plus - e_orig) / eps

# Test with a constraint
constraint_pos = np.array([1.0, 0.0, 0.0])
k = 10.0
atom_idx = 1  # First H atom

# Analytical gradient should be: k * (coords[atom_idx] - constraint_pos)
analytical_grad = k * (coords[atom_idx] - constraint_pos)
print(f"\nAnalytical gradient: {analytical_grad}")

# Numerical gradients
num_grad_x = numerical_gradient(coords, constraint_pos, k, atom_idx, 0)
num_grad_y = numerical_gradient(coords, constraint_pos, k, atom_idx, 1)
num_grad_z = numerical_gradient(coords, constraint_pos, k, atom_idx, 2)
num_grad = np.array([num_grad_x, num_grad_y, num_grad_z])
print(f"Numerical gradient: {num_grad}")
print(f"Difference: {analytical_grad - num_grad}")
print(f"Max relative error: {np.max(np.abs((analytical_grad - num_grad) / (num_grad + 1e-10)))}")
