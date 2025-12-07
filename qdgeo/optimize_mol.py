"""RDKit wrapper for QDGeo optimization"""

import numpy as np
from rdkit import Chem
from . import optimize

# Bond lengths (Angstroms)
BOND_LENGTHS = {
    ("C", "C"): 1.54, ("C", "H"): 1.09, ("C", "O"): 1.43,
    ("C", "N"): 1.47, ("O", "H"): 0.96, ("N", "H"): 1.01,
    ("O", "O"): 1.48, ("N", "N"): 1.45,
}

# Angle defaults by hybridization (SP3 = tetrahedral angle = arccos(-1/3) ≈ 109.471°)
ANGLE = {
    Chem.HybridizationType.SP3: np.arccos(-1.0/3.0),  # Exact tetrahedral angle
    Chem.HybridizationType.SP2: np.deg2rad(120.0),
    Chem.HybridizationType.SP: np.deg2rad(180.0),
}


def optimize_mol(mol, bond_k=1.5, angle_k=2.0, tolerance=1e-6, maxeval=5000, verbose=0):
    """Optimize molecular geometry using QDGeo.
    
    Args:
        mol: RDKit molecule object
        bond_k: Bond force constant (default: 1.5)
        angle_k: Angle force constant (default: 2.0)
        tolerance: Convergence tolerance (default: 1e-6)
        maxeval: Maximum function evaluations (default: 5000)
        verbose: If > 0, prints bond lengths and angles (default: 0)
    
    Returns:
        Optimized coordinates array, shape (n_atoms, 3)
    """
    n = mol.GetNumAtoms()
    
    # Extract bonds
    bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = tuple(sorted([mol.GetAtomWithIdx(a1).GetSymbol(), 
                           mol.GetAtomWithIdx(a2).GetSymbol()]))
        length = BOND_LENGTHS.get(key, 1.5)
        order = bond.GetBondTypeAsDouble()
        if order == 2.0:
            length *= 0.9
        elif order == 3.0:
            length *= 0.85
        bonds.append((a1, a2, length))
    
    # Extract angles
    # For each atom with 2+ neighbors, generate angles between all pairs of neighbors
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue  # Skip atoms with fewer than 2 neighbors (no angles to define)
        angle_val = ANGLE.get(atom.GetHybridization(), ANGLE[Chem.HybridizationType.SP3])
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    if verbose > 0:
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unnamed'
        print(f"\nMolecule: {name}")
        print(f"Atoms: {n}, Bonds: {len(bonds)}, Angles: {len(angles)}\n")
        print("Bond lengths (atom1, atom2, length):")
        for a1, a2, length in bonds:
            print(f"  ({a1}, {a2}): {mol.GetAtomWithIdx(a1).GetSymbol()}-"
                  f"{mol.GetAtomWithIdx(a2).GetSymbol()} = {length:.4f} Å")
        print(f"\nAngles (atom1, atom2, atom3, angle):")
        for a1, a2, a3, angle in angles:
            print(f"  ({a1}, {a2}, {a3}): {mol.GetAtomWithIdx(a1).GetSymbol()}-"
                  f"{mol.GetAtomWithIdx(a2).GetSymbol()}-"
                  f"{mol.GetAtomWithIdx(a3).GetSymbol()} = {np.rad2deg(angle):.2f}°")
        print()
    
    coords, converged, energy = optimize(
        n_atoms=n, bonds=bonds, angles=angles,
        bond_force_constant=bond_k, angle_force_constant=angle_k,
        tolerance=tolerance, maxeval=maxeval, verbose=verbose
    )
    
    if verbose > 0:
        print(f"Optimization converged: {converged}, Final energy: {energy:.6e}\n")
    
    return coords
