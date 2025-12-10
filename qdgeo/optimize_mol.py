"""RDKit wrapper for QDGeo optimization"""

import numpy as np
from typing import Optional
from rdkit import Chem
from . import optimize

BOND_LENGTHS = {
    ("C", "C"): 1.54, ("C", "H"): 1.09, ("C", "O"): 1.43,
    ("C", "N"): 1.47, ("O", "H"): 0.96, ("N", "H"): 1.01,
    ("O", "O"): 1.48, ("N", "N"): 1.45,
    ("C", "Br"): 1.94, ("C", "Cl"): 1.79,
}

ANGLE = {
    Chem.HybridizationType.SP3: np.arccos(-1.0/3.0),
    Chem.HybridizationType.SP2: np.deg2rad(120.0),
    Chem.HybridizationType.SP: np.deg2rad(180.0),
}


def _get_bond_length(mol, bond):
    """Get ideal bond length from bond type and atom symbols."""
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    key = tuple(sorted([mol.GetAtomWithIdx(a1).GetSymbol(), 
                       mol.GetAtomWithIdx(a2).GetSymbol()]))
    length = BOND_LENGTHS.get(key, 1.5)
    order = bond.GetBondTypeAsDouble()
    if order == 2.0:
        length *= 0.9
    elif order == 3.0:
        length *= 0.85
    return length


def _get_dihedral_atoms(mol, j, k):
    """Get first valid neighbor pair for dihedral (i, j, k, l)."""
    atom_j = mol.GetAtomWithIdx(j)
    atom_k = mol.GetAtomWithIdx(k)
    neighbors_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != k]
    neighbors_k = [n.GetIdx() for n in atom_k.GetNeighbors() if n.GetIdx() != j]
    if neighbors_j and neighbors_k:
        return neighbors_j[0], neighbors_k[0]
    return None, None


def optimize_mol(mol, bond_k=1.5, angle_k=2.0, tolerance=1e-6, maxeval=5000, verbose=0,
                 dihedral: Optional[dict[tuple[int, int, int, int], float]] = None,
                 dihedral_k=5.0, repulsion_k=0.0, repulsion_cutoff=3.0, n_starts=10):
    """Optimize molecular geometry using QDGeo.
    
    Args:
        mol: RDKit molecule object
        bond_k: Bond force constant (default: 1.5)
        angle_k: Angle force constant (default: 2.0)
        tolerance: Convergence tolerance (default: 1e-6)
        maxeval: Maximum function evaluations (default: 5000)
        verbose: If > 0, prints bond lengths and angles (default: 0)
        dihedral: Optional dictionary of dihedral angle constraints.
                  Keys are tuples of 4 atom indices (i, j, k, l), values are target angles in degrees.
        dihedral_k: Dihedral force constant (default: 5.0)
        repulsion_k: Non-bonded repulsion force constant (default: 0.0, disabled)
        repulsion_cutoff: Repulsion cutoff distance in Angstroms (default: 3.0)
        n_starts: Number of random starting points to try (default: 10)
    
    Returns:
        Optimized coordinates array, shape (n_atoms, 3)
    """
    n = mol.GetNumAtoms()
    
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _get_bond_length(mol, bond))
             for bond in mol.GetBonds()]
    
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            angle_val = ANGLE.get(atom.GetHybridization(), ANGLE[Chem.HybridizationType.SP3])
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    explicit_dihedrals = {}
    constrained_bonds = set()
    if dihedral is not None:
        for (i, j, k, l), angle_deg in dihedral.items():
            if not mol.GetBondBetweenAtoms(j, k).IsInRing():
                explicit_dihedrals[(i, j, k, l)] = np.deg2rad(angle_deg)
                constrained_bonds.add((min(j, k), max(j, k)))
    
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_key = (min(j, k), max(j, k))
        if bond_key in constrained_bonds:
            continue
        
        if bond.IsInRing():
            if bond.GetIsAromatic():
                i, l = _get_dihedral_atoms(mol, j, k)
                if i is not None and (i, j, k, l) not in explicit_dihedrals:
                    explicit_dihedrals[(i, j, k, l)] = 0.0
            continue
        
        i, l = _get_dihedral_atoms(mol, j, k)
        if i is not None and (i, j, k, l) not in explicit_dihedrals:
            explicit_dihedrals[(i, j, k, l)] = np.deg2rad(180.0)
    
    dihedrals = [(i, j, k, l, phi) for (i, j, k, l), phi in explicit_dihedrals.items()]
    
    if verbose > 0:
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unnamed'
        print(f"\nMolecule: {name}")
        print(f"Atoms: {n}, Bonds: {len(bonds)}, Angles: {len(angles)}, Dihedrals: {len(dihedrals)}\n")
        print("Bond lengths:")
        for a1, a2, length in bonds:
            sym1, sym2 = mol.GetAtomWithIdx(a1).GetSymbol(), mol.GetAtomWithIdx(a2).GetSymbol()
            print(f"  {sym1}-{sym2} ({a1}, {a2}): {length:.4f} Å")
        print("\nAngles:")
        for a1, a2, a3, angle in angles:
            sym1 = mol.GetAtomWithIdx(a1).GetSymbol()
            sym2 = mol.GetAtomWithIdx(a2).GetSymbol()
            sym3 = mol.GetAtomWithIdx(a3).GetSymbol()
            print(f"  {sym1}-{sym2}-{sym3} ({a1}, {a2}, {a3}): {np.rad2deg(angle):.2f}°")
        if dihedrals:
            print("\nDihedrals:")
            for i, j, k, l, angle in dihedrals:
                syms = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in [i, j, k, l]]
                print(f"  {'-'.join(syms)} ({i}, {j}, {k}, {l}): {np.rad2deg(angle):.2f}°")
        print()
    
    coords, converged, energy = optimize(
        n_atoms=n, bonds=bonds, angles=angles, dihedrals=dihedrals,
        bond_force_constant=bond_k, angle_force_constant=angle_k,
        dihedral_force_constant=dihedral_k,
        repulsion_force_constant=repulsion_k, repulsion_cutoff=repulsion_cutoff,
        tolerance=tolerance, maxeval=maxeval, verbose=verbose, n_starts=n_starts
    )
    
    if verbose > 0:
        print(f"Optimization converged: {converged}, Final energy: {energy:.6e}\n")
    
    return coords
