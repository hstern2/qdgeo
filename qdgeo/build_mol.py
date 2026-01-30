"""RDKit wrapper for QDGeo rigid-body molecule building"""

import numpy as np
from typing import Optional
from rdkit import Chem
from . import build_molecule

# Ideal bond lengths by atom pair (sorted alphabetically)
BOND_LENGTHS = {
    ("Br", "C"): 1.94,
    ("C", "C"): 1.54,
    ("C", "Cl"): 1.79,
    ("C", "F"): 1.35,
    ("C", "H"): 1.09,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "S"): 1.82,
    ("H", "N"): 1.01,
    ("H", "O"): 0.96,
    ("H", "S"): 1.34,
    ("N", "N"): 1.45,
    ("N", "O"): 1.40,
    ("O", "O"): 1.48,
    ("O", "S"): 1.43,
    ("S", "S"): 2.05,
}

# Ideal angles by hybridization
ANGLE_BY_HYBRIDIZATION = {
    Chem.HybridizationType.SP3: np.arccos(-1.0 / 3.0),  # 109.47°
    Chem.HybridizationType.SP2: np.deg2rad(120.0),
    Chem.HybridizationType.SP: np.deg2rad(180.0),
}

# Default angle for unknown hybridization
DEFAULT_ANGLE = np.arccos(-1.0 / 3.0)  # tetrahedral


def _get_bond_length(mol: Chem.Mol, bond: Chem.Bond) -> float:
    """Get ideal bond length based on atom types and bond order."""
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    sym1 = mol.GetAtomWithIdx(a1).GetSymbol()
    sym2 = mol.GetAtomWithIdx(a2).GetSymbol()
    key = tuple(sorted([sym1, sym2]))
    
    length = BOND_LENGTHS.get(key, 1.5)  # default to 1.5 Å
    
    # Adjust for bond order
    order = bond.GetBondTypeAsDouble()
    if order == 1.5:  # Aromatic
        length *= 0.92
    elif order == 2.0:  # Double
        length *= 0.87
    elif order == 3.0:  # Triple
        length *= 0.80
    
    return length


def _get_angle(mol: Chem.Mol, center_idx: int, ring_atoms: set, 
               five_ring_atoms: set, six_ring_atoms: set,
               nb1: int, nb2: int) -> float:
    """Get ideal angle for atoms nb1-center-nb2."""
    atom = mol.GetAtomWithIdx(center_idx)
    hyb = atom.GetHybridization()
    
    # Check for special ring geometries
    is_in_ring = center_idx in ring_atoms
    
    if is_in_ring:
        # Check if this specific angle is within a ring
        angle_in_5ring = (center_idx in five_ring_atoms and 
                          nb1 in five_ring_atoms and nb2 in five_ring_atoms)
        angle_in_6ring = (center_idx in six_ring_atoms and 
                          nb1 in six_ring_atoms and nb2 in six_ring_atoms)
        
        if angle_in_5ring:
            return np.deg2rad(108.0)  # internal angle of regular pentagon
        if angle_in_6ring:
            return np.deg2rad(120.0)  # internal angle of regular hexagon
    
    # Use hybridization-based angle
    return ANGLE_BY_HYBRIDIZATION.get(hyb, DEFAULT_ANGLE)


def _identify_rotatable_torsions(mol: Chem.Mol) -> list:
    """Identify all torsions around rotatable bonds.
    
    Returns list of (a1, a2, a3, a4) tuples where a2-a3 is the central bond.
    """
    torsions = []
    
    for bond in mol.GetBonds():
        # Skip ring bonds (non-rotatable) and terminal bonds
        if bond.IsInRing():
            continue
        
        a2 = bond.GetBeginAtomIdx()
        a3 = bond.GetEndAtomIdx()
        
        # Get neighbors of a2 (excluding a3)
        nbs_a2 = [n.GetIdx() for n in mol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a3]
        # Get neighbors of a3 (excluding a2)
        nbs_a3 = [n.GetIdx() for n in mol.GetAtomWithIdx(a3).GetNeighbors() if n.GetIdx() != a2]
        
        if not nbs_a2 or not nbs_a3:
            continue  # Terminal bond
        
        # Pick first neighbor on each side for the canonical torsion
        a1 = nbs_a2[0]
        a4 = nbs_a3[0]
        
        torsions.append((a1, a2, a3, a4))
    
    return torsions


def build_mol(mol: Chem.Mol, 
              torsions: Optional[dict[tuple[int, int, int, int], float]] = None,
              verbose: int = 0) -> np.ndarray:
    """Build molecular geometry using rigid-body construction.
    
    This function constructs 3D coordinates for a molecule using ideal bond
    lengths and angles, with specified torsion angles. This is much faster
    than force-field optimization and produces deterministic results.
    
    Args:
        mol: RDKit molecule object (can have implicit or explicit hydrogens)
        torsions: Optional dictionary mapping (i, j, k, l) atom index tuples
                  to torsion angles in degrees. The torsion is defined as the
                  dihedral angle i-j-k-l where j-k is the central bond.
                  Unspecified torsions default to 180° (anti/staggered).
        verbose: Verbosity level (0=silent, 1=info)
    
    Returns:
        numpy array of shape (n_atoms, 3) with Cartesian coordinates in Angstroms
    
    Example:
        >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
        >>> # Build with default (anti) torsions
        >>> coords = build_mol(mol)
        >>> # Build with gauche conformation
        >>> c_atoms = [i for i in range(mol.GetNumAtoms()) 
        ...            if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        >>> coords = build_mol(mol, torsions={(c_atoms[0], c_atoms[1], 
        ...                                     c_atoms[2], c_atoms[3]): 60.0})
    """
    n = mol.GetNumAtoms()
    
    # Get ring information
    ring_info = mol.GetRingInfo()
    all_rings = list(ring_info.AtomRings())
    ring_atoms = set(a for ring in all_rings for a in ring)
    five_ring_atoms = set(a for ring in all_rings if len(ring) == 5 for a in ring)
    six_ring_atoms = set(a for ring in all_rings if len(ring) == 6 for a in ring)
    
    # Build bonds list
    bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = _get_bond_length(mol, bond)
        bonds.append((a1, a2, length))
    
    if verbose > 0:
        print(f"Building molecule: {n} atoms, {len(bonds)} bonds")
    
    # Build angles list
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        
        if len(neighbors) < 2:
            continue
        
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                angle_val = _get_angle(mol, i, ring_atoms, five_ring_atoms,
                                       six_ring_atoms, neighbors[j], neighbors[k])
                angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    # Build torsions list
    torsion_list = []
    if torsions:
        for (a1, a2, a3, a4), angle_deg in torsions.items():
            torsion_list.append((a1, a2, a3, a4, np.deg2rad(angle_deg)))
    
    if verbose > 0:
        print(f"  Angles: {len(angles)}, User torsions: {len(torsion_list)}")
    
    # Build rings list
    rings_list = [list(ring) for ring in all_rings]
    
    if verbose > 0 and rings_list:
        print(f"  Rings: {len(rings_list)} ({[len(r) for r in rings_list]})")
    
    # Call C++ builder
    coords = build_molecule(
        n_atoms=n,
        bonds=bonds,
        angles=angles,
        torsions=torsion_list,
        rings=rings_list
    )
    
    if verbose > 0:
        print(f"  Built coordinates: {coords.shape}")
    
    return coords
